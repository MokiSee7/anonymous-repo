"""
exp3_alignment: Joint summary generation + frame selection with InternVideo2.5-Chat-8B.

Input:  All sampled frames labeled by shot ("Shot 1: [frames], Shot 2: [frames], ...")
        Frames are globally numbered 1..M across all shots.
        Each shot's frames are padded to a multiple of LOCAL_NUM_FRAMES=4 independently.
Task:   In a single call, the model:
          1. Generates a video summary with exactly N sentences (N = len(aligned_text_summary))
          2. Selects ceil(total_frames * 0.15) most important individual frames

Evaluation:
  A. Summary quality (exp1 metrics):
       ROUGE-1/2/L, BERTScore F1, BLEU-4, METEOR, CIDEr (corpus-level, post-hoc)

  B. Frame selection quality (exp2 metrics):
       F1, Kendall τ, Spearman ρ between model-selected frames and shot_level_gt frame set

  C. Cross-modal alignment:
       CLIPScore between generated summary sentences and selected frames
       CLIPScore = 2.5 * max(cos(mean_text_emb, mean_visual_emb), 0)

Optional (--with_doc): prepend aligned_full_document as plain text context.
  Ablation: measures how much textual grounding helps joint summarization + shot selection.

Note: <image> token count per shot in the prompt matches the padded frame count;
      text describes original (unpadded) frame counts.

Environment: internvideo
Usage:
    python run_internvideo.py --dataset summe
    python run_internvideo.py --dataset summe --with_doc
    python run_internvideo.py --dataset summe --video video_1  # smoke test
    python run_internvideo.py --dataset summe --max_frames_per_shot 4
"""

import sys
import json
import math
import argparse
import re
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_shots, load_aligned_document, DATASET_CONFIG, PROCESSED_DATA
from shared.evaluate_sets import set_f1
from shared.evaluate import compute_rank_correlations, get_gt_frame_indices
from shared.evaluate_text import compute_text_metrics
from shared.evaluate_clip import compute_clipscore

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID         = "OpenGVLab/InternVideo2_5_Chat_8B"
MODEL_DISPLAY    = "InternVideo2.5-Chat-8B"
HF_CACHE_DIR     = "/data/hf-cache/hub"
LOCAL_NUM_FRAMES = 4    # hardcoded in modeling_internvl_chat_hico2.py
MAX_NEW_TOKENS   = 1024
DEFAULT_MAX_FPS  = 4   # max frames per shot submitted to the model
INPUT_SIZE       = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _select_frames(frames, max_n, strategy):
    """Select up to max_n frames using the given strategy."""
    if max_n is None or len(frames) <= max_n:
        return frames
    if strategy == "truncate":
        return frames[:max_n]
    # default: uniform
    indices = np.linspace(0, len(frames) - 1, max_n, dtype=int)
    return [frames[i] for i in indices]


# ── Image preprocessing ───────────────────────────────────────────────────────
_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def pad_frames_to_multiple(frames):
    """Pad frame list to multiple of LOCAL_NUM_FRAMES by repeating last frame."""
    remainder = len(frames) % LOCAL_NUM_FRAMES
    if remainder:
        frames = frames + [frames[-1]] * (LOCAL_NUM_FRAMES - remainder)
    return frames


def preprocess_all_shots(shots, max_frames_per_shot, frame_strategy="uniform"):
    """
    Subsample each shot's frames, pad each shot independently to LOCAL_NUM_FRAMES multiple,
    stack all into one tensor.

    Returns:
        pixel_values:      (total_padded_frames, C, H, W) float tensor
        num_patches_list:  list of 1s, one per frame (for model.chat)
        n_orig_per_shot:   list of original (pre-pad) frame counts per shot
        n_padded_per_shot: list of padded frame counts per shot
    """
    all_tensors = []
    num_patches_list = []
    n_orig_per_shot = []
    n_padded_per_shot = []

    for shot in shots:
        frames = shot["frames"]
        # Subsample if needed
        frames = _select_frames(frames, max_frames_per_shot, frame_strategy)

        n_orig = len(frames)
        frames_padded = pad_frames_to_multiple(frames)
        n_padded = len(frames_padded)

        n_orig_per_shot.append(n_orig)
        n_padded_per_shot.append(n_padded)

        for frame in frames_padded:
            all_tensors.append(_transform(frame))
            num_patches_list.append(1)

    pixel_values = torch.stack(all_tensors)
    return pixel_values, num_patches_list, n_orig_per_shot, n_padded_per_shot


# ── Data loading ──────────────────────────────────────────────────────────────

# In-memory cache for processed JSONs
_processed_cache: dict = {}

def _load_processed(dataset_name):
    if dataset_name not in _processed_cache:
        path = PROCESSED_DATA.get(dataset_name)
        try:
            with open(path) as f:
                _processed_cache[dataset_name] = json.load(f)
        except Exception:
            _processed_cache[dataset_name] = {}
    return _processed_cache[dataset_name]


def load_exp3_data(dataset_name, video_id):
    """
    Load aligned_text_summary for a video.

    Returns:
        summary_sentences: list[str]  — GT summary sentences
        summary_raw_text:  str | None
    """
    data = _load_processed(dataset_name).get(video_id, {})
    summary = data.get("aligned_text_summary", {})
    return summary.get("sentences", []), summary.get("raw_text")


# ── CLIPScore: get representative frames from selected shots ──────────────────

# ── Prompt construction ───────────────────────────────────────────────────────

def _doc_prefix(doc_text):
    if not doc_text:
        return ""
    return (
        "The following is a detailed shot-by-shot description of this video for context:\n"
        f"{doc_text}\n\n"
    )


def build_prompt(shots, n_summary_sentences, n_select_frames, n_padded_per_shot, doc_text=None,
                 n_total_sampled=None, n_select_target=None):
    """
    Build prompt with globally-numbered frames (Frame 1, Frame 2, ...).
    The model selects individual frame numbers, not shot numbers.

    n_padded_per_shot: padded frame counts per shot (determines <image> token count).

    Returns:
        prompt_str:   the full prompt string
        frame_images: flat list of PIL images in submission order (orig, unpadded)
    """
    total_frames = sum(len(s["frames"]) for s in shots)
    parts = [f"Below are {total_frames} frames from a video, grouped by shot:\n"]
    frame_images = []
    global_idx = 1

    for shot, n_padded in zip(shots, n_padded_per_shot):
        orig_frames = shot["frames"]
        n_orig = len(orig_frames)
        if n_orig == 0:
            continue
        end_idx = global_idx + n_orig - 1
        label = (f"Frame {global_idx}" if n_orig == 1
                 else f"Frames {global_idx}–{end_idx}")
        parts.append(f"\nShot {shot['shot_idx'] + 1} ({label}):\n")
        parts.append("\n".join(["<image>"] * n_padded))
        frame_images.extend(orig_frames)
        global_idx += n_orig

    frame_list = ", ".join(f"Frame {i + 1}" for i in range(total_frames))
    budget_note = ""
    if n_total_sampled is not None and n_total_sampled != total_frames:
        budget_note = (
            f"The full sampled video contains {n_total_sampled} frames in total. "
            f"Only {total_frames} frames are shown here due to input limits. "
            f"The original 15% target is {n_select_target} frames, so select the best "
            f"{n_select_frames} frames from the submitted set.\n\n"
        )

    parts.append(
        "\n" + _doc_prefix(doc_text)
        + budget_note
        + f"Based on all {total_frames} frames above, complete TWO tasks:\n\n"
        f"TASK 1 — Write a summary of this video in EXACTLY {n_summary_sentences} sentences.\n"
        f"TASK 2 — Select EXACTLY {n_select_frames} most important frames (no more, no less) from: {frame_list}.\n\n"
        "Use EXACTLY this format (no extra text):\n"
        "SUMMARY:\n"
        + "\n".join(f"{i+1}. [sentence {i+1}]" for i in range(n_summary_sentences))
        + "\n\nIMPORTANT FRAMES: [Frame X, Frame Y, ...]"
    )
    return "\n".join(parts), frame_images


# ── Output parsing ────────────────────────────────────────────────────────────

def parse_output(text, n_summary_sentences, n_frames):
    """
    Parse SUMMARY sentences and IMPORTANT FRAMES from model output.

    Returns:
        summary_sentences: list[str]
        selected_frames:   list of 1-indexed frame positions in submitted list
        parse_ok:          bool
    """
    summary_sentences = []
    selected_frames = []
    parse_ok = True

    summ_match = re.search(r"SUMMARY\s*:\s*\n(.*?)(?=IMPORTANT FRAMES|$)", text,
                           re.DOTALL | re.IGNORECASE)
    if summ_match:
        block = summ_match.group(1).strip()
        lines = re.findall(r"^\d+\.\s*(.+?)$", block, re.MULTILINE)
        summary_sentences = [l.strip() for l in lines if l.strip()]
        if not summary_sentences:
            summary_sentences = [l.strip() for l in block.splitlines() if l.strip()]
    if not summary_sentences:
        parse_ok = False

    frames_match = re.search(r"IMPORTANT FRAMES\s*:\s*\[?([^\]\n]+)\]?",
                             text, re.IGNORECASE)
    if frames_match:
        raw = frames_match.group(1)
        nums = re.findall(r"\d+", raw)
        seen = set()
        for n in nums:
            pos = int(n)  # 1-indexed
            if 1 <= pos <= n_frames and pos not in seen:
                selected_frames.append(pos)
                seen.add(pos)
    if not selected_frames:
        parse_ok = False

    return summary_sentences, selected_frames, parse_ok


# ── Map selected frame positions → shot indices (for rank correlation) ─────────

def _frames_to_shots(selected_frames_1idx, shots):
    """Map 1-indexed frame positions to 0-indexed shot indices (any frame → shot included)."""
    frame_to_shot = {}
    pos = 1
    for shot in shots:
        for _ in shot["frames"]:
            frame_to_shot[pos] = shot["shot_idx"]
            pos += 1
    shot_set = set()
    for f in selected_frames_1idx:
        if f in frame_to_shot:
            shot_set.add(frame_to_shot[f])
    return sorted(shot_set)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {MODEL_DISPLAY} ...")
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    model.eval()
    print(f"  Loaded on: {next(model.parameters()).device}")
    return model, tokenizer


# ── Single inference call ─────────────────────────────────────────────────────

def _infer(model, tokenizer, shots, n_summary_sentences, n_select_frames,
           max_frames_per_shot, doc_text, frame_strategy="uniform",
           n_total_sampled=None, n_select_target=None):
    """
    Returns out dict with keys: text, n_output_tokens, was_output_truncated, error, frame_images.
    frame_images: flat list of PIL images in submission order (orig, unpadded) — for CLIPScore.
    """
    out = {"text": "", "n_output_tokens": None,
           "was_output_truncated": False, "error": None, "frame_images": []}
    try:
        pixel_values, num_patches_list, n_orig_per_shot, n_padded_per_shot = \
            preprocess_all_shots(shots, max_frames_per_shot, frame_strategy)
        pixel_values = pixel_values.to(dtype=torch.bfloat16, device=model.device)

        prompt, frame_images = build_prompt(
            shots, n_summary_sentences, n_select_frames, n_padded_per_shot, doc_text,
            n_total_sampled=n_total_sampled, n_select_target=n_select_target
        )
        out["frame_images"] = frame_images
        generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "repetition_penalty": 1.5}

        with torch.no_grad():
            response = model.chat(
                tokenizer, pixel_values, prompt, generation_config,
                num_patches_list=num_patches_list,
                history=None, return_history=False,
            )

        out["n_output_tokens"]      = len(tokenizer.encode(response))
        out["was_output_truncated"] = (out["n_output_tokens"] >= MAX_NEW_TOKENS - 2)
        out["text"] = response

    except Exception as e:
        out["error"] = str(e)
    return out


# ── Per-video inference ───────────────────────────────────────────────────────

def _make_empty_result(video_id, meta, n_shots, n_select, r, gt_shots, n_summ_sents,
                       total_frames=None, original_total_frames=None, was_frame_truncated=False,
                       n_select_target=None):
    null_eval = {"precision": None, "recall": None, "f1": None, "kendall_tau": None, "spearman_rho": None}
    return {
        "video_id":            video_id,
        "video_name":          meta["video_name"],
        "n_shots":             n_shots,
        "n_total_sampled":     original_total_frames,
        "n_original_frames":   original_total_frames,
        "total_frames_submitted": total_frames,
        "was_frame_truncated": was_frame_truncated,
        "n_select_target":     n_select_target,
        "n_select":            n_select,
        "n_output_tokens":     r["n_output_tokens"],
        "was_output_truncated": r["was_output_truncated"],
        "raw_output":          r["text"],
        "error":               r["error"],
        "parse_ok":            False,
        "generated_sentences": [],
        "selected_frames":     [],
        "gt_shots":            gt_shots,
        "eval_a_summary":        compute_text_metrics("", None),
        "eval_b_frame_selection": null_eval,
        "eval_c_alignment":      {"clip_score": None},
    }


def run_video(model, tokenizer, dataset_name, video_id, max_frames_per_shot,
             with_doc=False, frame_strategy="uniform", max_total_frames=None):
    meta, shots = load_all_shots(dataset_name, video_id)
    summ_sents, summ_raw = load_exp3_data(dataset_name, video_id)

    n_shots      = len(shots)
    gt_shots     = sorted(np.where(meta["shot_level_gt"] == 1)[0].tolist())
    n_summ_sents = len(summ_sents) if summ_sents else 3  # fallback to 3

    # Record original frame count before any per-shot truncation
    original_total_frames = sum(len(s["frames"]) for s in shots)

    # Per-shot frame selection via _select_frames (uniform subsampling up to max_frames_per_shot)
    processed_shots = []
    for shot in shots:
        frames = _select_frames(shot["frames"], max_frames_per_shot, frame_strategy)
        ns = dict(shot)
        ns["frames"] = frames
        processed_shots.append(ns)

    # Global budget across all shots
    if max_total_frames is not None:
        from shared.frame_utils import apply_global_shot_strategy
        processed_shots = apply_global_shot_strategy(processed_shots, max_total_frames,
                                                      frame_strategy if frame_strategy else "uniform")

    total_frames = sum(len(s["frames"]) for s in processed_shots)
    was_frame_truncated = (total_frames < original_total_frames)
    n_total_sampled = original_total_frames
    n_select_target = max(1, math.ceil(n_total_sampled * 0.15))
    n_select = min(total_frames, n_select_target)

    # Collect submitted frame indices (original frame space) in order
    submitted_frame_indices = [fi for shot in processed_shots
                                for fi in shot.get("frame_indices", [])]

    doc_context = load_aligned_document(dataset_name, video_id) if with_doc else None
    print(f"  Shots: {n_shots}  |  total_sampled: {n_total_sampled}  |  submitted: {total_frames}  "
          f"|  n_select_target: {n_select_target}  |  n_select: {n_select}  |  "
          f"gt_shots: {gt_shots}  |  doc: {'yes' if doc_context else 'no'}")
    print(f"  Summary sentences: {n_summ_sents}")

    r = _infer(
        model, tokenizer, processed_shots, n_summ_sents, n_select,
        max_frames_per_shot, doc_context, frame_strategy,
        n_total_sampled=n_total_sampled, n_select_target=n_select_target
    )

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return _make_empty_result(video_id, meta, n_shots, n_select, r,
                                  gt_shots, n_summ_sents, total_frames=total_frames,
                                  original_total_frames=original_total_frames,
                                  was_frame_truncated=was_frame_truncated,
                                  n_select_target=n_select_target)

    trunc = " [TRUNCATED]" if r["was_output_truncated"] else ""
    print(f"  out={r['n_output_tokens']}{trunc}")

    gen_sents, selected_frames, parse_ok = parse_output(r["text"], n_summ_sents, total_frames)
    print(f"  parse_ok={parse_ok}  |  gen_sents={len(gen_sents)}  |  selected_frames={selected_frames[:10]}")

    # ── Eval A: summary quality ───────────────────────────────────────────────
    gen_summary_text = " ".join(gen_sents)
    eval_a = compute_text_metrics(gen_summary_text, summ_raw)

    # ── Eval B: frame-level selection quality ─────────────────────────────────
    # pred_frame_set: original frame indices of selected frames
    pred_frame_set = {submitted_frame_indices[f - 1]
                      for f in selected_frames if 0 < f <= len(submitted_frame_indices)}
    gt_frame_set   = get_gt_frame_indices(meta["shot_level_gt"],
                                          meta["n_sampled_per_shot"], meta["picks"])
    eval_b = set_f1(pred_frame_set, gt_frame_set)
    # Rank correlations at shot level (frames mapped back to shots)
    predicted_shot_indices = _frames_to_shots(selected_frames, processed_shots)
    gt_shot_scores = [shot["gt_score"] for shot in shots]
    eval_b.update(compute_rank_correlations(predicted_shot_indices, n_shots, gt_shot_scores))
    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    print(f"  [B] frame F1={_fmt(eval_b['f1'])}  "
          f"τ={_fmt(eval_b['kendall_tau'])}  ρ={_fmt(eval_b['spearman_rho'])}")

    # ── Eval C: cross-modal alignment (CLIPScore) ──────────────────────────────
    frame_images = r["frame_images"]
    selected_images = [frame_images[f - 1]
                       for f in selected_frames if 0 < f <= len(frame_images)]
    clip_score = compute_clipscore(gen_sents, selected_images, device="cpu")
    eval_c = {"clip_score": clip_score}
    print(f"  [C] clip_score={clip_score}")

    return {
        "video_id":            video_id,
        "video_name":          meta["video_name"],
        "n_shots":             n_shots,
        "n_total_sampled":     n_total_sampled,
        "n_original_frames":   original_total_frames,
        "total_frames_submitted": total_frames,
        "was_frame_truncated": was_frame_truncated,
        "n_select_target":     n_select_target,
        "n_select":            n_select,
        "n_output_tokens":     r["n_output_tokens"],
        "was_output_truncated": r["was_output_truncated"],
        "raw_output":          r["text"],
        "error":               None,
        "parse_ok":            parse_ok,
        "generated_sentences": gen_sents,
        "selected_frames":     selected_frames,    # 1-indexed positions in submitted list
        "gt_shots":            gt_shots,           # 0-indexed, from shot_level_gt
        "eval_a_summary":        eval_a,
        "eval_b_frame_selection": eval_b,
        "eval_c_alignment":      eval_c,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",            default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--max_frames_per_shot", type=int, default=None,
                        help="Max frames per shot submitted to the model (default: None = all frames)")
    parser.add_argument("--max_total_frames",   type=int, default=None,
                        help="Global frame budget across all shots (default: None = no limit)")
    parser.add_argument("--with_doc",           action="store_true",
                        help="Prepend aligned_full_document as plain text context (ablation)")
    parser.add_argument("--video",              default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file",     default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    parser.add_argument("--frame_strategy", choices=["truncate", "uniform"], default="uniform",
                        help="Per-shot frame selection: truncate=first N, uniform=evenly-spaced N (default).")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_suffix = "_doc" if args.with_doc else ""
    strategy_suffix = "_truncate" if args.frame_strategy == "truncate" else ""
    subset_suffix = ""
    if args.video_ids_file:
        import os as _os
        subset_suffix = f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file   = results_dir / f"internvideo_{args.dataset}{doc_suffix}{strategy_suffix}{subset_suffix}_{timestamp}.json"

    model, tokenizer = load_model()

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_DISPLAY} | dataset={args.dataset} "
          f"| max_fps={args.max_frames_per_shot} | max_total={args.max_total_frames} "
          f"| frame_strategy={args.frame_strategy} ({len(video_ids)} videos)\n")

    all_results = []
    for i, vid in enumerate(video_ids):
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        result = None
        last_exc = None
        for attempt in range(1, 4):  # up to 3 attempts
            try:
                result = run_video(model, tokenizer, args.dataset, vid,
                                   args.max_frames_per_shot, args.with_doc, args.frame_strategy,
                                   max_total_frames=args.max_total_frames)
                break  # success
            except Exception as e:
                import traceback, gc
                last_exc = e
                print(f"  FATAL (attempt {attempt}/3): {e}")
                traceback.print_exc()
                # Clear CUDA memory fragmentation before retry
                gc.collect()
                torch.cuda.empty_cache()
        if result is None:
            null_eval = {"precision": None, "recall": None, "f1": None,
                         "kendall_tau": None, "spearman_rho": None}
            result = {
                "video_id": vid, "video_name": vid,
                "n_shots": None, "n_original_frames": None,
                "total_frames_submitted": None, "was_frame_truncated": False,
                "n_select": None, "n_output_tokens": None,
                "was_output_truncated": False, "raw_output": "",
                "error": f"FATAL (3 attempts): {last_exc}", "parse_ok": False,
                "generated_sentences": [], "selected_frames": [], "gt_shots": [],
                "eval_a_summary": compute_text_metrics("", None),
                "eval_b_frame_selection": null_eval,
                "eval_c_alignment": {"clip_score": None},
            }
        all_results.append(result)

    n_success  = sum(1 for r in all_results if not r["error"])
    n_error    = sum(1 for r in all_results if r["error"])
    n_parse_ok = sum(1 for r in all_results if r["parse_ok"])

    def avg(key_path):
        vals = []
        for r in all_results:
            v = r
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                vals.append(v)
        return round(float(np.mean(vals)), 4) if vals else None

    n_frame_truncated = sum(1 for r in all_results if r.get("was_frame_truncated"))
    truncated_ids = [r["video_id"] for r in all_results if r.get("was_frame_truncated")]
    print(f"\nDone: {n_success} success, {n_error} errors, parse_ok={n_parse_ok}/{len(all_results)}")
    if n_frame_truncated:
        print(f"  Frame-truncated videos ({n_frame_truncated}): {truncated_ids}")
    print(f"  [A] Avg ROUGE-L:       {avg(['eval_a_summary', 'rougeL'])}")
    print(f"  [A] Avg BERTScore:     {avg(['eval_a_summary', 'bert_score_f1'])}")
    print(f"  [A] Avg BLEU-4:        {avg(['eval_a_summary', 'bleu4'])}")
    print(f"  [A] Avg METEOR:        {avg(['eval_a_summary', 'meteor'])}")
    print(f"  [A] Avg G-Eval:        {avg(['eval_a_summary', 'g_eval_overall'])}")
    print(f"  [B] Avg frame F1:      {avg(['eval_b_frame_selection', 'f1'])}")
    print(f"  [B] Avg Kendall τ:     {avg(['eval_b_frame_selection', 'kendall_tau'])}")
    print(f"  [B] Avg Spearman ρ:    {avg(['eval_b_frame_selection', 'spearman_rho'])}")
    print(f"  [C] Avg CLIPScore:     {avg(['eval_c_alignment', 'clip_score'])}")

    output = {
        "model":               MODEL_DISPLAY,
        "dataset":             args.dataset,
        "max_frames_per_shot": args.max_frames_per_shot,
        "frame_strategy":      args.frame_strategy,
        "with_doc":            args.with_doc,
        "timestamp":           timestamp,
        "n_success":           n_success,
        "n_error":             n_error,
        "n_parse_ok":          n_parse_ok,
        "n_frame_truncated":   n_frame_truncated,
        "truncated_video_ids": truncated_ids,
        "avg_summary_rouge1":    avg(["eval_a_summary", "rouge1"]),
        "avg_summary_rouge2":    avg(["eval_a_summary", "rouge2"]),
        "avg_summary_rougeL":    avg(["eval_a_summary", "rougeL"]),
        "avg_summary_bertscore": avg(["eval_a_summary", "bert_score_f1"]),
        "avg_summary_bleu4":     avg(["eval_a_summary", "bleu4"]),
        "avg_summary_meteor":    avg(["eval_a_summary", "meteor"]),
        "avg_summary_geval":     avg(["eval_a_summary", "g_eval_overall"]),
        "avg_frame_f1":          avg(["eval_b_frame_selection", "f1"]),
        "avg_kendall_tau":       avg(["eval_b_frame_selection", "kendall_tau"]),
        "avg_spearman_rho":      avg(["eval_b_frame_selection", "spearman_rho"]),
        "avg_clip_score":        avg(["eval_c_alignment", "clip_score"]),
        "video_results":       all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
