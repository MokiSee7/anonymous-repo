"""
exp3_alignment: Joint summary generation + shot selection with VideoLLaMA3-7B.

Input:  All sampled frames (optionally capped), with a text prompt that labels
        which frames belong to which shot.
Task:   In a single call, the model:
          1. Generates a video summary with exactly N sentences (N = len(aligned_text_summary))
          2. Selects ceil(total_frames * 0.15) most important frame numbers

Evaluation:
  A. Summary quality: ROUGE-1/2/L, BERTScore F1, BLEU-4, METEOR, CIDEr
  B. Frame selection quality: F1, Kendall τ, Spearman ρ vs shot_level_gt
  C. Cross-modal alignment: CLIPScore between summary and selected frames

Frame input strategies (--frame_strategy):
  truncate : within each shot, take the first max_frames_per_shot frames
  uniform  : within each shot, uniformly sample max_frames_per_shot frames
  (default): use all frames (up to DEFAULT_MAX_FPS per shot, uniform)

Global budget (--max_total_frames): applied after per-shot selection.

Environment: videollama3
Usage:
    python run_videollama3.py --dataset summe
    python run_videollama3.py --dataset summe --frame_strategy truncate --max_frames_per_shot 4
    python run_videollama3.py --dataset summe --max_total_frames 200
    python run_videollama3.py --dataset summe --video video_1
"""

import sys
import json
import math
import argparse
import re
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import (
    get_video_ids, load_all_shots, load_aligned_document, DATASET_CONFIG, PROCESSED_DATA,
)
from shared.evaluate_sets import set_f1
from shared.evaluate import compute_rank_correlations, get_gt_frame_indices
from shared.evaluate_text import compute_text_metrics
from shared.evaluate_clip import compute_clipscore
from shared.frame_utils import apply_per_shot_strategy, apply_global_shot_strategy

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "/data/hf-cache/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B"
MODEL_DISPLAY = "VideoLLaMA3-7B"
MAX_NEW_TOKENS   = 1024
DEFAULT_MAX_FPS  = 4


# ── Data loading ──────────────────────────────────────────────────────────────
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
    data = _load_processed(dataset_name).get(video_id, {})
    summary = data.get("aligned_text_summary", {})
    return summary.get("sentences", []), summary.get("raw_text")


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(shots, n_summary_sentences, n_select_frames,
                 n_total_sampled=None, n_select_target=None):
    """
    Build text prompt + flat frame list for VideoLLaMA3.

    Returns:
        prompt_text:  str — describes shot groupings and tasks
        all_frames:   list[PIL.Image] — flat list in submission order
        frame_images: same as all_frames (for CLIPScore)
    """
    all_frames = []
    shot_labels = []
    global_idx = 1

    for shot in shots:
        frames = shot["frames"]
        if not frames:
            continue
        end_idx = global_idx + len(frames) - 1
        label = (f"Frame {global_idx}" if len(frames) == 1
                 else f"Frames {global_idx}-{end_idx}")
        shot_labels.append(f"Shot {shot['shot_idx'] + 1} ({label})")
        all_frames.extend(frames)
        global_idx += 1 + len(frames) - 1 + (1 if len(frames) > 1 else 0)
        # correct: just advance by len(frames)
    # redo this properly
    all_frames = []
    shot_labels = []
    global_idx = 1
    for shot in shots:
        frames = shot["frames"]
        if not frames:
            continue
        end_idx = global_idx + len(frames) - 1
        if len(frames) == 1:
            label = f"Frame {global_idx}"
        else:
            label = f"Frames {global_idx}-{end_idx}"
        shot_labels.append(f"Shot {shot['shot_idx'] + 1} ({label})")
        all_frames.extend(frames)
        global_idx = end_idx + 1

    total_frames = len(all_frames)
    shots_desc = "\n".join(shot_labels)
    frame_list = ", ".join(f"Frame {i + 1}" for i in range(total_frames))

    budget_note = ""
    if n_total_sampled is not None and n_total_sampled != total_frames:
        budget_note = (
            f"The full sampled video contains {n_total_sampled} frames in total. "
            f"Only {total_frames} frames are shown here due to input limits. "
            f"The original 15% target is {n_select_target} frames, so select the best "
            f"{n_select_frames} frames from the submitted set.\n\n"
        )

    prompt_text = (
        f"You are given {total_frames} frames from a video in chronological order, "
        f"grouped by shot as follows:\n{shots_desc}\n\n"
        f"{budget_note}"
        f"Based on all {total_frames} frames, complete TWO tasks:\n\n"
        f"TASK 1 — Write a summary of this video in EXACTLY {n_summary_sentences} sentences.\n"
        f"TASK 2 — Select EXACTLY {n_select_frames} most important frames (no more, no less) "
        f"from: {frame_list}.\n\n"
        "Use EXACTLY this format (no extra text):\n"
        "SUMMARY:\n"
        + "\n".join(f"{i+1}. [sentence {i+1}]" for i in range(n_summary_sentences))
        + "\n\nIMPORTANT FRAMES: [Frame X, Frame Y, ...]"
    )
    return prompt_text, all_frames


# ── Output parsing ────────────────────────────────────────────────────────────
def parse_output(text, n_summary_sentences, n_frames):
    summary_sentences = []
    selected_frames = []
    parse_ok = True

    # Accept both the instructed multiline format and the common one-line fallback.
    summ_match = re.search(r"SUMMARY\s*:\s*(.*?)(?=IMPORTANT FRAMES|$)", text,
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
            pos = int(n)
            if 1 <= pos <= n_frames and pos not in seen:
                selected_frames.append(pos)
                seen.add(pos)
    if not selected_frames:
        parse_ok = False

    return summary_sentences, selected_frames, parse_ok


def _frames_to_shots(selected_frames_1idx, shots):
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
    from transformers import AutoModelForCausalLM, AutoProcessor
    print(f"Loading {MODEL_DISPLAY} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    print(f"  Loaded on: {next(model.parameters()).device}")
    return model, processor


# ── Single inference call ─────────────────────────────────────────────────────
def _infer(model, processor, frames, text):
    out = {"text": "", "n_output_tokens": None, "was_output_truncated": False, "error": None}
    try:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "num_frames": len(frames)},
                {"type": "text", "text": text},
            ]},
        ]
        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated = output_ids  # generate() returns only new tokens
        out["n_output_tokens"]      = int(generated.shape[-1])
        out["was_output_truncated"] = (out["n_output_tokens"] >= MAX_NEW_TOKENS - 2)
        out["text"] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    except Exception as e:
        out["error"] = str(e)
    return out


# ── Per-video inference ───────────────────────────────────────────────────────
def run_video(model, processor, dataset_name, video_id,
              frame_strategy=None, max_frames_per_shot=DEFAULT_MAX_FPS,
              max_total_frames=None, test_mode=False):
    meta, shots = load_all_shots(dataset_name, video_id)
    summ_sents, summ_raw = load_exp3_data(dataset_name, video_id)

    n_shots      = len(shots)
    gt_shots     = sorted(np.where(meta["shot_level_gt"] == 1)[0].tolist())
    n_summ_sents = len(summ_sents) if summ_sents else 3

    original_total_frames = sum(len(s["frames"]) for s in shots)

    if frame_strategy is not None:
        shots = apply_per_shot_strategy(shots, max_frames_per_shot, frame_strategy)
    if max_total_frames is not None:
        strategy_for_global = frame_strategy if frame_strategy else "uniform"
        shots = apply_global_shot_strategy(shots, max_total_frames, strategy_for_global)

    total_frames = sum(len(s["frames"]) for s in shots)
    was_frame_truncated = (total_frames < original_total_frames)
    n_total_sampled = original_total_frames
    n_select_target = max(1, math.ceil(n_total_sampled * 0.15))
    n_select = min(total_frames, n_select_target)

    submitted_frame_indices = [fi for shot in shots for fi in shot.get("frame_indices", [])]

    print(f"  Shots: {n_shots}  |  total_sampled: {n_total_sampled}  |  submitted: {total_frames}  "
          f"|  n_select_target: {n_select_target}  |  n_select: {n_select}  |  "
          f"gt_shots: {gt_shots}")
    print(f"  Summary sentences: {n_summ_sents}")

    prompt_text, all_frames = build_prompt(
        shots, n_summ_sents, n_select,
        n_total_sampled=n_total_sampled, n_select_target=n_select_target
    )
    r = _infer(model, processor, all_frames, prompt_text)

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return _make_empty_result(video_id, meta, n_shots, n_select, r, gt_shots, n_summ_sents,
                                  frame_strategy=frame_strategy,
                                  max_frames_per_shot=max_frames_per_shot,
                                  max_total_frames=max_total_frames,
                                  total_frames=total_frames,
                                  original_total_frames=original_total_frames,
                                  was_frame_truncated=was_frame_truncated,
                                  n_select_target=n_select_target)

    trunc = " [TRUNCATED]" if r["was_output_truncated"] else ""
    print(f"  out={r['n_output_tokens']}{trunc}")

    gen_sents, selected_frames, parse_ok = parse_output(r["text"], n_summ_sents, total_frames)
    print(f"  parse_ok={parse_ok}  |  gen_sents={len(gen_sents)}  |  selected_frames={selected_frames[:10]}")

    gen_summary_text = " ".join(gen_sents)
    eval_a = compute_text_metrics(gen_summary_text, summ_raw)

    pred_frame_set = {submitted_frame_indices[f - 1]
                      for f in selected_frames if 0 < f <= len(submitted_frame_indices)}
    gt_frame_set   = get_gt_frame_indices(meta["shot_level_gt"],
                                          meta["n_sampled_per_shot"], meta["picks"])
    eval_b = set_f1(pred_frame_set, gt_frame_set)
    predicted_shot_indices = _frames_to_shots(selected_frames, shots)
    gt_shot_scores = [shot["gt_score"] for shot in shots]
    eval_b.update(compute_rank_correlations(predicted_shot_indices, n_shots, gt_shot_scores))
    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    print(f"  [B] frame F1={_fmt(eval_b['f1'])}  "
          f"τ={_fmt(eval_b['kendall_tau'])}  ρ={_fmt(eval_b['spearman_rho'])}")

    selected_images = [all_frames[f - 1] for f in selected_frames if 0 < f <= len(all_frames)]
    clip_score = compute_clipscore(gen_sents, selected_images, device="cpu")
    eval_c = {"clip_score": clip_score}
    print(f"  [C] clip_score={clip_score}")

    return {
        "video_id":               video_id,
        "video_name":             meta["video_name"],
        "frame_strategy":         frame_strategy,
        "max_frames_per_shot":    max_frames_per_shot,
        "max_total_frames":       max_total_frames,
        "n_total_sampled":        n_total_sampled,
        "n_original_frames":      original_total_frames,
        "total_frames_submitted": total_frames,
        "was_frame_truncated":    was_frame_truncated,
        "n_shots":                n_shots,
        "n_select_target":        n_select_target,
        "n_select":               n_select,
        "n_output_tokens":        r["n_output_tokens"],
        "was_output_truncated":   r["was_output_truncated"],
        "raw_output":             r["text"],
        "error":                  None,
        "parse_ok":               parse_ok,
        "generated_sentences":    gen_sents,
        "selected_frames":        selected_frames,
        "gt_shots":               gt_shots,
        "eval_a_summary":         eval_a,
        "eval_b_frame_selection": eval_b,
        "eval_c_alignment":       eval_c,
    }


def _make_empty_result(video_id, meta, n_shots, n_select, r, gt_shots, n_summ_sents,
                       frame_strategy=None, max_frames_per_shot=None, max_total_frames=None,
                       total_frames=None, original_total_frames=None, was_frame_truncated=False,
                       n_select_target=None):
    null_eval = {"precision": None, "recall": None, "f1": None, "kendall_tau": None, "spearman_rho": None}
    return {
        "video_id":               video_id,
        "video_name":             meta["video_name"],
        "frame_strategy":         frame_strategy,
        "max_frames_per_shot":    max_frames_per_shot,
        "max_total_frames":       max_total_frames,
        "n_total_sampled":        original_total_frames,
        "n_original_frames":      original_total_frames,
        "total_frames_submitted": total_frames,
        "was_frame_truncated":    was_frame_truncated,
        "n_shots":                n_shots,
        "n_select_target":        n_select_target,
        "n_select":               n_select,
        "n_output_tokens":        r["n_output_tokens"],
        "was_output_truncated":   r["was_output_truncated"],
        "raw_output":             r["text"],
        "error":                  r["error"],
        "parse_ok":               False,
        "generated_sentences":    [],
        "selected_frames":        [],
        "gt_shots":               gt_shots,
        "eval_a_summary":         compute_text_metrics("", None),
        "eval_b_frame_selection": null_eval,
        "eval_c_alignment":       {"clip_score": None},
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def _make_suffix(frame_strategy, max_frames_per_shot, max_total_frames):
    parts = []
    if frame_strategy:
        fps_tag = f"fps{max_frames_per_shot}" if max_frames_per_shot else ""
        parts.append(f"{frame_strategy}{fps_tag}")
    if max_total_frames:
        parts.append(f"total{max_total_frames}")
    return ("_" + "_".join(parts)) if parts else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",             default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--frame_strategy",      choices=["truncate", "uniform"], default=None)
    parser.add_argument("--max_frames_per_shot", type=int, default=None)
    parser.add_argument("--max_total_frames",    type=int, default=None)
    parser.add_argument("--video",               default=None, help="Single video ID for smoke test.")
    parser.add_argument("--video_ids_file",      default=None,
                        help="Path to a text file with one video ID per line.")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, processor = load_model()

    suffix       = _make_suffix(args.frame_strategy, args.max_frames_per_shot, args.max_total_frames)
    ids_tag      = f"_{Path(args.video_ids_file).stem}" if args.video_ids_file else ""
    base_name    = f"videollama3_{args.dataset}{suffix}{ids_tag}"
    partial_file = results_dir / f"{base_name}.partial.json"
    out_file     = results_dir / f"{base_name}_{timestamp}.json"

    all_results   = []
    processed_ids = set()
    if partial_file.exists():
        with open(partial_file) as f:
            ckpt = json.load(f)
        all_results   = ckpt.get("video_results", [])
        processed_ids = {r["video_id"] for r in all_results}
        print(f"Resuming from checkpoint: {len(processed_ids)} videos already done.")

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_DISPLAY} | dataset={args.dataset} "
          f"| frame_strategy={args.frame_strategy} max_fps={args.max_frames_per_shot} "
          f"max_total={args.max_total_frames} "
          f"({len(video_ids)} videos)\n")

    for i, vid in enumerate(video_ids):
        if vid in processed_ids:
            print(f"[{i+1}/{len(video_ids)}] {vid} (skipped)")
            continue
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        result = None
        last_exc = None
        for attempt in range(1, 4):
            try:
                result = run_video(
                    model, processor, args.dataset, vid,
                    frame_strategy=args.frame_strategy,
                    max_frames_per_shot=args.max_frames_per_shot,
                    max_total_frames=args.max_total_frames,
                )
                break
            except Exception as e:
                import traceback
                last_exc = e
                print(f"  FATAL (attempt {attempt}/3): {e}")
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
        if result is None:
            meta, _ = load_all_shots(args.dataset, vid)
            null_eval = {"precision": None, "recall": None, "f1": None,
                         "kendall_tau": None, "spearman_rho": None}
            result = {
                "video_id": vid, "video_name": meta.get("video_name", vid),
                "frame_strategy": args.frame_strategy,
                "max_frames_per_shot": args.max_frames_per_shot,
                "max_total_frames": args.max_total_frames,
                "n_original_frames": None, "total_frames_submitted": None,
                "was_frame_truncated": False, "n_shots": None, "n_select": None,
                "n_output_tokens": None, "was_output_truncated": False, "raw_output": "",
                "error": f"FATAL (3 attempts): {last_exc}", "parse_ok": False,
                "generated_sentences": [], "selected_frames": [], "gt_shots": [],
                "eval_a_summary": compute_text_metrics("", None),
                "eval_b_frame_selection": null_eval,
                "eval_c_alignment": {"clip_score": None},
            }
        all_results.append(result)
        with open(partial_file, "w") as f:
            json.dump({"video_results": all_results}, f)

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
    print(f"\nDone: {n_success} success, {n_error} errors, parse_ok={n_parse_ok}/{len(all_results)}")
    if n_frame_truncated:
        print(f"  Frame-truncated videos: {n_frame_truncated}")
    print(f"  [A] Avg ROUGE-L:       {avg(['eval_a_summary', 'rougeL'])}")
    print(f"  [A] Avg BERTScore:     {avg(['eval_a_summary', 'bert_score_f1'])}")
    print(f"  [A] Avg G-Eval:        {avg(['eval_a_summary', 'g_eval_overall'])}")
    print(f"  [B] Avg frame F1:      {avg(['eval_b_frame_selection', 'f1'])}")
    print(f"  [B] Avg Kendall τ:     {avg(['eval_b_frame_selection', 'kendall_tau'])}")
    print(f"  [C] Avg CLIPScore:     {avg(['eval_c_alignment', 'clip_score'])}")

    output = {
        "model":               MODEL_DISPLAY,
        "dataset":             args.dataset,
        "frame_strategy":      args.frame_strategy,
        "max_frames_per_shot": args.max_frames_per_shot,
        "max_total_frames":    args.max_total_frames,
        "timestamp":           timestamp,
        "n_success":           n_success,
        "n_error":             n_error,
        "n_parse_ok":          n_parse_ok,
        "n_frame_truncated":   n_frame_truncated,
        "avg_summary_rougeL":    avg(["eval_a_summary", "rougeL"]),
        "avg_summary_bertscore": avg(["eval_a_summary", "bert_score_f1"]),
        "avg_summary_geval":     avg(["eval_a_summary", "g_eval_overall"]),
        "avg_frame_f1":          avg(["eval_b_frame_selection", "f1"]),
        "avg_kendall_tau":       avg(["eval_b_frame_selection", "kendall_tau"]),
        "avg_spearman_rho":      avg(["eval_b_frame_selection", "spearman_rho"]),
        "avg_clip_score":        avg(["eval_c_alignment", "clip_score"]),
        "video_results":         all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    partial_file.unlink(missing_ok=True)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
