"""
exp3_alignment: Joint summary generation + shot selection with Qwen2.5-VL / Qwen3-VL.

Input:  All sampled frames labeled by shot ("Shot 1: [frames], Shot 2: [frames], ...")
Task:   In a single call, the model:
          1. Generates a video summary with exactly N sentences (N = len(aligned_text_summary))
          2. Selects ceil(n_shots * 0.15) most important shot numbers

Evaluation:
  A. Summary quality (exp1 metrics):
       ROUGE-1/2/L, BERTScore F1, BLEU-4, METEOR, CIDEr (corpus-level, post-hoc)

  B. Frame selection quality (exp2 metrics):
       F1, Kendall τ, Spearman ρ between model-selected shots and shot_level_gt

  C. Cross-modal alignment:
       CLIPScore between generated summary sentences and selected shot frames
       CLIPScore = 2.5 * max(cos(mean_text_emb, mean_visual_emb), 0)

Frame input strategies (--frame_strategy):
  truncate : within each shot, take the first max_frames_per_shot frames
  uniform  : within each shot, uniformly sample max_frames_per_shot frames (default behavior)
  (default): use existing max_frames_per_shot with uniform sampling

Global budget (--max_total_frames): after per-shot selection, if total frames still
  exceed this budget, apply the strategy globally across shots.

Optional (--shot_rep): for each shot, call the model to select the single most
  representative frame, then use 1 frame per shot as input for the main task.
  When --shot_rep is set, --max_frames_per_shot and --max_total_frames have no effect
  (n_shots frames are submitted, one per shot, in temporal order).

Optional (--with_doc): prepend aligned_full_document as plain text context.
  Ablation: measures how much textual grounding helps joint summarization + shot selection.

Environment: MLLM
Usage:
    python run_qwen.py --model qwen2.5 --dataset summe
    python run_qwen.py --model qwen2.5 --dataset summe --frame_strategy truncate --max_frames_per_shot 4
    python run_qwen.py --model qwen2.5 --dataset summe --frame_strategy uniform  --max_frames_per_shot 4
    python run_qwen.py --model qwen2.5 --dataset summe --max_total_frames 200
    python run_qwen.py --model qwen2.5 --dataset summe --shot_rep
    python run_qwen.py --model qwen2.5 --dataset summe --with_doc
    python run_qwen.py --model qwen2.5 --dataset summe --video video_1  # smoke test
"""

import sys
import json
import math
import argparse
import re
import numpy as np
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
from shared.frame_utils import apply_per_shot_strategy, apply_global_shot_strategy, compute_dynamic_max_frames
from shared.test_stats import compute_frame_stats, aggregate_frame_stats, aggregate_token_stats

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen2.5": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "display": "Qwen2.5-VL-7B-Instruct"},
    "qwen3":   {"model_id": "Qwen/Qwen3-VL-8B-Instruct",   "display": "Qwen3-VL-8B-Instruct"},
}
HF_CACHE_DIR         = "/data/hf-cache/hub"
MAX_NEW_TOKENS       = 1024
DEFAULT_MAX_FPS      = 4   # max frames per shot submitted to the model
# Default resolution: 448×448 (= 32×14 px, tiles perfectly with patch_size=14).
# tokens/frame = 448*448/784 = 256.  500 frames × 256 = 128K tokens (fits in Qwen's 128K ctx).
DEFAULT_MIN_PIXELS   = 448 * 448   # 200704
DEFAULT_MAX_PIXELS   = 448 * 448   # 200704


SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to summarize a video and identify its most important shots."
)


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


# ── Prompt builder ────────────────────────────────────────────────────────────

def _doc_prefix(doc_text):
    if not doc_text:
        return ""
    return (
        "The following is a detailed shot-by-shot description of this video for context:\n"
        f"{doc_text}\n\n"
    )


def build_content(shots, n_summary_sentences, n_select_frames, doc_text=None,
                  n_total_sampled=None, n_select_target=None):
    """
    Build multi-modal content with globally-numbered frames.

    Frames are numbered sequentially across all shots (Frame 1, Frame 2, ...).
    The model is asked to select individual frame numbers (not shot numbers).

    Returns:
        content:      list of content dicts for the model
        frame_images: flat list of PIL images in submission order (for CLIPScore)
    """
    n_shots = len(shots)
    total_frames = sum(len(s["frames"]) for s in shots)

    content = [{"type": "text", "text": f"Below are {total_frames} frames from a video, grouped by shot:\n"}]
    frame_images = []
    global_idx = 1  # 1-indexed frame counter shown in prompt

    for shot in shots:
        frames = shot["frames"]
        if not frames:
            continue
        end_idx = global_idx + len(frames) - 1
        label = (f"Frame {global_idx}" if len(frames) == 1
                 else f"Frames {global_idx}–{end_idx}")
        content.append({"type": "text", "text": f"\nShot {shot['shot_idx'] + 1} ({label}):\n"})
        for frame in frames:
            content.append({"type": "image", "image": frame})
            frame_images.append(frame)
            global_idx += 1

    frame_list = ", ".join(f"Frame {i + 1}" for i in range(total_frames))
    budget_note = ""
    if n_total_sampled is not None and n_total_sampled != total_frames:
        budget_note = (
            f"The full sampled video contains {n_total_sampled} frames in total. "
            f"Only {total_frames} frames are shown here due to input limits. "
            f"The original 15% target is {n_select_target} frames, so select the best "
            f"{n_select_frames} frames from the submitted set.\n\n"
        )

    content.append({
        "type": "text",
        "text": (
            "\n" + _doc_prefix(doc_text)
            + budget_note
            + f"Based on all {total_frames} frames above, complete TWO tasks:\n\n"
            f"TASK 1 — Write a summary of this video in EXACTLY {n_summary_sentences} sentences.\n"
            f"TASK 2 — Select EXACTLY {n_select_frames} most important frames (no more, no less) from: {frame_list}.\n\n"
            "Use EXACTLY this format (no extra text):\n"
            "SUMMARY:\n"
            + "\n".join(f"{i+1}. [sentence {i+1}]" for i in range(n_summary_sentences))
            + "\n\nIMPORTANT FRAMES: [Frame X, Frame Y, ...]"
        ),
    })
    return content, frame_images


# ── Output parsing ────────────────────────────────────────────────────────────

def parse_output(text, n_summary_sentences, n_frames):
    """
    Parse SUMMARY sentences and IMPORTANT FRAMES from model output.

    Returns:
        summary_sentences:  list[str]
        selected_frames:    list of 1-indexed frame positions (in submitted frame list)
        parse_ok:           bool
    """
    summary_sentences = []
    selected_frames = []
    parse_ok = True

    # Extract SUMMARY block
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

    # Extract IMPORTANT FRAMES
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
    """
    Map 1-indexed frame positions in submitted list to 0-indexed shot indices.
    A shot is included if any of its frames was selected.
    """
    # Build a per-frame → shot_idx lookup
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

def load_model(model_key, min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
    import torch

    cfg = MODEL_CONFIGS[model_key]
    print(f"Loading {cfg['display']} ...")
    if model_key == "qwen2.5":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE_DIR
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE_DIR
        )
    processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=HF_CACHE_DIR)
    if min_pixels is not None:
        processor.image_processor.min_pixels = min_pixels
    if max_pixels is not None:
        processor.image_processor.max_pixels = max_pixels
    print(f"  Loaded on: {next(model.parameters()).device}")
    tpf = (max_pixels // 784) if max_pixels is not None else "varies (original res)"
    print(f"  min_pixels={processor.image_processor.min_pixels}  "
          f"max_pixels={processor.image_processor.max_pixels}  "
          f"(≈{tpf} tokens/frame)")
    model.eval()
    return model, processor


# ── Single inference call ─────────────────────────────────────────────────────

def _infer(model, processor, content):
    from qwen_vl_utils import process_vision_info
    import torch

    out = {"text": "", "n_input_tokens": None, "n_output_tokens": None,
           "was_output_truncated": False, "error": None}
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        out["n_input_tokens"] = int(inputs.input_ids.shape[1])

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        out["n_output_tokens"]      = int(generated.shape[0])
        out["was_output_truncated"] = (out["n_output_tokens"] == MAX_NEW_TOKENS)
        out["text"] = processor.decode(generated, skip_special_tokens=True)

    except Exception as e:
        out["error"] = str(e)
    return out


# ── Per-shot representative frame selection ───────────────────────────────────
_SHOT_REP_SYSTEM = "You are a video analysis assistant."
_SHOT_REP_TMPL = (
    "These are {n} frames from a video shot in chronological order.\n"
    "Select the single frame that best represents the key visual content of this shot.\n"
    "Output ONLY a single integer (the 1-indexed frame number, from 1 to {n})."
)

def _select_rep_frame(model, processor, shot_frames):
    """
    Ask the model to select the most representative frame from a single shot.

    Returns (rep_frame, rep_idx, parse_ok).
    """
    from qwen_vl_utils import process_vision_info
    import torch

    n = len(shot_frames)
    if n == 0:
        return None, -1, False
    if n == 1:
        return shot_frames[0], 0, True

    content = [{"type": "text", "text": f"Below are {n} frames from a video shot:\n"}]
    for i, img in enumerate(shot_frames):
        content.append({"type": "text", "text": f"Frame {i + 1}:\n"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": "\n" + _SHOT_REP_TMPL.format(n=n)})

    messages = [
        {"role": "system", "content": _SHOT_REP_SYSTEM},
        {"role": "user",   "content": content},
    ]
    try:
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=16,
                do_sample=False, temperature=None, top_p=None,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        text = processor.decode(generated, skip_special_tokens=True).strip()
        m = re.search(r"\b(\d+)\b", text)
        if m:
            idx_1 = int(m.group(1))
            if 1 <= idx_1 <= n:
                return shot_frames[idx_1 - 1], idx_1 - 1, True
    except Exception:
        pass
    fallback = n // 2
    return shot_frames[fallback], fallback, False


def _apply_shot_rep(model, processor, shots):
    """
    For each shot, replace its frames with 1 model-selected representative frame.

    Returns (updated_shots, shot_rep_info).
    """
    from PIL import Image as PILImage

    rep_parse_ok = []
    new_shots = []
    for shot in shots:
        sf = shot["frames"]
        fi = shot.get("frame_indices", [])
        if not sf:
            ns = dict(shot)
            ns["frames"]        = [PILImage.new("RGB", (64, 64), color=(128, 128, 128))]
            ns["frame_indices"] = [-1]
            new_shots.append(ns)
            rep_parse_ok.append(False)
        else:
            rf, ri, ok = _select_rep_frame(model, processor, sf)
            ns = dict(shot)
            ns["frames"]        = [rf]
            ns["frame_indices"] = [fi[ri] if (ri >= 0 and ri < len(fi)) else -1]
            new_shots.append(ns)
            rep_parse_ok.append(ok)

    shot_rep_info = {
        "n_shots":        len(shots),
        "parse_ok_count": sum(rep_parse_ok),
    }
    return new_shots, shot_rep_info


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, processor, dataset_name, video_id,
              frame_strategy=None, max_frames_per_shot=DEFAULT_MAX_FPS,
              max_total_frames=None, shot_rep=False, with_doc=False, test_mode=False):
    """
    Run exp3 for a single video.

    Shot frame selection order:
      1. If shot_rep: replace each shot's frames with 1 model-selected rep frame
         (max_frames_per_shot and max_total_frames have no effect).
      2. Else:
         a. Apply per-shot frame limit via frame_strategy (or legacy uniform via build_content).
         b. Apply global budget (max_total_frames) if specified.
      3. Call build_content with the processed shots.
    """
    meta, shots = load_all_shots(dataset_name, video_id)
    summ_sents, summ_raw = load_exp3_data(dataset_name, video_id)

    n_shots      = len(shots)
    gt_shots     = sorted(np.where(meta["shot_level_gt"] == 1)[0].tolist())
    n_summ_sents = len(summ_sents) if summ_sents else 3  # fallback to 3

    shot_rep_info = None

    # Record original frame count before any truncation
    original_total_frames = sum(len(s["frames"]) for s in shots)

    if shot_rep:
        # Replace each shot's frames with 1 model-selected representative frame
        shots, shot_rep_info = _apply_shot_rep(model, processor, shots)
        print(f"  [shot_rep] {n_shots} shots, parse_ok={shot_rep_info['parse_ok_count']}/{n_shots}")
        total_frames = n_shots
    else:
        # Per-shot frame selection (only applied when frame_strategy is explicitly set;
        # default = all frames, consistent with exp1/exp2)
        if frame_strategy is not None:
            shots = apply_per_shot_strategy(shots, max_frames_per_shot, frame_strategy)

        # Global budget
        if max_total_frames is not None:
            strategy_for_global = frame_strategy if frame_strategy else "uniform"
            shots = apply_global_shot_strategy(shots, max_total_frames, strategy_for_global)

        total_frames = sum(len(s["frames"]) for s in shots)

    was_frame_truncated = (total_frames < original_total_frames)
    n_total_sampled = original_total_frames
    n_select_target = max(1, math.ceil(n_total_sampled * 0.15))
    n_select = min(total_frames, n_select_target)

    # Collect submitted frame indices (original frame space) and images in order
    submitted_frame_indices = [fi for shot in shots for fi in shot.get("frame_indices", [])]

    doc_context = load_aligned_document(dataset_name, video_id) if with_doc else None
    print(f"  Shots: {n_shots}  |  total_sampled: {n_total_sampled}  |  submitted: {total_frames}  "
          f"|  n_select_target: {n_select_target}  |  n_select: {n_select}  |  "
          f"gt_shots: {gt_shots}  |  doc: {'yes' if doc_context else 'no'}")
    print(f"  Summary sentences: {n_summ_sents}")

    content, frame_images = build_content(
        shots, n_summ_sents, n_select, doc_context,
        n_total_sampled=n_total_sampled, n_select_target=n_select_target
    )
    r = _infer(model, processor, content)

    test_stats = compute_frame_stats(submitted_frame_indices, meta) if test_mode else None

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return _make_empty_result(video_id, meta, n_shots, n_select, r,
                                  gt_shots, n_summ_sents,
                                  frame_strategy=frame_strategy,
                                  max_frames_per_shot=max_frames_per_shot,
                                  max_total_frames=max_total_frames,
                                  shot_rep=shot_rep, shot_rep_info=shot_rep_info,
                                  total_frames=total_frames,
                                  test_stats=test_stats,
                                  original_total_frames=original_total_frames,
                                  was_frame_truncated=was_frame_truncated,
                                  n_select_target=n_select_target)

    trunc = " [TRUNCATED]" if r["was_output_truncated"] else ""
    print(f"  in={r['n_input_tokens']} out={r['n_output_tokens']}{trunc}")

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
    predicted_shot_indices = _frames_to_shots(selected_frames, shots)
    gt_shot_scores = [shot["gt_score"] for shot in shots]
    eval_b.update(compute_rank_correlations(predicted_shot_indices, n_shots, gt_shot_scores))
    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    print(f"  [B] frame F1={_fmt(eval_b['f1'])}  "
          f"τ={_fmt(eval_b['kendall_tau'])}  ρ={_fmt(eval_b['spearman_rho'])}")

    # ── Eval C: cross-modal alignment (CLIPScore) ──────────────────────────────
    selected_images = [frame_images[f - 1]
                       for f in selected_frames if 0 < f <= len(frame_images)]
    clip_score = compute_clipscore(gen_sents, selected_images, device="cpu")
    eval_c = {"clip_score": clip_score}
    print(f"  [C] clip_score={clip_score}")

    return {
        "video_id":            video_id,
        "video_name":          meta["video_name"],
        "frame_strategy":      frame_strategy,
        "max_frames_per_shot": max_frames_per_shot,
        "max_total_frames":    max_total_frames,
        "shot_rep":            shot_rep,
        "shot_rep_info":       shot_rep_info,
        "n_total_sampled":     n_total_sampled,
        "n_original_frames":   original_total_frames,
        "total_frames_submitted": total_frames,
        "was_frame_truncated": was_frame_truncated,
        "test_stats":          test_stats,
        "n_shots":             n_shots,
        "n_select_target":     n_select_target,
        "n_select":            n_select,
        "n_input_tokens":      r["n_input_tokens"],
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


def _make_empty_result(video_id, meta, n_shots, n_select, r, gt_shots, n_summ_sents,
                       frame_strategy=None, max_frames_per_shot=None, max_total_frames=None,
                       shot_rep=False, shot_rep_info=None, total_frames=None, test_stats=None,
                       original_total_frames=None, was_frame_truncated=False,
                       n_select_target=None):
    null_eval = {"precision": None, "recall": None, "f1": None, "kendall_tau": None, "spearman_rho": None}
    return {
        "video_id":            video_id,
        "video_name":          meta["video_name"],
        "frame_strategy":      frame_strategy,
        "max_frames_per_shot": max_frames_per_shot,
        "max_total_frames":    max_total_frames,
        "shot_rep":            shot_rep,
        "shot_rep_info":       shot_rep_info,
        "n_total_sampled":     original_total_frames,
        "n_original_frames":   original_total_frames,
        "total_frames_submitted": total_frames,
        "was_frame_truncated": was_frame_truncated,
        "test_stats":          test_stats,
        "n_shots":             n_shots,
        "n_select_target":     n_select_target,
        "n_select":            n_select,
        "n_input_tokens":      r["n_input_tokens"],
        "n_output_tokens":     r["n_output_tokens"],
        "was_output_truncated": r["was_output_truncated"],
        "raw_output":          r["text"],
        "error":               r["error"],
        "parse_ok":            False,
        "generated_sentences": [],
        "selected_frames":     [],
        "gt_shots":            gt_shots,
        "eval_a_summary":         compute_text_metrics("", None),
        "eval_b_frame_selection": null_eval,
        "eval_c_alignment":       {"clip_score": None},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_suffix(frame_strategy, max_frames_per_shot, max_total_frames, shot_rep, with_doc):
    parts = []
    if shot_rep:
        parts.append("shotrep")
    if frame_strategy:
        fps_tag = f"fps{max_frames_per_shot}" if max_frames_per_shot else ""
        parts.append(f"{frame_strategy}{fps_tag}")
    if max_total_frames:
        parts.append(f"total{max_total_frames}")
    if with_doc:
        parts.append("doc")
    return ("_" + "_".join(parts)) if parts else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",               choices=["qwen2.5", "qwen3"], required=True)
    parser.add_argument("--dataset",             default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--frame_strategy",      choices=["truncate", "uniform"], default=None,
                        help="Per-shot frame selection: truncate=first N, uniform=evenly-spaced N. "
                             "Default (None): all frames submitted (consistent with exp1/exp2).")
    parser.add_argument("--max_frames_per_shot", type=int, default=None,
                        help="Max frames submitted per shot (default: None = all frames). "
                             "Only used when --frame_strategy is set.")
    parser.add_argument("--max_total_frames",    type=int, default=None,
                        help="Global frame budget across all shots. Applied after per-shot "
                             "selection. Uses --frame_strategy (or uniform if not set).")
    parser.add_argument("--shot_rep",            action="store_true",
                        help="For each shot, call model to select 1 representative frame, "
                             "then run the main task with 1 frame per shot.")
    parser.add_argument("--with_doc",            action="store_true",
                        help="Prepend aligned_full_document as plain text context (ablation).")
    parser.add_argument("--test_mode",           action="store_true",
                        help="Compute and save input quality statistics per video "
                             "(coverage, GT recall, gtscore ratio, shot coverage, token stats).")
    parser.add_argument("--min_pixels",          type=int, default=DEFAULT_MIN_PIXELS,
                        help=f"Processor min_pixels (default: {DEFAULT_MIN_PIXELS} = 448×448). "
                             "Controls minimum frame resolution submitted to the model.")
    parser.add_argument("--max_pixels",          type=int, default=DEFAULT_MAX_PIXELS,
                        help=f"Processor max_pixels (default: {DEFAULT_MAX_PIXELS} = 448×448). "
                             "Caps per-frame token count: tokens ≈ max_pixels/784. "
                             "At 448×448: 256 tokens/frame → 500 frames ≤ 128K tokens.")
    parser.add_argument("--video",               default=None, help="Single video ID for smoke test.")
    parser.add_argument("--video_ids_file",      default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    args = parser.parse_args()

    if args.frame_strategy == "truncate" and args.max_total_frames is None and args.with_doc:
        parser.error("--max_total_frames is required when using truncate with --with_doc.")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── shot_rep uses original pixel resolution; frame mode uses 448×448 default ──
    eff_min_pixels = None if args.shot_rep else args.min_pixels
    eff_max_pixels = None if args.shot_rep else args.max_pixels
    model, processor = load_model(args.model, min_pixels=eff_min_pixels, max_pixels=eff_max_pixels)

    # ── Dynamic truncation for global budget ─────────────────────────────────
    effective_max_total = args.max_total_frames
    max_total_dynamic   = False
    if args.frame_strategy == "truncate" and args.max_total_frames is None and not args.shot_rep:
        tokens_per_frame    = args.max_pixels // 784
        effective_max_total = compute_dynamic_max_frames(model, tokens_per_frame)
        max_total_dynamic   = True
        print(f"  Dynamic truncation: max_total_frames={effective_max_total} "
              f"(context={getattr(model.config, 'max_position_embeddings', 32768)}, "
              f"{tokens_per_frame} tokens/frame)")

    suffix       = _make_suffix(
        args.frame_strategy, args.max_frames_per_shot,
        effective_max_total, args.shot_rep, args.with_doc,
    )
    if args.video_ids_file:
        import os as _os
        suffix += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    base_name    = f"{args.model}_{args.dataset}{suffix}"
    partial_file = results_dir / f"{base_name}.partial.json"
    out_file     = results_dir / f"{base_name}_{timestamp}.json"

    # ── Resume from checkpoint if available ──────────────────────────────────
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
    print(f"\nRunning {MODEL_CONFIGS[args.model]['display']} | dataset={args.dataset} "
          f"| frame_strategy={args.frame_strategy} max_fps={args.max_frames_per_shot} "
          f"max_total={effective_max_total} shot_rep={args.shot_rep} "
          f"min_pixels={eff_min_pixels} max_pixels={eff_max_pixels} "
          f"({len(video_ids)} videos)\n")

    for i, vid in enumerate(video_ids):
        if vid in processed_ids:
            print(f"[{i+1}/{len(video_ids)}] {vid} (skipped)")
            continue
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        result = None
        last_exc = None
        for attempt in range(1, 4):  # up to 3 attempts
            try:
                result = run_video(
                    model, processor, args.dataset, vid,
                    frame_strategy=args.frame_strategy,
                    max_frames_per_shot=args.max_frames_per_shot,
                    max_total_frames=effective_max_total,
                    shot_rep=args.shot_rep,
                    with_doc=args.with_doc,
                    test_mode=args.test_mode,
                )
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
            meta, _ = load_all_shots(args.dataset, vid)
            null_eval = {"precision": None, "recall": None, "f1": None,
                         "kendall_tau": None, "spearman_rho": None}
            result = {
                "video_id": vid, "video_name": meta.get("video_name", vid),
                "frame_strategy": args.frame_strategy,
                "max_frames_per_shot": args.max_frames_per_shot,
                "max_total_frames": effective_max_total,
                "n_original_frames": None, "total_frames_submitted": None,
                "was_frame_truncated": False, "n_shots": None, "n_select": None,
                "n_input_tokens": None, "n_output_tokens": None,
                "was_output_truncated": False, "raw_output": "",
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

    # ── Test-mode aggregation ─────────────────────────────────────────────────
    test_stats_agg = None
    if args.test_mode:
        per_video_ts = [r["test_stats"] for r in all_results if r.get("test_stats")]
        test_stats_agg = {
            "frame_stats": aggregate_frame_stats(per_video_ts),
            "token_stats": aggregate_token_stats(all_results),
        }
        print("\n── Test-mode statistics (dataset aggregates) ──")
        for key, agg in test_stats_agg["frame_stats"].items():
            print(f"  {key:35s}  min={agg['min']:.4f}  max={agg['max']:.4f}  mean={agg['mean']:.4f}")
        if test_stats_agg["token_stats"]:
            ts = test_stats_agg["token_stats"]
            print(f"  {'n_input_tokens':35s}  min={ts['min']}  max={ts['max']}  mean={ts['mean']:.1f}")

    output = {
        "model":                    MODEL_CONFIGS[args.model]["display"],
        "dataset":                  args.dataset,
        "frame_strategy":           args.frame_strategy,
        "max_frames_per_shot":      args.max_frames_per_shot,
        "max_total_frames":         effective_max_total,
        "max_total_frames_dynamic": max_total_dynamic,
        "shot_rep":                 args.shot_rep,
        "with_doc":                 args.with_doc,
        "test_mode":                args.test_mode,
        "min_pixels":               eff_min_pixels,
        "max_pixels":               eff_max_pixels,
        "timestamp":                timestamp,
        "test_stats_agg":      test_stats_agg,
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
        "video_results":            all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    partial_file.unlink(missing_ok=True)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
