"""
exp1_v2t: Whole-video text summarization with Qwen2.5-VL / Qwen3-VL.

Two tasks per video (two model calls):
  1. Overall summary   – free-form 3-5 sentence summary of the full video
  2. Temporal segments – structured description of first / middle / last third

Logged per video:
  n_frames_submitted, n_input_tokens, n_output_tokens, was_output_truncated,
  generated text, ROUGE-1/2/L + BERTScore vs aligned_text_summary (if available)

Frame input strategies (--frame_strategy):
  truncate : submit the first max_frames frames (cap, no resampling)
  uniform  : uniformly downsample to max_frames frames
  (default): submit all frames — no cap

Optional (--shot_rep): for each shot, call the model to select the single most
  representative frame, then submit those representative frames (one per shot,
  in temporal order) as the input for the two main tasks.
  Can be combined with --frame_strategy / --max_frames; when combined, the
  strategy is applied after rep-frame collection (rarely changes anything since
  n_shots << typical max_frames).

Optional (--with_doc): prepend aligned_full_document as plain text context to
  each prompt. Ablation: measures how much textual grounding helps.

Environment: MLLM
Usage:
    python run_qwen.py --model qwen2.5 --dataset summe
    python run_qwen.py --model qwen2.5 --dataset summe --frame_strategy truncate --max_frames 500
    python run_qwen.py --model qwen2.5 --dataset summe --frame_strategy uniform  --max_frames 500
    python run_qwen.py --model qwen2.5 --dataset summe --shot_rep
    python run_qwen.py --model qwen2.5 --dataset summe --with_doc
    python run_qwen.py --model qwen2.5 --dataset summe --video video_1
"""

import sys
import json
import argparse
import re
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import (
    get_video_ids, load_all_frames, load_all_shots,
    load_aligned_summary, load_aligned_document,
)
from shared.evaluate_text import compute_text_metrics
from shared.frame_utils import apply_frame_strategy, compute_dynamic_max_frames
from shared.test_stats import compute_frame_stats, aggregate_frame_stats, aggregate_token_stats

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen2.5": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "display": "Qwen2.5-VL-7B-Instruct"},
    "qwen3":   {"model_id": "Qwen/Qwen3-VL-8B-Instruct",   "display": "Qwen3-VL-8B-Instruct"},
}
HF_CACHE_DIR    = "/data/hf-cache/hub"
MAX_NEW_TOKENS  = 512   # large enough to avoid false truncation


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a video analysis assistant. Your task is to describe the content "
    "of a video based on its frames."
)

def _doc_prefix(doc_text):
    if not doc_text:
        return ""
    return (
        "The following is a detailed shot-by-shot description of this video for context:\n"
        f"{doc_text}\n\n"
    )

def overall_prompt(n_frames, doc_text=None):
    return (
        f"You are given {n_frames} frames from a video shown in chronological order.\n\n"
        + _doc_prefix(doc_text)
        + "Please write a concise summary of the video, describing the main events, "
        "actions, scenes, and topics from beginning to end.\n\n"
        "Your summary should be 3-5 sentences."
    )

def segments_prompt(n_frames, doc_text=None):
    n1 = n_frames // 3
    n2 = 2 * n_frames // 3
    return (
        f"You are given {n_frames} frames from a video shown in chronological order.\n\n"
        + _doc_prefix(doc_text)
        + f"Frames 1-{n1} cover the FIRST THIRD of the video.\n"
        f"Frames {n1+1}-{n2} cover the MIDDLE THIRD of the video.\n"
        f"Frames {n2+1}-{n_frames} cover the LAST THIRD of the video.\n\n"
        "Describe what happens in each section. Use exactly this format:\n"
        "FIRST THIRD: [your description]\n"
        "MIDDLE THIRD: [your description]\n"
        "LAST THIRD: [your description]"
    )

def parse_segments(text):
    """Extract first/middle/last third descriptions from model output."""
    parts = {"first_third": "", "middle_third": "", "last_third": ""}
    patterns = [
        ("first_third",  r"FIRST THIRD\s*:\s*(.+?)(?=MIDDLE THIRD|LAST THIRD|$)"),
        ("middle_third", r"MIDDLE THIRD\s*:\s*(.+?)(?=LAST THIRD|$)"),
        ("last_third",   r"LAST THIRD\s*:\s*(.+)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            parts[key] = m.group(1).strip()
    return parts


# ── Model loading ─────────────────────────────────────────────────────────────
# Default resolution: 448×448 (= 32×14 px per side, tiles perfectly with patch_size=14).
# Tokens per frame at 448×448: 200704 / 784 = 256  (patch=14×14, merge=2×2 → 784 px/token).
# 500 frames × 256 = 128K tokens — fits within Qwen's 128K context window.
DEFAULT_MIN_PIXELS = 448 * 448   # 200704
DEFAULT_MAX_PIXELS = 448 * 448   # 200704


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
def _infer(model, processor, frames, prompt_text):
    """
    One model call: frames + prompt_text → generated string.

    Returns dict:
        text, n_input_tokens, n_output_tokens, was_output_truncated, error
    """
    from qwen_vl_utils import process_vision_info
    import torch

    out = {"text": "", "n_input_tokens": None, "n_output_tokens": None,
           "was_output_truncated": False, "error": None}
    try:
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt_text})
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

    Returns:
        rep_frame : PIL.Image
        rep_idx   : 0-indexed position within shot_frames
        parse_ok  : bool
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
    # Fallback: middle frame
    fallback = n // 2
    return shot_frames[fallback], fallback, False


def _collect_shot_reps(model, processor, dataset_name, video_id):
    """
    For each shot in a video, call the model to select 1 representative frame.

    Returns:
        meta              : video metadata dict
        rep_frames        : list[PIL.Image]  — one per shot, in temporal order
        rep_frame_indices : list[int]        — original frame index of each rep
        rep_parse_ok      : list[bool]       — whether model output was parsed cleanly
    """
    from PIL import Image as PILImage

    meta, shots = load_all_shots(dataset_name, video_id)
    rep_frames, rep_frame_indices, rep_parse_ok = [], [], []

    for shot in shots:
        sf = shot["frames"]
        fi = shot["frame_indices"]
        if not sf:
            rep_frames.append(PILImage.new("RGB", (64, 64), color=(128, 128, 128)))
            rep_frame_indices.append(-1)
            rep_parse_ok.append(False)
        else:
            rf, ri, ok = _select_rep_frame(model, processor, sf)
            rep_frames.append(rf)
            rep_frame_indices.append(fi[ri] if (ri >= 0 and ri < len(fi)) else -1)
            rep_parse_ok.append(ok)

    return meta, rep_frames, rep_frame_indices, rep_parse_ok


# ── Per-video inference ───────────────────────────────────────────────────────
def run_video(model, processor, dataset_name, video_id,
              frame_strategy=None, max_frames=None, shot_rep=False, with_doc=False,
              test_mode=False):
    """
    Run exp1 tasks for a single video.

    Frame selection order:
      1. If shot_rep: collect 1 model-selected rep frame per shot → frames list
      2. Apply frame_strategy (truncate/uniform) if specified
      3. Run the two summarization tasks unchanged
    """
    shot_rep_info = None

    if shot_rep:
        meta, frames, frame_indices, rep_parse_ok = _collect_shot_reps(
            model, processor, dataset_name, video_id
        )
        shot_rep_info = {
            "n_shots":              len(frames),
            "parse_ok_count":       sum(rep_parse_ok),
            "selected_frame_indices": frame_indices,
        }
        print(f"  [shot_rep] {len(frames)} shots, parse_ok={sum(rep_parse_ok)}/{len(frames)}")
    else:
        meta, frames, frame_indices = load_all_frames(dataset_name, video_id)

    # Apply frame strategy (may be a no-op if max_frames is None or n <= max_frames)
    if frame_strategy and max_frames:
        frames, frame_indices = apply_frame_strategy(frames, frame_indices, max_frames, frame_strategy)

    n_frames  = len(frames)
    reference = load_aligned_summary(dataset_name, video_id)
    doc_text  = load_aligned_document(dataset_name, video_id) if with_doc else None
    print(f"  Frames: {n_frames}  |  reference: {'yes' if reference else 'no'}  |  doc: {'yes' if doc_text else 'no'}")

    # ── Task 1: overall summary ───────────────────────────────────────────────
    r1 = _infer(model, processor, frames, overall_prompt(n_frames, doc_text))
    if r1["error"]:
        print(f"  [overall] ERROR: {r1['error'][:100]}")
    else:
        trunc = " [TRUNCATED]" if r1["was_output_truncated"] else ""
        print(f"  [overall] in={r1['n_input_tokens']} out={r1['n_output_tokens']}{trunc}")
        print(f"            {r1['text'][:100]}...")

    # ── Task 2: temporal segments ─────────────────────────────────────────────
    r2 = _infer(model, processor, frames, segments_prompt(n_frames, doc_text))
    if r2["error"]:
        print(f"  [segments] ERROR: {r2['error'][:100]}")
    else:
        trunc = " [TRUNCATED]" if r2["was_output_truncated"] else ""
        print(f"  [segments] in={r2['n_input_tokens']} out={r2['n_output_tokens']}{trunc}")
    segments = parse_segments(r2["text"]) if not r2["error"] else {}

    # ── Text evaluation ───────────────────────────────────────────────────────
    text_eval = compute_text_metrics(r1["text"], reference)

    # ── Test-mode statistics ──────────────────────────────────────────────────
    test_stats = compute_frame_stats(frame_indices, meta) if test_mode else None

    return {
        "video_id":           video_id,
        "video_name":         meta["video_name"],
        "frame_strategy":     frame_strategy,
        "max_frames":         max_frames,
        "shot_rep":           shot_rep,
        "shot_rep_info":      shot_rep_info,
        "n_frames_submitted": n_frames,
        "with_doc":           with_doc,
        "test_stats":         test_stats,
        "overall_summary": {
            "n_input_tokens":       r1["n_input_tokens"],
            "n_output_tokens":      r1["n_output_tokens"],
            "was_output_truncated": r1["was_output_truncated"],
            "text":                 r1["text"],
            "error":                r1["error"],
        },
        "temporal_segments": {
            "n_input_tokens":       r2["n_input_tokens"],
            "n_output_tokens":      r2["n_output_tokens"],
            "was_output_truncated": r2["was_output_truncated"],
            "first_third":          segments.get("first_third", ""),
            "middle_third":         segments.get("middle_third", ""),
            "last_third":           segments.get("last_third", ""),
            "raw_text":             r2["text"],
            "error":                r2["error"],
        },
        "text_eval": text_eval,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def _make_suffix(frame_strategy, max_frames, shot_rep, with_doc):
    parts = []
    if shot_rep:
        parts.append("shotrep")
    if frame_strategy:
        parts.append(f"{frame_strategy}{max_frames}" if max_frames else frame_strategy)
    if with_doc:
        parts.append("doc")
    return ("_" + "_".join(parts)) if parts else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          choices=["qwen2.5", "qwen3"], required=True)
    parser.add_argument("--dataset",        default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--frame_strategy", choices=["truncate", "uniform"], default=None,
                        help="Frame selection strategy. "
                             "truncate: first N frames (N auto-computed from context if --max_frames omitted); "
                             "uniform: evenly-spaced N frames (--max_frames required).")
    parser.add_argument("--max_frames",     type=int, default=None,
                        help="Frame budget. Required for 'uniform'. "
                             "For 'truncate': if omitted, computed from model context window. "
                             "Not applicable with --with_doc (document length is variable).")
    parser.add_argument("--shot_rep",       action="store_true",
                        help="For each shot, call the model to select 1 representative frame, "
                             "then use those frames as input (one per shot, temporal order).")
    parser.add_argument("--with_doc",       action="store_true",
                        help="Prepend aligned_full_document as plain text context (ablation).")
    parser.add_argument("--test_mode",      action="store_true",
                        help="Compute and save input quality statistics per video "
                             "(coverage, GT recall, gtscore ratio, shot coverage, token stats).")
    parser.add_argument("--min_pixels",     type=int, default=DEFAULT_MIN_PIXELS,
                        help=f"Processor min_pixels (default: {DEFAULT_MIN_PIXELS} = 448×448). "
                             "Controls minimum frame resolution submitted to the model.")
    parser.add_argument("--max_pixels",     type=int, default=DEFAULT_MAX_PIXELS,
                        help=f"Processor max_pixels (default: {DEFAULT_MAX_PIXELS} = 448×448). "
                             "Caps per-frame token count: tokens ≈ max_pixels/784. "
                             "At 448×448: 256 tokens/frame → 500 frames ≤ 128K tokens.")
    parser.add_argument("--video",          default=None, help="Single video ID for debugging.")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed (overrides full dataset).")
    args = parser.parse_args()

    if args.frame_strategy == "uniform" and args.max_frames is None:
        parser.error("--max_frames is required when --frame_strategy is 'uniform'.")
    if args.frame_strategy == "truncate" and args.max_frames is None and args.with_doc:
        parser.error("--max_frames is required when using truncate with --with_doc "
                     "(document length is variable; cannot auto-compute safe limit).")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── shot_rep uses original pixel resolution; frame mode uses 448×448 default ──
    eff_min_pixels = None if args.shot_rep else args.min_pixels
    eff_max_pixels = None if args.shot_rep else args.max_pixels
    model, processor = load_model(args.model, min_pixels=eff_min_pixels, max_pixels=eff_max_pixels)

    # ── Dynamic truncation: auto-compute max_frames from model context window ──
    effective_max_frames = args.max_frames
    if args.frame_strategy == "truncate" and args.max_frames is None:
        tokens_per_frame    = args.max_pixels // 784   # e.g. 200704//784 = 256 for 448×448
        effective_max_frames = compute_dynamic_max_frames(model, tokens_per_frame)
        print(f"  Dynamic truncation: max_frames={effective_max_frames} "
              f"(context={getattr(model.config, 'max_position_embeddings', 32768)}, "
              f"{tokens_per_frame} tokens/frame)")

    suffix       = _make_suffix(args.frame_strategy, effective_max_frames, args.shot_rep, args.with_doc)
    if args.video_ids_file:
        import os
        tag = os.path.splitext(os.path.basename(args.video_ids_file))[0]  # e.g. mrhisum_100_seed42
        suffix += f"_{tag}"
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
    print(f"\nRunning {MODEL_CONFIGS[args.model]['display']} on {args.dataset} "
          f"| frame_strategy={args.frame_strategy} max_frames={effective_max_frames} "
          f"shot_rep={args.shot_rep} with_doc={args.with_doc} "
          f"min_pixels={eff_min_pixels} max_pixels={eff_max_pixels} "
          f"({len(video_ids)} videos)\n")

    for i, vid in enumerate(video_ids):
        if vid in processed_ids:
            print(f"[{i+1}/{len(video_ids)}] {vid} (skipped)")
            continue
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        all_results.append(run_video(
            model, processor, args.dataset, vid,
            frame_strategy=args.frame_strategy,
            max_frames=effective_max_frames,
            shot_rep=args.shot_rep,
            with_doc=args.with_doc,
            test_mode=args.test_mode,
        ))
        with open(partial_file, "w") as f:
            json.dump({"video_results": all_results}, f)

    n_success  = sum(1 for r in all_results if not r["overall_summary"]["error"])
    n_error    = sum(1 for r in all_results if r["overall_summary"]["error"])
    avg_frames = float(np.mean([r["n_frames_submitted"] for r in all_results]))
    has_eval   = sum(1 for r in all_results if r["text_eval"]["has_reference"])
    print(f"\nDone: {n_success} success, {n_error} errors, avg_frames={avg_frames:.1f}, "
          f"with_text_eval={has_eval}/{len(all_results)}")
    
    # ── Aggregate text metrics ──────────────────────────────────────────────────
    def _aggregate_metric(key):
        vals = [r["text_eval"][key] for r in all_results 
                if r["text_eval"].get("has_reference") and r["text_eval"].get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    
    text_metrics_agg = {
        "avg_rouge1": _aggregate_metric("rouge1"),
        "avg_rouge2": _aggregate_metric("rouge2"),
        "avg_rougeL": _aggregate_metric("rougeL"),
        "avg_bert_score_f1": _aggregate_metric("bert_score_f1"),
        "avg_bleu4": _aggregate_metric("bleu4"),
        "avg_meteor": _aggregate_metric("meteor"),
        "avg_g_eval_overall": _aggregate_metric("g_eval_overall"),
    }

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
        "model":                MODEL_CONFIGS[args.model]["display"],
        "dataset":              args.dataset,
        "frame_strategy":       args.frame_strategy,
        "max_frames":           effective_max_frames,
        "max_frames_dynamic":   args.frame_strategy == "truncate" and args.max_frames is None,
        "shot_rep":             args.shot_rep,
        "with_doc":             args.with_doc,
        "test_mode":            args.test_mode,
        "min_pixels":           eff_min_pixels,
        "max_pixels":           eff_max_pixels,
        "timestamp":            timestamp,
        "max_new_tokens":       MAX_NEW_TOKENS,
        "n_success":            n_success,
        "n_error":              n_error,
        "avg_frames_submitted": avg_frames,
        "avg_rouge1":           text_metrics_agg["avg_rouge1"],
        "avg_rouge2":           text_metrics_agg["avg_rouge2"],
        "avg_rougeL":           text_metrics_agg["avg_rougeL"],
        "avg_bert_score_f1":    text_metrics_agg["avg_bert_score_f1"],
        "avg_bleu4":            text_metrics_agg["avg_bleu4"],
        "avg_meteor":           text_metrics_agg["avg_meteor"],
        "avg_g_eval_overall":   text_metrics_agg["avg_g_eval_overall"],
        "test_stats_agg":       test_stats_agg,
        "video_results":        all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    partial_file.unlink(missing_ok=True)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
