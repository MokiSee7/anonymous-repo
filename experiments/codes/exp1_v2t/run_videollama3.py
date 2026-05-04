"""
exp1_v2t: Whole-video text summarization with VideoLLaMA3-7B.

Two tasks per video (two model calls):
  1. Overall summary   – free-form 3-5 sentence summary of the full video
  2. Temporal segments – structured description of first / middle / last third

Logged per video:
  n_frames_submitted, n_output_tokens, was_output_truncated,
  generated text, ROUGE-1/2/L + BERTScore vs aligned_text_summary (if available)

Frame input strategies (--frame_strategy):
  truncate : submit the first max_frames frames
  uniform  : uniformly downsample to max_frames frames
  (default): submit all frames — no cap

Environment: videollama3
Usage:
    python run_videollama3.py --dataset summe
    python run_videollama3.py --dataset summe --frame_strategy truncate --max_frames 128
    python run_videollama3.py --dataset summe --frame_strategy uniform  --max_frames 128
    python run_videollama3.py --dataset summe --video video_1
"""

import sys
import json
import argparse
import re
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_frames, load_aligned_summary
from shared.evaluate_text import compute_text_metrics
from shared.frame_utils import apply_frame_strategy

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "/data/hf-cache/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B"
MODEL_DISPLAY = "VideoLLaMA3-7B"
MAX_NEW_TOKENS = 512


# ── Prompts ───────────────────────────────────────────────────────────────────
def overall_prompt(n_frames):
    return (
        f"You are given {n_frames} frames from a video shown in chronological order.\n\n"
        "Please write a concise summary of the video, describing the main events, "
        "actions, scenes, and topics from beginning to end.\n\n"
        "Your summary should be 3-5 sentences."
    )


def segments_prompt(n_frames):
    n1 = n_frames // 3
    n2 = 2 * n_frames // 3
    return (
        f"You are given {n_frames} frames from a video shown in chronological order.\n\n"
        f"Frames 1-{n1} cover the FIRST THIRD of the video.\n"
        f"Frames {n1+1}-{n2} cover the MIDDLE THIRD of the video.\n"
        f"Frames {n2+1}-{n_frames} cover the LAST THIRD of the video.\n\n"
        "Describe what happens in each section. Use exactly this format:\n"
        "FIRST THIRD: [your description]\n"
        "MIDDLE THIRD: [your description]\n"
        "LAST THIRD: [your description]"
    )


def parse_segments(text):
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
def run_video(model, processor, dataset_name, video_id, frame_strategy=None, max_frames=None):
    meta, frames, frame_indices = load_all_frames(dataset_name, video_id)

    if frame_strategy and max_frames:
        frames, frame_indices = apply_frame_strategy(frames, frame_indices, max_frames, frame_strategy)

    n_frames  = len(frames)
    reference = load_aligned_summary(dataset_name, video_id)
    print(f"  Frames: {n_frames}  |  reference: {'yes' if reference else 'no'}")

    # Task 1: overall summary
    r1 = _infer(model, processor, frames, overall_prompt(n_frames))
    if r1["error"]:
        print(f"  [overall] ERROR: {r1['error'][:100]}")
    else:
        trunc = " [TRUNCATED]" if r1["was_output_truncated"] else ""
        print(f"  [overall] out={r1['n_output_tokens']}{trunc}")
        print(f"            {r1['text'][:100]}...")

    # Task 2: temporal segments
    r2 = _infer(model, processor, frames, segments_prompt(n_frames))
    if r2["error"]:
        print(f"  [segments] ERROR: {r2['error'][:100]}")
    else:
        trunc = " [TRUNCATED]" if r2["was_output_truncated"] else ""
        print(f"  [segments] out={r2['n_output_tokens']}{trunc}")
    segments = parse_segments(r2["text"]) if not r2["error"] else {}

    text_eval = compute_text_metrics(r1["text"], reference)

    return {
        "video_id":           video_id,
        "video_name":         meta["video_name"],
        "frame_strategy":     frame_strategy,
        "max_frames":         max_frames,
        "n_frames_submitted": n_frames,
        "overall_summary": {
            "n_output_tokens":      r1["n_output_tokens"],
            "was_output_truncated": r1["was_output_truncated"],
            "text":                 r1["text"],
            "error":                r1["error"],
        },
        "temporal_segments": {
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
def _make_suffix(frame_strategy, max_frames):
    if frame_strategy:
        return f"_{frame_strategy}{max_frames}" if max_frames else f"_{frame_strategy}"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--frame_strategy", choices=["truncate", "uniform"], default=None)
    parser.add_argument("--max_frames",     type=int, default=None)
    parser.add_argument("--video",          default=None, help="Single video ID for debugging.")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line.")
    args = parser.parse_args()

    if args.frame_strategy == "uniform" and args.max_frames is None:
        parser.error("--max_frames is required when --frame_strategy is 'uniform'.")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, processor = load_model()

    suffix       = _make_suffix(args.frame_strategy, args.max_frames)
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
    print(f"\nRunning {MODEL_DISPLAY} on {args.dataset} "
          f"| frame_strategy={args.frame_strategy} max_frames={args.max_frames} "
          f"({len(video_ids)} videos)\n")

    for i, vid in enumerate(video_ids):
        if vid in processed_ids:
            print(f"[{i+1}/{len(video_ids)}] {vid} (skipped)")
            continue
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        all_results.append(run_video(
            model, processor, args.dataset, vid,
            frame_strategy=args.frame_strategy,
            max_frames=args.max_frames,
        ))
        with open(partial_file, "w") as f:
            json.dump({"video_results": all_results}, f)

    n_success  = sum(1 for r in all_results if not r["overall_summary"]["error"])
    n_error    = sum(1 for r in all_results if r["overall_summary"]["error"])
    avg_frames = float(np.mean([r["n_frames_submitted"] for r in all_results]))
    has_eval   = sum(1 for r in all_results if r["text_eval"]["has_reference"])
    print(f"\nDone: {n_success} success, {n_error} errors, avg_frames={avg_frames:.1f}, "
          f"with_text_eval={has_eval}/{len(all_results)}")

    output = {
        "model":                MODEL_DISPLAY,
        "dataset":              args.dataset,
        "frame_strategy":       args.frame_strategy,
        "max_frames":           args.max_frames,
        "timestamp":            timestamp,
        "max_new_tokens":       MAX_NEW_TOKENS,
        "n_success":            n_success,
        "n_error":              n_error,
        "avg_frames_submitted": avg_frames,
        "video_results":        all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    partial_file.unlink(missing_ok=True)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
