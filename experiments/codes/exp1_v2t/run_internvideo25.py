"""
exp1_v2t: Whole-video text summarization with InternVideo2.5-Chat-8B.

Two tasks per video (two model calls):
  1. Overall summary   – free-form 3-5 sentence summary of the full video
  2. Temporal segments – structured description of first / middle / last third

Note: frames are padded to a multiple of 4 (local_num_frames architectural constraint).
n_frames_submitted always reflects the original count (before padding).

Logged per video:
  n_frames_submitted, n_frames_padded, n_input_tokens, n_output_tokens,
  was_output_truncated, generated text, ROUGE-1/2/L + BERTScore (if reference available)

Environment: internvideo
Usage:
    python run_internvideo25.py --dataset summe
    python run_internvideo25.py --dataset summe --video video_1
"""

import sys
import json
import argparse
import re
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_frames, load_aligned_summary
from shared.evaluate_text import compute_text_metrics
from shared.frame_utils import apply_frame_strategy
from shared.test_stats import compute_frame_stats, aggregate_frame_stats, aggregate_token_stats

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID         = "OpenGVLab/InternVideo2_5_Chat_8B"
MODEL_DISPLAY    = "InternVideo2.5-Chat-8B"
HF_CACHE_DIR     = "/data/hf-cache/hub"
LOCAL_NUM_FRAMES = 4    # hardcoded in modeling_internvl_chat_hico2.py
MAX_NEW_TOKENS   = 512

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE    = 448

# ── Image preprocessing ───────────────────────────────────────────────────────
_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def preprocess_frames(pil_images):
    tensors = [_transform(img) for img in pil_images]
    return torch.stack(tensors), [1] * len(pil_images)

def pad_to_multiple(frames):
    """Repeat last frame until count is a multiple of LOCAL_NUM_FRAMES."""
    remainder = len(frames) % LOCAL_NUM_FRAMES
    if remainder:
        frames = frames + [frames[-1]] * (LOCAL_NUM_FRAMES - remainder)
    return frames


# ── Prompts ───────────────────────────────────────────────────────────────────
def overall_prompt(n_frames_padded, n_frames_orig):
    # <image> tokens must match actual frames passed (padded count)
    # text description uses original count so the model isn't confused by padding
    image_tokens = "\n".join(["<image>"] * n_frames_padded)
    text = (
        f"You are given {n_frames_orig} frames from a video shown in chronological order.\n\n"
        "Please write a concise summary of the video, describing the main events, "
        "actions, scenes, and topics from beginning to end.\n\n"
        "Your summary should be 3-5 sentences."
    )
    return f"{image_tokens}\n{text}"

def segments_prompt(n_frames_padded, n_frames_orig):
    n1, n2 = n_frames_orig // 3, 2 * n_frames_orig // 3
    image_tokens = "\n".join(["<image>"] * n_frames_padded)
    text = (
        f"You are given {n_frames_orig} frames from a video shown in chronological order.\n\n"
        f"Frames 1-{n1} cover the FIRST THIRD of the video.\n"
        f"Frames {n1+1}-{n2} cover the MIDDLE THIRD of the video.\n"
        f"Frames {n2+1}-{n_frames_orig} cover the LAST THIRD of the video.\n\n"
        "Describe what happens in each section. Use exactly this format:\n"
        "FIRST THIRD: [your description]\n"
        "MIDDLE THIRD: [your description]\n"
        "LAST THIRD: [your description]"
    )
    return f"{image_tokens}\n{text}"

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
MAX_RETRIES = 3  # retry when model returns empty string (no error)

def _infer(model, tokenizer, frames_padded, n_frames_orig, prompt_fn):
    """
    One model call.
    frames_padded: already padded list of PIL images
    n_frames_orig: original count (used in prompt text)
    prompt_fn:     overall_prompt or segments_prompt (takes n_frames_orig)
    Retries up to MAX_RETRIES times if the model returns an empty response.
    """
    out = {"text": "", "n_output_tokens": None,
           "was_output_truncated": False, "error": None}
    try:
        pixel_values, num_patches_list = preprocess_frames(frames_padded)
        pixel_values = pixel_values.to(dtype=torch.bfloat16, device=model.device)

        n_frames_padded = len(frames_padded)
        prompt_text = prompt_fn(n_frames_padded, n_frames_orig)

        response = ""
        for attempt in range(MAX_RETRIES):
            generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "repetition_penalty": 1.5}
            with torch.no_grad():
                response = model.chat(
                    tokenizer, pixel_values, prompt_text,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None, return_history=False,
                )
            if response.strip():
                break
            if attempt < MAX_RETRIES - 1:
                print(f"    [retry {attempt+1}/{MAX_RETRIES-1}] empty response, retrying ...")

        # model.chat returns a string; estimate output tokens via tokenizer
        out["n_output_tokens"]      = len(tokenizer.encode(response))
        out["was_output_truncated"] = (out["n_output_tokens"] >= MAX_NEW_TOKENS - 2)
        out["text"] = response

    except Exception as e:
        out["error"] = str(e)
    return out


# ── Per-video inference ───────────────────────────────────────────────────────
def run_video(model, tokenizer, dataset_name, video_id,
              frame_strategy=None, max_frames=None, test_mode=False):
    meta, frames, frame_indices = load_all_frames(dataset_name, video_id)

    if frame_strategy and max_frames:
        frames, frame_indices = apply_frame_strategy(frames, frame_indices, max_frames, frame_strategy)

    n_frames_orig   = len(frames)
    frames_padded   = pad_to_multiple(frames)
    n_frames_padded = len(frames_padded)
    reference       = load_aligned_summary(dataset_name, video_id)

    print(f"  Frames: {n_frames_orig} (padded→{n_frames_padded})  |  reference: {'yes' if reference else 'no'}")

    # ── Task 1: overall summary ───────────────────────────────────────────────
    r1 = _infer(model, tokenizer, frames_padded, n_frames_orig, overall_prompt)
    if r1["error"]:
        print(f"  [overall] ERROR: {r1['error'][:100]}")
    else:
        trunc = " [TRUNCATED]" if r1["was_output_truncated"] else ""
        print(f"  [overall] out={r1['n_output_tokens']}{trunc}")
        print(f"            {r1['text'][:100]}...")

    # ── Task 2: temporal segments ─────────────────────────────────────────────
    r2 = _infer(model, tokenizer, frames_padded, n_frames_orig, segments_prompt)
    if r2["error"]:
        print(f"  [segments] ERROR: {r2['error'][:100]}")
    else:
        trunc = " [TRUNCATED]" if r2["was_output_truncated"] else ""
        print(f"  [segments] out={r2['n_output_tokens']}{trunc}")
    segments = parse_segments(r2["text"]) if not r2["error"] else {}

    # ── Text evaluation ───────────────────────────────────────────────────────
    text_eval  = compute_text_metrics(r1["text"], reference)
    test_stats = compute_frame_stats(frame_indices, meta) if test_mode else None

    return {
        "video_id":           video_id,
        "video_name":         meta["video_name"],
        "frame_strategy":     frame_strategy,
        "max_frames":         max_frames,
        "n_frames_submitted": n_frames_orig,
        "n_frames_padded":    n_frames_padded,
        "test_stats":         test_stats,
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--frame_strategy", choices=["truncate", "uniform"], default=None,
                        help="truncate: first N frames; uniform: evenly-spaced N frames.")
    parser.add_argument("--max_frames",     type=int, default=None,
                        help="Frame budget (required for uniform; optional cap for truncate).")
    parser.add_argument("--test_mode",      action="store_true",
                        help="Compute and save input quality statistics per video.")
    parser.add_argument("--video",          default=None, help="Single video ID for debugging.")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    args = parser.parse_args()

    if args.frame_strategy == "uniform" and args.max_frames is None:
        parser.error("--max_frames is required when --frame_strategy is 'uniform'.")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    suffix = ""
    if args.frame_strategy:
        suffix = f"_{args.frame_strategy}{args.max_frames or ''}"
    if args.video_ids_file:
        import os
        tag = os.path.splitext(os.path.basename(args.video_ids_file))[0]
        suffix += f"_{tag}"

    base_name    = f"internvideo25_{args.dataset}{suffix}"
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

    model, tokenizer = load_model()

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
            model, tokenizer, args.dataset, vid,
            frame_strategy=args.frame_strategy,
            max_frames=args.max_frames,
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

    output = {
        "model":                MODEL_DISPLAY,
        "dataset":              args.dataset,
        "frame_strategy":       args.frame_strategy,
        "max_frames":           args.max_frames,
        "test_mode":            args.test_mode,
        "timestamp":            timestamp,
        "max_new_tokens":       MAX_NEW_TOKENS,
        "n_success":            n_success,
        "n_error":              n_error,
        "avg_frames_submitted": avg_frames,
        "test_stats_agg":       test_stats_agg,
        "video_results":        all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    partial_file.unlink(missing_ok=True)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
