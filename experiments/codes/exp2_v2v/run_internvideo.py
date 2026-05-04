"""
exp2_v2v: Video-to-video key frame selection with InternVideo2.5-Chat-8B.

Two modes (--mode):
  frame (default): Submit all sampled frames labeled "Frame 1, Frame 2, ...", ask the
                   model to select the most important ceil(n_frames * 0.15) frame indices.
                   Selected frames are mapped back to shots for evaluation vs shot_level_gt.
  shot:            Alternative: for each shot, uniformly sample 20% of its frames (min 1)
                   as representative frames. Submit all representative frames labeled
                   "Frame 1, Frame 2, ...". Ask model to select ceil(n_submitted * 0.15)
                   frames. GT is the top-gtscore 15% of the same submitted frames.
                   Evaluation: F1 between model selection and GT (both as 1-indexed positions
                   within the submitted representative frames).

Output format requested from model: JSON array of 1-indexed integers, e.g. [1, 5, 12, ...]

Note: frames are padded to a multiple of LOCAL_NUM_FRAMES=4 (architectural constraint).
Frame labels and n_select use original (unpadded) counts.

Frame input strategies (--frame_strategy, frame mode only):
  truncate : submit the first max_frames frames
  uniform  : uniformly downsample to max_frames frames
  (default): submit all frames — no cap

Optional (--with_doc): prepend aligned_full_document as plain text context to each prompt.
  Ablation: measures how much textual grounding helps frame selection.

Environment: internvideo
Usage:
    python run_internvideo.py --dataset summe
    python run_internvideo.py --dataset summe --mode shot
    python run_internvideo.py --dataset summe --frame_strategy truncate --max_frames 636
    python run_internvideo.py --dataset summe --frame_strategy uniform --max_frames 636
    python run_internvideo.py --dataset summe --with_doc
    python run_internvideo.py --dataset summe --video video_1  # smoke test
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
from shared.data_loader import get_video_ids, load_all_frames, load_all_shots, load_aligned_document, get_gt_shot_scores
from shared.evaluate_sets import set_f1
from shared.evaluate import compute_rank_correlations, get_gt_frame_indices
from shared.frame_utils import apply_frame_strategy
from shared.test_stats import compute_frame_stats, aggregate_frame_stats, aggregate_token_stats

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID         = "OpenGVLab/InternVideo2_5_Chat_8B"
MODEL_DISPLAY    = "InternVideo2.5-Chat-8B"
HF_CACHE_DIR     = "/data/hf-cache/hub"
LOCAL_NUM_FRAMES = 4    # hardcoded in modeling_internvl_chat_hico2.py
MAX_NEW_TOKENS   = 2048

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


def pad_to_multiple(frames):
    """Repeat last frame until count is a multiple of LOCAL_NUM_FRAMES."""
    remainder = len(frames) % LOCAL_NUM_FRAMES
    if remainder:
        frames = frames + [frames[-1]] * (LOCAL_NUM_FRAMES - remainder)
    return frames


def preprocess_frames(pil_images):
    tensors = [_transform(img) for img in pil_images]
    return torch.stack(tensors), [1] * len(pil_images)


# ── Shared utility functions ───────────────────────────────────────────────────

def parse_index_list(text, n_submitted, n_select):
    """
    Parse a JSON array of 1-indexed integers from model output.

    Returns:
        selected:  list of valid 1-indexed integers (may be shorter than n_select
                   if model output is malformed)
        parse_ok:  bool, True if parsing succeeded cleanly
    """
    # Try strict JSON array first
    m = re.search(r"\[[\d,\s]+\]", text)
    if m:
        try:
            raw = json.loads(m.group())
            valid = [int(x) for x in raw if 1 <= int(x) <= n_submitted]
            # Deduplicate while preserving order
            seen = set()
            unique = [x for x in valid if not (x in seen or seen.add(x))]
            return unique[:n_select], len(unique) > 0
        except (ValueError, TypeError):
            pass

    # Fallback: extract all integers from the text
    nums = [int(x) for x in re.findall(r"\b\d+\b", text) if 1 <= int(x) <= n_submitted]
    seen = set()
    unique = [x for x in nums if not (x in seen or seen.add(x))]
    return unique[:n_select], len(unique) > 0


def frames_to_shots(selected_frame_positions, meta):
    """
    Map selected frame positions (1-indexed, in picks order) to shot indices.

    Returns:
        shot_indices: sorted list of unique 0-indexed shot indices
    """
    cumsum = np.cumsum([0] + list(meta["n_sampled_per_shot"]))
    shot_indices = set()
    for pos in selected_frame_positions:
        idx = pos - 1  # convert to 0-indexed
        for shot_i in range(len(meta["n_sampled_per_shot"])):
            if cumsum[shot_i] <= idx < cumsum[shot_i + 1]:
                shot_indices.add(shot_i)
                break
    return sorted(shot_indices)


def _make_result(video_id, meta, mode, n_submitted, n_select, r,
                 selected_raw, predicted, gt, parse_ok, eval_result=None, extra=None,
                 frame_strategy=None, max_frames=None, test_stats=None,
                 n_total_sampled=None, n_select_target=None):
    result = {
        "video_id":        video_id,
        "video_name":      meta["video_name"],
        "mode":            mode,
        "frame_strategy":  frame_strategy,
        "max_frames":      max_frames,
        "n_total_sampled": n_total_sampled if n_total_sampled is not None else n_submitted,
        "n_submitted":     n_submitted,
        "n_select_target": n_select_target if n_select_target is not None else n_select,
        "n_select":        n_select,
        "n_output_tokens": r["n_output_tokens"],
        "was_output_truncated": r["was_output_truncated"],
        "raw_output":      r["text"],
        "error":           r["error"],
        "parse_ok":        parse_ok,
        # frame mode: 1-indexed frame positions → 0-indexed shot indices
        # shot mode:  1-indexed submitted-frame positions (both predicted and gt)
        "selected_raw":    selected_raw,
        "predicted":       predicted,
        "gt":              gt,
        "eval":            eval_result or {"precision": None, "recall": None, "f1": None, "kendall_tau": None, "spearman_rho": None},
        "test_stats":      test_stats,
    }
    if extra:
        result.update(extra)
    return result


# ── Prompt builders ───────────────────────────────────────────────────────────

def _doc_prefix(doc_text):
    if not doc_text:
        return ""
    return (
        "The following is a detailed shot-by-shot description of this video for context:\n"
        f"{doc_text}\n\n"
    )


def build_frame_mode_prompt(n_frames_padded, n_frames_orig, n_select, doc_text=None,
                            n_total_sampled=None, n_select_target=None):
    """
    Frame mode: <image> tokens (padded count) + text using original count for frame labels.
    """
    image_tokens = "\n".join(["<image>"] * n_frames_padded)
    text = (
        f"Below are {n_frames_orig} frames from a video in chronological order "
        f"(labeled Frame 1 through Frame {n_frames_orig}).\n\n"
        + _doc_prefix(doc_text)
        + (
            f"The full sampled video contains {n_total_sampled} frames, so the 15% target budget is "
            f"{n_select_target} frame(s). Based on the {n_frames_orig} frames shown here, select exactly "
            f"{n_select} important frame(s).\n\n"
            if n_total_sampled is not None and n_select_target is not None and n_total_sampled != n_frames_orig
            else ""
        )
        + f"From these {n_frames_orig} frames, select the {n_select} most important frames "
        "that best represent the key moments of the video.\n\n"
        "Output ONLY a JSON array of frame numbers (1-indexed integers), "
        f"with exactly {n_select} elements. Example: [1, 5, 12]\n"
        "Do not include any explanation."
    )
    return f"{image_tokens}\n{text}"


def build_shot_mode_prompt(n_frames_padded, n_frames_orig, n_select, doc_text=None):
    """
    Shot mode: <image> tokens (padded count) + text using original count for frame labels.
    """
    image_tokens = "\n".join(["<image>"] * n_frames_padded)
    text = (
        f"Below are {n_frames_orig} representative frames from a video in chronological order "
        f"(labeled Frame 1 through Frame {n_frames_orig}).\n\n"
        + _doc_prefix(doc_text)
        + f"From these {n_frames_orig} frames, select the {n_select} most important frames "
        "that best represent the key moments of the video.\n\n"
        "Output ONLY a JSON array of frame numbers (1-indexed integers), "
        f"with exactly {n_select} elements. Example: [1, 5, 12]\n"
        "Do not include any explanation."
    )
    return f"{image_tokens}\n{text}"


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

def _infer(model, tokenizer, frames_padded, prompt):
    """
    One model call.
    frames_padded: already padded list of PIL images
    prompt:        full prompt string with <image> tokens
    """
    out = {"text": "", "n_output_tokens": None,
           "was_output_truncated": False, "error": None}
    try:
        pixel_values, num_patches_list = preprocess_frames(frames_padded)
        pixel_values = pixel_values.to(dtype=torch.bfloat16, device=model.device)

        generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}

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

def run_video_frame_mode(model, tokenizer, dataset_name, video_id,
                         frame_strategy=None, max_frames=None, with_doc=False, test_mode=False):
    """Frame mode: submit frames (optionally capped), ask to select 15%."""
    meta, frames, frame_indices = load_all_frames(dataset_name, video_id)
    n_total_sampled = len(frame_indices)

    if frame_strategy and max_frames:
        frames, frame_indices = apply_frame_strategy(frames, frame_indices, max_frames, frame_strategy)
        # Rebuild n_sampled_per_shot for the subsampled picks
        sub_picks = np.array(frame_indices)
        n_sampled_sub = []
        for start, end in meta["change_points"]:
            n_sampled_sub.append(int(np.sum((sub_picks >= start) & (sub_picks <= end))))
        meta["n_sampled_per_shot"] = np.array(n_sampled_sub)

    n_submitted     = len(frames)
    frames_padded   = pad_to_multiple(frames)
    n_frames_padded = len(frames_padded)

    n_select_target = max(1, math.ceil(n_total_sampled * 0.15))
    n_select = min(n_submitted, n_select_target)
    gt_shots = sorted(np.where(meta["shot_level_gt"] == 1)[0].tolist())

    doc_text = load_aligned_document(dataset_name, video_id) if with_doc else None
    print(f"  Frames: {n_submitted} / total_sampled={n_total_sampled} (padded→{n_frames_padded})  |  "
          f"select: {n_select} (target={n_select_target})  |  gt_shots: {gt_shots}  |  "
          f"doc: {'yes' if doc_text else 'no'}")

    prompt = build_frame_mode_prompt(
        n_frames_padded, n_submitted, n_select, doc_text,
        n_total_sampled=n_total_sampled, n_select_target=n_select_target
    )
    r = _infer(model, tokenizer, frames_padded, prompt)

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return _make_result(video_id, meta, "frame", n_submitted, n_select, r,
                            [], [], gt_shots, False, None,
                            frame_strategy=frame_strategy, max_frames=max_frames,
                            extra={"n_frames_padded": n_frames_padded},
                            n_total_sampled=n_total_sampled, n_select_target=n_select_target)

    trunc = " [TRUNCATED]" if r["was_output_truncated"] else ""
    print(f"  out={r['n_output_tokens']}{trunc}  raw: {r['text'][:80]}")

    selected_frames, parse_ok = parse_index_list(r["text"], n_submitted, n_select)
    # Frame-level F1: compare selected pick indices vs GT pick indices
    pred_frame_set  = {frame_indices[f - 1] for f in selected_frames if 0 <= f - 1 < len(frame_indices)}
    gt_frame_set    = get_gt_frame_indices(meta["shot_level_gt"], meta["n_sampled_per_shot"], meta["picks"])
    eval_result     = set_f1(pred_frame_set, gt_frame_set)
    # Rank correlations at shot level
    predicted_shots = frames_to_shots(selected_frames, meta)
    gt_shot_scores  = get_gt_shot_scores(meta["gtscore"], meta["n_sampled_per_shot"])
    eval_result.update(compute_rank_correlations(predicted_shots, len(meta["n_sampled_per_shot"]), gt_shot_scores))
    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    print(f"  predicted shots: {predicted_shots}  |  F1: {_fmt(eval_result['f1'])}  "
          f"τ={_fmt(eval_result['kendall_tau'])}  ρ={_fmt(eval_result['spearman_rho'])}  parse_ok: {parse_ok}")

    test_stats = compute_frame_stats(frame_indices, meta) if test_mode else None

    return _make_result(video_id, meta, "frame", n_submitted, n_select, r,
                        selected_frames, predicted_shots, gt_shots, parse_ok, eval_result,
                        frame_strategy=frame_strategy, max_frames=max_frames,
                        test_stats=test_stats,
                        extra={"n_frames_padded": n_frames_padded},
                        n_total_sampled=n_total_sampled, n_select_target=n_select_target)


def run_video_shot_mode(model, tokenizer, dataset_name, video_id,
                        with_doc=False, test_mode=False):
    """
    Shot mode: uniformly sample 20% of each shot's frames (min 1) as representative
    frames, submit them all (padded), ask model to select 15%.

    GT: top-gtscore 15% of the same submitted representative frames.
    """
    from PIL import Image as PILImage

    meta, shots = load_all_shots(dataset_name, video_id)

    rep_frames        = []
    rep_gtscores      = []
    rep_frame_indices = []

    gtscore      = meta["gtscore"]  # (n_steps,) aligned with picks order
    picks        = meta["picks"]
    picks_cursor = 0                # tracks position in picks / gtscore

    for shot in shots:
        n_in_shot = int(meta["n_sampled_per_shot"][shot["shot_idx"]])
        n_rep     = max(1, math.ceil(n_in_shot * 0.20))

        if n_in_shot == 0:
            # Empty shot: placeholder image with gtscore 0
            rep_frames.append(PILImage.new("RGB", (64, 64), color=(128, 128, 128)))
            rep_gtscores.append(0.0)
            rep_frame_indices.append(-1)
            continue

        # Uniform sample indices within this shot's sampled frames
        sample_idx     = np.linspace(0, n_in_shot - 1, n_rep, dtype=int)
        frames_in_shot = shot["frames"]  # already loaded PIL Images

        for si in sample_idx:
            if si < len(frames_in_shot):
                rep_frames.append(frames_in_shot[si])
            else:
                rep_frames.append(PILImage.new("RGB", (64, 64), color=(128, 128, 128)))
            gs_idx = picks_cursor + int(si)
            rep_gtscores.append(float(gtscore[gs_idx]) if gs_idx < len(gtscore) else 0.0)
            rep_frame_indices.append(int(picks[gs_idx]) if gs_idx < len(picks) else -1)

        picks_cursor += n_in_shot

    n_submitted     = len(rep_frames)
    frames_padded   = pad_to_multiple(rep_frames)
    n_frames_padded = len(frames_padded)
    n_select        = max(1, math.ceil(n_submitted * 0.15))

    # GT: top-gtscore 15% of submitted representative frames (1-indexed positions)
    sorted_by_score   = sorted(range(n_submitted), key=lambda i: rep_gtscores[i], reverse=True)
    gt_positions      = sorted(sorted_by_score[:n_select])          # 0-indexed
    gt_positions_1idx = [p + 1 for p in gt_positions]              # 1-indexed for F1

    doc_text = load_aligned_document(dataset_name, video_id) if with_doc else None
    print(f"  Shots: {len(shots)}  |  rep_frames: {n_submitted} (padded→{n_frames_padded})  |  select: {n_select}  |  gt_pos (1-idx): {gt_positions_1idx[:8]}...  |  doc: {'yes' if doc_text else 'no'}")

    prompt = build_shot_mode_prompt(n_frames_padded, n_submitted, n_select, doc_text)
    r = _infer(model, tokenizer, frames_padded, prompt)

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return _make_result(video_id, meta, "shot", n_submitted, n_select, r,
                            [], [], gt_positions_1idx, False,
                            extra={"n_frames_padded": n_frames_padded, "rep_gtscores": rep_gtscores})

    trunc = " [TRUNCATED]" if r["was_output_truncated"] else ""
    print(f"  out={r['n_output_tokens']}{trunc}  raw: {r['text'][:80]}")

    selected_1idx, parse_ok = parse_index_list(r["text"], n_submitted, n_select)
    eval_result = set_f1(selected_1idx, gt_positions_1idx)
    eval_result.update(compute_rank_correlations([p - 1 for p in selected_1idx], n_submitted, rep_gtscores))
    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    print(f"  selected: {selected_1idx[:8]}  |  F1: {_fmt(eval_result['f1'])}  "
          f"τ={_fmt(eval_result['kendall_tau'])}  ρ={_fmt(eval_result['spearman_rho'])}  parse_ok: {parse_ok}")

    test_stats = compute_frame_stats(rep_frame_indices, meta) if test_mode else None

    return _make_result(video_id, meta, "shot", n_submitted, n_select, r,
                        selected_1idx, selected_1idx, gt_positions_1idx, parse_ok, eval_result,
                        test_stats=test_stats,
                        extra={"n_frames_padded": n_frames_padded, "rep_gtscores": rep_gtscores})


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_suffix(mode, frame_strategy, max_frames, with_doc):
    parts = []
    if frame_strategy:
        parts.append(f"{frame_strategy}{max_frames}" if max_frames else frame_strategy)
    if with_doc:
        parts.append("doc")
    suffix = "_".join(parts)
    return f"_{mode}_{suffix}" if suffix else f"_{mode}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--mode",           choices=["frame", "shot"], default="frame",
                        help="'frame': select from all sampled frames; "
                             "'shot': select from 20%%-per-shot representative frames.")
    parser.add_argument("--frame_strategy", choices=["truncate", "uniform"], default=None,
                        help="Frame selection strategy (frame mode only). "
                             "truncate: first N frames; uniform: evenly-spaced N frames.")
    parser.add_argument("--max_frames",     type=int, default=None,
                        help="Frame budget (frame mode only). Required for 'uniform'.")
    parser.add_argument("--with_doc",       action="store_true",
                        help="Prepend aligned_full_document as plain text context (ablation).")
    parser.add_argument("--test_mode",      action="store_true",
                        help="Compute and save input quality statistics per video.")
    parser.add_argument("--video",          default=None, help="Single video ID for smoke test.")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    args = parser.parse_args()

    if args.frame_strategy == "uniform" and args.max_frames is None:
        parser.error("--max_frames is required when --frame_strategy is 'uniform'.")
    if args.mode == "shot" and args.frame_strategy:
        parser.error("--frame_strategy is only supported for --mode frame.")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    suffix       = _make_suffix(args.mode, args.frame_strategy, args.max_frames, args.with_doc)
    if args.video_ids_file:
        import os as _os
        suffix += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    base_name    = f"internvideo_{args.dataset}{suffix}"
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
    print(f"\nRunning {MODEL_DISPLAY} | mode={args.mode} | "
          f"frame_strategy={args.frame_strategy} max_frames={args.max_frames} "
          f"dataset={args.dataset} ({len(video_ids)} videos)\n")

    for i, vid in enumerate(video_ids):
        if vid in processed_ids:
            print(f"[{i+1}/{len(video_ids)}] {vid} (skipped)")
            continue
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        if args.mode == "frame":
            all_results.append(run_video_frame_mode(
                model, tokenizer, args.dataset, vid,
                frame_strategy=args.frame_strategy,
                max_frames=args.max_frames,
                with_doc=args.with_doc,
                test_mode=args.test_mode,
            ))
        else:
            all_results.append(run_video_shot_mode(
                model, tokenizer, args.dataset, vid,
                with_doc=args.with_doc,
                test_mode=args.test_mode,
            ))
        with open(partial_file, "w") as f:
            json.dump({"video_results": all_results}, f)

    n_success  = sum(1 for r in all_results if not r["error"])
    n_error    = sum(1 for r in all_results if r["error"])
    n_parse_ok = sum(1 for r in all_results if r["parse_ok"])
    f1_vals    = [r["eval"]["f1"] for r in all_results if r["eval"]["f1"] is not None]
    tau_vals   = [r["eval"]["kendall_tau"] for r in all_results if r["eval"].get("kendall_tau") is not None]
    rho_vals   = [r["eval"]["spearman_rho"] for r in all_results if r["eval"].get("spearman_rho") is not None]
    avg_f1     = float(np.mean(f1_vals)) if f1_vals else None
    avg_tau    = float(np.mean(tau_vals)) if tau_vals else None
    avg_rho    = float(np.mean(rho_vals)) if rho_vals else None

    print(f"\nDone: {n_success} success, {n_error} errors, "
          f"parse_ok={n_parse_ok}/{len(all_results)}, "
          f"avg_F1={f'{avg_f1:.3f}' if avg_f1 is not None else 'N/A'}  "
          f"avg_τ={f'{avg_tau:.3f}' if avg_tau is not None else 'N/A'}  "
          f"avg_ρ={f'{avg_rho:.3f}' if avg_rho is not None else 'N/A'}")

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
        "model":           MODEL_DISPLAY,
        "dataset":         args.dataset,
        "mode":            args.mode,
        "frame_strategy":  args.frame_strategy,
        "max_frames":      args.max_frames,
        "with_doc":        args.with_doc,
        "test_mode":       args.test_mode,
        "timestamp":       timestamp,
        "n_success":       n_success,
        "n_error":         n_error,
        "n_parse_ok":      n_parse_ok,
        "avg_f1":          avg_f1,
        "avg_kendall_tau": avg_tau,
        "avg_spearman_rho": avg_rho,
        "test_stats_agg":  test_stats_agg,
        "video_results":   all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    partial_file.unlink(missing_ok=True)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
