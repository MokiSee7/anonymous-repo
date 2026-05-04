"""
exp2_v2v: Video-to-video key frame selection with Qwen2.5-VL / Qwen3-VL.

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

Frame input strategies (--frame_strategy, frame mode only):
  truncate : submit the first max_frames frames
  uniform  : uniformly downsample to max_frames frames
  (default): submit all frames — no cap

Optional (--shot_rep, frame mode only): for each shot, call the model to select the
  single most representative frame, then submit those frames (one per shot, temporal order)
  for the selection task. The frame-to-shot mapping simplifies to frame i → shot i-1.
  Can be combined with --frame_strategy / --max_frames (applied after rep collection).

Optional (--with_doc): prepend aligned_full_document as plain text context to each prompt.
  Ablation: measures how much textual grounding helps frame selection.

Environment: MLLM
Usage:
    python run_qwen.py --model qwen2.5 --dataset summe
    python run_qwen.py --model qwen2.5 --dataset summe --mode shot
    python run_qwen.py --model qwen2.5 --dataset summe --frame_strategy truncate --max_frames 500
    python run_qwen.py --model qwen2.5 --dataset summe --frame_strategy uniform  --max_frames 500
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
from shared.data_loader import get_video_ids, load_all_frames, load_all_shots, load_aligned_document, get_gt_shot_scores
from shared.evaluate_sets import set_f1
from shared.evaluate import compute_rank_correlations, get_gt_frame_indices
from shared.frame_utils import apply_frame_strategy, compute_dynamic_max_frames
from shared.test_stats import compute_frame_stats, aggregate_frame_stats, aggregate_token_stats

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen2.5": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "display": "Qwen2.5-VL-7B-Instruct"},
    "qwen3":   {"model_id": "Qwen/Qwen3-VL-8B-Instruct",   "display": "Qwen3-VL-8B-Instruct"},
}
HF_CACHE_DIR   = "/data/hf-cache/hub"
MAX_NEW_TOKENS = 2048
# Default resolution: 448×448 (= 32×14 px, tiles perfectly with patch_size=14).
# tokens/frame = 448*448/784 = 256.  500 frames × 256 = 128K tokens (fits in Qwen's 128K ctx).
DEFAULT_MIN_PIXELS = 448 * 448   # 200704
DEFAULT_MAX_PIXELS = 448 * 448   # 200704

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to identify the most important frames in a video."
)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _doc_prefix(doc_text):
    if not doc_text:
        return ""
    return (
        "The following is a detailed shot-by-shot description of this video for context:\n"
        f"{doc_text}\n\n"
    )

def build_frame_mode_content(frames, n_select, doc_text=None, n_total_sampled=None, n_select_target=None):
    """Frame mode: submitted frames labeled Frame 1..N."""
    n = len(frames)
    content = [{"type": "text", "text": f"Below are {n} frames from a video in chronological order:\n"}]
    for i, img in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {i + 1}:\n"})
        content.append({"type": "image", "image": img})
    content.append({
        "type": "text",
        "text": (
            "\n" + _doc_prefix(doc_text)
            + (
                f"The full sampled video contains {n_total_sampled} frames, so the 15% target budget is "
                f"{n_select_target} frame(s). Based on the {n} frames shown here, select exactly {n_select} "
                "important frame(s).\n\n"
                if n_total_sampled is not None and n_select_target is not None and n_total_sampled != n
                else ""
            )
            + f"From these {n} frames, select the {n_select} most important frames "
            "that best represent the key moments of the video.\n\n"
            "Output ONLY a JSON array of frame numbers (1-indexed integers), "
            f"with exactly {n_select} elements. Example: [1, 5, 12]\n"
            "Do not include any explanation."
        ),
    })
    return content


def build_shot_mode_content(rep_frames, n_select, doc_text=None):
    """
    Shot mode (alternative): representative frames sampled at 20% per shot (min 1),
    all labeled as Frame 1..N in chronological order.
    rep_frames: list of PIL.Image in chronological order across all shots
    """
    n = len(rep_frames)
    content = [{"type": "text", "text": f"Below are {n} representative frames from a video in chronological order:\n"}]
    for i, img in enumerate(rep_frames):
        content.append({"type": "text", "text": f"Frame {i + 1}:\n"})
        content.append({"type": "image", "image": img})
    content.append({
        "type": "text",
        "text": (
            "\n" + _doc_prefix(doc_text)
            + f"From these {n} frames, select the {n_select} most important frames "
            "that best represent the key moments of the video.\n\n"
            "Output ONLY a JSON array of frame numbers (1-indexed integers), "
            f"with exactly {n_select} elements. Example: [1, 5, 12]\n"
            "Do not include any explanation."
        ),
    })
    return content


# ── Output parsing ─────────────────────────────────────────────────────────────

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


# ── Frame-to-shot mapping ─────────────────────────────────────────────────────

def frames_to_shots(selected_frame_positions, meta):
    """
    Map selected frame positions (1-indexed, in picks order) to shot indices.

    Args:
        selected_frame_positions: list of 1-indexed frame positions
        meta: dict from load_video_meta with n_sampled_per_shot

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

    Returns (rep_frame, rep_idx, parse_ok):
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
    fallback = n // 2
    return shot_frames[fallback], fallback, False


def _collect_shot_reps(model, processor, dataset_name, video_id):
    """
    Collect 1 model-selected representative frame per shot.

    Returns:
        meta              : video metadata dict (from load_all_shots)
        rep_frames        : list[PIL.Image]  — one per shot
        rep_frame_indices : list[int]        — original frame index of each rep
        rep_parse_ok      : list[bool]
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

def run_video_frame_mode(model, processor, dataset_name, video_id,
                         frame_strategy=None, max_frames=None, shot_rep=False, with_doc=False,
                         test_mode=False):
    """
    Frame mode: submit frames, ask to select 15%.

    Frame selection:
      - If shot_rep: 1 model-selected rep frame per shot; frame i → shot i-1.
      - Else: all sampled frames (with optional frame_strategy cap).
    """
    shot_rep_info = None

    if shot_rep:
        meta, frames, frame_indices, rep_parse_ok = _collect_shot_reps(
            model, processor, dataset_name, video_id
        )
        n_shots = len(frames)
        shot_rep_info = {
            "n_shots":              n_shots,
            "parse_ok_count":       sum(rep_parse_ok),
            "selected_frame_indices": frame_indices,
        }
        print(f"  [shot_rep] {n_shots} shots, parse_ok={sum(rep_parse_ok)}/{n_shots}")
    else:
        meta, frames, frame_indices = load_all_frames(dataset_name, video_id)
        n_total_sampled = len(frame_indices)
        if frame_strategy and max_frames:
            frames, frame_indices = apply_frame_strategy(
                frames, frame_indices, max_frames, frame_strategy
            )
            # Rebuild n_sampled_per_shot for the subsampled picks
            sub_picks = np.array(frame_indices)
            n_sampled_sub = []
            for start, end in meta["change_points"]:
                n_sampled_sub.append(int(np.sum((sub_picks >= start) & (sub_picks <= end))))
            meta["n_sampled_per_shot"] = np.array(n_sampled_sub)

    n_submitted = len(frames)
    n_total_sampled = n_submitted if shot_rep else n_total_sampled
    n_select_target = max(1, math.ceil(n_total_sampled * 0.15))
    n_select = min(n_submitted, n_select_target)
    gt_shots    = sorted(np.where(meta["shot_level_gt"] == 1)[0].tolist())

    doc_text = load_aligned_document(dataset_name, video_id) if with_doc else None
    print(f"  Frames: {n_submitted} / total_sampled={n_total_sampled}  |  "
          f"select: {n_select} (target={n_select_target})  |  gt_shots: {gt_shots}  |  "
          f"doc: {'yes' if doc_text else 'no'}")

    content = build_frame_mode_content(
        frames, n_select, doc_text, n_total_sampled=n_total_sampled, n_select_target=n_select_target
    )
    r = _infer(model, processor, content)

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return _make_result(video_id, meta, "frame", n_submitted, n_select, r,
                            [], [], gt_shots, False, None,
                            frame_strategy=frame_strategy, max_frames=max_frames,
                            shot_rep=shot_rep, shot_rep_info=shot_rep_info,
                            n_total_sampled=n_total_sampled, n_select_target=n_select_target)

    trunc = " [TRUNCATED]" if r["was_output_truncated"] else ""
    print(f"  in={r['n_input_tokens']} out={r['n_output_tokens']}{trunc}  raw: {r['text'][:80]}")

    selected_frames, parse_ok = parse_index_list(r["text"], n_submitted, n_select)

    if shot_rep:
        # Each submitted frame i (1-indexed) corresponds directly to shot i-1 (0-indexed)
        predicted_shots = sorted(set(f - 1 for f in selected_frames if 1 <= f <= len(frames)))
    else:
        predicted_shots = frames_to_shots(selected_frames, meta)

    # Frame-level F1: compare selected pick indices vs GT pick indices
    pred_frame_set  = {frame_indices[f - 1] for f in selected_frames if 0 <= f - 1 < len(frame_indices)}
    gt_frame_set    = get_gt_frame_indices(meta["shot_level_gt"], meta["n_sampled_per_shot"], meta["picks"])
    eval_result     = set_f1(pred_frame_set, gt_frame_set)
    # Rank correlations at shot level
    gt_shot_scores  = get_gt_shot_scores(meta["gtscore"], meta["n_sampled_per_shot"])
    eval_result.update(compute_rank_correlations(predicted_shots, len(meta["n_sampled_per_shot"]), gt_shot_scores))
    print(f"DEBUG: eval_result = {eval_result}")
    
    # Format correlation values safely
    kendall_str = f"{eval_result['kendall_tau']:.3f}" if eval_result['kendall_tau'] is not None else 'N/A'
    spearman_str = f"{eval_result['spearman_rho']:.3f}" if eval_result['spearman_rho'] is not None else 'N/A'
    
    print(f"  predicted shots: {predicted_shots}  |  F1: {eval_result['f1']:.3f}  "
          f"τ={kendall_str}  ρ={spearman_str}  parse_ok: {parse_ok}")

    test_stats = compute_frame_stats(frame_indices, meta) if test_mode else None

    return _make_result(video_id, meta, "frame", n_submitted, n_select, r,
                        selected_frames, predicted_shots, gt_shots, parse_ok, eval_result,
                        frame_strategy=frame_strategy, max_frames=max_frames,
                        shot_rep=shot_rep, shot_rep_info=shot_rep_info,
                        test_stats=test_stats,
                        n_total_sampled=n_total_sampled, n_select_target=n_select_target)


def run_video_shot_mode(model, processor, dataset_name, video_id, with_doc=False, test_mode=False):
    """
    Shot mode (alternative): uniformly sample 20% of each shot's frames (min 1) as
    representative frames, submit them all, ask model to select 15% of submitted frames.

    GT: the top-gtscore 15% of the same submitted representative frames.
    Evaluation: F1 between model selection and GT (both as 1-indexed submitted positions).
    """
    from PIL import Image as PILImage

    meta, shots = load_all_shots(dataset_name, video_id)

    # Build representative frames: 20% per shot (min 1), uniform sampling
    rep_frames        = []   # PIL Images in chronological order
    rep_gtscores      = []   # gtscore for each representative frame
    rep_frame_indices = []   # original frame index (picks value) for test_mode stats

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
        sample_idx = np.linspace(0, n_in_shot - 1, n_rep, dtype=int)
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

    n_submitted = len(rep_frames)
    n_select    = max(1, math.ceil(n_submitted * 0.15))

    # GT: top-gtscore 15% of submitted representative frames (1-indexed positions)
    sorted_by_score = sorted(range(n_submitted), key=lambda i: rep_gtscores[i], reverse=True)
    gt_positions    = sorted(sorted_by_score[:n_select])          # 0-indexed
    gt_positions_1idx = [p + 1 for p in gt_positions]            # 1-indexed for F1

    doc_text = load_aligned_document(dataset_name, video_id) if with_doc else None
    print(f"  Shots: {len(shots)}  |  rep_frames: {n_submitted}  |  select: {n_select}  |  gt_pos (1-idx): {gt_positions_1idx[:8]}...  |  doc: {'yes' if doc_text else 'no'}")

    content = build_shot_mode_content(rep_frames, n_select, doc_text)
    r = _infer(model, processor, content)

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return _make_result(video_id, meta, "shot", n_submitted, n_select, r,
                            [], [], gt_positions_1idx, False,
                            extra={"rep_gtscores": rep_gtscores})

    trunc = " [TRUNCATED]" if r["was_output_truncated"] else ""
    print(f"  in={r['n_input_tokens']} out={r['n_output_tokens']}{trunc}  raw: {r['text'][:80]}")

    selected_1idx, parse_ok = parse_index_list(r["text"], n_submitted, n_select)
    eval_result = set_f1(selected_1idx, gt_positions_1idx)
    eval_result.update(compute_rank_correlations([p - 1 for p in selected_1idx], n_submitted, rep_gtscores))
    print(f"  selected: {selected_1idx[:8]}  |  F1: {eval_result['f1']:.3f}  "
          f"τ={eval_result['kendall_tau']:.3f if eval_result['kendall_tau'] is not None else 'N/A'}  "
          f"ρ={eval_result['spearman_rho']:.3f if eval_result['spearman_rho'] is not None else 'N/A'}  parse_ok: {parse_ok}")

    test_stats = compute_frame_stats(rep_frame_indices, meta) if test_mode else None

    return _make_result(video_id, meta, "shot", n_submitted, n_select, r,
                        selected_1idx, selected_1idx, gt_positions_1idx, parse_ok, eval_result,
                        extra={"rep_gtscores": rep_gtscores},
                        test_stats=test_stats)


def _make_result(video_id, meta, mode, n_submitted, n_select, r,
                 selected_raw, predicted, gt, parse_ok, eval_result=None, extra=None,
                 frame_strategy=None, max_frames=None, shot_rep=False, shot_rep_info=None,
                 test_stats=None, n_total_sampled=None, n_select_target=None):
    result = {
        "video_id":        video_id,
        "video_name":      meta["video_name"],
        "mode":            mode,
        "frame_strategy":  frame_strategy,
        "max_frames":      max_frames,
        "shot_rep":        shot_rep,
        "shot_rep_info":   shot_rep_info,
        "n_total_sampled": n_total_sampled if n_total_sampled is not None else n_submitted,
        "n_submitted":     n_submitted,
        "n_select_target": n_select_target if n_select_target is not None else n_select,
        "n_select":        n_select,
        "n_input_tokens":  r["n_input_tokens"],
        "n_output_tokens": r["n_output_tokens"],
        "was_output_truncated": r["was_output_truncated"],
        "raw_output":      r["text"],
        "error":           r["error"],
        "parse_ok":        parse_ok,
        # frame mode: 1-indexed frame positions → 0-indexed shot indices (or 1-indexed for shot mode)
        "selected_raw":    selected_raw,
        "predicted":       predicted,
        "gt":              gt,
        "eval":            eval_result or {"precision": None, "recall": None, "f1": None, "kendall_tau": None, "spearman_rho": None},
        "test_stats":      test_stats,
    }
    if extra:
        result.update(extra)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_suffix(mode, frame_strategy, max_frames, shot_rep, with_doc):
    parts = []
    if shot_rep:
        parts.append("shotrep")
    if frame_strategy:
        parts.append(f"{frame_strategy}{max_frames}" if max_frames else frame_strategy)
    if with_doc:
        parts.append("doc")
    suffix = "_".join(parts)
    return f"_{mode}_{suffix}" if suffix else f"_{mode}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          choices=["qwen2.5", "qwen3"], required=True)
    parser.add_argument("--dataset",        default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--mode",           choices=["frame", "shot"], default="frame",
                        help="'frame': select from all sampled frames; "
                             "'shot': select from 20%%-per-shot representative frames.")
    parser.add_argument("--frame_strategy", choices=["truncate", "uniform"], default=None,
                        help="Frame selection strategy (frame mode only). "
                             "truncate: first N (N auto-computed from context if --max_frames omitted); "
                             "uniform: evenly-spaced N (--max_frames required).")
    parser.add_argument("--max_frames",     type=int, default=None,
                        help="Frame budget (frame mode only). Required for 'uniform'. "
                             "For 'truncate': auto-computed from context window if omitted.")
    parser.add_argument("--shot_rep",       action="store_true",
                        help="(frame mode only) For each shot, call model to select 1 "
                             "representative frame, then run frame selection on those.")
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
    parser.add_argument("--video",          default=None, help="Single video ID for smoke test.")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    args = parser.parse_args()

    if args.frame_strategy == "uniform" and args.max_frames is None:
        parser.error("--max_frames is required when --frame_strategy is 'uniform'.")
    if args.frame_strategy == "truncate" and args.max_frames is None and args.with_doc:
        parser.error("--max_frames is required when using truncate with --with_doc.")
    if args.mode == "shot" and (args.frame_strategy or args.shot_rep):
        parser.error("--frame_strategy and --shot_rep are only supported for --mode frame.")

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
        tokens_per_frame     = args.max_pixels // 784
        effective_max_frames = compute_dynamic_max_frames(model, tokens_per_frame)
        print(f"  Dynamic truncation: max_frames={effective_max_frames} "
              f"(context={getattr(model.config, 'max_position_embeddings', 32768)}, "
              f"{tokens_per_frame} tokens/frame)")

    suffix       = _make_suffix(args.mode, args.frame_strategy, effective_max_frames, args.shot_rep, args.with_doc)
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
    print(f"\nRunning {MODEL_CONFIGS[args.model]['display']} | mode={args.mode} | "
          f"frame_strategy={args.frame_strategy} max_frames={effective_max_frames} "
          f"shot_rep={args.shot_rep} dataset={args.dataset} "
          f"min_pixels={eff_min_pixels} max_pixels={eff_max_pixels} "
          f"({len(video_ids)} videos)\n")

    for i, vid in enumerate(video_ids):
        if vid in processed_ids:
            print(f"[{i+1}/{len(video_ids)}] {vid} (skipped)")
            continue
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        if args.mode == "frame":
            all_results.append(run_video_frame_mode(
                model, processor, args.dataset, vid,
                frame_strategy=args.frame_strategy,
                max_frames=effective_max_frames,
                shot_rep=args.shot_rep,
                with_doc=args.with_doc,
                test_mode=args.test_mode,
            ))
        else:
            all_results.append(run_video_shot_mode(
                model, processor, args.dataset, vid,
                with_doc=args.with_doc,
                test_mode=args.test_mode,
            ))
        with open(partial_file, "w") as f:
            json.dump({"video_results": all_results}, f)

    n_success   = sum(1 for r in all_results if not r["error"])
    n_error     = sum(1 for r in all_results if r["error"])
    n_parse_ok  = sum(1 for r in all_results if r["parse_ok"])
    f1_vals     = [r["eval"]["f1"] for r in all_results if r["eval"]["f1"] is not None]
    tau_vals    = [r["eval"]["kendall_tau"] for r in all_results if r["eval"].get("kendall_tau") is not None]
    rho_vals    = [r["eval"]["spearman_rho"] for r in all_results if r["eval"].get("spearman_rho") is not None]
    avg_f1      = float(np.mean(f1_vals)) if f1_vals else None
    avg_tau     = float(np.mean(tau_vals)) if tau_vals else None
    avg_rho     = float(np.mean(rho_vals)) if rho_vals else None

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
        if test_stats_agg["token_stats"]:
            ts = test_stats_agg["token_stats"]
            print(f"  {'n_input_tokens':35s}  min={ts['min']}  max={ts['max']}  mean={ts['mean']:.1f}")

    output = {
        "model":               MODEL_CONFIGS[args.model]["display"],
        "dataset":             args.dataset,
        "mode":                args.mode,
        "frame_strategy":      args.frame_strategy,
        "max_frames":          effective_max_frames,
        "max_frames_dynamic":  args.frame_strategy == "truncate" and args.max_frames is None,
        "shot_rep":            args.shot_rep,
        "with_doc":            args.with_doc,
        "test_mode":           args.test_mode,
        "min_pixels":          eff_min_pixels,
        "max_pixels":          eff_max_pixels,
        "timestamp":           timestamp,
        "n_success":           n_success,
        "n_error":             n_error,
        "n_parse_ok":          n_parse_ok,
        "avg_f1":              avg_f1,
        "avg_kendall_tau":     avg_tau,
        "avg_spearman_rho":    avg_rho,
        "test_stats_agg":      test_stats_agg,
        "video_results":       all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    partial_file.unlink(missing_ok=True)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
