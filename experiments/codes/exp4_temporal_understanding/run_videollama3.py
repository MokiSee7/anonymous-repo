"""
exp4_temporal_understanding: Temporal reordering with VideoLLaMA3-7B.

Two modes:
  --mode shot (default):
      Shuffled shot frames → model reorders shots.
      --no_doc: shot-frames only (ablation: does document help?)
      Output: "Shot X > Shot Y > ..."

  --mode sentence:
      Shuffled sentences + representative frames (SENTENCE_FRAMES_PER_SHOT per shot).
      --text_only: sentences only, no frames.
      Output: "Sentence X > Sentence Y > ..."

Evaluation: Spearman ρ, Kendall τ, Top-K Precision.

Environment: videollama3
Usage:
    python run_videollama3.py --dataset summe --video video_1
    python run_videollama3.py --dataset summe --mode shot
    python run_videollama3.py --dataset summe --mode shot --no_doc
    python run_videollama3.py --dataset summe --mode sentence
    python run_videollama3.py --dataset summe --mode sentence --text_only
"""

import sys
import gc
import json
import random
import argparse
import re
import traceback
import hashlib
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_shot_frames, load_video_meta
from shared.evaluate_ranking import compute_ranking_metrics

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "/data/hf-cache/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B"
MODEL_DISPLAY = "VideoLLaMA3-7B"
MAX_NEW_TOKENS           = 128
N_SHOTS                  = 5
N_SENTENCES              = 5
RANDOM_SEED              = 42
SENTENCE_FRAMES_PER_SHOT = 4


# ── Per-video deterministic RNG ────────────────────────────────────────────────
def _video_rng(global_seed, video_id):
    h = int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16)
    return random.Random((global_seed + h) % (2 ** 32))


# ── Frame sampling ─────────────────────────────────────────────────────────────
def _sample_frames_uniform(frames, k):
    if not frames:
        return []
    if len(frames) <= k:
        return frames + [frames[-1]] * (k - len(frames))
    indices = [int(i * len(frames) / k) for i in range(k)]
    return [frames[i] for i in indices]


# ── Parsing ────────────────────────────────────────────────────────────────────
def _parse_ranking(text, n, primary_pattern, label_name):
    if not text or not text.strip():
        return None, "empty_output"
    nums = re.findall(primary_pattern, text, re.IGNORECASE)
    if not nums:
        nums = re.findall(r'\b(\d+)\b', text)
        if not nums:
            return None, "wrong_format"
    nums = [int(x) for x in nums]
    seen, deduped, has_dup = set(), [], False
    for x in nums:
        if x in seen:
            has_dup = True
        else:
            seen.add(x)
            deduped.append(x)
    if has_dup:
        return None, "duplicated_items"
    expected = set(range(1, n + 1))
    if set(deduped) != expected:
        return None, "missing_items" if (expected - set(deduped)) else "wrong_format"
    if len(deduped) != n:
        return None, "wrong_format"
    return deduped, None


def parse_shot_ranking(text, n):
    return _parse_ranking(text, n, r'Shot\s+(\d+)', "Shot")


def parse_sentence_ranking(text, n):
    return _parse_ranking(text, n, r'Sentence\s+(\d+)', "Sentence")


# ── Data loading ───────────────────────────────────────────────────────────────
_data_cache: dict = {}

def _get_video_data(dataset_name, video_id):
    if dataset_name not in _data_cache:
        import os
        path = f"/data/MMS_Benchmark/data/processed/{dataset_name}_data.json"
        if os.path.exists(path):
            with open(path) as f:
                _data_cache[dataset_name] = json.load(f)
        else:
            _data_cache[dataset_name] = {}
    return _data_cache[dataset_name].get(video_id, {})


def load_sentences(dataset_name, video_id):
    doc = _get_video_data(dataset_name, video_id).get("aligned_full_document", {})
    sentences = doc.get("sentences", []) if isinstance(doc, dict) else []
    return [s for s in sentences if isinstance(s, str) and s.strip()]


def load_shot_all_frames(dataset_name, video_id, shot_indices, meta):
    result = []
    for si in shot_indices:
        frames, _ = load_shot_frames(dataset_name, video_id, si, meta)
        result.append((si, frames))
    return result


# ── Model loading ──────────────────────────────────────────────────────────────
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


# ── Inference ──────────────────────────────────────────────────────────────────
def _infer(model, processor, frames, text):
    """Visual inference: flat list of PIL frames + text prompt."""
    out = {"text": "", "n_output_tokens": None, "error": None}
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
        out["n_output_tokens"] = int(generated.shape[-1])
        out["text"] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    except Exception as e:
        out["error"] = str(e)
    return out


def _infer_text(model, processor, text):
    """Text-only inference."""
    out = {"text": "", "n_output_tokens": None, "error": None}
    try:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]
        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated = output_ids  # generate() returns only new tokens
        out["n_output_tokens"] = int(generated.shape[-1])
        out["text"] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    except Exception as e:
        out["error"] = str(e)
    return out


# ── Prompt builders ────────────────────────────────────────────────────────────
def load_full_document_text(dataset_name, video_id):
    sentences = load_sentences(dataset_name, video_id)
    return " ".join(sentences) if sentences else None


def build_shot_reorder_prompt(shuffled_shot_frames, shot_frame_ranges, document_text):
    """
    Text prompt for shot reordering.
    shot_frame_ranges: list of (start_1idx, end_1idx) for each shot label
    """
    n = len(shuffled_shot_frames)
    lines = []
    for label_idx, (start, end) in enumerate(shot_frame_ranges):
        if start == end:
            lines.append(f"Shot {label_idx + 1}: Frame {start}")
        else:
            lines.append(f"Shot {label_idx + 1}: Frames {start}-{end}")
    shot_list = ", ".join(f"Shot {i}" for i in range(1, n + 1))
    prompt = (
        f"You are given frames from {n} video shots, shown in shuffled order.\n"
        "The frame layout is:\n" + "\n".join(lines) + "\n\n"
    )
    if document_text:
        prompt += f"The video's full text description:\n{document_text}\n\n"
    prompt += (
        f"You have been shown {n} shots ({shot_list}), presented in shuffled order.\n"
        "Reorder ALL shots to match the actual temporal sequence of the video, from first to last.\n\n"
        "Output ONLY the ordering in this exact format (use > between shots):\n"
        "Shot X > Shot Y > Shot Z > ...\n"
        "Include all shots exactly once."
    )
    return prompt


def build_sentence_reorder_prompt(shuffled_sentences, n_frames_total):
    n = len(shuffled_sentences)
    lines = [f"Sentence {i + 1}: {s}" for i, s in enumerate(shuffled_sentences)]
    sentence_list = ", ".join(f"Sentence {i}" for i in range(1, n + 1))
    return (
        f"Below are {n} sentences describing events in a video, shown in shuffled order:\n"
        + "\n".join(lines)
        + f"\n\nYou are also shown {n_frames_total} representative frames from the video.\n\n"
        f"You have been shown {n} sentences ({sentence_list}), presented in shuffled order.\n"
        "Reorder ALL sentences to match the actual temporal sequence of the video, from first to last.\n\n"
        "Output ONLY the ordering in this exact format (use > between sentences):\n"
        "Sentence X > Sentence Y > Sentence Z > ...\n"
        "Include all sentences exactly once."
    )


def build_sentence_text_only_prompt(shuffled_sentences):
    n = len(shuffled_sentences)
    lines = [f"Sentence {i + 1}: {s}" for i, s in enumerate(shuffled_sentences)]
    sentence_list = ", ".join(f"Sentence {i}" for i in range(1, n + 1))
    return (
        f"Below are {n} sentences describing events in a video, shown in shuffled order:\n"
        + "\n".join(lines)
        + f"\n\nYou have been shown {n} sentences ({sentence_list}), presented in shuffled order.\n"
        "Reorder ALL sentences to match the actual temporal sequence of the video, from first to last.\n\n"
        "Output ONLY the ordering in this exact format (use > between sentences):\n"
        "Sentence X > Sentence Y > Sentence Z > ...\n"
        "Include all sentences exactly once."
    )


# ── Per-video: shot reordering ─────────────────────────────────────────────────
def run_video_shot(model, processor, dataset_name, video_id, rng, with_doc=True):
    meta = load_video_meta(dataset_name, video_id)
    n_shots_total = len(meta["n_frame_per_seg"])

    if n_shots_total == 0:
        return _skip_result(video_id, "no_shots")

    valid_shot_indices = [i for i in range(n_shots_total) if meta["n_sampled_per_shot"][i] > 0]
    if not valid_shot_indices:
        return _skip_result(video_id, "no_sampled_frames")

    n = min(N_SHOTS, len(valid_shot_indices))
    selected_orig_indices = sorted(rng.sample(valid_shot_indices, n))
    shuffled_order = list(range(n))
    rng.shuffle(shuffled_order)
    shuffled_shot_orig_indices = [selected_orig_indices[i] for i in shuffled_order]

    print(f"  shots: {n_shots_total} total, {n} selected | orig={selected_orig_indices}")

    shot_frames_list = load_shot_all_frames(dataset_name, video_id, shuffled_shot_orig_indices, meta)
    n_frames_per_shot = [len(f) for _, f in shot_frames_list]

    # Flatten frames and compute ranges
    all_frames = []
    shot_frame_ranges = []
    cursor = 1
    for _, frames in shot_frames_list:
        start = cursor
        end = cursor + len(frames) - 1
        shot_frame_ranges.append((start, end))
        all_frames.extend(frames)
        cursor = end + 1

    n_frames_original = len(all_frames)
    print(f"  total_frames={n_frames_original}")

    document_text = load_full_document_text(dataset_name, video_id) if with_doc else None
    prompt = build_shot_reorder_prompt(shot_frames_list, shot_frame_ranges, document_text)
    infer_out = _infer(model, processor, all_frames, prompt)

    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    error_log = None

    if infer_out["error"]:
        print(f"  ERROR: {infer_out['error'][:120]}")
        parsed_ranking, parse_failure_reason = None, None
        metrics = {"spearman_r": None, "kendall_tau": None}
        gt_scores = []
        error_log = {"video_id": video_id, "error_type": "InferenceError", "error_message": infer_out["error"]}
    else:
        print(f"  out={infer_out['n_output_tokens']}  frames={n_frames_original}")
        print(f"  raw: {infer_out['text'][:120]}")
        parsed_ranking, parse_failure_reason = parse_shot_ranking(infer_out["text"], n)
        if parsed_ranking is None:
            print(f"  [WARN] parse_fail={parse_failure_reason}")
            metrics = {"spearman_r": None, "kendall_tau": None}
            gt_scores = []
        else:
            max_idx = max(shuffled_shot_orig_indices)
            gt_scores = [float(max_idx - shuffled_shot_orig_indices[k]) for k in range(n)]
            metrics = compute_ranking_metrics(parsed_ranking, gt_scores)
            k_key = f"top{min(3, n)}_precision"
            print(f"  {' > '.join(f'Shot {x}' for x in parsed_ranking)}")
            print(f"  Spearman={_fmt(metrics['spearman_r'])}  Kendall={_fmt(metrics['kendall_tau'])}"
                  f"  Top{min(3,n)}-P={_fmt(metrics.get(k_key))}")

    return {
        "video_id":                   video_id,
        "mode":                       "shot",
        "with_doc":                   with_doc,
        "skipped":                    False,
        "skip_reason":                None,
        "n_shots_total":              n_shots_total,
        "n_shots_selected":           n,
        "selected_orig_shot_indices": selected_orig_indices,
        "shuffled_shot_orig_indices": shuffled_shot_orig_indices,
        "n_frames_per_shot":          n_frames_per_shot,
        "n_frames_original":          n_frames_original,
        "has_document":               document_text is not None,
        "n_output_tokens":            infer_out["n_output_tokens"],
        "raw_output":                 infer_out["text"],
        "parsed_ranking":             parsed_ranking,
        "parse_failure_reason":       parse_failure_reason,
        "gt_scores":                  gt_scores,
        "spearman_r":                 metrics["spearman_r"],
        "kendall_tau":                metrics["kendall_tau"],
        "top3_precision":             metrics.get(f"top{min(3, n)}_precision"),
        "error":                      infer_out["error"],
        "error_log":                  error_log,
    }


# ── Per-video: sentence reordering ────────────────────────────────────────────
def run_video_sentence(model, processor, dataset_name, video_id, text_only, rng):
    sentences = load_sentences(dataset_name, video_id)
    if not sentences:
        return _skip_result(video_id, "no_sentences")

    n = min(N_SENTENCES, len(sentences))
    selected_orig_indices = sorted(rng.sample(range(len(sentences)), n))
    selected_sentences = [sentences[i] for i in selected_orig_indices]
    shuffled_indices = list(range(n))
    rng.shuffle(shuffled_indices)
    shuffled_sentences = [selected_sentences[i] for i in shuffled_indices]
    shuffled_labels = [selected_orig_indices[i] for i in shuffled_indices]

    if text_only:
        n_frames_per_shot, n_frames_submitted = [], 0
        print(f"  sentences: {len(sentences)} total, {n} selected | text_only")
        prompt = build_sentence_text_only_prompt(shuffled_sentences)
        infer_out = _infer_text(model, processor, prompt)
    else:
        meta = load_video_meta(dataset_name, video_id)
        n_shots_total = len(meta["n_frame_per_seg"])
        all_frames, n_frames_per_shot = [], []
        for si in range(n_shots_total):
            frames, _ = load_shot_frames(dataset_name, video_id, si, meta)
            if frames:
                sampled = _sample_frames_uniform(frames, SENTENCE_FRAMES_PER_SHOT)
                all_frames.extend(sampled)
                n_frames_per_shot.append(len(sampled))
        n_frames_submitted = len(all_frames)
        print(f"  sentences: {len(sentences)} total, {n} selected | frames={n_frames_submitted}")
        prompt = build_sentence_reorder_prompt(shuffled_sentences, n_frames_submitted)
        infer_out = _infer(model, processor, all_frames, prompt)

    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    error_log = None

    if infer_out["error"]:
        print(f"  ERROR: {infer_out['error'][:120]}")
        parsed_ranking, parse_failure_reason = None, None
        metrics = {"spearman_r": None, "kendall_tau": None}
        gt_scores = []
        error_log = {"video_id": video_id, "error_type": "InferenceError", "error_message": infer_out["error"]}
    else:
        print(f"  out={infer_out['n_output_tokens']}")
        print(f"  raw: {infer_out['text'][:120]}")
        parsed_ranking, parse_failure_reason = parse_sentence_ranking(infer_out["text"], n)
        if parsed_ranking is None:
            print(f"  [WARN] parse_fail={parse_failure_reason}")
            metrics = {"spearman_r": None, "kendall_tau": None}
            gt_scores = []
        else:
            max_pos = max(shuffled_labels)
            gt_scores = [float(max_pos - shuffled_labels[k]) for k in range(n)]
            metrics = compute_ranking_metrics(parsed_ranking, gt_scores)
            k_key = f"top{min(3, n)}_precision"
            print(f"  {' > '.join(f'Sentence {x}' for x in parsed_ranking)}")
            print(f"  Spearman={_fmt(metrics['spearman_r'])}  Kendall={_fmt(metrics['kendall_tau'])}"
                  f"  Top{min(3,n)}-P={_fmt(metrics.get(k_key))}")

    return {
        "video_id":               video_id,
        "mode":                   "sentence",
        "text_only":              text_only,
        "skipped":                False,
        "skip_reason":            None,
        "n_sentences_total":      len(sentences),
        "n_sentences_selected":   n,
        "shuffled_labels":        shuffled_labels,
        "original_positions":     selected_orig_indices,
        "n_frames_per_shot":      n_frames_per_shot,
        "n_frames_submitted":     n_frames_submitted if not text_only else 0,
        "n_output_tokens":        infer_out["n_output_tokens"],
        "sentences_shown":        shuffled_sentences,
        "raw_output":             infer_out["text"],
        "parsed_ranking":         parsed_ranking,
        "parse_failure_reason":   parse_failure_reason,
        "gt_scores":              gt_scores,
        "spearman_r":             metrics["spearman_r"],
        "kendall_tau":            metrics["kendall_tau"],
        "top3_precision":         metrics.get(f"top{min(3, n)}_precision"),
        "error":                  infer_out["error"],
        "error_log":              error_log,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _skip_result(video_id, reason):
    return {
        "video_id": video_id, "skipped": True, "skip_reason": reason,
        "n_output_tokens": None, "raw_output": None,
        "parsed_ranking": None, "parse_failure_reason": None,
        "gt_scores": [], "spearman_r": None, "kendall_tau": None,
        "top3_precision": None, "error": None, "error_log": None,
    }


def _fatal_error_result(video_id, mode, err_info):
    return {
        "video_id": video_id, "mode": mode, "skipped": False, "skip_reason": None,
        "fatal_error": True, "parsed_ranking": None, "parse_failure_reason": None,
        "gt_scores": [], "spearman_r": None, "kendall_tau": None, "top3_precision": None,
        "n_output_tokens": None, "raw_output": None,
        "error": err_info["error_message"], "error_log": err_info,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--mode",      choices=["shot", "sentence"], default="shot")
    parser.add_argument("--no_doc",    action="store_true",
                        help="(shot mode) Ablation: no document text, shot frames only")
    parser.add_argument("--text_only", action="store_true",
                        help="(sentence mode) Ablation: no frames, sentences only")
    parser.add_argument("--video",          default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line.")
    parser.add_argument("--seed",           type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if args.no_doc and args.mode != "shot":
        parser.error("--no_doc only applies to --mode shot")
    if args.text_only and args.mode != "sentence":
        parser.error("--text_only only applies to --mode sentence")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "shot":
        mode_tag = "shot_nodoc" if args.no_doc else "shot"
    elif args.text_only:
        mode_tag = "sentence_text_only"
    else:
        mode_tag = "sentence"

    stem = f"videollama3_{args.dataset}_{mode_tag}_seed{args.seed}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    if args.video:
        out_file     = results_dir / f"{stem}_{args.video}_{timestamp}.json"
        partial_file = None
    else:
        out_file     = results_dir / f"{stem}_{timestamp}.json"
        partial_file = results_dir / f"{stem}.partial.json"

    completed_ids: set = set()
    all_results: list = []
    if partial_file is not None and partial_file.exists():
        with open(partial_file) as f:
            all_results = json.load(f)
        completed_ids = {r["video_id"] for r in all_results}
        print(f"  Resuming: {len(completed_ids)} videos already done")

    model, processor = load_model()

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_DISPLAY} on {args.dataset} "
          f"({len(video_ids)} videos, mode={mode_tag}, seed={args.seed})\n")

    error_log = []
    for i, vid in enumerate(video_ids):
        if vid in completed_ids:
            print(f"[{i+1}/{len(video_ids)}] {vid}  [already done]")
            continue
        print(f"[{i+1}/{len(video_ids)}] {vid}")
        result = None
        last_exc = None
        for attempt in range(1, 4):
            rng = _video_rng(args.seed, vid)
            try:
                if args.mode == "shot":
                    result = run_video_shot(model, processor, args.dataset, vid, rng,
                                            with_doc=not args.no_doc)
                else:
                    result = run_video_sentence(model, processor, args.dataset, vid,
                                                args.text_only, rng)
                break
            except Exception as e:
                last_exc = e
                print(f"  [attempt {attempt}/3] {type(e).__name__}: {e}")
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

        if result is None:
            err_info = {"video_id": vid, "error_type": type(last_exc).__name__,
                        "error_message": str(last_exc), "attempt_count": 3}
            error_log.append(err_info)
            result = _fatal_error_result(vid, args.mode, err_info)
        elif result.get("error_log") is not None:
            error_log.append(result["error_log"])

        all_results.append(result)
        if partial_file is not None:
            with open(partial_file, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    valid    = [r for r in all_results
                if not r["skipped"] and not r.get("error") and not r.get("fatal_error")]
    n_skip   = sum(1 for r in all_results if r["skipped"])
    n_error  = sum(1 for r in all_results
                   if not r["skipped"] and (r.get("error") or r.get("fatal_error")))
    n_parsed = sum(1 for r in valid if r["parsed_ranking"] is not None)

    spearman_vals = [r["spearman_r"]     for r in valid if r["spearman_r"]     is not None]
    kendall_vals  = [r["kendall_tau"]    for r in valid if r["kendall_tau"]     is not None]
    top3_vals     = [r["top3_precision"] for r in valid if r["top3_precision"]  is not None]
    avg_spearman  = float(np.mean(spearman_vals)) if spearman_vals else None
    avg_kendall   = float(np.mean(kendall_vals))  if kendall_vals  else None
    avg_top3      = float(np.mean(top3_vals))     if top3_vals     else None

    def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
    print(f"\nDone: {len(valid)} success, {n_parsed} parsed, {n_error} errors, {n_skip} skipped")
    print(f"  Spearman={fmt(avg_spearman)}  Kendall={fmt(avg_kendall)}  Top3-P={fmt(avg_top3)}")

    output = {
        "model":                    MODEL_DISPLAY,
        "dataset":                  args.dataset,
        "mode":                     args.mode,
        "mode_tag":                 mode_tag,
        "no_doc":                   args.no_doc,
        "text_only":                args.text_only,
        "timestamp":                timestamp,
        "seed":                     args.seed,
        "n_shots_max":              N_SHOTS,
        "n_sentences_max":          N_SENTENCES,
        "sentence_frames_per_shot": SENTENCE_FRAMES_PER_SHOT,
        "n_total":                  len(all_results),
        "n_success":                len(valid),
        "n_parsed":                 n_parsed,
        "n_error":                  n_error,
        "n_skipped":                n_skip,
        "avg_spearman":             avg_spearman,
        "avg_kendall":              avg_kendall,
        "avg_top3_prec":            avg_top3,
        "error_log":                error_log,
        "video_results":            all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_file}")

    if partial_file is not None and partial_file.exists():
        partial_file.unlink()
        print("Partial checkpoint removed.")


if __name__ == "__main__":
    main()
