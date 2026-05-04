"""
exp4_temporal_understanding — text-only GPT baseline.

Mirrors exp4 shot-reorder mode (with_doc=True), replacing video frames with
shot text descriptions (clips[i].summary from step2/LLaVA).

Same shot selection, shuffle order, GT scores, and evaluation as the multimodal runs.

Models: gpt-5.4  (Foundry/OpenAI-compatible endpoint)
Environment: exp7  (conda activate exp7)

Setup:
    source /data/MMS_Benchmark/.secrets/openai.env

Usage:
    python run_gpt_baseline.py --model gpt-5.4 --dataset summe
    python run_gpt_baseline.py --model gpt-5.4 --dataset summe --video video_1
    python run_gpt_baseline.py --model gpt-5.4 --dataset mrhisum --video_ids_file mrhisum_100_seed42.txt
"""

import sys
import os
import re
import json
import time
import random
import hashlib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_video_meta, PROCESSED_DATA
from shared.evaluate_ranking import compute_ranking_metrics
from shared.openai_clients import (
    build_chat_completion_kwargs,
    get_chat_client,
    is_content_filter_error,
    sanitize_messages,
)

VALID_MODELS = ["gpt-5.4"]

# ── Constants (match exp4 exactly) ───────────────────────────────────────────
N_SHOTS               = 5
RANDOM_SEED           = 42
MAX_ATTEMPTS          = 5
RATE_LIMIT_SLEEP      = 1.0
MAX_COMPLETION_TOKENS = 128

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to understand the temporal sequence of events in a video."
)

# ── In-memory data cache ──────────────────────────────────────────────────────
_data_cache: dict = {}

def _get_video_data(dataset_name, video_id):
    if dataset_name not in _data_cache:
        path = f"/data/MMS_Benchmark/data/processed/{dataset_name}_data.json"
        if os.path.exists(path):
            with open(path) as f:
                _data_cache[dataset_name] = json.load(f)
        else:
            _data_cache[dataset_name] = {}
    return _data_cache[dataset_name].get(video_id, {})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_clip_descriptions(dataset_name, video_id):
    """Return list of shot description strings (clips[i].summary), one per shot."""
    clips = _get_video_data(dataset_name, video_id).get("clips", [])
    return [c.get("summary", "") for c in clips]


def load_full_document_text(dataset_name, video_id):
    """Return aligned_full_document as a single string (same as exp4)."""
    doc = _get_video_data(dataset_name, video_id).get("aligned_full_document", {})
    sentences = doc.get("sentences", []) if isinstance(doc, dict) else []
    sentences = [s for s in sentences if isinstance(s, str) and s.strip()]
    return " ".join(sentences) if sentences else None


# ── Per-video deterministic RNG (identical to exp4) ──────────────────────────

def _video_rng(global_seed, video_id):
    h = int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16)
    return random.Random((global_seed + h) % (2 ** 32))


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(shuffled_descriptions, document_text):
    """
    Text-only equivalent of build_content_shot_reorder.
    shuffled_descriptions: list of (orig_shot_idx, description_str) in shuffled display order.
    """
    n = len(shuffled_descriptions)
    lines = [f"Below are {n} shots from a video, presented in shuffled order:\n"]

    for label_idx, (_, desc) in enumerate(shuffled_descriptions):
        shot_desc = desc.strip() if desc.strip() else "[no description available]"
        lines.append(f"Shot {label_idx + 1}: {shot_desc}")

    if document_text:
        lines.append(f"\nThe video's full text description:\n{document_text}")

    shot_list = ", ".join(f"Shot {i}" for i in range(1, n + 1))
    lines.append(
        f"\nYou have been shown {n} shots ({shot_list}), presented in shuffled order.\n"
        "Reorder ALL shots to match the actual temporal sequence of the video, from first to last.\n\n"
        "Output ONLY the ordering in this exact format (use > between shots):\n"
        "Shot X > Shot Y > Shot Z > ...\n"
        "Include all shots exactly once."
    )
    return "\n".join(lines)


# ── Response parsing (identical logic to exp4) ────────────────────────────────

def parse_shot_ranking(text, n):
    """
    Parse 'Shot X > Shot Y > ...' — same logic as exp4's parse_shot_ranking.

    Returns (parsed_ranking, failure_reason).
    parsed_ranking: list of 0-indexed positions (display label - 1), or None on failure.
    """
    if not text or not text.strip():
        return None, "empty_output"

    # Primary: "Shot X > Shot Y > ..."
    nums = re.findall(r'Shot\s+(\d+)', text, re.IGNORECASE)
    if not nums:
        # Fallback: plain numbers separated by >
        nums = re.findall(r'\d+', text)

    if not nums:
        return None, "wrong_format"

    parsed = [int(x) - 1 for x in nums]   # convert to 0-indexed

    if len(parsed) != n:
        return None, f"wrong_count:{len(parsed)}_expected_{n}"
    if len(set(parsed)) != n:
        return None, "duplicated_items"
    if any(x < 0 or x >= n for x in parsed):
        return None, "out_of_range"

    return parsed, None


# ── Azure OpenAI client ───────────────────────────────────────────────────────

_client = None
_client_model = None

def get_client(model):
    global _client, _client_model
    if _client is None or _client_model != model:
        _client = get_chat_client(model)
        _client_model = model
    return _client


def _api_call(model, prompt):
    import openai
    client = get_client(model)
    backoff = 2.0
    used_sanitized_retry = False
    for attempt in range(MAX_ATTEMPTS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        if used_sanitized_retry:
            messages = sanitize_messages(messages)
        try:
            response = client.chat.completions.create(
                **build_chat_completion_kwargs(
                    model=model,
                    messages=messages,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    temperature=0.0,
                )
            )
            return response, None
        except openai.RateLimitError:
            wait = backoff * (2 ** attempt)
            print(f"    [RateLimit] sleeping {wait:.0f}s ...")
            time.sleep(wait)
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                wait = backoff * (2 ** attempt)
                print(f"    [ServerError {e.status_code}] sleeping {wait:.0f}s ...")
                time.sleep(wait)
            elif is_content_filter_error(str(e)) and not used_sanitized_retry:
                print("    [ContentFilter] retrying once with softened wording ...")
                used_sanitized_retry = True
            else:
                return None, str(e)
        except Exception as e:
            return None, str(e)
    return None, "max_attempts_exceeded"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skip_result(video_id, reason):
    return {
        "video_id": video_id, "skipped": True, "skip_reason": reason,
        "n_input_tokens": None, "n_output_tokens": None, "raw_output": None,
        "parsed_ranking": None, "parse_failure_reason": None,
        "gt_scores": [], "spearman_r": None, "kendall_tau": None,
        "top3_precision": None, "error": None,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, dataset_name, video_id, rng, with_doc=True):
    meta = load_video_meta(dataset_name, video_id)
    n_shots_total = len(meta["n_frame_per_seg"])

    if n_shots_total == 0:
        return _skip_result(video_id, "no_shots")

    clip_descriptions = load_clip_descriptions(dataset_name, video_id)

    # Pad or trim descriptions to match n_shots_total
    while len(clip_descriptions) < n_shots_total:
        clip_descriptions.append("")

    # Shot selection — identical to exp4 run_video_shot
    valid_shot_indices = [i for i in range(n_shots_total) if meta["n_sampled_per_shot"][i] > 0]
    if not valid_shot_indices:
        return _skip_result(video_id, "no_sampled_frames")

    n = min(N_SHOTS, len(valid_shot_indices))
    selected_orig_indices = sorted(rng.sample(valid_shot_indices, n))
    shuffled_order = list(range(n))
    rng.shuffle(shuffled_order)
    shuffled_shot_orig_indices = [selected_orig_indices[i] for i in shuffled_order]

    print(f"  shots: {n_shots_total} total, {n} selected | orig={selected_orig_indices}")

    # Build (orig_idx, description) pairs in shuffled display order
    shuffled_descriptions = [
        (si, clip_descriptions[si]) for si in shuffled_shot_orig_indices
    ]

    document_text = load_full_document_text(dataset_name, video_id) if with_doc else None
    prompt = build_prompt(shuffled_descriptions, document_text)

    response, err = _api_call(model, prompt)
    if err:
        print(f"  ERROR: {err[:120]}")
        return {
            "video_id": video_id, "skipped": False, "skip_reason": None,
            "n_shots_total": n_shots_total, "n_shots_selected": n,
            "selected_orig_shot_indices": selected_orig_indices,
            "shuffled_shot_orig_indices": shuffled_shot_orig_indices,
            "has_document": document_text is not None,
            "n_input_tokens": None, "n_output_tokens": None,
            "raw_output": "", "parsed_ranking": None,
            "parse_failure_reason": None, "gt_scores": [],
            "spearman_r": None, "kendall_tau": None, "top3_precision": None,
            "error": err,
        }

    raw_text        = response.choices[0].message.content or ""
    n_input_tokens  = response.usage.prompt_tokens
    n_output_tokens = response.usage.completion_tokens
    print(f"  in={n_input_tokens}  out={n_output_tokens}")
    print(f"  raw: {raw_text[:120]}")

    time.sleep(RATE_LIMIT_SLEEP)

    parsed_ranking, parse_failure_reason = parse_shot_ranking(raw_text, n)

    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    if parsed_ranking is None:
        print(f"  [WARN] parse_fail={parse_failure_reason}")
        metrics = {"spearman_r": None, "kendall_tau": None}
        gt_scores = []
    else:
        max_idx = max(shuffled_shot_orig_indices)
        gt_scores = [float(max_idx - shuffled_shot_orig_indices[k]) for k in range(n)]
        metrics = compute_ranking_metrics(parsed_ranking, gt_scores)
        k_key = f"top{min(3, n)}_precision"
        print(f"  {' > '.join(f'Shot {x+1}' for x in parsed_ranking)}")
        print(f"  Spearman={_fmt(metrics['spearman_r'])}  Kendall={_fmt(metrics['kendall_tau'])}"
              f"  Top{min(3,n)}-P={_fmt(metrics.get(k_key))}")

    return {
        "video_id":                   video_id,
        "mode":                       "shot_nodoc" if not with_doc else "shot",
        "baseline":                   "text_only_gpt",
        "skipped":                    False,
        "skip_reason":                None,
        "n_shots_total":              n_shots_total,
        "n_shots_selected":           n,
        "selected_orig_shot_indices": selected_orig_indices,
        "shuffled_shot_orig_indices": shuffled_shot_orig_indices,
        "has_document":               document_text is not None,
        "n_input_tokens":             n_input_tokens,
        "n_output_tokens":            n_output_tokens,
        "raw_output":                 raw_text,
        "parsed_ranking":             parsed_ranking,
        "parse_failure_reason":       parse_failure_reason,
        "gt_scores":                  gt_scores,
        "spearman_r":                 metrics["spearman_r"],
        "kendall_tau":                metrics["kendall_tau"],
        "top3_precision":             metrics.get(f"top{min(3, n)}_precision"),
        "error":                      None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=VALID_MODELS, required=True)
    parser.add_argument("--dataset", default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--video",   default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line.")
    parser.add_argument("--no_doc", action="store_true",
                        help="Ablation: omit document text, shot descriptions only")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "shot_nodoc" if args.no_doc else "shot"
    stem = f"baseline_gpt_{args.model}_{args.dataset}_{mode_tag}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        video_ids = [l.strip() for l in open(args.video_ids_file) if l.strip()]
    else:
        video_ids = get_video_ids(args.dataset)

    # Resume from partial checkpoint
    completed_ids: set = set()
    all_results: list = []
    if not args.video and partial.exists():
        all_results = json.load(open(partial))
        completed_ids = {r["video_id"] for r in all_results}
        print(f"  Resuming: {len(completed_ids)} videos already done")

    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}  ({len(video_ids)} videos)")
    print(f"Output:  {out_file}")

    for i, vid in enumerate(video_ids):
        if vid in completed_ids:
            print(f"[{i+1}/{len(video_ids)}] {vid}  [already done]")
            continue
        print(f"\n[{i+1}/{len(video_ids)}] {vid}")
        rng = _video_rng(args.seed, vid)
        result = run_video(args.model, args.dataset, vid, rng, with_doc=not args.no_doc)
        result["dataset"] = args.dataset
        all_results.append(result)
        if not args.video:
            json.dump(all_results, open(partial, "w"), indent=2, ensure_ascii=False)

    # Aggregate
    valid = [r for r in all_results if not r.get("skipped") and r.get("spearman_r") is not None]
    _avg = lambda vals: float(np.mean(vals)) if vals else None
    _fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"

    avg_spearman = _avg([r["spearman_r"]    for r in valid])
    avg_kendall  = _avg([r["kendall_tau"]   for r in valid])
    avg_top3     = _avg([r["top3_precision"] for r in valid if r.get("top3_precision") is not None])
    n_parsed     = sum(1 for r in valid if r.get("parsed_ranking") is not None)

    print(f"\n{'='*60}")
    print(f"  {args.dataset}  valid={len(valid)}/{len(video_ids)}  parsed={n_parsed}")
    print(f"  Spearman={_fmt(avg_spearman)}  Kendall={_fmt(avg_kendall)}  Top3-P={_fmt(avg_top3)}")
    print(f"{'='*60}")

    output = {
        "model":        args.model,
        "dataset":      args.dataset,
        "baseline":     "text_only_gpt",
        "mode":         mode_tag,
        "seed":         args.seed,
        "n_videos":     len(video_ids),
        "n_valid":      len(valid),
        "avg_spearman": avg_spearman,
        "avg_kendall":  avg_kendall,
        "avg_top3_prec": avg_top3,
        "video_results": all_results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
