"""
exp5_direct_alignment — text-only GPT baseline.

Instead of video frames, all shot descriptions (clips[i].summary from step2/LLaVA)
are given as text.  The model assigns clip IDs to each summary sentence.

Same sentence sampling, shuffling, and evaluation as the multimodal exp5-direct runs.

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
import hashlib
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, PROCESSED_DATA
from shared.openai_clients import (
    build_chat_completion_kwargs,
    get_chat_client,
    is_content_filter_error,
    sanitize_messages,
)

VALID_MODELS = ["gpt-5.4"]

# ── Constants ─────────────────────────────────────────────────────────────────
N_SENTENCES           = 3     # max summary sentences to sample (same as exp5-direct)
RANDOM_SEED           = 42
MAX_ATTEMPTS          = 5
RATE_LIMIT_SLEEP      = 1.0
MAX_COMPLETION_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to match summary sentences to the video clips they describe, "
    "using only the provided text descriptions of each clip."
)

# ── In-memory JSON cache ──────────────────────────────────────────────────────
_processed_cache: dict = {}

def _load_processed(dataset_name):
    if dataset_name not in _processed_cache:
        path = PROCESSED_DATA.get(dataset_name, "")
        try:
            with open(path) as f:
                _processed_cache[dataset_name] = json.load(f)
        except Exception:
            _processed_cache[dataset_name] = {}
    return _processed_cache[dataset_name]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_clip_descriptions(dataset_name, video_id):
    """Return list of shot description strings (clips[i].summary), one per shot."""
    data = _load_processed(dataset_name).get(video_id, {})
    clips = data.get("clips", [])
    return [c.get("summary", "") for c in clips]


def load_exp5_data(dataset_name, video_id):
    """Load summary sentences and their GT shot_ids (identical to exp5-direct)."""
    data = _load_processed(dataset_name).get(video_id, {})
    summary    = data.get("aligned_text_summary", {})
    sentences  = summary.get("sentences", [])
    alignments = summary.get("alignments", [])
    shot_ids = []
    for i in range(len(sentences)):
        entry = next((a for a in alignments if a.get("sentence_id") == i), None)
        shot_ids.append(entry["shot_ids"] if entry else [])
    return sentences, shot_ids


# ── Metrics (identical to exp5-direct) ───────────────────────────────────────

def set_metrics(pred, gt):
    pred, gt = set(pred), set(gt)
    inter = len(pred & gt)
    union = len(pred | gt)
    p   = inter / len(pred) if pred else 0.0
    r   = inter / len(gt)   if gt   else 0.0
    f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = inter / union if union > 0 else 1.0
    hit = 1 if inter > 0 else 0
    return {"p": p, "r": r, "f1": f1, "iou": iou, "hit": hit}


def _avg(vals):
    return float(np.mean(vals)) if vals else None


# ── Per-video deterministic RNG (same scheme as exp5-direct) ─────────────────

def _video_rng(global_seed, dataset, video_id):
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(clip_descriptions, shuffled_sentences):
    """Construct the text-only prompt for GPT."""
    n_clips = len(clip_descriptions)
    n_sents = len(shuffled_sentences)

    clips_block = "\n".join(
        f"Clip {i + 1}: {desc}" if desc.strip()
        else f"Clip {i + 1}: [no description available]"
        for i, desc in enumerate(clip_descriptions)
    )

    sents_block = "\n".join(
        f"Summary {k + 1}: {s}" for k, s in enumerate(shuffled_sentences)
    )

    format_example = "\n".join(
        f"Summary {k + 1}: Clip X, Clip Y" for k in range(n_sents)
    )

    return (
        f"Below are text descriptions of {n_clips} clips from a video, "
        f"labeled Clip 1 through Clip {n_clips}:\n\n"
        f"{clips_block}\n\n"
        f"Here are {n_sents} sentences from a text summary of this video "
        f"(the order has been shuffled):\n\n"
        f"{sents_block}\n\n"
        "For each summary sentence, identify which clip(s) it corresponds to.\n"
        "Use exactly this format (list clip numbers separated by commas):\n"
        f"{format_example}"
    )


# ── Response parsing (identical logic to exp5-direct) ─────────────────────────

def parse_assignments(text, n_sentences, n_clips):
    """
    Parse 'Summary K: Clip A, Clip B, ...' for K = 1..n_sentences.

    Returns:
        assignments: list[list[int]] — 0-indexed clip IDs per sentence
        parse_info:  dict
    """
    assignments = [[] for _ in range(n_sentences)]
    failure_reasons = []

    for k in range(1, n_sentences + 1):
        pattern = rf"Summary\s+{k}\s*:\s*(.*?)(?=Summary\s+\d+\s*:|$)"
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not m:
            failure_reasons.append(f"missing Summary {k}")
            continue
        raw = m.group(1).strip()
        nums = re.findall(r"\d+", raw)
        valid = []
        for s in nums:
            idx = int(s) - 1   # convert 1-indexed label → 0-indexed
            if 0 <= idx < n_clips:
                valid.append(idx)
        assignments[k - 1] = valid

    nonempty = sum(1 for a in assignments if a)
    format_ok = (nonempty == n_sentences) and not failure_reasons
    return assignments, {
        "format_ok": format_ok,
        "nonempty_predictions_count": nonempty,
        "failure_reasons": failure_reasons if failure_reasons else None,
    }


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
    """Single API call with exponential backoff on rate-limit / server errors."""
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


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, dataset_name, video_id, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    clip_descriptions = load_clip_descriptions(dataset_name, video_id)
    sentences, gt_shot_ids = load_exp5_data(dataset_name, video_id)
    n_clips = len(clip_descriptions)

    if not sentences:
        print(f"  [SKIP] No summary sentences")
        return {
            "video_id": video_id, "n_clips": n_clips,
            "n_sentences_sampled": 0, "sentence_results": [],
            "avg_p": None, "avg_r": None, "avg_f1": None,
            "avg_iou": None, "avg_hit": None,
            "n_input_tokens": None, "n_output_tokens": None,
            "raw_output": "",
            "format_ok": None, "nonempty_predictions_count": None,
            "failure_reasons": None, "error": "no_summary",
        }

    if n_clips == 0:
        print(f"  [SKIP] No clip descriptions")
        return {
            "video_id": video_id, "n_clips": 0,
            "n_sentences_sampled": 0, "sentence_results": [],
            "avg_p": None, "avg_r": None, "avg_f1": None,
            "avg_iou": None, "avg_hit": None,
            "n_input_tokens": None, "n_output_tokens": None,
            "raw_output": "",
            "format_ok": None, "nonempty_predictions_count": None,
            "failure_reasons": None, "error": "no_clips",
        }

    # Sample + shuffle — same RNG scheme as exp5-direct
    n_sample = min(N_SENTENCES, len(sentences))
    sampled_indices = sorted(rng.sample(range(len(sentences)), n_sample))
    sampled_sents   = [sentences[i]   for i in sampled_indices]
    sampled_gt      = [gt_shot_ids[i] for i in sampled_indices]

    shuffle_order = list(range(n_sample))
    rng.shuffle(shuffle_order)
    shuffled_sents = [sampled_sents[k] for k in shuffle_order]
    inv_shuffle = [0] * n_sample
    for pos, orig in enumerate(shuffle_order):
        inv_shuffle[pos] = orig

    prompt = build_prompt(clip_descriptions, shuffled_sents)
    print(f"  clips: {n_clips}  |  sentences: {n_sample}")

    response, err = _api_call(model, prompt)
    if err:
        print(f"  ERROR: {err[:120]}")
        return {
            "video_id": video_id, "n_clips": n_clips,
            "n_sentences_sampled": n_sample,
            "sampled_sentence_indices": sampled_indices,
            "shuffle_order": shuffle_order,
            "shuffled_sentence_texts": shuffled_sents,
            "sentence_results": [],
            "avg_p": None, "avg_r": None, "avg_f1": None,
            "avg_iou": None, "avg_hit": None,
            "n_input_tokens": None, "n_output_tokens": None,
            "raw_output": "",
            "format_ok": None, "nonempty_predictions_count": None,
            "failure_reasons": None, "error": err,
        }

    raw_text        = response.choices[0].message.content or ""
    n_input_tokens  = response.usage.prompt_tokens
    n_output_tokens = response.usage.completion_tokens
    print(f"  in={n_input_tokens}  out={n_output_tokens}")

    time.sleep(RATE_LIMIT_SLEEP)

    assignments_shuffled, parse_info = parse_assignments(raw_text, n_sample, n_clips)

    # Remap to original sampled order
    assignments = [[] for _ in range(n_sample)]
    for pos in range(n_sample):
        assignments[inv_shuffle[pos]] = assignments_shuffled[pos]

    # Per-sentence metrics
    sentence_results = []
    p_vals, r_vals, f1_vals, iou_vals, hit_vals = [], [], [], [], []
    for j in range(n_sample):
        m = set_metrics(set(assignments[j]), set(sampled_gt[j]))
        p_vals.append(m["p"]); r_vals.append(m["r"]); f1_vals.append(m["f1"])
        iou_vals.append(m["iou"]); hit_vals.append(m["hit"])
        sentence_results.append({
            "sentence_idx":  sampled_indices[j],
            "sentence_text": sampled_sents[j],
            "gt_clips":      sorted(sampled_gt[j]),
            "pred_clips":    sorted(assignments[j]),
            "n_pred_clips":  len(assignments[j]),
            "p": m["p"], "r": m["r"], "f1": m["f1"],
            "iou": m["iou"], "hit": m["hit"],
        })

    avg_p   = _avg(p_vals);   avg_r   = _avg(r_vals)
    avg_f1  = _avg(f1_vals);  avg_iou = _avg(iou_vals);  avg_hit = _avg(hit_vals)
    print(f"  parse={parse_info['format_ok']} nonempty={parse_info['nonempty_predictions_count']}  "
          f"P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f} IoU={avg_iou:.3f} Hit={avg_hit:.3f}"
          if avg_f1 is not None else f"  parse={parse_info['format_ok']}")

    return {
        "video_id":                   video_id,
        "n_clips":                    n_clips,
        "n_sentences_sampled":        n_sample,
        "sampled_sentence_indices":   sampled_indices,
        "shuffle_order":              shuffle_order,
        "shuffled_sentence_texts":    shuffled_sents,
        "sentence_results":           sentence_results,
        "avg_p":                      avg_p,
        "avg_r":                      avg_r,
        "avg_f1":                     avg_f1,
        "avg_iou":                    avg_iou,
        "avg_hit":                    avg_hit,
        "n_input_tokens":             n_input_tokens,
        "n_output_tokens":            n_output_tokens,
        "raw_output":                 raw_text,
        "format_ok":                  parse_info["format_ok"],
        "nonempty_predictions_count": parse_info["nonempty_predictions_count"],
        "failure_reasons":            parse_info["failure_reasons"],
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
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"direct_baseline_gpt_{args.model}_{args.dataset}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    # Determine video IDs to process
    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        video_ids = [l.strip() for l in open(args.video_ids_file) if l.strip()]
    else:
        video_ids = get_video_ids(args.dataset)

    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}  ({len(video_ids)} videos)")
    print(f"Output:  {out_file}")

    all_results = []
    for i, vid in enumerate(video_ids):
        print(f"\n[{i+1}/{len(video_ids)}] {vid}")
        result = run_video(args.model, args.dataset, vid, args.seed)
        result["dataset"] = args.dataset
        all_results.append(result)

        # Checkpoint after every video
        json.dump({"video_results": all_results}, open(partial, "w"),
                  indent=2, ensure_ascii=False)

    def collect(key):
        return [r[key] for r in all_results if r.get(key) is not None]

    fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
    avg_p   = _avg(collect("avg_p"));   avg_r   = _avg(collect("avg_r"))
    avg_f1  = _avg(collect("avg_f1"));  avg_iou = _avg(collect("avg_iou"))
    avg_hit = _avg(collect("avg_hit"))
    n_success = sum(1 for r in all_results if r.get("avg_f1") is not None)

    print(f"\n{'='*60}")
    print(f"  {args.dataset}  n={n_success}/{len(video_ids)}")
    print(f"  P={fmt(avg_p)}  R={fmt(avg_r)}  F1={fmt(avg_f1)}  "
          f"IoU={fmt(avg_iou)}  Hit={fmt(avg_hit)}")
    print(f"{'='*60}")

    output = {
        "model":        args.model,
        "dataset":      args.dataset,
        "baseline":     "text_only_gpt",
        "seed":         args.seed,
        "n_videos":     len(video_ids),
        "n_success":    n_success,
        "avg_p":        avg_p,
        "avg_r":        avg_r,
        "avg_f1":       avg_f1,
        "avg_iou":      avg_iou,
        "avg_hit":      avg_hit,
        "n_sentences_max": N_SENTENCES,
        "video_results": all_results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    partial.unlink(missing_ok=True)
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
