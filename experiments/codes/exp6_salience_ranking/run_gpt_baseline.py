"""
exp6_salience_ranking — text-only GPT baseline.

Instead of video frames, shot text descriptions (clips[i].summary from step2/LLaVA)
are provided as text.  The model ranks shots by importance using only text.

Same shot selection, RNG, and evaluation as the multimodal exp6 runs.

Models: gpt-5.4  (Foundry/OpenAI-compatible endpoint)
Environment: exp7  (conda activate exp7)

Setup:
    source /data/MMS_Benchmark/.secrets/openai.env

Usage:
    python run_gpt_baseline.py --model gpt-5.4 --dataset summe
    python run_gpt_baseline.py --model gpt-5.4 --dataset summe --video video_1
    python run_gpt_baseline.py --model gpt-5.4 --dataset mrhisum --video_ids_file mrhisum_100_seed42.txt
    python run_gpt_baseline.py --model gpt-5.4 --dataset summe --with_doc
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
from shared.data_loader import get_video_ids, load_all_shots, load_aligned_document, PROCESSED_DATA
from shared.evaluate_ranking import compute_ranking_metrics
from shared.openai_clients import (
    build_chat_completion_kwargs,
    get_chat_client,
    is_content_filter_error,
    sanitize_messages,
)

VALID_MODELS = ["gpt-5.4"]

# ── Constants (identical to multimodal exp6) ──────────────────────────────────
N_SHOTS          = 8
RANDOM_SEED      = 42
MAX_ATTEMPTS     = 3
MAX_TOKENS       = 256

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to rank video shots by their importance to the overall video content."
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


def load_shot_descriptions(dataset_name, video_id):
    """Return dict of shot_idx → description string from clips[i].summary."""
    data = _load_processed(dataset_name).get(video_id, {})
    clips = data.get("clips", [])
    return {i: c.get("summary", "") for i, c in enumerate(clips)}


# ── Per-video deterministic RNG (identical to multimodal exp6) ────────────────

def _video_rng(global_seed, dataset, video_id):
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


# ── Prompt construction ───────────────────────────────────────────────────────

def build_ranking_instruction(n):
    shot_list = ", ".join(f"Shot {i}" for i in range(1, n + 1))
    return (
        f"\nYou have read descriptions of {n} shots from a video ({shot_list}).\n"
        "Rank ALL of these shots from most important to least important "
        "based on their content and relevance to the overall video.\n\n"
        "Output ONLY the ranking in this exact format (use > between shots):\n"
        "Shot X > Shot Y > Shot Z > ...\n"
        "Include all shots exactly once."
    )


def build_prompt(selected_shots, shot_descriptions, doc_text):
    """Build text-only prompt substituting descriptions for frames."""
    lines = []

    if doc_text:
        lines.append(f"Background document describing the video:\n{doc_text}\n")

    lines.append(
        f"Below are {len(selected_shots)} shots from the video, "
        f"labeled Shot 1 through Shot {len(selected_shots)}:\n"
    )

    for i, shot in enumerate(selected_shots):
        desc = shot_descriptions.get(shot["shot_idx"], "").strip()
        if not desc:
            desc = "[no description available]"
        lines.append(f"\nShot {i + 1}:\n{desc}")

    lines.append(build_ranking_instruction(len(selected_shots)))
    return "\n".join(lines)


# ── Parsing (identical to multimodal exp6) ────────────────────────────────────

def parse_ranking(text, n):
    nums = re.findall(r'Shot\s+(\d+)', text, re.IGNORECASE)
    nums = [int(x) for x in nums]
    seen = set()
    nums = [x for x in nums if not (x in seen or seen.add(x))]
    if len(nums) == n and set(nums) == set(range(1, n + 1)):
        return nums
    return None


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
                    max_completion_tokens=MAX_TOKENS,
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

_null_metrics = {"spearman_r": None, "kendall_tau": None, "top3_precision": None}


def run_video(model, dataset_name, video_id, with_doc, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    meta, shots = load_all_shots(dataset_name, video_id)
    doc_text = load_aligned_document(dataset_name, video_id) if with_doc else None
    shot_descriptions = load_shot_descriptions(dataset_name, video_id)

    # Same shot selection as multimodal (valid_shots = shots with at least 1 frame)
    valid_shots = [s for s in shots if len(s.get("frames", [])) > 0]
    n = min(N_SHOTS, len(valid_shots))
    selected = sorted(rng.sample(valid_shots, n), key=lambda s: s["shot_idx"])

    print(f"  shots: {len(shots)} total, {n} selected  |  doc: {'yes' if doc_text else 'no'}")

    prompt = build_prompt(selected, shot_descriptions, doc_text)

    response, err = _api_call(model, prompt)

    metrics = dict(_null_metrics)
    parsed_ranking = None

    if err:
        print(f"  ERROR: {err[:120]}")
        return {
            "video_id": video_id, "video_name": meta.get("video_name", ""),
            "n_shots_total": len(shots), "n_shots_selected": n,
            "selected_shot_indices": [s["shot_idx"] for s in selected],
            "n_input_tokens": None, "n_output_tokens": None,
            "raw_output": "", "parsed_ranking": None, "gt_scores": [],
            **_null_metrics, "error": err,
        }

    raw_text = response.choices[0].message.content or ""
    n_in  = response.usage.prompt_tokens
    n_out = response.usage.completion_tokens
    print(f"  in={n_in}  out={n_out}")
    print(f"  raw: {raw_text[:120]}")
    time.sleep(1)

    parsed_ranking = parse_ranking(raw_text, n)
    if parsed_ranking is None:
        print(f"  [WARN] Could not parse ranking")
    else:
        gt_scores = [s["gt_score"] for s in selected]
        metrics = compute_ranking_metrics(parsed_ranking, gt_scores)
        print(f"  ranking: {' > '.join(f'Shot {x}' for x in parsed_ranking)}")
        if metrics["spearman_r"] is not None:
            print(f"  Spearman={metrics['spearman_r']:.3f}  "
                  f"Kendall={metrics['kendall_tau']:.3f}  "
                  f"Top3-P={metrics['top3_precision']:.2f}")

    return {
        "video_id":              video_id,
        "video_name":            meta.get("video_name", ""),
        "n_shots_total":         len(shots),
        "n_shots_selected":      n,
        "selected_shot_indices": [s["shot_idx"] for s in selected],
        "n_input_tokens":        n_in,
        "n_output_tokens":       n_out,
        "raw_output":            raw_text,
        "parsed_ranking":        parsed_ranking,
        "gt_scores":             [s["gt_score"] for s in selected],
        "spearman_r":            metrics["spearman_r"],
        "kendall_tau":           metrics["kendall_tau"],
        "top3_precision":        metrics["top3_precision"],
        "error":                 None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=VALID_MODELS, required=True)
    parser.add_argument("--dataset", default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--with_doc", action="store_true",
                        help="Include aligned_document as background context")
    parser.add_argument("--video",   default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line.")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_tag = "_with_doc" if args.with_doc else ""
    stem = f"baseline_gpt_{args.model}_{args.dataset}{doc_tag}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)

    print(f"\nRunning text-only baseline: {args.model} | dataset={args.dataset} "
          f"| with_doc={args.with_doc} ({len(video_ids)} videos)\n")

    # Resume from partial checkpoint
    all_results = []
    done_ids = set()
    if partial.exists():
        try:
            prev = json.load(open(partial))
            all_results = prev.get("video_results", [])
            done_ids = {r["video_id"] for r in all_results}
            print(f"  Resuming from partial: {len(done_ids)} videos already done")
        except Exception:
            pass

    for i, vid in enumerate(video_ids):
        if vid in done_ids:
            continue
        print(f"[{i+1}/{len(video_ids)}] {vid}")

        error_log = []
        result = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                result = run_video(args.model, args.dataset, vid, args.with_doc, args.seed)
                break
            except Exception as e:
                msg = f"attempt {attempt+1}: {e}"
                print(f"  [RETRY] {msg}")
                error_log.append(msg)
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(10)

        if result is None:
            result = {
                "video_id": vid, "n_shots_total": None, "n_shots_selected": None,
                "selected_shot_indices": [], "n_input_tokens": None,
                "n_output_tokens": None, "raw_output": "", "parsed_ranking": None,
                "gt_scores": [], **_null_metrics, "error": "; ".join(error_log),
            }
        if error_log:
            result["error_log"] = error_log

        all_results.append(result)
        json.dump({"video_results": all_results}, open(partial, "w"),
                  indent=2, ensure_ascii=False)

    # ── Aggregate ──────────────────────────────────────────────────────────
    n_success = sum(1 for r in all_results if not r.get("error"))
    n_parsed  = sum(1 for r in all_results if r.get("parsed_ranking") is not None)

    def avg(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    avg_spearman = avg("spearman_r")
    avg_kendall  = avg("kendall_tau")
    avg_top3     = avg("top3_precision")

    def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
    print(f"\nDone: {n_success}/{len(all_results)} success, {n_parsed} parsed")
    print(f"  Spearman={fmt(avg_spearman)}  Kendall={fmt(avg_kendall)}  Top3-P={fmt(avg_top3)}")

    output = {
        "model":        args.model,
        "dataset":      args.dataset,
        "baseline":     "text_only_gpt",
        "with_doc":     args.with_doc,
        "timestamp":    timestamp,
        "n_shots_max":  N_SHOTS,
        "seed":         args.seed,
        "n_success":    n_success,
        "n_parsed":     n_parsed,
        "avg_spearman": avg_spearman,
        "avg_kendall":  avg_kendall,
        "avg_top3_prec": avg_top3,
        "video_results": all_results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
