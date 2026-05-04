"""
text_proxy_baseline (exp7): Text-only joint summarization + frame selection with GPT.

Input:  Text descriptions of ALL sampled frames, listed as Frame 1..N in
        chronological order. No visual input, no document context.
Task:   The model jointly:
          1. Generates a text summary (exactly M sentences, M = len(GT sentences))
          2. Selects ceil(N * 0.15) most important individual frames

Evaluation:
  A. Summary quality (exp1 metrics):
       ROUGE-1/2/L, BERTScore F1, BLEU-4, METEOR
  B. Frame selection quality (exp2 metrics):
       F1 (precision/recall) vs GT frame set;
       Kendall τ and Spearman ρ vs frame-level gtscore
  C. Cross-modal alignment:
       CLIPScore between generated summary text and selected frame images
       (requires shared/evaluate_clip — skipped gracefully if not installed)

Models: gpt-5.4  (Foundry/OpenAI-compatible endpoint)
Environment: exp7  (conda activate exp7)

Setup:
    source /data/MMS_Benchmark/.secrets/openai.env

Usage:
    python run_gpt.py --dataset summe --model gpt-5.4
    python run_gpt.py --dataset summe --model gpt-5.4 --video video_1
"""

import sys
import gc
import os
import re
import json
import math
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import (
    get_video_ids, load_video_meta, PROCESSED_DATA, DATASET_CONFIG
)
from shared.evaluate import get_gt_frame_indices, compute_rank_correlations
from shared.evaluate_sets import set_f1
from shared.evaluate_text import compute_text_metrics

from shared.openai_clients import (
    build_chat_completion_kwargs,
    get_chat_client,
    is_content_filter_error,
    sanitize_messages,
)

VALID_MODELS = ["gpt-5.4"]

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_ATTEMPTS          = 5       # retries per video (includes rate-limit backoffs)
RATE_LIMIT_SLEEP      = 1.0     # seconds between successful API calls
MAX_COMPLETION_TOKENS = 2048

# ── In-memory JSON cache ───────────────────────────────────────────────────────
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


# ── Data loading ───────────────────────────────────────────────────────────────

def load_exp7_data(dataset_name, video_id):
    """
    Returns:
        frame_descs:       list of (frame_id: int, description: str), length = n_steps
        summary_sentences: list[str], GT summary sentences (for prompt length hint)
        summary_raw_text:  str, full GT summary text (for eval A)
    """
    video_data = _load_processed(dataset_name).get(video_id, {})

    frames_raw = video_data.get("frames", [])
    frame_descs = []
    for fr in frames_raw:
        raw_desc = fr.get("description", "")
        # Guard against NaN / non-string values left by failed step2 runs
        desc = str(raw_desc).strip() if isinstance(raw_desc, str) else ""
        frame_descs.append((int(fr["frame_id"]), desc))

    summary = video_data.get("aligned_text_summary", {})
    if isinstance(summary, str):
        import ast
        try:
            summary = ast.literal_eval(summary)
        except Exception:
            summary = {}
    sentences = summary.get("sentences", []) if isinstance(summary, dict) else []
    raw_text  = summary.get("raw_text", "")   if isinstance(summary, dict) else ""
    if not raw_text and sentences:
        raw_text = " ".join(sentences)

    return frame_descs, sentences, raw_text.strip()


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(frame_descs, n_summary_sentences, n_select_frames):
    """
    Build the strict text-only prompt.

    The output format is heavily constrained to improve parse reliability:
      - SUMMARY: followed by N numbered sentences, one per line
      - IMPORTANT FRAMES: followed by a bracketed list of integers
    """
    n = len(frame_descs)
    lines = [
        f"Below are text descriptions of {n} sampled frames from a video, "
        f"listed in chronological order as Frame 1 to Frame {n}:\n"
    ]
    for i, (_, desc) in enumerate(frame_descs):
        lines.append(f"Frame {i + 1}: {desc}" if desc else f"Frame {i + 1}: [no description]")

    lines.append(
        f"\n"
        f"Based on the frame descriptions above, complete BOTH tasks below.\n"
        f"Do not output anything other than the two labeled sections.\n\n"
        f"TASK 1 — VIDEO SUMMARY:\n"
        f"Write a concise factual summary of the video in exactly "
        f"{n_summary_sentences} sentence(s). "
        f"Number each sentence (\"1. ...\", \"2. ...\", etc.), one per line. "
        f"Output only the label and the numbered sentences.\n"
        f"SUMMARY:\n\n"
        f"TASK 2 — IMPORTANT FRAMES:\n"
        f"Select exactly {n_select_frames} frame number(s) (integers from 1 to {n}) "
        f"that best capture the key content and events of the video. "
        f"Output only the label and a single bracketed, comma-separated list of integers.\n"
        f"IMPORTANT FRAMES: [X, Y, ...]"
    )
    return "\n".join(lines)


# ── Parsing ───────────────────────────────────────────────────────────────────

def _sentence_split(text):
    """Split text into sentences on '. ', '! ', '? ' boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def parse_output(text, n_summary_sentences, n_frames, n_select):
    """
    Two-level parse of SUMMARY and IMPORTANT FRAMES sections.

    Summary parse:
      Level 1 — line-by-line: split section on newlines, strip bullets/numbers.
      Level 2 — sentence-split fallback: if level 1 yields fewer than
                 n_summary_sentences, rejoin and split on sentence boundaries.

    Args:
        n_frames:  total frame count (used for bounds check: 1 <= p <= n_frames)
        n_select:  expected number of selected frames (used for count-mismatch check)

    Returns:
        gen_sentences:       list[str] of generated summary sentences
        selected_positions:  list[int] of 1-indexed selected frame positions
        parse_ok:            bool  (True iff frames were parsed successfully)
        parse_failure_reason: str or None
    """
    gen_sentences: list = []
    selected_positions: list = []
    parse_ok = False
    failure_reasons: list = []

    # ── SUMMARY section ────────────────────────────────────────────────────────
    sum_match = re.search(
        r'SUMMARY\s*:\s*\n(.*?)(?=(?:TASK\s*2|IMPORTANT\s*FRAMES)|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    if not sum_match:
        failure_reasons.append("no_summary_section")
    else:
        block = sum_match.group(1)

        # Level 1: line-by-line
        raw_lines = [l.strip() for l in block.split("\n") if l.strip()]
        cleaned = [re.sub(r'^[\d]+[.)]\s*|^[-*•]\s*', '', l) for l in raw_lines]
        cleaned = [s for s in cleaned if s]

        if len(cleaned) >= n_summary_sentences:
            gen_sentences = cleaned[:n_summary_sentences]
        elif cleaned:
            # Level 2: sentence-split fallback on the whole block
            joined = " ".join(cleaned)
            split = _sentence_split(joined)
            gen_sentences = split if len(split) >= len(cleaned) else cleaned
        # else: leave gen_sentences empty (block was whitespace-only)

    # ── IMPORTANT FRAMES section ───────────────────────────────────────────────
    frame_match = re.search(
        r'IMPORTANT\s*FRAMES\s*:\s*\[?([^\]\n]+)\]?',
        text, re.IGNORECASE
    )
    if not frame_match:
        failure_reasons.append("no_frames_section")
    else:
        nums = re.findall(r'\d+', frame_match.group(1))
        seen: set = set()
        for x in nums:
            p = int(x)
            if 1 <= p <= n_frames and p not in seen:
                seen.add(p)
                selected_positions.append(p)

        if not selected_positions:
            failure_reasons.append("no_valid_frame_numbers")
        else:
            parse_ok = True
            if len(selected_positions) != n_select:
                # Non-fatal: record but still accept
                failure_reasons.append(
                    f"frames_count_mismatch(got={len(selected_positions)},expected={n_select})"
                )

    parse_failure_reason = "; ".join(failure_reasons) if failure_reasons else None
    return gen_sentences, selected_positions, parse_ok, parse_failure_reason


# ── CLIPScore (optional, graceful fallback) ────────────────────────────────────

def _compute_clipscore(sentences, selected_positions, picks, dataset_name, video_id):
    """
    Load selected frame images and compute CLIPScore.
    Returns float or None if clip is not available.
    """
    try:
        from PIL import Image
        from shared.evaluate_clip import compute_clipscore

        frames_root = DATASET_CONFIG[dataset_name]["frames"]
        images = []
        for pos in selected_positions:
            frame_idx = int(picks[pos - 1])
            img_path = os.path.join(frames_root, video_id, f"frame_{frame_idx:05d}.jpg")
            if os.path.exists(img_path):
                images.append(Image.open(img_path).convert("RGB"))
        return compute_clipscore(sentences, images, device="cpu") if images else None
    except Exception:
        return None


# ── Azure OpenAI client ────────────────────────────────────────────────────────

_client = None
_client_model = None


def get_client(model):
    global _client, _client_model
    if _client is None or _client_model != model:
        _client = get_chat_client(model)
        _client_model = model
    return _client


def _api_call(model, prompt):
    """Single API call with exponential backoff on rate-limit / server errors.

    Returns:
        (response, error_str)  — one of them is always None.
    """
    import openai
    client = get_client(model)
    backoff = 2.0
    used_sanitized_retry = False
    for attempt in range(MAX_ATTEMPTS):
        messages = [{"role": "user", "content": prompt}]
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


# ── Null result helpers ────────────────────────────────────────────────────────

def _null_eval_a():
    return {
        "rouge1": None, "rouge2": None, "rougeL": None,
        "bert_score_f1": None, "bleu4": None, "meteor": None,
    }


def _null_eval_b():
    return {
        "f1": None, "precision": None, "recall": None,
        "kendall_tau": None, "spearman_rho": None,
    }


def _make_empty_result(video_id, video_name, error_msg):
    return {
        "video_id":               video_id,
        "video_name":             video_name,
        "n_frames":               None,
        "n_select":               None,
        "n_summary_sents":        None,
        "n_input_tokens":         None,
        "n_output_tokens":        None,
        "raw_output":             "",
        "gen_summary":            None,
        "selected_positions":     None,
        "parse_ok":               False,
        "parse_failure_reason":   error_msg,
        "eval_a_summary":         _null_eval_a(),
        "eval_b_frame_selection": _null_eval_b(),
        "eval_c_alignment":       {"clip_score": None},
        "error":                  error_msg,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, dataset_name, video_id):
    meta = load_video_meta(dataset_name, video_id)
    picks   = meta["picks"]
    n_steps = len(picks)
    gtscore = meta["gtscore"]

    frame_descs, summary_sentences, summary_raw_text = load_exp7_data(dataset_name, video_id)
    video_name = meta["video_name"]

    if not frame_descs:
        print("  [SKIP] No frame descriptions")
        return _make_empty_result(video_id, video_name, "no_frame_descriptions")

    if len(frame_descs) != n_steps:
        print(f"  [WARN] frame_descs ({len(frame_descs)}) != picks ({n_steps}); truncating to shared prefix")
    n_frames  = min(len(frame_descs), n_steps)
    frame_descs = frame_descs[:n_frames]
    n_select  = max(1, math.ceil(n_frames * 0.15))
    n_summary = max(1, len(summary_sentences)) if summary_sentences else 3

    print(f"  frames={n_frames}  select={n_select}  summary_sents={n_summary}"
          f"  |  gt_text={'yes' if summary_raw_text else 'no'}")

    prompt = build_prompt(frame_descs, n_summary, n_select)

    response, api_err = _api_call(model, prompt)
    if api_err or response is None:
        print(f"  [API ERROR] {api_err}")
        return _make_empty_result(video_id, video_name, api_err or "api_error")

    raw_text      = response.choices[0].message.content or ""
    n_in_tokens   = response.usage.prompt_tokens     if response.usage else None
    n_out_tokens  = response.usage.completion_tokens if response.usage else None
    print(f"  in_tokens={n_in_tokens}  out_tokens={n_out_tokens}")
    print(f"  raw[:200]: {raw_text[:200]!r}")

    gen_sentences, selected_positions, parse_ok, parse_failure_reason = \
        parse_output(raw_text, n_summary, n_frames, n_select)

    if not parse_ok:
        print(f"  [WARN] parse failed — {parse_failure_reason}")
    elif parse_failure_reason:
        print(f"  [WARN] partial parse — {parse_failure_reason}")

    gen_summary = " ".join(gen_sentences) if gen_sentences else ""

    # ── Eval A: summary quality ────────────────────────────────────────────────
    eval_a = _null_eval_a()
    if gen_summary and summary_raw_text:
        try:
            eval_a = compute_text_metrics(gen_summary, summary_raw_text)
        except Exception as e:
            print(f"  [WARN] eval_a failed: {e}")

    # ── Eval B: frame selection quality ───────────────────────────────────────
    eval_b = _null_eval_b()
    if selected_positions:
        pred_frame_indices = {int(picks[pos - 1]) for pos in selected_positions}
        gt_frame_indices   = get_gt_frame_indices(
            meta["shot_level_gt"], meta["n_sampled_per_shot"], picks
        )
        fb = set_f1(pred_frame_indices, gt_frame_indices)
        eval_b["f1"]        = round(fb["f1"], 4)
        eval_b["precision"] = round(fb["precision"], 4)
        eval_b["recall"]    = round(fb["recall"], 4)

        selected_0idx = [pos - 1 for pos in selected_positions]
        rc = compute_rank_correlations(selected_0idx, n_steps, gtscore)
        eval_b["kendall_tau"]  = rc["kendall_tau"]
        eval_b["spearman_rho"] = rc["spearman_rho"]

        print(f"  F1={eval_b['f1']:.3f}  τ={eval_b['kendall_tau']:.3f}"
              f"  ρ={eval_b['spearman_rho']:.3f}")

    # ── Eval C: CLIPScore ──────────────────────────────────────────────────────
    clip_score = None
    if selected_positions and gen_sentences:
        clip_score = _compute_clipscore(
            gen_sentences, selected_positions, picks, dataset_name, video_id
        )

    time.sleep(RATE_LIMIT_SLEEP)   # polite pacing between API calls

    return {
        "video_id":               video_id,
        "video_name":             video_name,
        "n_frames":               n_frames,
        "n_select":               n_select,
        "n_summary_sents":        n_summary,
        "n_input_tokens":         n_in_tokens,
        "n_output_tokens":        n_out_tokens,
        "raw_output":             raw_text,
        "gen_summary":            gen_summary,
        "selected_positions":     selected_positions,   # 1-indexed frame positions
        "parse_ok":               parse_ok,
        "parse_failure_reason":   parse_failure_reason,
        "eval_a_summary":         eval_a,
        "eval_b_frame_selection": eval_b,
        "eval_c_alignment":       {"clip_score": clip_score},
        "error":                  None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--model",   default="gpt-5.4", choices=VALID_MODELS)
    parser.add_argument("--video",   default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line.")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace(".", "-")
    stem       = f"text_proxy_baseline_{safe_model}_{args.dataset}_{timestamp}"
    if args.video_ids_file:
        import os as _os2
        stem += f"_{_os2.path.splitext(_os2.path.basename(args.video_ids_file))[0]}"
    out_file   = results_dir / f"{stem}.json"
    partial    = results_dir / f"{stem}.partial.json"

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\ntext_proxy_baseline | model={args.model} | dataset={args.dataset} "
          f"({len(video_ids)} videos)\n")

    # Resume from partial checkpoint
    all_results: list = []
    done_ids:    set  = set()
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
                result = run_video(args.model, args.dataset, vid)
                break
            except Exception as e:
                msg = f"attempt {attempt+1}: {e}"
                print(f"  [RETRY] {msg}")
                error_log.append(msg)
                if attempt < MAX_ATTEMPTS - 1:
                    gc.collect()
                    time.sleep(2 ** attempt)

        if result is None:
            result = _make_empty_result(vid, vid, "; ".join(error_log))
        if error_log:
            result["error_log"] = error_log

        all_results.append(result)
        json.dump({"video_results": all_results}, open(partial, "w"),
                  indent=2, ensure_ascii=False)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def collect(key_path):
        vals = []
        for r in all_results:
            v = r
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                vals.append(v)
        return vals

    def avg(key_path):
        vals = collect(key_path)
        return round(float(np.mean(vals)), 4) if vals else None

    n_success  = sum(1 for r in all_results if not r.get("error"))
    n_parse_ok = sum(1 for r in all_results if r.get("parse_ok"))

    # Token / cost summary
    total_in_tokens  = sum(r.get("n_input_tokens")  or 0 for r in all_results)
    total_out_tokens = sum(r.get("n_output_tokens") or 0 for r in all_results)
    def fmt(v): return f"{v:.4f}" if v is not None else "N/A"

    print(f"\nDone: {n_success}/{len(all_results)} success, parse_ok={n_parse_ok}")
    print(f"  [A] ROUGE-L:      {fmt(avg(['eval_a_summary', 'rougeL']))}")
    print(f"  [A] BERTScore:    {fmt(avg(['eval_a_summary', 'bert_score_f1']))}")
    print(f"  [A] BLEU-4:       {fmt(avg(['eval_a_summary', 'bleu4']))}")
    print(f"  [A] METEOR:       {fmt(avg(['eval_a_summary', 'meteor']))}")
    print(f"  [B] Frame F1:     {fmt(avg(['eval_b_frame_selection', 'f1']))}")
    print(f"  [B] Kendall τ:    {fmt(avg(['eval_b_frame_selection', 'kendall_tau']))}")
    print(f"  [B] Spearman ρ:   {fmt(avg(['eval_b_frame_selection', 'spearman_rho']))}")
    print(f"  [C] CLIPScore:    {fmt(avg(['eval_c_alignment', 'clip_score']))}")
    print(f"\nToken usage:")
    print(f"  Total input  tokens: {total_in_tokens:,}")
    print(f"  Total output tokens: {total_out_tokens:,}")
    print(f"  Total tokens:        {total_in_tokens + total_out_tokens:,}")

    # Parse failure breakdown
    failure_counts: dict = {}
    for r in all_results:
        reason = r.get("parse_failure_reason")
        if reason:
            for part in reason.split("; "):
                key = re.sub(r'\(.*\)', '', part)   # strip variable parts like (got=X,expected=Y)
                failure_counts[key] = failure_counts.get(key, 0) + 1
    if failure_counts:
        print(f"\nParse failure breakdown:")
        for k, v in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    output = {
        "experiment":        "text_proxy_baseline",
        "model":             args.model,
        "dataset":           args.dataset,
        "timestamp":         timestamp,
        "n_success":         n_success,
        "n_parse_ok":        n_parse_ok,
        "total_input_tokens":  total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "avg_rougeL":        avg(["eval_a_summary", "rougeL"]),
        "avg_rouge1":        avg(["eval_a_summary", "rouge1"]),
        "avg_rouge2":        avg(["eval_a_summary", "rouge2"]),
        "avg_bertscore":     avg(["eval_a_summary", "bert_score_f1"]),
        "avg_bleu4":         avg(["eval_a_summary", "bleu4"]),
        "avg_meteor":        avg(["eval_a_summary", "meteor"]),
        "avg_frame_f1":      avg(["eval_b_frame_selection", "f1"]),
        "avg_kendall_tau":   avg(["eval_b_frame_selection", "kendall_tau"]),
        "avg_spearman_rho":  avg(["eval_b_frame_selection", "spearman_rho"]),
        "avg_clip_score":    avg(["eval_c_alignment", "clip_score"]),
        "video_results":     all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
