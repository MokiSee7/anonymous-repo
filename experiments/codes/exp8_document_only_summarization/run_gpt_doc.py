"""
exp8_document_only_summarization: Text-only summarization baseline using the
aligned full document (no visual input, no frame descriptions).

Input:  aligned_full_document sentences for each video — a structured textual
        abstraction of the video content.
Task:   Generate a concise summary in exactly M sentences, where M = number of
        GT summary sentences from aligned_text_summary.
Evaluation:
  ROUGE-1/2/L, BERTScore F1, BLEU-4, METEOR vs aligned_text_summary.raw_text

Purpose:
  Control experiment for exp1.  Measures how much of summary quality is
  attributable to pure language modelling vs visual understanding.

Models: gpt-5.4  (Foundry/OpenAI-compatible endpoint)
Environment: exp7  (conda activate exp7)

Setup:
    source /data/MMS_Benchmark/.secrets/openai.env

Usage:
    python run_gpt_doc.py --dataset summe --model gpt-5.4
    python run_gpt_doc.py --dataset summe --model gpt-5.4 --video video_1
"""

import gc
import sys
import os
import re
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_video_meta, PROCESSED_DATA
from shared.evaluate_text import compute_text_metrics

from shared.openai_clients import (
    build_chat_completion_kwargs,
    get_chat_client,
    is_content_filter_error,
    sanitize_messages,
)

VALID_MODELS = ["gpt-5.4"]

# ── Constants ──────────────────────────────────────────────────────────────────
API_MAX_RETRIES       = 5   # retries inside _api_call for RateLimitError / 5xx
VIDEO_MAX_RETRIES     = 2   # retries in main() for unexpected Python exceptions
RATE_LIMIT_SLEEP      = 1.0
MAX_COMPLETION_TOKENS = 1024

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

def load_exp8_data(dataset_name, video_id):
    """
    Returns:
        doc_sentences:     list[str], sentences from aligned_full_document
        summary_sentences: list[str], GT summary sentences (for prompt length hint)
        summary_raw_text:  str, full GT summary text (for evaluation)
    """
    video_data = _load_processed(dataset_name).get(video_id, {})

    # aligned_full_document
    doc = video_data.get("aligned_full_document", {})
    if isinstance(doc, str):
        import ast
        try:
            doc = ast.literal_eval(doc)
        except Exception:
            doc = {}
    doc_sentences = doc.get("sentences", []) if isinstance(doc, dict) else []
    if not isinstance(doc_sentences, list):
        doc_sentences = []

    # GT summary (used only for evaluation, NOT as input)
    summary = video_data.get("aligned_text_summary", {})
    if isinstance(summary, str):
        import ast
        try:
            summary = ast.literal_eval(summary)
        except Exception:
            summary = {}
    sum_sentences = summary.get("sentences", []) if isinstance(summary, dict) else []
    raw_text      = summary.get("raw_text", "")   if isinstance(summary, dict) else ""
    if not raw_text and sum_sentences:
        raw_text = " ".join(sum_sentences)

    return doc_sentences, sum_sentences, raw_text.strip()


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(doc_sentences, n_summary_sentences):
    """
    Build the document-to-summary prompt.

    The output format is constrained to a single labeled SUMMARY section with
    numbered sentences for reliable parsing.
    """
    n = len(doc_sentences)
    lines = [
        f"Below is a structured textual description of a video, consisting of "
        f"{n} sentences in chronological order:\n"
    ]
    for i, sent in enumerate(doc_sentences):
        lines.append(f"{i + 1}. {sent}" if sent.strip() else f"{i + 1}. [no content]")

    lines.append(
        f"\n"
        f"Based on the document above, complete the task below.\n"
        f"Do not output anything other than the labeled section.\n\n"
        f"TASK — VIDEO SUMMARY:\n"
        f"Write a concise factual summary of the video in exactly "
        f"{n_summary_sentences} sentence(s). "
        f"Number each sentence (\"1. ...\", \"2. ...\", etc.), one per line. "
        f"Output only the label and the numbered sentences.\n"
        f"SUMMARY:"
    )
    return "\n".join(lines)


# ── Parsing ───────────────────────────────────────────────────────────────────

def _sentence_split(text):
    """Split text into sentences on '. ', '! ', '? ' boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def parse_output(text, n_summary_sentences):
    """
    Two-level parse of the SUMMARY section.

    Level 1 — line-by-line: split on newlines, strip numbering/bullets.
    Level 2 — sentence-split fallback: if level 1 yields fewer than
               n_summary_sentences, rejoin and split on sentence boundaries.

    Returns:
        gen_sentences:        list[str] of generated summary sentences
        parse_ok:             bool (True iff at least one sentence extracted)
        parse_failure_reason: str or None
        parse_mode:           "line" | "sentence_fallback" | None
    """
    gen_sentences: list = []
    failure_reasons: list = []
    parse_mode = None

    sum_match = re.search(
        r'SUMMARY\s*:\s*\n?(.*?)$',
        text, re.DOTALL | re.IGNORECASE
    )
    if not sum_match:
        failure_reasons.append("no_summary_section")
        return [], False, "; ".join(failure_reasons), None

    block = sum_match.group(1)

    # Level 1: line-by-line
    raw_lines = [l.strip() for l in block.split("\n") if l.strip()]
    cleaned   = [re.sub(r'^[\d]+[.)]\s*|^[-*•]\s*', '', l) for l in raw_lines]
    cleaned   = [s for s in cleaned if s]

    if len(cleaned) >= n_summary_sentences:
        gen_sentences = cleaned[:n_summary_sentences]
        parse_mode = "line"
    elif cleaned:
        # Level 2: sentence-split fallback
        joined = " ".join(cleaned)
        split  = _sentence_split(joined)
        gen_sentences = split if len(split) >= len(cleaned) else cleaned
        parse_mode = "sentence_fallback"
    else:
        failure_reasons.append("empty_summary_block")

    if not gen_sentences:
        failure_reasons.append("no_sentences_extracted")
        return [], False, "; ".join(failure_reasons), None

    parse_ok = True
    if len(gen_sentences) != n_summary_sentences:
        failure_reasons.append(
            f"sentence_count_mismatch(got={len(gen_sentences)},expected={n_summary_sentences})"
        )

    parse_failure_reason = "; ".join(failure_reasons) if failure_reasons else None
    return gen_sentences, parse_ok, parse_failure_reason, parse_mode


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
    """Single API call with exponential backoff on rate-limit / server errors."""
    import openai
    client = get_client(model)
    backoff = 2.0
    used_sanitized_retry = False
    for attempt in range(API_MAX_RETRIES):
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

def _null_eval():
    return {
        "rouge1": None, "rouge2": None, "rougeL": None,
        "bert_score_f1": None, "bleu4": None, "meteor": None,
    }


def _make_empty_result(video_id, video_name, error_msg):
    return {
        "video_id":             video_id,
        "video_name":           video_name,
        "n_doc_sents":          None,
        "n_summary_sents":      None,
        "n_input_tokens":       None,
        "n_output_tokens":      None,
        "raw_output":           "",
        "gen_summary":          None,
        "parse_ok":             False,
        "format_ok":            False,
        "parse_mode":           None,
        "parse_failure_reason": error_msg,
        "eval_summary":         _null_eval(),
        "error":                error_msg,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, dataset_name, video_id):
    meta       = load_video_meta(dataset_name, video_id)
    video_name = meta["video_name"]

    doc_sentences, sum_sentences, summary_raw_text = load_exp8_data(dataset_name, video_id)

    if not doc_sentences:
        print("  [SKIP] No aligned_full_document sentences")
        return _make_empty_result(video_id, video_name, "no_document_sentences")

    n_doc     = len(doc_sentences)
    n_summary = max(1, len(sum_sentences)) if sum_sentences else 3

    print(f"  doc_sents={n_doc}  summary_sents={n_summary}"
          f"  |  gt_text={'yes' if summary_raw_text else 'no'}")

    prompt = build_prompt(doc_sentences, n_summary)

    response, api_err = _api_call(model, prompt)
    if api_err or response is None:
        print(f"  [API ERROR] {api_err}")
        return _make_empty_result(video_id, video_name, api_err or "api_error")

    raw_text     = response.choices[0].message.content or ""
    n_in_tokens  = response.usage.prompt_tokens     if response.usage else None
    n_out_tokens = response.usage.completion_tokens if response.usage else None
    print(f"  in_tokens={n_in_tokens}  out_tokens={n_out_tokens}")
    print(f"  raw[:200]: {raw_text[:200]!r}")

    gen_sentences, parse_ok, parse_failure_reason, parse_mode = parse_output(raw_text, n_summary)

    if not parse_ok:
        print(f"  [WARN] parse failed — {parse_failure_reason}")
    elif parse_failure_reason:
        print(f"  [WARN] partial parse — {parse_failure_reason}")

    gen_summary = " ".join(gen_sentences) if gen_sentences else ""

    # ── Evaluation ─────────────────────────────────────────────────────────────
    eval_summary = _null_eval()
    if gen_summary and summary_raw_text:
        try:
            eval_summary = compute_text_metrics(gen_summary, summary_raw_text)
        except Exception as e:
            print(f"  [WARN] eval failed: {e}")

    if gen_summary and summary_raw_text:
        r = eval_summary.get("rougeL")
        b = eval_summary.get("bert_score_f1")
        print(f"  ROUGE-L={r:.3f}  BERTScore={b:.3f}" if r and b else "  eval partial")

    time.sleep(RATE_LIMIT_SLEEP)

    return {
        "video_id":             video_id,
        "video_name":           video_name,
        "n_doc_sents":          n_doc,
        "n_summary_sents":      n_summary,
        "n_input_tokens":       n_in_tokens,
        "n_output_tokens":      n_out_tokens,
        "raw_output":           raw_text,
        "gen_summary":          gen_summary,
        "parse_ok":             parse_ok,
        "format_ok":            len(gen_sentences) == n_summary,
        "parse_mode":           parse_mode,
        "parse_failure_reason": parse_failure_reason,
        "eval_summary":         eval_summary,
        "error":                None,
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
    stem       = f"doc_only_{safe_model}_{args.dataset}_{timestamp}"
    if args.video_ids_file:
        import os as _os2
        stem += f"_{_os2.path.splitext(_os2.path.basename(args.video_ids_file))[0]}"
    out_file   = results_dir / f"{stem}.json"
    partial    = results_dir / f"doc_only_{safe_model}_{args.dataset}.partial.json"

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nexp8_document_only | model={args.model} | dataset={args.dataset} "
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
        for attempt in range(VIDEO_MAX_RETRIES):
            try:
                result = run_video(args.model, args.dataset, vid)
                break
            except Exception as e:
                msg = f"attempt {attempt+1}: {e}"
                print(f"  [RETRY] {msg}")
                error_log.append(msg)
                if attempt < VIDEO_MAX_RETRIES - 1:
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

    n_success   = sum(1 for r in all_results if not r.get("error"))
    n_parse_ok  = sum(1 for r in all_results if r.get("parse_ok"))
    n_format_ok = sum(1 for r in all_results if r.get("format_ok"))
    n_line_mode = sum(1 for r in all_results if r.get("parse_mode") == "line")
    n_fallback  = sum(1 for r in all_results if r.get("parse_mode") == "sentence_fallback")

    total_in_tokens  = sum(r.get("n_input_tokens")  or 0 for r in all_results)
    total_out_tokens = sum(r.get("n_output_tokens") or 0 for r in all_results)

    def fmt(v): return f"{v:.4f}" if v is not None else "N/A"

    print(f"\nDone: {n_success}/{len(all_results)} success, "
          f"parse_ok={n_parse_ok}, format_ok={n_format_ok} "
          f"(line={n_line_mode}, fallback={n_fallback})")
    print(f"  ROUGE-1:   {fmt(avg(['eval_summary', 'rouge1']))}")
    print(f"  ROUGE-2:   {fmt(avg(['eval_summary', 'rouge2']))}")
    print(f"  ROUGE-L:   {fmt(avg(['eval_summary', 'rougeL']))}")
    print(f"  BERTScore: {fmt(avg(['eval_summary', 'bert_score_f1']))}")
    print(f"  BLEU-4:    {fmt(avg(['eval_summary', 'bleu4']))}")
    print(f"  METEOR:    {fmt(avg(['eval_summary', 'meteor']))}")
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
                key = re.sub(r'\(.*\)', '', part)
                failure_counts[key] = failure_counts.get(key, 0) + 1
    if failure_counts:
        print(f"\nParse failure breakdown:")
        for k, v in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    output = {
        "experiment":          "exp8_document_only_summarization",
        "model":               args.model,
        "dataset":             args.dataset,
        "timestamp":           timestamp,
        "n_success":           n_success,
        "n_parse_ok":          n_parse_ok,
        "n_format_ok":         n_format_ok,
        "n_parse_mode_line":   n_line_mode,
        "n_parse_mode_fallback": n_fallback,
        "total_input_tokens":  total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "avg_rouge1":          avg(["eval_summary", "rouge1"]),
        "avg_rouge2":          avg(["eval_summary", "rouge2"]),
        "avg_rougeL":          avg(["eval_summary", "rougeL"]),
        "avg_bertscore":       avg(["eval_summary", "bert_score_f1"]),
        "avg_bleu4":           avg(["eval_summary", "bleu4"]),
        "avg_meteor":          avg(["eval_summary", "meteor"]),
        "video_results":       all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
