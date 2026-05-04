"""
Frame-description-only summarization baseline using all sampled frame captions.

Input:  processed_data[video_id]["frames"][*]["description"] in chronological order
Task:   Generate a concise summary in exactly M sentences, where M = number of
        GT summary sentences from aligned_text_summary.
Evaluation:
  ROUGE-1/2/L, BERTScore F1, BLEU-4, METEOR vs aligned_text_summary.raw_text

Purpose:
  Control baseline for Exp1 that removes visual pixels but keeps the full set of
  frame-level textual descriptions. This isolates the effect of turning all
  sampled frames into text before summarization.

Models: gpt-5.4
Environment: exp7

Usage:
    python run_gpt_frame_desc.py --dataset summe --model gpt-5.4
    python run_gpt_frame_desc.py --dataset videoxum --video_ids_file ../exp1_v2t/videoxum_100_seed42.txt
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
from shared.evaluate_text import compute_rouge, compute_meteor
from shared.openai_clients import (
    build_chat_completion_kwargs,
    get_chat_client,
    is_content_filter_error,
    sanitize_messages,
)

VALID_MODELS = ["gpt-5.4"]

API_MAX_RETRIES = 5
VIDEO_MAX_RETRIES = 2
RATE_LIMIT_SLEEP = 1.0
MAX_COMPLETION_TOKENS = 1024

_processed_cache: dict = {}
_client = None
_client_model = None
_bert_scorer = None


def _load_processed(dataset_name):
    if dataset_name not in _processed_cache:
        path = PROCESSED_DATA.get(dataset_name, "")
        try:
            with open(path) as f:
                _processed_cache[dataset_name] = json.load(f)
        except Exception:
            _processed_cache[dataset_name] = {}
    return _processed_cache[dataset_name]


def load_frame_desc_data(dataset_name, video_id):
    """
    Returns:
        frame_descs:        list[str], frame descriptions in chronological order
        summary_sentences:  list[str], GT summary sentences (for prompt length hint)
        summary_raw_text:   str, GT summary raw text for evaluation
    """
    video_data = _load_processed(dataset_name).get(video_id, {})

    frames = video_data.get("frames", [])
    frame_descs = []
    if isinstance(frames, list):
        for item in frames:
            if not isinstance(item, dict):
                continue
            desc = (item.get("description") or "").strip()
            if desc:
                frame_descs.append(desc)

    summary = video_data.get("aligned_text_summary", {})
    if isinstance(summary, str):
        import ast
        try:
            summary = ast.literal_eval(summary)
        except Exception:
            summary = {}
    sum_sentences = summary.get("sentences", []) if isinstance(summary, dict) else []
    raw_text = summary.get("raw_text", "") if isinstance(summary, dict) else ""
    if not raw_text and sum_sentences:
        raw_text = " ".join(sum_sentences)

    return frame_descs, sum_sentences, raw_text.strip()


def build_prompt(frame_descs, n_summary_sentences):
    n = len(frame_descs)
    lines = [
        f"Below is a chronological list of {n} frame-level descriptions sampled from a video.\n",
        "Each line describes one sampled frame. Use the full sequence to infer the overall video content.\n",
    ]
    for i, sent in enumerate(frame_descs):
        lines.append(f"{i + 1}. {sent}" if sent.strip() else f"{i + 1}. [no content]")

    lines.append(
        f"\n"
        f"Based on the frame descriptions above, complete the task below.\n"
        f"Do not output anything other than the labeled section.\n\n"
        f"TASK — VIDEO SUMMARY:\n"
        f"Write a concise factual summary of the full video in exactly "
        f"{n_summary_sentences} sentence(s). "
        f"Number each sentence (\"1. ...\", \"2. ...\", etc.), one per line. "
        f"Output only the label and the numbered sentences.\n"
        f"SUMMARY:"
    )
    return "\n".join(lines)


def _sentence_split(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def parse_output(text, n_summary_sentences):
    gen_sentences = []
    failure_reasons = []
    parse_mode = None

    sum_match = re.search(r"SUMMARY\s*:\s*\n?(.*?)$", text, re.DOTALL | re.IGNORECASE)
    if not sum_match:
        failure_reasons.append("no_summary_section")
        return [], False, "; ".join(failure_reasons), None

    block = sum_match.group(1)
    raw_lines = [l.strip() for l in block.split("\n") if l.strip()]
    cleaned = [re.sub(r'^[\d]+[.)]\s*|^[-*•]\s*', '', l) for l in raw_lines]
    cleaned = [s for s in cleaned if s]

    if len(cleaned) >= n_summary_sentences:
        gen_sentences = cleaned[:n_summary_sentences]
        parse_mode = "line"
    elif cleaned:
        joined = " ".join(cleaned)
        split = _sentence_split(joined)
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

    return gen_sentences, parse_ok, "; ".join(failure_reasons) if failure_reasons else None, parse_mode


def get_client(model):
    global _client, _client_model
    if _client is None or _client_model != model:
        _client = get_chat_client(model)
        _client_model = model
    return _client


def get_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(
            lang="en",
            rescale_with_baseline=False,
            model_type="roberta-large",
        )
    return _bert_scorer


def compute_text_metrics_fast(hypothesis: str, reference: str | None) -> dict:
    if not reference or not hypothesis:
        return {
            "has_reference": reference is not None,
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
            "bert_score_f1": None,
            "bleu4": None,
            "meteor": None,
            "cider": None,
        }

    rouge = compute_rouge(hypothesis, reference)
    meteor = compute_meteor(hypothesis, reference)

    bert = None
    try:
        scorer = get_bert_scorer()
        _, _, f1 = scorer.score([hypothesis], [reference], verbose=False)
        bert = round(float(f1[0]), 4)
    except Exception:
        bert = None

    return {
        "has_reference": True,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bert_score_f1": bert,
        "bleu4": None,
        "meteor": meteor,
        "cider": None,
    }


def _api_call(model, prompt):
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


def _null_eval():
    return {
        "rouge1": None,
        "rouge2": None,
        "rougeL": None,
        "bert_score_f1": None,
        "bleu4": None,
        "meteor": None,
    }


def _make_empty_result(video_id, video_name, error_msg):
    return {
        "video_id": video_id,
        "video_name": video_name,
        "n_frame_descs": None,
        "n_summary_sents": None,
        "n_input_tokens": None,
        "n_output_tokens": None,
        "raw_output": "",
        "gen_summary": None,
        "parse_ok": False,
        "format_ok": False,
        "parse_mode": None,
        "parse_failure_reason": error_msg,
        "eval_summary": _null_eval(),
        "error": error_msg,
    }


def run_video(model, dataset_name, video_id):
    meta = load_video_meta(dataset_name, video_id)
    video_name = meta["video_name"]
    frame_descs, sum_sentences, summary_raw_text = load_frame_desc_data(dataset_name, video_id)

    if not frame_descs:
        print("  [SKIP] No frame descriptions")
        return _make_empty_result(video_id, video_name, "no_frame_descriptions")

    n_desc = len(frame_descs)
    n_summary = max(1, len(sum_sentences)) if sum_sentences else 3
    print(f"  frame_descs={n_desc}  summary_sents={n_summary}  |  gt_text={'yes' if summary_raw_text else 'no'}")

    prompt = build_prompt(frame_descs, n_summary)
    response, api_err = _api_call(model, prompt)
    if api_err or response is None:
        print(f"  [API ERROR] {api_err}")
        return _make_empty_result(video_id, video_name, api_err or "api_error")

    raw_text = response.choices[0].message.content or ""
    n_in_tokens = response.usage.prompt_tokens if response.usage else None
    n_out_tokens = response.usage.completion_tokens if response.usage else None
    print(f"  in_tokens={n_in_tokens}  out_tokens={n_out_tokens}")
    print(f"  raw[:200]: {raw_text[:200]!r}")

    gen_sentences, parse_ok, parse_failure_reason, parse_mode = parse_output(raw_text, n_summary)
    if not parse_ok:
        print(f"  [WARN] parse failed — {parse_failure_reason}")
    elif parse_failure_reason:
        print(f"  [WARN] partial parse — {parse_failure_reason}")

    gen_summary = " ".join(gen_sentences) if gen_sentences else ""
    eval_summary = _null_eval()
    if gen_summary and summary_raw_text:
        try:
            eval_summary = compute_text_metrics_fast(gen_summary, summary_raw_text)
        except Exception as e:
            print(f"  [WARN] eval failed: {e}")

    if gen_summary and summary_raw_text:
        r = eval_summary.get("rougeL")
        b = eval_summary.get("bert_score_f1")
        print(f"  ROUGE-L={r:.3f}  BERTScore={b:.3f}" if r and b else "  eval partial")

    time.sleep(RATE_LIMIT_SLEEP)

    return {
        "video_id": video_id,
        "video_name": video_name,
        "n_frame_descs": n_desc,
        "n_summary_sents": n_summary,
        "n_input_tokens": n_in_tokens,
        "n_output_tokens": n_out_tokens,
        "raw_output": raw_text,
        "gen_summary": gen_summary,
        "parse_ok": parse_ok,
        "format_ok": len(gen_sentences) == n_summary,
        "parse_mode": parse_mode,
        "parse_failure_reason": parse_failure_reason,
        "eval_summary": eval_summary,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="summe", choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--model", default="gpt-5.4", choices=VALID_MODELS)
    parser.add_argument("--video", default=None)
    parser.add_argument("--video_ids_file", default=None, help="Path to text file with one video ID per line.")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace(".", "-")
    stem = f"frame_desc_only_{safe_model}_{args.dataset}_{timestamp}"
    if args.video_ids_file:
        stem += f"_{Path(args.video_ids_file).stem}"
    out_file = results_dir / f"{stem}.json"
    partial = results_dir / f"frame_desc_only_{safe_model}_{args.dataset}.partial.json"

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)

    print(f"\nframe_desc_only | model={args.model} | dataset={args.dataset} ({len(video_ids)} videos)\n")

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
        json.dump({"video_results": all_results}, open(partial, "w"), indent=2, ensure_ascii=False)

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

    n_success = sum(1 for r in all_results if not r.get("error"))
    n_parse_ok = sum(1 for r in all_results if r.get("parse_ok"))
    n_format_ok = sum(1 for r in all_results if r.get("format_ok"))
    n_line_mode = sum(1 for r in all_results if r.get("parse_mode") == "line")
    n_fallback = sum(1 for r in all_results if r.get("parse_mode") == "sentence_fallback")
    total_in_tokens = sum(r.get("n_input_tokens") or 0 for r in all_results)
    total_out_tokens = sum(r.get("n_output_tokens") or 0 for r in all_results)

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print(f"\nDone: {n_success}/{len(all_results)} success, parse_ok={n_parse_ok}, format_ok={n_format_ok} (line={n_line_mode}, fallback={n_fallback})")
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

    output = {
        "experiment": "exp8_frame_desc_only_summarization",
        "model": args.model,
        "dataset": args.dataset,
        "timestamp": timestamp,
        "n_success": n_success,
        "n_parse_ok": n_parse_ok,
        "n_format_ok": n_format_ok,
        "n_parse_mode_line": n_line_mode,
        "n_parse_mode_fallback": n_fallback,
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "avg_rouge1": avg(["eval_summary", "rouge1"]),
        "avg_rouge2": avg(["eval_summary", "rouge2"]),
        "avg_rougeL": avg(["eval_summary", "rougeL"]),
        "avg_bertscore": avg(["eval_summary", "bert_score_f1"]),
        "avg_bleu4": avg(["eval_summary", "bleu4"]),
        "avg_meteor": avg(["eval_summary", "meteor"]),
        "video_results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
