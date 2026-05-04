"""
exp6_salience_ranking: Shot salience ranking with GPT-5.2 / GPT-4.1-mini.

Multimodal version: actual video frames submitted via Azure OpenAI vision API.
All settings (shot selection, RNG, evaluation) are identical to run_qwen.py.

Usage (conda env: exp7):
    python run_gpt.py --model gpt-5.2 --dataset summe
    python run_gpt.py --model gpt-5.2 --dataset summe --with_doc
    python run_gpt.py --model gpt-5.2 --dataset summe --video video_1  # smoke test
"""

import sys
import base64
import io
import json
import hashlib
import random
import argparse
import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_shots, load_aligned_document
from shared.evaluate_ranking import compute_ranking_metrics

# ── Constants ──────────────────────────────────────────────────────────────────
VALID_MODELS      = ["gpt-5.2", "gpt-4.1-mini"]
AZURE_ENDPOINT    = _os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_VERSION = "2024-12-01-preview"

MAX_TOKENS       = 256
N_SHOTS          = 8
RANDOM_SEED      = 42
MAX_ATTEMPTS     = 3
AZURE_MAX_IMAGES = 50   # hard limit for this Azure OpenAI deployment

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to rank video shots by their importance to the overall video content."
)

import os as _os
AZURE_KEY = _os.environ.get("AZURE_OPENAI_KEY", "")


# ── Azure OpenAI client ────────────────────────────────────────────────────────

def _make_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key=AZURE_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )


# ── Frame helpers ──────────────────────────────────────────────────────────────

def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _frame_msg(img):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(img)}"}}


# ── Per-video deterministic RNG ────────────────────────────────────────────────

def _video_rng(global_seed, dataset, video_id):
    """MD5-based deterministic per-video+dataset seed, independent of processing order."""
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


# ── Prompt construction ───────────────────────────────────────────────────────

def build_ranking_instruction(n):
    shot_list = ", ".join(f"Shot {i}" for i in range(1, n + 1))
    return (
        f"\nYou have seen {n} shots from a video ({shot_list}).\n"
        "Rank ALL of these shots from most important to least important "
        "based on their visual content and relevance to the overall video.\n\n"
        "Output ONLY the ranking in this exact format (use > between shots):\n"
        "Shot X > Shot Y > Shot Z > ...\n"
        "Include all shots exactly once."
    )


def build_messages(selected_shots, doc_text):
    """Build OpenAI messages with interleaved text and base64 frames."""
    content = []

    if doc_text:
        content.append({
            "type": "text",
            "text": f"Background document describing the video:\n{doc_text}\n\n"
        })

    content.append({
        "type": "text",
        "text": (f"Below are {len(selected_shots)} shots from the video, "
                 f"labeled Shot 1 through Shot {len(selected_shots)}:\n")
    })

    for i, shot in enumerate(selected_shots):
        content.append({"type": "text", "text": f"\nShot {i + 1}:\n"})
        for frame in shot["frames"]:
            content.append(_frame_msg(frame))

    content.append({"type": "text", "text": build_ranking_instruction(len(selected_shots))})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_ranking(text, n):
    """
    Parse "Shot X > Shot Y > ..." into a list of 1-indexed shot numbers.
    Returns list of ints, or None if parsing fails.
    """
    nums = re.findall(r'Shot\s+(\d+)', text, re.IGNORECASE)
    nums = [int(x) for x in nums]
    seen = set()
    nums = [x for x in nums if not (x in seen or seen.add(x))]
    if len(nums) == n and set(nums) == set(range(1, n + 1)):
        return nums
    return None


# ── Single inference call ─────────────────────────────────────────────────────

def _infer(client, model_name, selected_shots, doc_text):
    out = {"text": "", "n_input_tokens": None, "n_output_tokens": None, "error": None}
    try:
        messages = build_messages(selected_shots, doc_text)

        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=MAX_TOKENS,
                    temperature=0,
                )
                break
            except Exception as e:
                if attempt < MAX_ATTEMPTS - 1 and "rate" in str(e).lower():
                    print(f"  [rate limit] sleeping 30s ...")
                    time.sleep(30)
                else:
                    raise

        out["n_input_tokens"]  = resp.usage.prompt_tokens
        out["n_output_tokens"] = resp.usage.completion_tokens
        out["text"] = resp.choices[0].message.content or ""
        time.sleep(1)

    except Exception as e:
        out["error"] = str(e)
    return out


# ── Result helpers ─────────────────────────────────────────────────────────────

_null_metrics = {"spearman_r": None, "kendall_tau": None, "top3_precision": None}


def _fatal_error_result(video_id, error_msg):
    return {
        "video_id": video_id, "n_shots_total": None, "n_shots_selected": None,
        "selected_shot_indices": [], "total_frames_submitted": None,
        "n_input_tokens": None, "n_output_tokens": None,
        "raw_output": "", "parsed_ranking": None, "gt_scores": [],
        **_null_metrics,
        "error": "fatal_error", "error_message": error_msg,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(client, model_name, dataset_name, video_id, with_doc, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    meta, shots = load_all_shots(dataset_name, video_id)
    doc_text = load_aligned_document(dataset_name, video_id) if with_doc else None

    valid_shots = [s for s in shots if len(s["frames"]) > 0]
    n = min(N_SHOTS, len(valid_shots))
    selected = sorted(rng.sample(valid_shots, n), key=lambda s: s["shot_idx"])

    # Cap total frames to AZURE_MAX_IMAGES by uniformly subsampling each shot
    total_frames = sum(len(s["frames"]) for s in selected)
    if total_frames > AZURE_MAX_IMAGES:
        per_shot = max(1, AZURE_MAX_IMAGES // n)
        capped = []
        for s in selected:
            fs = s["frames"]
            if len(fs) > per_shot:
                idxs = [int(i * len(fs) / per_shot) for i in range(per_shot)]
                fs = [fs[i] for i in idxs]
            capped.append({**s, "frames": fs})
        selected = capped
        total_frames = sum(len(s["frames"]) for s in selected)

    print(f"  shots: {len(shots)} total, {n} selected  |  frames: {total_frames}"
          f"  |  doc: {'yes' if doc_text else 'no'}")

    result = _infer(client, model_name, selected, doc_text)

    metrics = dict(_null_metrics)
    parsed_ranking = None

    if result["error"]:
        print(f"  ERROR: {result['error'][:120]}")
    else:
        print(f"  in={result['n_input_tokens']}  out={result['n_output_tokens']}")
        print(f"  raw: {result['text'][:120]}")
        parsed_ranking = parse_ranking(result["text"], n)
        if parsed_ranking is None:
            print(f"  [WARN] Could not parse ranking")
        else:
            gt_scores = [s["gt_score"] for s in selected]
            metrics = compute_ranking_metrics(parsed_ranking, gt_scores)
            print(f"  ranking: {' > '.join(f'Shot {x}' for x in parsed_ranking)}")
            print(f"  Spearman={metrics['spearman_r']:.3f}  Kendall={metrics['kendall_tau']:.3f}"
                  f"  Top3-P={metrics['top3_precision']:.2f}"
                  if metrics['spearman_r'] is not None else "  metrics: N/A")

    return {
        "video_id":               video_id,
        "video_name":             meta["video_name"],
        "n_shots_total":          len(shots),
        "n_shots_selected":       n,
        "selected_shot_indices":  [s["shot_idx"] for s in selected],
        "total_frames_submitted": total_frames,
        "n_input_tokens":         result["n_input_tokens"],
        "n_output_tokens":        result["n_output_tokens"],
        "raw_output":             result["text"],
        "parsed_ranking":         parsed_ranking,
        "gt_scores":              [s["gt_score"] for s in selected],
        "spearman_r":             metrics["spearman_r"],
        "kendall_tau":            metrics["kendall_tau"],
        "top3_precision":         metrics["top3_precision"],
        "error":                  result["error"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=VALID_MODELS, required=True)
    parser.add_argument("--dataset",  default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--with_doc", action="store_true",
                        help="Include aligned_document as background context")
    parser.add_argument("--video",    default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line.")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if not AZURE_KEY:
        raise RuntimeError("AZURE_OPENAI_KEY environment variable is not set")

    client = _make_client()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_tag  = "_with_doc" if args.with_doc else ""
    stem     = f"{args.model}_{args.dataset}{doc_tag}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os2
        stem += f"_{_os2.path.splitext(_os2.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {args.model} | dataset={args.dataset} "
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
                result = run_video(client, args.model, args.dataset, vid, args.with_doc, args.seed)
                break
            except Exception as e:
                msg = f"attempt {attempt+1}: {e}"
                print(f"  [RETRY] {msg}")
                error_log.append(msg)
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(10)

        if result is None:
            result = _fatal_error_result(vid, "; ".join(error_log))
        if error_log:
            result["error_log"] = error_log

        all_results.append(result)
        json.dump({"video_results": all_results}, open(partial, "w"), indent=2, ensure_ascii=False)

    # ── Aggregate ──────────────────────────────────────────────────────────────
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
        "model":         args.model,
        "dataset":       args.dataset,
        "with_doc":      args.with_doc,
        "timestamp":     timestamp,
        "n_shots_max":   N_SHOTS,
        "seed":          args.seed,
        "n_success":     n_success,
        "n_parsed":      n_parsed,
        "avg_spearman":  avg_spearman,
        "avg_kendall":   avg_kendall,
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
