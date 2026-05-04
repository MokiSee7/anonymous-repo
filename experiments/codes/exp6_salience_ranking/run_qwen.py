"""
exp6_salience_ranking: Shot salience ranking with Qwen2.5-VL / Qwen3-VL.

Randomly selects up to 8 shots per video, labels them Shot 1..N sequentially,
and asks the model to rank them by importance.

Optional: provide aligned_document as background context (--with_doc).

Logged per video:
  selected shot indices, model ranking output, parsed ranking,
  Spearman / Kendall / Top3-Precision vs GT shot scores.

Environment: MLLM
Usage:
    python run_qwen.py --model qwen2.5 --dataset summe
    python run_qwen.py --model qwen2.5 --dataset summe --with_doc
    python run_qwen.py --model qwen2.5 --dataset summe --video video_1  # smoke test
"""

import sys
import gc
import json
import hashlib
import random
import argparse
import re
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_shots, load_aligned_document
from shared.evaluate_ranking import compute_ranking_metrics

MODEL_CONFIGS = {
    "qwen2.5": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "display": "Qwen2.5-VL-7B-Instruct"},
    "qwen3":   {"model_id": "Qwen/Qwen3-VL-8B-Instruct",   "display": "Qwen3-VL-8B-Instruct"},
}
HF_CACHE_DIR   = "/data/hf-cache/hub"
MAX_NEW_TOKENS = 256
N_SHOTS        = 8
RANDOM_SEED    = 42
MAX_ATTEMPTS   = 3

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to rank video shots by their importance to the overall video content."
)


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


def build_content(selected_shots, doc_text):
    """Build Qwen message content list (interleaved text + images)."""
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
            content.append({"type": "image", "image": frame})

    content.append({"type": "text", "text": build_ranking_instruction(len(selected_shots))})
    return content


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


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_key):
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
    model.eval()
    print(f"  Loaded on: {next(model.parameters()).device}")
    return model, processor


# ── Single inference call ─────────────────────────────────────────────────────

def _infer(model, processor, selected_shots, doc_text):
    from qwen_vl_utils import process_vision_info
    import torch

    device = next(model.parameters()).device
    out = {"text": "", "n_input_tokens": None, "n_output_tokens": None, "error": None}
    try:
        content = build_content(selected_shots, doc_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)

        out["n_input_tokens"] = int(inputs.input_ids.shape[1])

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        out["n_output_tokens"] = int(generated.shape[0])
        out["text"] = processor.decode(generated, skip_special_tokens=True)

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

def run_video(model, processor, dataset_name, video_id, with_doc, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    meta, shots = load_all_shots(dataset_name, video_id)
    doc_text = load_aligned_document(dataset_name, video_id) if with_doc else None

    valid_shots = [s for s in shots if len(s["frames"]) > 0]
    n = min(N_SHOTS, len(valid_shots))
    selected = sorted(rng.sample(valid_shots, n), key=lambda s: s["shot_idx"])

    total_frames = sum(len(s["frames"]) for s in selected)
    print(f"  shots: {len(shots)} total, {n} selected  |  frames: {total_frames}"
          f"  |  doc: {'yes' if doc_text else 'no'}")

    result = _infer(model, processor, selected, doc_text)

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
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["qwen2.5", "qwen3"], required=True)
    parser.add_argument("--dataset",  default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--with_doc", action="store_true",
                        help="Include aligned_document as background context")
    parser.add_argument("--video",    default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_tag  = "_with_doc" if args.with_doc else ""
    stem     = f"{args.model}_{args.dataset}{doc_tag}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    model, processor = load_model(args.model)

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_CONFIGS[args.model]['display']} | dataset={args.dataset} "
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
                result = run_video(model, processor, args.dataset, vid, args.with_doc, args.seed)
                break
            except Exception as e:
                msg = f"attempt {attempt+1}: {e}"
                print(f"  [RETRY] {msg}")
                error_log.append(msg)
                if attempt < MAX_ATTEMPTS - 1:
                    gc.collect()
                    torch.cuda.empty_cache()

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
        "model":         MODEL_CONFIGS[args.model]["display"],
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
