"""
exp6_pairwise_ranking: Pairwise shot salience comparison with Qwen2.5-VL.

Diagnostic experiment to determine whether Qwen2.5's failure in full ranking
(exp6) stems from visual understanding failure or global reasoning failure.

Design:
  1. Select up to N_SHOTS shots per video (same seed/selection as exp6).
  2. For each pair (i, j), present ONLY those two shots' frames and ask:
       "Which shot is more important to the video, Shot A or Shot B?"
  3. Aggregate pairwise win counts → full ranking.
  4. Evaluate with Spearman / Kendall / Top3-Precision vs GT shot scores.

Each inference call sees exactly 2 shots — no position bias, no global
ordering burden. If Qwen2.5 still fails pairwise, the problem is visual
understanding itself. If it succeeds, the problem was reasoning/ordering.

N_SHOTS=8 → C(8,2)=28 pairs per video.
Same N_SHOTS and seed as exp6/run_qwen.py → identical shot selection per video.

Environment: MLLM
Usage:
    python run_qwen_pairwise.py --dataset summe
    python run_qwen_pairwise.py --dataset tvsum
    python run_qwen_pairwise.py --dataset summe --video video_1  # smoke test
"""

import sys
import gc
import json
import hashlib
import random
import argparse
import re
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_shots
from shared.evaluate_ranking import compute_ranking_metrics

MODEL_CONFIGS = {
    "qwen2.5": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "display": "Qwen2.5-VL-7B-Instruct"},
    "qwen3":   {"model_id": "Qwen/Qwen3-VL-8B-Instruct",   "display": "Qwen3-VL-8B-Instruct"},
}
HF_CACHE_DIR = "/data/hf-cache/hub"
MAX_NEW_TOKENS = 16    # answer is just "Shot A" or "Shot B"
N_SHOTS      = 8       # C(8,2) = 28 pairs per video; matches exp6/run_qwen.py exactly
RANDOM_SEED  = 42
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to judge which of two video shots is more important "
    "to the overall content of the video."
)


# ── Per-video deterministic RNG ────────────────────────────────────────────────

def _video_rng(global_seed, dataset, video_id):
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


# ── Prompt construction ───────────────────────────────────────────────────────

def build_pair_content(shot_a, shot_b):
    """
    Present exactly two shots (Shot A and Shot B) and ask which is more important.
    Returns Qwen message content list.
    """
    content = [
        {"type": "text", "text": "Below are two shots from a video:\n\nShot A:\n"}
    ]
    for frame in shot_a["frames"]:
        content.append({"type": "image", "image": frame})

    content.append({"type": "text", "text": "\nShot B:\n"})
    for frame in shot_b["frames"]:
        content.append({"type": "image", "image": frame})

    content.append({
        "type": "text",
        "text": (
            "\nWhich shot is more important to the overall content of the video?\n"
            "Answer with exactly one of: Shot A  or  Shot B"
        )
    })
    return content


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_winner(text):
    """
    Parse model output to determine winner.
    Returns 'A', 'B', or None if ambiguous.
    """
    text = text.strip()
    has_a = bool(re.search(r'\bShot\s*A\b', text, re.IGNORECASE))
    has_b = bool(re.search(r'\bShot\s*B\b', text, re.IGNORECASE))
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    # Tie-break: first mentioned wins
    m = re.search(r'\bShot\s*([AB])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
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


# ── Single pair inference ─────────────────────────────────────────────────────

def _infer_pair(model, processor, shot_a, shot_b):
    from qwen_vl_utils import process_vision_info
    import torch

    device = next(model.parameters()).device
    out = {"text": "", "winner": None, "n_input_tokens": None, "error": None}
    try:
        content = build_pair_content(shot_a, shot_b)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
        out["text"]   = processor.decode(generated, skip_special_tokens=True)
        out["winner"] = parse_winner(out["text"])

    except Exception as e:
        out["error"] = str(e)
    return out


# ── Result helpers ─────────────────────────────────────────────────────────────

_null_metrics = {"spearman_r": None, "kendall_tau": None, "top3_precision": None}


def _fatal_error_result(video_id, error_msg):
    return {
        "video_id": video_id, "n_shots_total": None, "n_shots_selected": None,
        "selected_shot_indices": [], "n_pairs": None,
        "n_pairs_parsed": None, "win_counts": [],
        "pairwise_results": [], "derived_ranking": None, "gt_scores": [],
        **_null_metrics,
        "error": "fatal_error", "error_message": error_msg,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, processor, dataset_name, video_id, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    meta, shots = load_all_shots(dataset_name, video_id)
    valid_shots = [s for s in shots if len(s["frames"]) > 0]
    n = min(N_SHOTS, len(valid_shots))
    # Same selection logic as exp6 run_qwen.py (same seed → same shots)
    selected = sorted(rng.sample(valid_shots, n), key=lambda s: s["shot_idx"])

    pairs = list(itertools.combinations(range(n), 2))  # (i, j), i < j
    print(f"  shots={n}  pairs={len(pairs)}")

    # Win count matrix: wins[i] = number of pairs shot i won
    wins = [0] * n
    pairwise_results = []
    n_parsed = 0

    for pair_idx, (i, j) in enumerate(pairs):
        r = _infer_pair(model, processor, selected[i], selected[j])

        winner_local = None  # 'A'=shot i wins, 'B'=shot j wins
        if r["error"]:
            print(f"  pair ({i+1},{j+1}) ERROR: {r['error'][:80]}")
        else:
            winner_local = r["winner"]
            if winner_local == "A":
                wins[i] += 1
                n_parsed += 1
            elif winner_local == "B":
                wins[j] += 1
                n_parsed += 1
            else:
                print(f"  pair ({i+1},{j+1}) ambiguous: {r['text'][:60]!r}")

        pairwise_results.append({
            "pair_idx":       pair_idx,
            "shot_i":         i + 1,            # 1-indexed local position
            "shot_j":         j + 1,
            "shot_i_orig":    selected[i]["shot_idx"],
            "shot_j_orig":    selected[j]["shot_idx"],
            "raw_output":     r["text"],
            "winner":         winner_local,      # 'A', 'B', or None
            "n_input_tokens": r["n_input_tokens"],
            "error":          r["error"],
        })

    # Derive ranking from win counts (higher wins = more important = rank 1)
    # Ties broken by original order (stable sort descending)
    order = sorted(range(n), key=lambda k: -wins[k])
    derived_ranking = [order[rank] + 1 for rank in range(n)]  # 1-indexed positions

    gt_scores = [s["gt_score"] for s in selected]
    metrics = dict(_null_metrics)
    if n_parsed > 0:
        metrics = compute_ranking_metrics(derived_ranking, gt_scores)
        sp = metrics["spearman_r"]
        kd = metrics["kendall_tau"]
        t3 = metrics["top3_precision"]
        print(f"  parsed={n_parsed}/{len(pairs)}  wins={wins}")
        print(f"  ranking: {derived_ranking}  "
              f"Spearman={sp:.3f}  Kendall={kd:.3f}  Top3-P={t3:.2f}"
              if sp is not None else f"  ranking: {derived_ranking}")
    else:
        print(f"  [WARN] No pairs parsed")

    return {
        "video_id":              video_id,
        "video_name":            meta["video_name"],
        "n_shots_total":         len(shots),
        "n_shots_selected":      n,
        "selected_shot_indices": [s["shot_idx"] for s in selected],
        "gt_scores":             gt_scores,
        "n_pairs":               len(pairs),
        "n_pairs_parsed":        n_parsed,
        "win_counts":            wins,
        "derived_ranking":       derived_ranking,
        "pairwise_results":      pairwise_results,
        "spearman_r":            metrics["spearman_r"],
        "kendall_tau":           metrics["kendall_tau"],
        "top3_precision":        metrics["top3_precision"],
        "error":                 None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["qwen2.5", "qwen3"], required=True)
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
    stem     = f"{args.model}_pairwise_{args.dataset}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    model, processor = load_model(args.model)
    model_display = MODEL_CONFIGS[args.model]["display"]

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {model_display} pairwise | dataset={args.dataset} "
          f"| N_SHOTS={N_SHOTS} ({len(video_ids)} videos, "
          f"{N_SHOTS*(N_SHOTS-1)//2} pairs/video)\n")

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
                result = run_video(model, processor, args.dataset, vid, args.seed)
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
    def avg(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    n_success = sum(1 for r in all_results if not r.get("error"))
    total_pairs  = sum(r.get("n_pairs", 0) or 0 for r in all_results)
    parsed_pairs = sum(r.get("n_pairs_parsed", 0) or 0 for r in all_results)

    avg_spearman = avg("spearman_r")
    avg_kendall  = avg("kendall_tau")
    avg_top3     = avg("top3_precision")

    def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
    print(f"\nDone: {n_success}/{len(all_results)} success")
    print(f"  pairs: {parsed_pairs}/{total_pairs} parsed")
    print(f"  Spearman={fmt(avg_spearman)}  Kendall={fmt(avg_kendall)}  Top3-P={fmt(avg_top3)}")

    output = {
        "model":            model_display,
        "dataset":          args.dataset,
        "mode":             "pairwise",
        "timestamp":        timestamp,
        "n_shots_max":       N_SHOTS,
        "n_pairs_per_video": N_SHOTS * (N_SHOTS - 1) // 2,
        "seed":             args.seed,
        "n_success":        n_success,
        "total_pairs":      total_pairs,
        "parsed_pairs":     parsed_pairs,
        "avg_spearman":     avg_spearman,
        "avg_kendall":      avg_kendall,
        "avg_top3_prec":    avg_top3,
        "video_results":    all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
