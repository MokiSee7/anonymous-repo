"""
exp6_pairwise_ranking: Pairwise shot salience comparison with VideoLLaMA3-7B.

Diagnostic experiment to determine whether VideoLLaMA3's failure in full ranking
(exp6) stems from visual understanding failure or global reasoning failure.

Design:
  1. Select up to N_SHOTS shots per video (same seed/selection as exp6).
  2. For each pair (i, j), present ONLY those two shots' frames and ask:
       "Which shot is more important to the video, Shot A or Shot B?"
  3. Aggregate pairwise win counts -> full ranking.
  4. Evaluate with Spearman / Kendall / Top3-Precision vs GT shot scores.

Environment: videollama3
Usage:
    python run_videollama3_pairwise.py --dataset summe
    python run_videollama3_pairwise.py --dataset tvsum
    python run_videollama3_pairwise.py --dataset summe --video video_1
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
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_shots
from shared.evaluate_ranking import compute_ranking_metrics

MODEL_PATH      = "/data/hf-cache/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B"
MODEL_DISPLAY   = "VideoLLaMA3-7B"
MAX_NEW_TOKENS  = 16
N_SHOTS         = 8
RANDOM_SEED     = 42
MAX_ATTEMPTS    = 3


def _video_rng(global_seed, dataset, video_id):
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


def build_pair_prompt(shot_a, shot_b):
    """
    Build text prompt + flat frame list for VideoLLaMA3.
    Returns (prompt_text, all_frames).
    """
    all_frames = []
    shot_labels = []
    cursor = 1

    for label, shot in [("Shot A", shot_a), ("Shot B", shot_b)]:
        frames = shot["frames"]
        start = cursor
        end = cursor + len(frames) - 1
        if len(frames) == 1:
            shot_labels.append(f"{label}: Frame {start}")
        else:
            shot_labels.append(f"{label}: Frames {start}-{end}")
        all_frames.extend(frames)
        cursor = end + 1

    prompt_text = (
        f"You are given {len(all_frames)} frames from two video shots, organized as follows:\n"
        f"{shot_labels[0]}\n{shot_labels[1]}\n\n"
        "Which shot is more important to the overall content of the video?\n"
        "Answer with exactly one of: Shot A  or  Shot B"
    )
    return prompt_text, all_frames


def parse_winner(text):
    text = text.strip()
    has_a = bool(re.search(r"\bShot\s*A\b", text, re.IGNORECASE))
    has_b = bool(re.search(r"\bShot\s*B\b", text, re.IGNORECASE))
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    m = re.search(r"\bShot\s*([AB])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def load_model():
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"Loading {MODEL_DISPLAY} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    print(f"  Loaded on: {next(model.parameters()).device}")
    return model, processor


def _infer_pair(model, processor, shot_a, shot_b):
    out = {"text": "", "winner": None, "n_output_tokens": None, "n_frames_submitted": None, "error": None}
    try:
        prompt_text, all_frames = build_pair_prompt(shot_a, shot_b)
        out["n_frames_submitted"] = len(all_frames)
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": all_frames, "num_frames": len(all_frames)},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        out["n_output_tokens"] = int(output_ids.shape[-1])
        out["text"] = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        out["winner"] = parse_winner(out["text"])
    except Exception as e:
        out["error"] = str(e)
    return out


_null_metrics = {"spearman_r": None, "kendall_tau": None, "top3_precision": None}


def _fatal_error_result(video_id, error_msg):
    return {
        "video_id": video_id,
        "n_shots_total": None,
        "n_shots_selected": None,
        "selected_shot_indices": [],
        "n_pairs": None,
        "n_pairs_parsed": None,
        "win_counts": [],
        "pairwise_results": [],
        "derived_ranking": None,
        "gt_scores": [],
        **_null_metrics,
        "error": "fatal_error",
        "error_message": error_msg,
    }


def run_video(model, processor, dataset_name, video_id, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    meta, shots = load_all_shots(dataset_name, video_id)
    valid_shots = [s for s in shots if len(s["frames"]) > 0]
    n = min(N_SHOTS, len(valid_shots))
    selected = sorted(rng.sample(valid_shots, n), key=lambda s: s["shot_idx"])

    pairs = list(itertools.combinations(range(n), 2))
    print(f"  shots={n}  pairs={len(pairs)}")

    wins = [0] * n
    pairwise_results = []
    n_parsed = 0

    for pair_idx, (i, j) in enumerate(pairs):
        r = _infer_pair(model, processor, selected[i], selected[j])

        winner_local = None
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
            "pair_idx": pair_idx,
            "shot_i": i + 1,
            "shot_j": j + 1,
            "shot_i_orig": selected[i]["shot_idx"],
            "shot_j_orig": selected[j]["shot_idx"],
            "raw_output": r["text"],
            "winner": winner_local,
            "n_output_tokens": r["n_output_tokens"],
            "n_frames_submitted": r["n_frames_submitted"],
            "error": r["error"],
        })

    order = sorted(range(n), key=lambda k: -wins[k])
    derived_ranking = [order[rank] + 1 for rank in range(n)]

    gt_scores = [s["gt_score"] for s in selected]
    metrics = dict(_null_metrics)
    if n_parsed > 0:
        metrics = compute_ranking_metrics(derived_ranking, gt_scores)
        sp = metrics["spearman_r"]
        kd = metrics["kendall_tau"]
        t3 = metrics["top3_precision"]
        print(f"  parsed={n_parsed}/{len(pairs)}  wins={wins}")
        print(
            f"  ranking: {derived_ranking}  Spearman={sp:.3f}  Kendall={kd:.3f}  Top3-P={t3:.2f}"
            if sp is not None else f"  ranking: {derived_ranking}"
        )
    else:
        print("  [WARN] No pairs parsed")

    return {
        "video_id": meta["video_id"] if "video_id" in meta else video_id,
        "video_name": meta["video_name"],
        "n_shots_total": len(shots),
        "n_shots_selected": n,
        "selected_shot_indices": [s["shot_idx"] for s in selected],
        "gt_scores": gt_scores,
        "n_pairs": len(pairs),
        "n_pairs_parsed": n_parsed,
        "win_counts": wins,
        "derived_ranking": derived_ranking,
        "pairwise_results": pairwise_results,
        "spearman_r": metrics["spearman_r"],
        "kendall_tau": metrics["kendall_tau"],
        "top3_precision": metrics["top3_precision"],
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--video", default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"videollama3_pairwise_{args.dataset}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial = results_dir / f"{stem}.partial.json"

    model, processor = load_model()

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_DISPLAY} pairwise | dataset={args.dataset} "
          f"| N_SHOTS={N_SHOTS} ({len(video_ids)} videos, {N_SHOTS*(N_SHOTS-1)//2} pairs/video)\n")

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

    def avg(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    n_success = sum(1 for r in all_results if not r.get("error"))
    total_pairs = sum(r.get("n_pairs", 0) or 0 for r in all_results)
    parsed_pairs = sum(r.get("n_pairs_parsed", 0) or 0 for r in all_results)
    avg_spearman = avg("spearman_r")
    avg_kendall = avg("kendall_tau")
    avg_top3 = avg("top3_precision")

    def fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    print(f"\nDone: {n_success}/{len(all_results)} success")
    print(f"  pairs: {parsed_pairs}/{total_pairs} parsed")
    print(f"  Spearman={fmt(avg_spearman)}  Kendall={fmt(avg_kendall)}  Top3-P={fmt(avg_top3)}")

    output = {
        "model": MODEL_DISPLAY,
        "dataset": args.dataset,
        "mode": "pairwise",
        "timestamp": timestamp,
        "n_shots_max": N_SHOTS,
        "n_pairs_per_video": N_SHOTS * (N_SHOTS - 1) // 2,
        "seed": args.seed,
        "n_success": n_success,
        "total_pairs": total_pairs,
        "parsed_pairs": parsed_pairs,
        "avg_spearman": avg_spearman,
        "avg_kendall": avg_kendall,
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
