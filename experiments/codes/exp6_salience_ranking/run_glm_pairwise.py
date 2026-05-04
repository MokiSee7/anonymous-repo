"""
exp6_pairwise_ranking: Pairwise shot salience comparison with GLM-4.6V-Flash.

Design:
  1. Select up to N_SHOTS shots per video (same seed/selection as exp6/run_glm.py).
  2. For each pair (i, j), present only those two shots' frames and ask:
       "Which shot is more important to the overall content of the video?"
  3. Aggregate pairwise win counts -> full ranking.
  4. Evaluate with Spearman / Kendall / Top3-Precision vs GT shot scores.

Environment: GLM
Usage:
    python run_glm_pairwise.py --dataset summe
    python run_glm_pairwise.py --dataset summe --video video_1
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

MODEL_ID = "zai-org/GLM-4.6V-Flash"
MODEL_DISPLAY = "GLM-4.6V-Flash"
HF_CACHE_DIR = "/data/hf-cache"

MAX_NEW_TOKENS = 64
N_SHOTS = 8
RANDOM_SEED = 42
MAX_ATTEMPTS = 3
FRAMES_MIN = 1
FRAMES_MAX = 1
FRAMES_PCT = 0.10


def compute_n_frames_for_shot(n_frames_in_shot):
    return min(n_frames_in_shot, max(FRAMES_MIN, min(FRAMES_MAX, round(n_frames_in_shot * FRAMES_PCT))))

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to judge which of two video shots is more important "
    "to the overall content of the video. "
    "Do not output reasoning, chain-of-thought, or <think> blocks. "
    "Output only the final answer."
)


def _video_rng(global_seed, dataset, video_id):
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


def _strip_thinking(text):
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*<think>\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def build_pair_content(shot_a, shot_b):
    content = [{"type": "text", "text": "Below are two shots from a video.\n\nShot A:\n"}]
    for frame in shot_a["frames"][:compute_n_frames_for_shot(len(shot_a["frames"]))]:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": "\nShot B:\n"})
    for frame in shot_b["frames"][:compute_n_frames_for_shot(len(shot_b["frames"]))]:
        content.append({"type": "image", "image": frame})
    content.append({
        "type": "text",
        "text": (
            "\nWhich shot is more important to the overall content of the video?\n"
            "Answer with exactly one of: Shot A  or  Shot B\n"
            "Do not output any explanation or thinking process."
        )
    })
    return content


def parse_winner(text):
    text = (text or "").strip()
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
    from transformers import AutoProcessor, Glm4vForConditionalGeneration

    print(f"Loading {MODEL_DISPLAY} ...")
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)
    model.eval()
    print(f"  Loaded on: {next(model.parameters()).device}")
    return model, processor


def _infer_pair(model, processor, shot_a, shot_b):
    device = next(model.parameters()).device
    out = {"text": "", "winner": None, "n_input_tokens": None, "n_output_tokens": None, "error": None}
    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": build_pair_content(shot_a, shot_b)},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,
        ).to(device)
        inputs.pop("token_type_ids", None)

        out["n_input_tokens"] = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        out["n_output_tokens"] = int(generated.shape[0])
        out["text"] = _strip_thinking(processor.decode(generated, skip_special_tokens=True))
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
            "n_input_tokens": r["n_input_tokens"],
            "n_output_tokens": r["n_output_tokens"],
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
        "video_id": video_id,
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
    stem = f"glm4.6v-flash_pairwise_{args.dataset}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os2
        stem += f"_{_os2.path.splitext(_os2.path.basename(args.video_ids_file))[0]}"
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
                    if torch.cuda.is_available():
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
        "avg_spearman": avg("spearman_r"),
        "avg_kendall": avg("kendall_tau"),
        "avg_top3_prec": avg("top3_precision"),
        "video_results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
