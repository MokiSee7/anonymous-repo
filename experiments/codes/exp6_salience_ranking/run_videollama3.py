"""
exp6_salience_ranking: Shot salience ranking with VideoLLaMA3-7B.

Randomly selects up to 8 shots per video, labels them Shot 1..N sequentially,
and asks the model to rank them by importance. All shot frames are concatenated
into one flat list; the text prompt describes which frames belong to which shot.

Optional: provide aligned_document as background context (--with_doc).

Environment: videollama3
Usage:
    python run_videollama3.py --dataset summe
    python run_videollama3.py --dataset summe --with_doc
    python run_videollama3.py --dataset summe --video video_1
"""

import sys
import gc
import json
import hashlib
import random
import argparse
import re
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_shots, load_aligned_document
from shared.evaluate_ranking import compute_ranking_metrics

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "/data/hf-cache/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B"
MODEL_DISPLAY = "VideoLLaMA3-7B"
MAX_NEW_TOKENS = 256
N_SHOTS        = 8
RANDOM_SEED    = 42
MAX_ATTEMPTS   = 3


# ── Per-video deterministic RNG ────────────────────────────────────────────────
def _video_rng(global_seed, dataset, video_id):
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


# ── Prompt construction ───────────────────────────────────────────────────────
def build_prompt(selected_shots, doc_text):
    """
    Build text prompt + flat frame list for VideoLLaMA3.
    Returns (prompt_text, all_frames).
    """
    all_frames = []
    shot_labels = []
    cursor = 1

    for i, shot in enumerate(selected_shots):
        frames = shot["frames"]
        start = cursor
        end = cursor + len(frames) - 1
        if len(frames) == 1:
            shot_labels.append(f"Shot {i + 1}: Frame {start}")
        else:
            shot_labels.append(f"Shot {i + 1}: Frames {start}-{end}")
        all_frames.extend(frames)
        cursor = end + 1

    n = len(selected_shots)
    shot_list = ", ".join(f"Shot {i}" for i in range(1, n + 1))
    shots_desc = "\n".join(shot_labels)

    doc_prefix = ""
    if doc_text:
        doc_prefix = f"Background document describing the video:\n{doc_text}\n\n"

    prompt_text = (
        f"{doc_prefix}"
        f"You are given {len(all_frames)} frames from {n} video shots, "
        f"organized as follows:\n{shots_desc}\n\n"
        f"You have seen {n} shots from a video ({shot_list}).\n"
        "Rank ALL of these shots from most important to least important "
        "based on their visual content and relevance to the overall video.\n\n"
        "Output ONLY the ranking in this exact format (use > between shots):\n"
        "Shot X > Shot Y > Shot Z > ...\n"
        "Include all shots exactly once."
    )
    return prompt_text, all_frames


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_ranking(text, n):
    nums = re.findall(r'Shot\s+(\d+)', text, re.IGNORECASE)
    nums = [int(x) for x in nums]
    seen = set()
    nums = [x for x in nums if not (x in seen or seen.add(x))]
    if len(nums) == n and set(nums) == set(range(1, n + 1)):
        return nums
    return None


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    from transformers import AutoModelForCausalLM, AutoProcessor
    print(f"Loading {MODEL_DISPLAY} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    print(f"  Loaded on: {next(model.parameters()).device}")
    return model, processor


# ── Single inference call ─────────────────────────────────────────────────────
def _infer(model, processor, frames, text):
    out = {"text": "", "n_output_tokens": None, "error": None}
    try:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "num_frames": len(frames)},
                {"type": "text", "text": text},
            ]},
        ]
        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated = output_ids  # generate() returns only new tokens
        out["n_output_tokens"] = int(generated.shape[-1])
        out["text"] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    except Exception as e:
        out["error"] = str(e)
    return out


# ── Result helpers ─────────────────────────────────────────────────────────────
_null_metrics = {"spearman_r": None, "kendall_tau": None, "top3_precision": None}


def _fatal_error_result(video_id, error_msg):
    return {
        "video_id": video_id, "n_shots_total": None, "n_shots_selected": None,
        "selected_shot_indices": [], "total_frames_submitted": None,
        "n_output_tokens": None,
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

    prompt_text, all_frames = build_prompt(selected, doc_text)
    total_frames = len(all_frames)
    print(f"  shots: {len(shots)} total, {n} selected  |  frames: {total_frames}"
          f"  |  doc: {'yes' if doc_text else 'no'}")

    result = _infer(model, processor, all_frames, prompt_text)

    metrics = dict(_null_metrics)
    parsed_ranking = None

    if result["error"]:
        print(f"  ERROR: {result['error'][:120]}")
    else:
        print(f"  out={result['n_output_tokens']}")
        print(f"  raw: {result['text'][:120]}")
        parsed_ranking = parse_ranking(result["text"], n)
        if parsed_ranking is None:
            print(f"  [WARN] Could not parse ranking")
        else:
            gt_scores = [s["gt_score"] for s in selected]
            metrics = compute_ranking_metrics(parsed_ranking, gt_scores)
            print(f"  ranking: {' > '.join(f'Shot {x}' for x in parsed_ranking)}")
            if metrics["spearman_r"] is not None:
                print(f"  Spearman={metrics['spearman_r']:.3f}  Kendall={metrics['kendall_tau']:.3f}"
                      f"  Top3-P={metrics['top3_precision']:.2f}")

    return {
        "video_id":               video_id,
        "video_name":             meta["video_name"],
        "n_shots_total":          len(shots),
        "n_shots_selected":       n,
        "selected_shot_indices":  [s["shot_idx"] for s in selected],
        "total_frames_submitted": total_frames,
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
    parser.add_argument("--dataset",  default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--with_doc", action="store_true",
                        help="Include aligned_document as background context")
    parser.add_argument("--video",          default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line.")
    parser.add_argument("--seed",           type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_tag  = "_with_doc" if args.with_doc else ""
    stem     = f"videollama3_{args.dataset}{doc_tag}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    model, processor = load_model()

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_DISPLAY} | dataset={args.dataset} "
          f"| with_doc={args.with_doc} ({len(video_ids)} videos)\n")

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
        "model":         MODEL_DISPLAY,
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
