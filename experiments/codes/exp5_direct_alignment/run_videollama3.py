"""
exp5_direct_alignment: Direct multimodal clip-to-sentence alignment with VideoLLaMA3-7B.

Setup:
  - Sample up to N_SENTENCES sentences from aligned_text_summary, shuffle their order.
  - Present ALL shots labeled "Clip 1", "Clip 2", ... with a flat frame list.
  - Ask the model to assign clip IDs to each summary sentence.

Output format requested:
  Summary 1: Clip A, Clip B
  Summary 2: Clip C
  ...

Evaluation (per sentence, per video, per dataset):
  - precision, recall, f1, iou (Jaccard), hit@any, n_pred_clips

Frame budget per shot: 30% of shot length, clamped to [4, 8].

Environment: videollama3
Usage:
    python run_videollama3.py --dataset summe
    python run_videollama3.py --dataset summe --video video_1  # smoke test
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
from shared.data_loader import get_video_ids, load_all_shots, PROCESSED_DATA

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH    = "/data/hf-cache/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B"
MODEL_DISPLAY = "VideoLLaMA3-7B"
MAX_NEW_TOKENS = 512
N_SENTENCES    = 3
RANDOM_SEED    = 42
MAX_ATTEMPTS   = 3
FRAMES_MIN     = 4
FRAMES_MAX     = 8
FRAMES_PCT     = 0.30


# ── Metrics ────────────────────────────────────────────────────────────────────

def set_metrics(pred, gt):
    pred, gt = set(pred), set(gt)
    inter = len(pred & gt)
    union = len(pred | gt)
    p   = inter / len(pred) if pred else 0.0
    r   = inter / len(gt)   if gt   else 0.0
    f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = inter / union if union > 0 else 1.0
    hit = 1 if inter > 0 else 0
    return {"p": p, "r": r, "f1": f1, "iou": iou, "hit": hit}


def _avg(vals):
    return float(np.mean(vals)) if vals else None


# ── Per-video deterministic RNG ────────────────────────────────────────────────

def _video_rng(global_seed, dataset, video_id):
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


# ── Frame budget ───────────────────────────────────────────────────────────────

def compute_n_frames_for_shot(n_frames_in_shot):
    return max(FRAMES_MIN, min(FRAMES_MAX, round(n_frames_in_shot * FRAMES_PCT)))


# ── Data loading ───────────────────────────────────────────────────────────────

_processed_cache: dict = {}

def _load_processed(dataset_name):
    if dataset_name not in _processed_cache:
        path = PROCESSED_DATA.get(dataset_name)
        try:
            with open(path) as f:
                _processed_cache[dataset_name] = json.load(f)
        except Exception:
            _processed_cache[dataset_name] = {}
    return _processed_cache[dataset_name]


def load_exp5_data(dataset_name, video_id):
    data = _load_processed(dataset_name).get(video_id, {})
    summary = data.get("aligned_text_summary", {})
    sentences  = summary.get("sentences", [])
    alignments = summary.get("alignments", [])
    shot_ids = []
    for i in range(len(sentences)):
        entry = next((a for a in alignments if a.get("sentence_id") == i), None)
        shot_ids.append(entry["shot_ids"] if entry else [])
    return sentences, shot_ids, summary.get("raw_text")


# ── Prompt / content building ─────────────────────────────────────────────────

def build_prompt_and_frames(shots, shuffled_sentences):
    """
    Build text prompt + flat frame list for VideoLLaMA3.

    Returns:
        prompt_text:  str
        all_frames:   flat list of PIL.Image
        total_frames: int
    """
    all_frames  = []
    clip_labels = []
    cursor = 1
    n_capped_per_shot = []

    for i, shot in enumerate(shots):
        k = compute_n_frames_for_shot(len(shot["frames"]))
        frames = shot["frames"][:k]
        n_capped_per_shot.append(len(frames))
        start = cursor
        end = cursor + len(frames) - 1
        if len(frames) == 1:
            clip_labels.append(f"Clip {i + 1}: Frame {start}")
        else:
            clip_labels.append(f"Clip {i + 1}: Frames {start}-{end}")
        all_frames.extend(frames)
        cursor = end + 1

    n_clips = len(shots)
    total_frames = len(all_frames)
    n_sents = len(shuffled_sentences)

    sents_formatted = "\n".join(
        f"Summary {k + 1}: {s}" for k, s in enumerate(shuffled_sentences)
    )

    prompt_text = (
        f"Below are {total_frames} frames from a video, organized into {n_clips} clips:\n"
        + "\n".join(clip_labels) + "\n\n"
        f"Here are {n_sents} sentences from a text summary of this video "
        f"(the order has been shuffled):\n\n"
        f"{sents_formatted}\n\n"
        "For each summary sentence, identify which clip(s) it corresponds to.\n"
        "Use exactly this format (list clip numbers separated by commas):\n"
        + "\n".join(f"Summary {k + 1}: Clip X, Clip Y" for k in range(n_sents))
    )
    return prompt_text, all_frames, total_frames


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_assignments(text, n_sentences, n_clips):
    assignments = [[] for _ in range(n_sentences)]
    format_ok = True
    nonempty  = 0
    failures  = []

    for k in range(1, n_sentences + 1):
        pat = rf"Summary\s*{k}\s*:\s*(.+?)(?=Summary\s*\d|$)"
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if not m:
            format_ok = False
            failures.append(f"missing_line_{k}")
        else:
            nums  = re.findall(r'\d+', m.group(1))
            clips = [int(x) - 1 for x in nums if 1 <= int(x) <= n_clips]
            assignments[k - 1] = clips
            if clips:
                nonempty += 1
            else:
                failures.append(f"no_valid_clip_ids_line_{k}")

    return assignments, {
        "format_ok":                  format_ok,
        "nonempty_predictions_count": nonempty,
        "failure_reasons":            failures if failures else None,
    }


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


# ── Inference ─────────────────────────────────────────────────────────────────

def _infer(model, processor, shots, shuffled_sentences):
    out = {"text": "", "n_output_tokens": None, "total_frames_submitted": None, "error": None}
    try:
        prompt_text, all_frames, total_frames = build_prompt_and_frames(shots, shuffled_sentences)
        out["total_frames_submitted"] = total_frames

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

def _null_metrics():
    return {"avg_p": None, "avg_r": None, "avg_f1": None, "avg_iou": None, "avg_hit": None}


def _fatal_error_result(video_id, error_msg):
    return {
        "video_id": video_id, "n_sentences_sampled": 0,
        "sentence_results": [],
        **_null_metrics(),
        "n_output_tokens": None, "total_frames_submitted": None,
        "raw_output": "",
        "format_ok": None, "nonempty_predictions_count": None, "failure_reasons": None,
        "error": "fatal_error", "error_message": error_msg,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, processor, dataset_name, video_id, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    meta, shots = load_all_shots(dataset_name, video_id)
    sentences, gt_shot_ids, raw_text = load_exp5_data(dataset_name, video_id)
    n_shots = len(shots)

    if not sentences:
        print(f"  [SKIP] No summary sentences available")
        return {
            "video_id": video_id, "video_name": meta["video_name"],
            "n_clips": n_shots, "n_sentences_sampled": 0,
            "sentence_results": [],
            **_null_metrics(),
            "n_output_tokens": None, "total_frames_submitted": None,
            "raw_output": "",
            "format_ok": None, "nonempty_predictions_count": None, "failure_reasons": None,
            "error": "no_summary",
        }

    n_sample = min(N_SENTENCES, len(sentences))
    sampled_indices = sorted(rng.sample(range(len(sentences)), n_sample))
    sampled_sents   = [sentences[i]   for i in sampled_indices]
    sampled_gt      = [gt_shot_ids[i] for i in sampled_indices]

    shuffle_order = list(range(n_sample))
    rng.shuffle(shuffle_order)
    shuffled_sents = [sampled_sents[k] for k in shuffle_order]
    inv_shuffle = [0] * n_sample
    for pos, orig in enumerate(shuffle_order):
        inv_shuffle[pos] = orig

    r = _infer(model, processor, shots, shuffled_sents)
    print(f"  clips: {n_shots}  |  sentences: {n_sample}  |  frames: {r['total_frames_submitted']}")

    parse_info = {"format_ok": None, "nonempty_predictions_count": 0, "failure_reasons": None}
    assignments_shuffled = [[] for _ in range(n_sample)]

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return {
            "video_id": video_id, "video_name": meta["video_name"],
            "n_clips": n_shots, "n_sentences_sampled": n_sample,
            "sampled_sentence_indices": sampled_indices,
            "shuffle_order": shuffle_order,
            "shuffled_sentence_texts": shuffled_sents,
            "sentence_results": [],
            **_null_metrics(),
            "n_output_tokens": r["n_output_tokens"],
            "total_frames_submitted": r["total_frames_submitted"],
            "raw_output": r["text"],
            "format_ok": None, "nonempty_predictions_count": None, "failure_reasons": None,
            "error": r["error"],
        }

    print(f"  out={r['n_output_tokens']}")
    assignments_shuffled, parse_info = parse_assignments(r["text"], n_sample, n_shots)

    assignments = [[] for _ in range(n_sample)]
    for pos in range(n_sample):
        assignments[inv_shuffle[pos]] = assignments_shuffled[pos]

    sentence_results = []
    p_vals, r_vals, f1_vals, iou_vals, hit_vals = [], [], [], [], []
    for j in range(n_sample):
        m = set_metrics(set(assignments[j]), set(sampled_gt[j]))
        p_vals.append(m["p"]); r_vals.append(m["r"]); f1_vals.append(m["f1"])
        iou_vals.append(m["iou"]); hit_vals.append(m["hit"])
        sentence_results.append({
            "sentence_idx":  sampled_indices[j],
            "sentence_text": sampled_sents[j],
            "gt_clips":      sorted(sampled_gt[j]),
            "pred_clips":    sorted(assignments[j]),
            "n_pred_clips":  len(assignments[j]),
            "p":             m["p"], "r": m["r"], "f1": m["f1"],
            "iou":           m["iou"], "hit": m["hit"],
        })

    avg_p   = _avg(p_vals);   avg_r   = _avg(r_vals)
    avg_f1  = _avg(f1_vals);  avg_iou = _avg(iou_vals);  avg_hit = _avg(hit_vals)
    print(f"  parse={parse_info['format_ok']} nonempty={parse_info['nonempty_predictions_count']}  "
          f"P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f} IoU={avg_iou:.3f} Hit={avg_hit:.3f}"
          if avg_f1 is not None else
          f"  parse={parse_info['format_ok']}")

    return {
        "video_id":                   video_id,
        "video_name":                 meta["video_name"],
        "n_clips":                    n_shots,
        "n_sentences_sampled":        n_sample,
        "sampled_sentence_indices":   sampled_indices,
        "shuffle_order":              shuffle_order,
        "shuffled_sentence_texts":    shuffled_sents,
        "sentence_results":           sentence_results,
        "avg_p":                      avg_p,
        "avg_r":                      avg_r,
        "avg_f1":                     avg_f1,
        "avg_iou":                    avg_iou,
        "avg_hit":                    avg_hit,
        "n_output_tokens":            r["n_output_tokens"],
        "total_frames_submitted":     r["total_frames_submitted"],
        "raw_output":                 r["text"],
        "format_ok":                  parse_info["format_ok"],
        "nonempty_predictions_count": parse_info["nonempty_predictions_count"],
        "failure_reasons":            parse_info["failure_reasons"],
        "error":                      r["error"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--video",   default=None, help="Single video ID for smoke test")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem     = f"direct_videollama3_{args.dataset}_seed{args.seed}_{timestamp}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    model, processor = load_model()

    video_ids = [args.video] if args.video else get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_DISPLAY} | dataset={args.dataset} ({len(video_ids)} videos)\n")

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

    def collect(key):
        return [r[key] for r in all_results if r.get(key) is not None]

    n_success = sum(1 for r in all_results if not r.get("error"))
    avg_p   = _avg(collect("avg_p"));   avg_r   = _avg(collect("avg_r"))
    avg_f1  = _avg(collect("avg_f1"));  avg_iou = _avg(collect("avg_iou"))
    avg_hit = _avg(collect("avg_hit"))

    def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
    print(f"\nDone: {n_success}/{len(all_results)} success")
    print(f"  P={fmt(avg_p)}  R={fmt(avg_r)}  F1={fmt(avg_f1)}  IoU={fmt(avg_iou)}  Hit={fmt(avg_hit)}")

    output = {
        "model":           MODEL_DISPLAY,
        "dataset":         args.dataset,
        "mode":            "direct",
        "timestamp":       timestamp,
        "n_sentences_max": N_SENTENCES,
        "frames_pct":      FRAMES_PCT,
        "frames_min":      FRAMES_MIN,
        "frames_max":      FRAMES_MAX,
        "seed":            args.seed,
        "n_success":       n_success,
        "avg_p":           avg_p,
        "avg_r":           avg_r,
        "avg_f1":          avg_f1,
        "avg_iou":         avg_iou,
        "avg_hit":         avg_hit,
        "video_results":   all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
