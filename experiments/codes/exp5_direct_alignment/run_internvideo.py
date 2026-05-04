"""
exp5_direct_alignment: Direct multimodal clip-to-sentence alignment with InternVideo2.5-Chat-8B.

Setup:
  - Sample up to N_SENTENCES sentences from aligned_text_summary, shuffle their order.
  - Present ALL shots labeled "Clip 1", "Clip 2", ... with their frames.
  - Ask the model to assign clip IDs to each summary sentence.

Output format requested:
  Summary 1: Clip A, Clip B
  Summary 2: Clip C
  ...

Evaluation (per sentence, per video, per dataset):
  - precision, recall, f1, iou (Jaccard), hit@any, n_pred_clips

Frame budget per shot: 30% of shot length, clamped to [4, 8], padded to LOCAL_NUM_FRAMES multiple.

Environment: internvideo
Usage:
    python run_internvideo.py --dataset summe
    python run_internvideo.py --dataset summe --video video_1  # smoke test
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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.data_loader import get_video_ids, load_all_shots, PROCESSED_DATA

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID         = "OpenGVLab/InternVideo2_5_Chat_8B"
MODEL_DISPLAY    = "InternVideo2.5-Chat-8B"
HF_CACHE_DIR     = "/data/hf-cache/hub"
LOCAL_NUM_FRAMES = 4    # hardcoded in modeling_internvl_chat_hico2.py
MAX_NEW_TOKENS   = 512
N_SENTENCES      = 3    # max summary sentences to sample
RANDOM_SEED      = 42
INPUT_SIZE       = 448
MAX_ATTEMPTS     = 3
FRAMES_MIN       = 4    # per-shot frame budget: min
FRAMES_MAX       = 8    # per-shot frame budget: max
FRAMES_PCT       = 0.30 # per-shot frame budget: proportion of shot length

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

SYSTEM_PROMPT = (
    "You are a video analysis assistant. "
    "Your task is to match summary sentences to the video clips they describe."
)

# ── Image preprocessing ───────────────────────────────────────────────────────
_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Metrics ────────────────────────────────────────────────────────────────────

def set_metrics(pred, gt):
    """Compute P, R, F1, IoU (Jaccard), Hit@Any. IoU=1.0 when both empty."""
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
    """MD5-based deterministic per-video+dataset seed, independent of processing order."""
    h = hashlib.md5(f"{global_seed}:{dataset}:{video_id}".encode()).hexdigest()
    return random.Random(int(h, 16) & 0xFFFFFFFF)


# ── Frame budget ───────────────────────────────────────────────────────────────

def compute_n_frames_for_shot(n_frames_in_shot):
    """30% of shot length, clamped to [FRAMES_MIN, FRAMES_MAX]."""
    return max(FRAMES_MIN, min(FRAMES_MAX, round(n_frames_in_shot * FRAMES_PCT)))


def pad_frames_to_multiple(frames):
    """Pad frame list to multiple of LOCAL_NUM_FRAMES by repeating last frame."""
    remainder = len(frames) % LOCAL_NUM_FRAMES
    if remainder:
        frames = frames + [frames[-1]] * (LOCAL_NUM_FRAMES - remainder)
    return frames


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
    """Load summary sentences and their GT shot_ids."""
    data = _load_processed(dataset_name).get(video_id, {})
    summary = data.get("aligned_text_summary", {})
    sentences  = summary.get("sentences", [])
    alignments = summary.get("alignments", [])
    shot_ids = []
    for i in range(len(sentences)):
        entry = next((a for a in alignments if a.get("sentence_id") == i), None)
        shot_ids.append(entry["shot_ids"] if entry else [])
    return sentences, shot_ids, summary.get("raw_text")


# ── Prompt / tensor building ─────────────────────────────────────────────────

def preprocess_and_build_prompt(shots, shuffled_sentences):
    """
    Build prompt string with <image> tokens + stack frame tensors.
    Per-shot frame budget: compute_n_frames_for_shot, padded to LOCAL_NUM_FRAMES multiple.

    Returns:
        prompt_text:        str
        pixel_values:       (total_padded_frames, C, H, W)
        num_patches_list:   list of 1s
        n_padded_per_shot:  list[int]
        total_frames_capped: int  (pre-padding)
        total_frames_padded: int
    """
    n_clips = len(shots)
    parts = [f"Below are {n_clips} clips from a video, labeled Clip 1 through Clip {n_clips}:\n"]
    all_tensors = []
    num_patches_list = []
    n_padded_per_shot = []
    total_capped = 0

    for i, shot in enumerate(shots):
        k = compute_n_frames_for_shot(len(shot["frames"]))
        frames = shot["frames"][:k]
        total_capped += len(frames)
        frames_padded = pad_frames_to_multiple(frames)
        n_padded_per_shot.append(len(frames_padded))

        parts.append(f"\nClip {i + 1}:\n")
        parts.append("\n".join(["<image>"] * len(frames_padded)))

        for frame in frames_padded:
            all_tensors.append(_transform(frame))
            num_patches_list.append(1)

    pixel_values = torch.stack(all_tensors) if all_tensors else torch.zeros(0)
    total_padded = sum(n_padded_per_shot)

    n_sents = len(shuffled_sentences)
    sents_formatted = "\n".join(
        f"Summary {k + 1}: {s}" for k, s in enumerate(shuffled_sentences)
    )
    parts.append(
        f"\nHere are {n_sents} sentences from a text summary of this video "
        f"(the order has been shuffled):\n\n"
        f"{sents_formatted}\n\n"
        "For each summary sentence, identify which clip(s) it corresponds to.\n"
        "Use exactly this format (list clip numbers separated by commas):\n"
        + "\n".join(f"Summary {k + 1}: Clip X, Clip Y" for k in range(n_sents))
    )
    return "\n".join(parts), pixel_values, num_patches_list, n_padded_per_shot, total_capped, total_padded


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_assignments(text, n_sentences, n_clips):
    """
    Parse "Summary K: Clip A, Clip B, ..." for K=1..n_sentences.

    Returns:
        assignments: list[list[int]] — 0-indexed clip IDs per sentence
        parse_info:  dict with format_ok, nonempty_predictions_count, failure_reasons
    """
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
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {MODEL_DISPLAY} ...")
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    model.eval()
    print(f"  Loaded on: {next(model.parameters()).device}")
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def _infer(model, tokenizer, shots, shuffled_sentences):
    out = {"text": "", "n_output_tokens": None,
           "total_frames_capped": None, "total_frames_padded": None, "error": None}
    try:
        device = next(model.parameters()).device
        prompt_text, pixel_values, num_patches_list, _, total_capped, total_padded = \
            preprocess_and_build_prompt(shots, shuffled_sentences)
        out["total_frames_capped"] = total_capped
        out["total_frames_padded"] = total_padded

        pv = pixel_values.to(dtype=torch.bfloat16, device=device)
        generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_text}"

        with torch.no_grad():
            response = model.chat(
                tokenizer, pixel_values=pv, question=full_prompt,
                generation_config=generation_config, num_patches_list=num_patches_list,
                history=None, return_history=False,
            )
        out["n_output_tokens"] = len(tokenizer.encode(response))
        out["text"] = response

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
        "n_output_tokens": None,
        "total_frames_capped": None, "total_frames_padded": None,
        "raw_output": "",
        "format_ok": None, "nonempty_predictions_count": None, "failure_reasons": None,
        "error": "fatal_error", "error_message": error_msg,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, tokenizer, dataset_name, video_id, seed):
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
            "n_output_tokens": None,
            "total_frames_capped": None, "total_frames_padded": None,
            "raw_output": "",
            "format_ok": None, "nonempty_predictions_count": None, "failure_reasons": None,
            "error": "no_summary",
        }

    # Sample + shuffle (deterministic per video+dataset)
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

    r = _infer(model, tokenizer, shots, shuffled_sents)
    print(f"  clips: {n_shots}  |  sentences: {n_sample}  |  "
          f"frames_capped={r['total_frames_capped']} (padded={r['total_frames_padded']})")

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
            "total_frames_capped": r["total_frames_capped"],
            "total_frames_padded": r["total_frames_padded"],
            "raw_output": r["text"],
            "format_ok": None, "nonempty_predictions_count": None, "failure_reasons": None,
            "error": r["error"],
        }

    print(f"  out={r['n_output_tokens']}")
    assignments_shuffled, parse_info = parse_assignments(r["text"], n_sample, n_shots)

    # Remap to original sampled order
    assignments = [[] for _ in range(n_sample)]
    for pos in range(n_sample):
        assignments[inv_shuffle[pos]] = assignments_shuffled[pos]

    # Per-sentence metrics
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
        "total_frames_capped":        r["total_frames_capped"],
        "total_frames_padded":        r["total_frames_padded"],
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
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem     = f"direct_internvideo_{args.dataset}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file = results_dir / f"{stem}.json"
    partial  = results_dir / f"{stem}.partial.json"

    model, tokenizer = load_model()

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_DISPLAY} | dataset={args.dataset} ({len(video_ids)} videos)\n")

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
                result = run_video(model, tokenizer, args.dataset, vid, args.seed)
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
        "model":       MODEL_DISPLAY,
        "dataset":     args.dataset,
        "mode":        "direct",
        "timestamp":   timestamp,
        "n_sentences_max": N_SENTENCES,
        "frames_pct":  FRAMES_PCT,
        "frames_min":  FRAMES_MIN,
        "frames_max":  FRAMES_MAX,
        "seed":        args.seed,
        "n_success":   n_success,
        "avg_p":       avg_p,
        "avg_r":       avg_r,
        "avg_f1":      avg_f1,
        "avg_iou":     avg_iou,
        "avg_hit":     avg_hit,
        "video_results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
