"""
exp6_salience_ranking: Shot salience ranking with InternVideo2.5-Chat-8B.

Randomly selects up to N_SHOTS shots per video, labels them Shot 1..N sequentially,
and asks the model to rank them by importance.

Shot ordering mode (--shot_order):
  ordered  (default): shots presented in original temporal order (re-labeled 1..N)
  shuffled:           shots presented in shuffled order (re-labeled 1..N after shuffle)
  Used for ablation: does the model exhibit position bias (first/last shots preferred)?

Experiment settings:
  1. ordered  + no_doc  (default)
  2. shuffled + no_doc
  3. ordered  + with_doc

Frame budget per shot: 30% of shot length, clamped to [FRAMES_MIN, FRAMES_MAX],
then padded to LOCAL_NUM_FRAMES=4 multiple (architectural constraint).

Features:
  - Per-video deterministic RNG: seed derived from (global_seed, dataset, video_id) via MD5
  - Partial checkpoint: resumes from crash (saved after every video)
  - Retry: up to MAX_ATTEMPTS attempts per video; gc + empty_cache between retries
  - Robust parse: primary "Shot X" pattern + fallback to bare numbers; dedup + completeness check
  - Top-K Precision: K = min(3, n); reported as top_k_precision with k stored separately

Environment: internvideo
Usage:
    python run_internvideo.py --dataset summe
    python run_internvideo.py --dataset summe --shot_order shuffled
    python run_internvideo.py --dataset summe --with_doc
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
from shared.data_loader import get_video_ids, load_all_shots, load_aligned_document
from shared.evaluate_ranking import compute_ranking_metrics

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID         = "OpenGVLab/InternVideo2_5_Chat_8B"
MODEL_DISPLAY    = "InternVideo2.5-Chat-8B"
HF_CACHE_DIR     = "/data/hf-cache/hub"
LOCAL_NUM_FRAMES = 4    # hardcoded in modeling_internvl_chat_hico2.py
MAX_NEW_TOKENS   = 256
N_SHOTS          = 8
RANDOM_SEED      = 42
INPUT_SIZE       = 448
MAX_ATTEMPTS     = 3
FRAMES_MIN       = 4    # per-shot frame budget: min
FRAMES_MAX       = 8    # per-shot frame budget: max
FRAMES_PCT       = 0.30 # per-shot frame budget: proportion of shot length

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Per-video deterministic RNG ────────────────────────────────────────────────

def _video_rng(global_seed, dataset, video_id):
    """MD5-based deterministic per-video seed — independent of processing order."""
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


# ── Prompt construction ───────────────────────────────────────────────────────

def _ranking_instruction(n):
    shot_list = ", ".join(f"Shot {i}" for i in range(1, n + 1))
    return (
        f"\nYou have seen {n} shots from a video ({shot_list}).\n"
        "Rank ALL of these shots from most important to least important "
        "based on how essential each shot is for understanding the main content "
        "and key events of the overall video.\n\n"
        "Output ONLY the ranking in this exact format (use > between shots):\n"
        "Shot X > Shot Y > Shot Z > ...\n"
        "Include all shots exactly once."
    )


def build_prompt(selected_shots, doc_text):
    """
    Build prompt string with interleaved <image> tokens.
    Applies per-shot adaptive frame budget (compute_n_frames_for_shot).

    Returns:
        prompt:              str
        pixel_values:        (total_padded_frames, C, H, W) tensor
        num_patches_list:    list of 1s
        n_padded_per_shot:   list[int]
        total_frames_capped: int   (pre-padding)
        total_frames_padded: int
    """
    parts = []
    all_tensors = []
    num_patches_list = []
    n_padded_per_shot = []
    total_capped = 0

    if doc_text:
        parts.append(f"Background document describing the video:\n{doc_text}\n")

    parts.append(
        f"Below are {len(selected_shots)} shots from the video, "
        f"labeled Shot 1 through Shot {len(selected_shots)}:\n"
    )

    for i, shot in enumerate(selected_shots):
        k = compute_n_frames_for_shot(len(shot["frames"]))
        frames = shot["frames"][:k]
        total_capped += len(frames)
        frames_padded = pad_frames_to_multiple(frames)
        n_padded_per_shot.append(len(frames_padded))

        parts.append(f"\nShot {i + 1}:\n")
        parts.append("\n".join(["<image>"] * len(frames_padded)))

        for frame in frames_padded:
            all_tensors.append(_transform(frame))
            num_patches_list.append(1)

    parts.append(_ranking_instruction(len(selected_shots)))
    prompt = "\n".join(parts)

    pixel_values = torch.stack(all_tensors) if all_tensors else torch.zeros(0)
    total_padded = sum(n_padded_per_shot)

    return prompt, pixel_values, num_patches_list, n_padded_per_shot, total_capped, total_padded


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_ranking(text, n):
    """
    Robust parser: tries "Shot X" primary pattern first, falls back to bare integers.
    Deduplicates, validates completeness.

    Returns:
        ranking:        list[int] of 1-indexed shot labels (best→worst), or None
        failure_reason: None | "empty_output" | "wrong_format" | "duplicated_items" |
                        "missing_items"
    """
    if not text or not text.strip():
        return None, "empty_output"

    nums = re.findall(r'Shot\s+(\d+)', text, re.IGNORECASE)
    if not nums:
        # Fallback: bare numbers in order of appearance
        nums = re.findall(r'\b(\d+)\b', text)
        if not nums:
            return None, "wrong_format"

    nums = [int(x) for x in nums]
    seen, deduped, has_dup = set(), [], False
    for x in nums:
        if x in seen:
            has_dup = True
        else:
            seen.add(x)
            deduped.append(x)

    if has_dup:
        return None, "duplicated_items"

    expected = set(range(1, n + 1))
    if set(deduped) != expected:
        missing = expected - set(deduped)
        return None, "missing_items" if missing else "wrong_format"
    if len(deduped) != n:
        return None, "wrong_format"

    return deduped, None


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


# ── Single inference call ─────────────────────────────────────────────────────

def _infer(model, tokenizer, selected_shots, doc_text):
    out = {"text": "", "n_output_tokens": None,
           "total_frames_capped": None, "total_frames_padded": None, "error": None}
    try:
        prompt, pixel_values, num_patches_list, _, total_capped, total_padded = \
            build_prompt(selected_shots, doc_text)
        out["total_frames_capped"] = total_capped
        out["total_frames_padded"] = total_padded

        pv = pixel_values.to(dtype=torch.bfloat16, device=model.device)
        generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}

        with torch.no_grad():
            response = model.chat(
                tokenizer, pv, prompt, generation_config,
                num_patches_list=num_patches_list,
                history=None, return_history=False,
            )

        out["n_output_tokens"] = len(tokenizer.encode(response))
        out["text"] = response

    except Exception as e:
        out["error"] = str(e)
    return out


# ── Result helpers ─────────────────────────────────────────────────────────────

def _null_metrics():
    return {"spearman_r": None, "kendall_tau": None, "top_k_precision": None, "top_k": None}


def _fatal_error_result(video_id, error_msg):
    return {
        "video_id":               video_id,
        "n_shots_total":          None,
        "n_shots_selected":       None,
        "selected_shot_indices":  None,
        "shot_order":             None,
        "shuffled_order":         None,
        "total_frames_capped":    None,
        "total_frames_padded":    None,
        "n_output_tokens":        None,
        "raw_output":             "",
        "parsed_ranking":         None,
        "failure_reason":         None,
        "gt_scores":              None,
        **_null_metrics(),
        "position_bias":          {"top1_display_position": None, "top1_original_index": None},
        "error":                  "fatal_error",
        "error_message":          error_msg,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, tokenizer, dataset_name, video_id, with_doc, shot_order, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    meta, shots = load_all_shots(dataset_name, video_id)
    doc_text = load_aligned_document(dataset_name, video_id) if with_doc else None

    valid_shots = [s for s in shots if len(s["frames"]) > 0]
    n = min(N_SHOTS, len(valid_shots))

    # Sample shots (deterministic per video)
    selected = sorted(rng.sample(valid_shots, n), key=lambda s: s["shot_idx"])
    original_indices = [s["shot_idx"] for s in selected]

    # Apply shot ordering
    if shot_order == "shuffled":
        # shuffled_order[display_pos] = index into the sorted 'selected' list
        shuffled_order = list(range(n))
        rng.shuffle(shuffled_order)
        selected = [selected[i] for i in shuffled_order]
    else:
        shuffled_order = list(range(n))  # identity — temporal order unchanged

    print(f"  shots: {len(shots)} total, {n} selected"
          f"  |  order: {shot_order}  |  doc: {'yes' if doc_text else 'no'}")

    r = _infer(model, tokenizer, selected, doc_text)
    print(f"  frames_capped={r['total_frames_capped']} (padded={r['total_frames_padded']})")

    # gt_scores_display[i] = gt score of the shot shown as Shot (i+1) in the prompt
    gt_scores_display = [selected[i]["gt_score"] for i in range(n)]

    # gt_scores_orig: scores in original selection order (index matches original_indices)
    gt_scores_orig = [None] * n
    for display_pos, orig_pos in enumerate(shuffled_order):
        gt_scores_orig[orig_pos] = selected[display_pos]["gt_score"]

    if r["error"]:
        print(f"  ERROR: {r['error'][:120]}")
        return {
            "video_id":               video_id,
            "video_name":             meta["video_name"],
            "n_shots_total":          len(shots),
            "n_shots_selected":       n,
            "selected_shot_indices":  original_indices,
            "shot_order":             shot_order,
            "shuffled_order":         shuffled_order,
            "total_frames_capped":    r["total_frames_capped"],
            "total_frames_padded":    r["total_frames_padded"],
            "n_output_tokens":        r["n_output_tokens"],
            "raw_output":             r["text"],
            "parsed_ranking":         None,
            "failure_reason":         None,
            "gt_scores":              gt_scores_orig,
            **_null_metrics(),
            "position_bias":          {"top1_display_position": None, "top1_original_index": None},
            "error":                  r["error"],
        }

    print(f"  out={r['n_output_tokens']}")
    print(f"  raw output: {r['text'][:200]}")

    parsed_ranking, failure_reason = parse_ranking(r["text"], n)

    spearman_r = kendall_tau = top_k_prec = top_k = None
    position_bias = {"top1_display_position": None, "top1_original_index": None}
    if parsed_ranking is not None:
        # parsed_ranking is in display-label space (1..n); gt_scores_display matches that space
        top_k = min(3, n)
        metrics = compute_ranking_metrics(parsed_ranking, gt_scores_display, top_k=top_k)
        spearman_r  = metrics["spearman_r"]
        kendall_tau = metrics["kendall_tau"]
        top_k_prec  = metrics.get(f"top{top_k}_precision")

        # Position bias: which display position and which original shot did the model rank #1?
        top1_display_pos  = parsed_ranking[0]                         # 1-indexed display label
        top1_orig_pos     = shuffled_order[top1_display_pos - 1]      # 0-indexed in sorted selection
        position_bias = {
            "top1_display_position": top1_display_pos,                # 1=first shown, n=last shown
            "top1_original_index":   original_indices[top1_orig_pos], # actual shot_idx in video
        }

        print(f"  ranking: {' > '.join(f'Shot {x}' for x in parsed_ranking)}")
        print(f"  Spearman={spearman_r:.3f}  Kendall={kendall_tau:.3f}  Top{top_k}-P={top_k_prec:.2f}"
              f"  top1_display={top1_display_pos}/{n}")
    else:
        print(f"  [WARN] Could not parse ranking: {failure_reason}")

    return {
        "video_id":               video_id,
        "video_name":             meta["video_name"],
        "n_shots_total":          len(shots),
        "n_shots_selected":       n,
        "selected_shot_indices":  original_indices,
        "shot_order":             shot_order,
        "shuffled_order":         shuffled_order,
        "total_frames_capped":    r["total_frames_capped"],
        "total_frames_padded":    r["total_frames_padded"],
        "n_output_tokens":        r["n_output_tokens"],
        "raw_output":             r["text"],
        "parsed_ranking":         parsed_ranking,
        "failure_reason":         failure_reason,
        "gt_scores":              gt_scores_orig,
        "spearman_r":             spearman_r,
        "kendall_tau":            kendall_tau,
        "top_k_precision":        top_k_prec,
        "top_k":                  top_k,
        "position_bias":          position_bias,
        "error":                  r["error"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="summe",
                        choices=["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"])
    parser.add_argument("--shot_order", default="ordered", choices=["ordered", "shuffled"],
                        help="ordered: temporal order (default); shuffled: randomised order")
    parser.add_argument("--with_doc",   action="store_true",
                        help="Prepend aligned_document as background text context")
    parser.add_argument("--video",      default=None, help="Single video ID for smoke test")
    parser.add_argument("--video_ids_file", default=None,
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_suffix = "_doc" if args.with_doc else ""
    stem       = f"internvideo_{args.dataset}_{args.shot_order}{doc_suffix}_seed{args.seed}_{timestamp}"
    if args.video_ids_file:
        import os as _os
        stem += f"_{_os.path.splitext(_os.path.basename(args.video_ids_file))[0]}"
    out_file   = results_dir / f"{stem}.json"
    partial    = results_dir / f"{stem}.partial.json"

    model, tokenizer = load_model()

    if args.video:
        video_ids = [args.video]
    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = get_video_ids(args.dataset)
    print(f"\nRunning {MODEL_DISPLAY} | dataset={args.dataset} "
          f"| shot_order={args.shot_order} | with_doc={args.with_doc} "
          f"| seed={args.seed} ({len(video_ids)} videos)\n")

    # Resume from partial checkpoint
    all_results = []
    done_ids: set = set()
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
                result = run_video(
                    model, tokenizer, args.dataset, vid,
                    args.with_doc, args.shot_order, args.seed,
                )
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

    n_success      = sum(1 for r in all_results if not r.get("error"))
    n_parsed       = sum(1 for r in all_results if r.get("parsed_ranking") is not None)
    spearman_vals  = collect("spearman_r")
    kendall_vals   = collect("kendall_tau")
    top_k_vals     = collect("top_k_precision")
    top_k_list     = collect("top_k")
    avg_spearman   = float(np.mean(spearman_vals))  if spearman_vals  else None
    avg_kendall    = float(np.mean(kendall_vals))   if kendall_vals   else None
    avg_top_k_prec = float(np.mean(top_k_vals))     if top_k_vals     else None
    top_k_used     = max(set(top_k_list), key=top_k_list.count) if top_k_list else None

    failure_counts: dict = {}
    for r in all_results:
        fr = r.get("failure_reason")
        if fr:
            failure_counts[fr] = failure_counts.get(fr, 0) + 1

    # Position bias aggregate: average display position of the model's top-1 pick
    # (1 = always picks first-shown shot, N_SHOTS = always picks last-shown)
    top1_display_vals = [
        r["position_bias"]["top1_display_position"]
        for r in all_results
        if r.get("position_bias", {}).get("top1_display_position") is not None
    ]
    avg_top1_display_pos = float(np.mean(top1_display_vals)) if top1_display_vals else None

    def fmt(v): return f"{v:.4f}" if v is not None else "N/A"
    def fmt2(v): return f"{v:.2f}" if v is not None else "N/A"
    print(f"\nDone: {n_success}/{len(all_results)} success, {n_parsed} parsed")
    print(f"  avg Spearman={fmt(avg_spearman)}  Kendall={fmt(avg_kendall)}"
          f"  Top{top_k_used}-P={fmt(avg_top_k_prec)}")
    print(f"  avg top1 display position={fmt2(avg_top1_display_pos)} "
          f"(neutral={fmt2(N_SHOTS / 2 + 0.5)} for n={N_SHOTS})")
    if failure_counts:
        print(f"  parse failures: {failure_counts}")

    output = {
        "model":                  MODEL_DISPLAY,
        "dataset":                args.dataset,
        "shot_order":             args.shot_order,
        "with_doc":               args.with_doc,
        "timestamp":              timestamp,
        "n_shots_max":            N_SHOTS,
        "frames_pct":             FRAMES_PCT,
        "frames_min":             FRAMES_MIN,
        "frames_max":             FRAMES_MAX,
        "seed":                   args.seed,
        "n_success":              n_success,
        "n_parsed":               n_parsed,
        "avg_spearman":           avg_spearman,
        "avg_kendall":            avg_kendall,
        "avg_top_k_prec":         avg_top_k_prec,
        "top_k_used":             top_k_used,
        "avg_top1_display_pos":   avg_top1_display_pos,
        "position_bias_neutral":  (N_SHOTS + 1) / 2,
        "failure_counts":         failure_counts,
        "video_results":          all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
