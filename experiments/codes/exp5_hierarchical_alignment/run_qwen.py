"""
exp5_hierarchical_alignment: Two-step hierarchical alignment with Qwen2.5-VL / Qwen3-VL.

Two-step process per video:

  Step 1 — Text-only (no frames):
    Input:  Full aligned_full_document sentences labeled "Doc 1..M" +
            N shuffled summary sentences labeled "Summary 1..N".
    Task:   For each summary sentence, identify which document sentence(s) it corresponds to.
    Output: "Summary K: Doc X, Doc Y"
    Eval:   P/R/F1 vs GT doc sentence indices (those sharing shot_ids with the summary sentence).

  Step 2 — Visual:
    Input:  All video clips labeled "Clip 1..C" with their frames +
            The union of doc sentences predicted in Step 1, RE-NUMBERED as Candidate Doc 1, 2, ...
    Task:   For each candidate doc sentence, identify which clip(s) it corresponds to.
    Output: "Candidate Doc K: Clip A, Clip B"
    Eval:   Final chain P/R/F1/IoU/Hit — for each summary sentence, union the Step-2 predicted
            clips from its Step-1 predicted docs, compare with GT shot_ids.

  Oracle Step 2:
    Same as Step 2 but using GT doc sentences (upper bound for chain).

Metrics (per sentence, averaged per video, averaged across videos):
  Step 1 (doc set):   precision, recall, f1
  Final chain (clips): precision, recall, f1, iou (Jaccard), hit@any, n_pred_clips
  Oracle chain (clips): same set

Frame sampling: 30% of shot length, clamped to [4, 8].

Environment: MLLM
Usage:
    python run_qwen.py --model qwen2.5 --dataset summe
    python run_qwen.py --model qwen3 --dataset summe
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
from shared.data_loader import get_video_ids, load_all_shots, PROCESSED_DATA
from shared.evaluate_sets import set_f1  # kept for compatibility; use set_metrics below

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen2.5": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "display": "Qwen2.5-VL-7B-Instruct"},
    "qwen3":   {"model_id": "Qwen/Qwen3-VL-8B-Instruct",   "display": "Qwen3-VL-8B-Instruct"},
}
HF_CACHE_DIR   = "/data/hf-cache/hub"
MAX_NEW_TOKENS = 512
N_SENTENCES    = 3    # max summary sentences to sample
RANDOM_SEED    = 42
MAX_ATTEMPTS   = 3
FRAMES_MIN     = 4    # per-shot frame budget: min
FRAMES_MAX     = 8    # per-shot frame budget: max
FRAMES_PCT     = 0.30 # per-shot frame budget: proportion of shot length

SYSTEM_PROMPT_STEP1 = (
    "You are a video analysis assistant. "
    "Your task is to match summary sentences to the document sentences they correspond to."
)
SYSTEM_PROMPT_STEP2 = (
    "You are a video analysis assistant. "
    "Your task is to match document sentences to the video clips they describe."
)


# ── Metrics ────────────────────────────────────────────────────────────────────

def set_metrics(pred, gt):
    """
    Compute precision, recall, F1, IoU (Jaccard), and Hit@Any for set predictions.

    IoU = |pred ∩ gt| / |pred ∪ gt|; returns 1.0 when both sets are empty.
    Hit = 1 if pred ∩ gt ≠ ∅ else 0.
    """
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
    """
    Load aligned_text_summary and aligned_full_document with alignment info.

    Returns:
        summ_sents:    list[str]
        summ_shot_ids: list[list[int]]  — GT shots per summary sentence (0-indexed)
        doc_sents:     list[str]
        doc_shot_ids:  list[list[int]]  — shots per doc sentence (0-indexed)
        raw_text:      str | None
    """
    data = _load_processed(dataset_name).get(video_id, {})
    summary = data.get("aligned_text_summary", {})
    doc     = data.get("aligned_full_document", {})

    def extract(block):
        sentences  = block.get("sentences", [])
        alignments = block.get("alignments", [])
        shot_ids   = []
        for i in range(len(sentences)):
            entry = next((a for a in alignments if a.get("sentence_id") == i), None)
            shot_ids.append(entry["shot_ids"] if entry else [])
        return sentences, shot_ids

    summ_sents, summ_shot_ids = extract(summary)
    doc_sents,  doc_shot_ids  = extract(doc)

    return summ_sents, summ_shot_ids, doc_sents, doc_shot_ids, summary.get("raw_text")


def gt_doc_indices(summ_shot_ids_i, doc_shot_ids):
    """GT doc sentence indices (0-indexed) whose shot_ids overlap with summary sentence."""
    s = set(summ_shot_ids_i)
    return [j for j, dsi in enumerate(doc_shot_ids) if s.intersection(dsi)]


# ── Prompt / content building ─────────────────────────────────────────────────

def build_step1_content(doc_sents, shuffled_sents):
    """Text-only content: document sentences + shuffled summary sentences."""
    doc_block  = "\n".join(f"Doc {j + 1}: {s}" for j, s in enumerate(doc_sents))
    n_sents    = len(shuffled_sents)
    summ_block = "\n".join(f"Summary {k + 1}: {s}" for k, s in enumerate(shuffled_sents))
    instruction = (
        "\n\nFor each summary sentence, identify which document sentence(s) it corresponds to.\n"
        "A summary sentence may correspond to one or more document sentences.\n"
        "Use exactly this format (list doc numbers separated by commas):\n"
        + "\n".join(f"Summary {k + 1}: Doc X, Doc Y" for k in range(n_sents))
    )
    return (
        f"Here is a document describing the video ({len(doc_sents)} sentences):\n\n"
        f"{doc_block}\n\n"
        f"Here are {n_sents} sentences from a text summary of this video "
        f"(the order has been shuffled):\n\n"
        f"{summ_block}"
        f"{instruction}"
    )


def build_step2_content(shots, doc_indices_0indexed, doc_sents):
    """
    Visual content: all clips with frames (dynamic budget) + re-numbered candidate doc sentences.

    doc_indices_0indexed: sorted list of 0-indexed doc sentence indices.
    Frame budget per shot: compute_n_frames_for_shot.

    Returns:
        content:           list of Qwen content dicts
        n_capped_per_shot: list of frame counts actually used per shot
    """
    content = []
    n_clips = len(shots)
    n_capped_per_shot = []

    content.append({
        "type": "text",
        "text": f"Below are {n_clips} clips from a video, labeled Clip 1 through Clip {n_clips}:\n"
    })

    for i, shot in enumerate(shots):
        k = compute_n_frames_for_shot(len(shot["frames"]))
        frames = shot["frames"][:k]
        n_capped_per_shot.append(len(frames))
        content.append({"type": "text", "text": f"\nClip {i + 1}:\n"})
        for frame in frames:
            content.append({"type": "image", "image": frame})

    if not doc_indices_0indexed:
        content.append({
            "type": "text",
            "text": "\n[No document sentences were identified in the previous step.]\n"
        })
        return content, n_capped_per_shot

    # Re-number candidates: Candidate Doc 1, 2, 3, ...
    cand_lines = "\n".join(
        f"Candidate Doc {cand_k}: {doc_sents[j]}"
        for cand_k, j in enumerate(doc_indices_0indexed, 1)
        if j < len(doc_sents)
    )
    n_cands = len(doc_indices_0indexed)
    instruction = (
        f"\n\nBelow are {n_cands} candidate document sentence(s). "
        "For each, identify which clip(s) it corresponds to.\n"
        "Use exactly this format (list clip numbers separated by commas):\n"
        + "\n".join(f"Candidate Doc {k}: Clip A, Clip B" for k in range(1, n_cands + 1))
    )
    content.append({
        "type": "text",
        "text": (
            f"\nHere are {n_cands} candidate document sentence(s) to match to the clips above:\n\n"
            f"{cand_lines}"
            f"{instruction}"
        )
    })
    return content, n_capped_per_shot


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_step1(text, n_sentences, n_docs):
    """
    Parse "Summary K: Doc X, Doc Y, ..." for K=1..n_sentences.

    Returns:
        assignments: list[list[int]] — 0-indexed doc IDs per sentence
        parse_info:  dict with format_ok, nonempty_predictions_count, failure_reasons
    """
    assignments = [[] for _ in range(n_sentences)]
    format_ok   = True
    nonempty    = 0
    failures    = []

    for k in range(1, n_sentences + 1):
        pat = rf"Summary\s*{k}\s*:\s*(.+?)(?=Summary\s*\d|$)"
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if not m:
            format_ok = False
            failures.append(f"missing_line_{k}")
        else:
            nums = re.findall(r'\d+', m.group(1))
            docs = [int(x) - 1 for x in nums if 1 <= int(x) <= n_docs]
            assignments[k - 1] = docs
            if docs:
                nonempty += 1
            else:
                failures.append(f"no_valid_doc_ids_line_{k}")

    return assignments, {
        "format_ok":                  format_ok,
        "nonempty_predictions_count": nonempty,
        "failure_reasons":            failures if failures else None,
    }


def parse_step2(text, n_candidates, n_clips):
    """
    Parse "Candidate Doc K: Clip A, Clip B, ..." for K=1..n_candidates.

    Returns:
        clip_lists:  list[list[int]] — 0-indexed clip IDs per candidate (length = n_candidates)
        parse_info:  dict
    """
    clip_lists = [[] for _ in range(n_candidates)]
    format_ok  = True
    nonempty   = 0
    failures   = []

    for k in range(1, n_candidates + 1):
        pat = rf"Candidate\s+Doc\s*{k}\s*:\s*(.+?)(?=Candidate\s+Doc\s*\d|$)"
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if not m:
            format_ok = False
            failures.append(f"missing_line_{k}")
        else:
            nums  = re.findall(r'\d+', m.group(1))
            clips = [int(x) - 1 for x in nums if 1 <= int(x) <= n_clips]
            clip_lists[k - 1] = clips
            if clips:
                nonempty += 1
            else:
                failures.append(f"no_valid_clip_ids_line_{k}")

    return clip_lists, {
        "format_ok":                  format_ok,
        "nonempty_predictions_count": nonempty,
        "failure_reasons":            failures if failures else None,
    }


def _build_doc_to_clips(doc_indices_0indexed, clip_lists):
    """Map candidate position back to original 0-indexed doc idx."""
    return {doc_indices_0indexed[k]: clip_lists[k] for k in range(len(doc_indices_0indexed))}


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


# ── Inference helpers ─────────────────────────────────────────────────────────

def _infer_text(model, processor, system_prompt, user_text):
    """Text-only inference (step 1)."""
    import torch

    device = next(model.parameters()).device
    out = {"text": "", "n_input_tokens": None, "n_output_tokens": None, "error": None}
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_input], padding=True, return_tensors="pt",
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


def _infer_visual(model, processor, system_prompt, content):
    """Visual inference (step 2)."""
    from qwen_vl_utils import process_vision_info
    import torch

    device = next(model.parameters()).device
    out = {"text": "", "n_input_tokens": None, "n_output_tokens": None, "error": None}
    try:
        messages = [
            {"role": "system", "content": system_prompt},
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


def _run_step2(model, processor, shots, doc_indices_0indexed, doc_sents, label="step2"):
    """Run step 2 for a given set of doc indices. Returns result dict."""
    if not doc_indices_0indexed:
        return {
            "skipped": True, "skip_reason": "no_doc_indices",
            "raw_output": None, "n_input_tokens": None, "n_output_tokens": None, "error": None,
            "format_ok": None, "nonempty_predictions_count": None, "failure_reasons": None,
            "parse_ok": None,
            "total_frames_submitted": None,
            "doc_to_clips": {},
        }

    content, n_capped_per_shot = build_step2_content(shots, doc_indices_0indexed, doc_sents)
    total_frames = sum(n_capped_per_shot)
    print(f"  [{label}] cands={len(doc_indices_0indexed)}, frames_submitted={total_frames}")

    r = _infer_visual(model, processor, SYSTEM_PROMPT_STEP2, content)

    if r["error"]:
        print(f"  [{label}] ERROR: {r['error'][:100]}")
        return {
            "skipped": False, "raw_output": None,
            "n_input_tokens": None, "n_output_tokens": None,
            "error": r["error"],
            "format_ok": None, "nonempty_predictions_count": None, "failure_reasons": None,
            "parse_ok": None,
            "total_frames_submitted": total_frames,
            "doc_to_clips": {},
        }

    print(f"  [{label}] in={r['n_input_tokens']}  out={r['n_output_tokens']}")
    print(f"            {r['text'][:200]}")

    n_shots = len(shots)
    clip_lists, parse_info = parse_step2(r["text"], len(doc_indices_0indexed), n_shots)
    doc_to_clips = _build_doc_to_clips(doc_indices_0indexed, clip_lists)

    return {
        "skipped":                    False,
        "raw_output":                 r["text"],
        "n_input_tokens":             r["n_input_tokens"],
        "n_output_tokens":            r["n_output_tokens"],
        "error":                      None,
        "format_ok":                  parse_info["format_ok"],
        "nonempty_predictions_count": parse_info["nonempty_predictions_count"],
        "failure_reasons":            parse_info["failure_reasons"],
        "parse_ok":                   parse_info["format_ok"] and parse_info["nonempty_predictions_count"] > 0,
        "total_frames_submitted":     total_frames,
        "doc_to_clips":               doc_to_clips,
    }


# ── Result helpers ─────────────────────────────────────────────────────────────

def _null_metrics():
    return {
        "step1_avg_p": None, "step1_avg_r": None, "step1_avg_f1": None,
        "final_avg_p": None, "final_avg_r": None, "final_avg_f1": None,
        "final_avg_iou": None, "final_avg_hit": None,
        "oracle_final_avg_p": None, "oracle_final_avg_r": None, "oracle_final_avg_f1": None,
        "oracle_final_avg_iou": None, "oracle_final_avg_hit": None,
    }


def _fatal_error_result(video_id, error_msg):
    return {
        "video_id": video_id, "skipped": False,
        "step1_result": {}, "step2_result": {}, "step2_oracle_result": {},
        "sentence_results": [],
        **_null_metrics(),
        "error": "fatal_error", "error_message": error_msg,
    }


# ── Per-video inference ───────────────────────────────────────────────────────

def run_video(model, processor, dataset_name, video_id, seed):
    rng = _video_rng(seed, dataset_name, video_id)

    meta, shots = load_all_shots(dataset_name, video_id)
    summ_sents, summ_shot_ids, doc_sents, doc_shot_ids, raw_text = \
        load_exp5_data(dataset_name, video_id)

    n_shots = len(shots)
    n_docs  = len(doc_sents)

    if not summ_sents or not doc_sents:
        print(f"  [SKIP] Missing summary or document sentences")
        return {
            "video_id": video_id, "video_name": meta["video_name"],
            "n_clips": n_shots, "n_doc_sents": n_docs, "n_sentences_sampled": 0,
            "step1_result": {}, "step2_result": {}, "step2_oracle_result": {},
            "sentence_results": [],
            **_null_metrics(),
            "error": "no_data",
        }

    # Sample + shuffle summary sentences (deterministic per video+dataset)
    n_sample = min(N_SENTENCES, len(summ_sents))
    sampled_indices  = sorted(rng.sample(range(len(summ_sents)), n_sample))
    sampled_sents    = [summ_sents[i]    for i in sampled_indices]
    sampled_gt_shots = [summ_shot_ids[i] for i in sampled_indices]
    sampled_gt_docs  = [gt_doc_indices(summ_shot_ids[i], doc_shot_ids) for i in sampled_indices]

    shuffle_order = list(range(n_sample))
    rng.shuffle(shuffle_order)
    shuffled_sents = [sampled_sents[k] for k in shuffle_order]
    inv_shuffle    = [0] * n_sample
    for pos, orig in enumerate(shuffle_order):
        inv_shuffle[pos] = orig

    print(f"  clips: {n_shots}  |  doc_sents: {n_docs}  |  sentences: {n_sample}")

    # ── Step 1: text-only ─────────────────────────────────────────────────────
    step1_text = build_step1_content(doc_sents, shuffled_sents)
    r1 = _infer_text(model, processor, SYSTEM_PROMPT_STEP1, step1_text)

    s1_assignments_shuffled = [[] for _ in range(n_sample)]
    s1_parse_info = {"format_ok": None, "nonempty_predictions_count": 0, "failure_reasons": None}

    if r1["error"]:
        print(f"  [Step1] ERROR: {r1['error'][:100]}")
    else:
        print(f"  [Step1] in={r1['n_input_tokens']}  out={r1['n_output_tokens']}")
        s1_assignments_shuffled, s1_parse_info = parse_step1(r1["text"], n_sample, n_docs)

    # Remap to original sample order
    step1_assignments = [[] for _ in range(n_sample)]
    for pos in range(n_sample):
        step1_assignments[inv_shuffle[pos]] = s1_assignments_shuffled[pos]

    # Step 1 metrics (doc-set P/R/F1)
    s1_p_vals, s1_r_vals, s1_f1_vals = [], [], []
    for j in range(n_sample):
        m = set_metrics(set(step1_assignments[j]), set(sampled_gt_docs[j]))
        s1_p_vals.append(m["p"]); s1_r_vals.append(m["r"]); s1_f1_vals.append(m["f1"])
    step1_avg_p  = _avg(s1_p_vals)
    step1_avg_r  = _avg(s1_r_vals)
    step1_avg_f1 = _avg(s1_f1_vals)
    print(f"  [Step1] parse={s1_parse_info['format_ok']} nonempty={s1_parse_info['nonempty_predictions_count']}  "
          f"P={step1_avg_p:.3f} R={step1_avg_r:.3f} F1={step1_avg_f1:.3f}"
          if step1_avg_f1 is not None else
          f"  [Step1] parse={s1_parse_info['format_ok']} nonempty={s1_parse_info['nonempty_predictions_count']}")

    # ── Step 2: predicted docs ─────────────────────────────────────────────────
    pred_doc_indices = sorted(set(d for preds in step1_assignments for d in preds))
    s2 = _run_step2(model, processor, shots, pred_doc_indices, doc_sents, label="Step2-pred")

    # ── Step 2 oracle: GT docs ─────────────────────────────────────────────────
    oracle_doc_indices = sorted(set(j for gt_docs_i in sampled_gt_docs for j in gt_docs_i))
    s2_oracle = _run_step2(model, processor, shots, oracle_doc_indices, doc_sents, label="Step2-oracle")

    # ── Chain metrics ──────────────────────────────────────────────────────────
    def compute_chain_metrics(doc_to_clips_map, use_gt_docs=False):
        """
        Compute per-sentence clip prediction metrics.
        use_gt_docs=False → uses step1_assignments (predicted docs)
        use_gt_docs=True  → uses sampled_gt_docs (oracle upper bound)
        """
        per_sent = []
        p_vals, r_vals, f1_vals, iou_vals, hit_vals = [], [], [], [], []
        for j in range(n_sample):
            gt_clips   = set(sampled_gt_shots[j])
            src_docs   = sampled_gt_docs[j] if use_gt_docs else step1_assignments[j]
            pred_clips = set(c for d in src_docs for c in doc_to_clips_map.get(d, []))
            m = set_metrics(pred_clips, gt_clips)
            p_vals.append(m["p"]); r_vals.append(m["r"]); f1_vals.append(m["f1"])
            iou_vals.append(m["iou"]); hit_vals.append(m["hit"])
            per_sent.append({
                "pred_clips": sorted(pred_clips), "n_pred_clips": len(pred_clips),
                "p": m["p"], "r": m["r"], "f1": m["f1"], "iou": m["iou"], "hit": m["hit"],
            })
        return per_sent, _avg(p_vals), _avg(r_vals), _avg(f1_vals), _avg(iou_vals), _avg(hit_vals)

    pred_per_sent, final_avg_p, final_avg_r, final_avg_f1, final_avg_iou, final_avg_hit = \
        compute_chain_metrics(s2["doc_to_clips"], use_gt_docs=False)
    oracle_per_sent, oracle_avg_p, oracle_avg_r, oracle_avg_f1, oracle_avg_iou, oracle_avg_hit = \
        compute_chain_metrics(s2_oracle["doc_to_clips"], use_gt_docs=True)

    print(f"  [Final]  P={final_avg_p:.3f} R={final_avg_r:.3f} F1={final_avg_f1:.3f} "
          f"IoU={final_avg_iou:.3f} Hit={final_avg_hit:.3f}"
          if final_avg_f1 is not None else f"  [Final] final={final_avg_f1}")
    print(f"  [Oracle] P={oracle_avg_p:.3f} R={oracle_avg_r:.3f} F1={oracle_avg_f1:.3f} "
          f"IoU={oracle_avg_iou:.3f} Hit={oracle_avg_hit:.3f}"
          if oracle_avg_f1 is not None else f"  [Oracle] oracle={oracle_avg_f1}")

    # ── Build per-sentence results ─────────────────────────────────────────────
    sentence_results = []
    for j in range(n_sample):
        pm = pred_per_sent[j]
        om = oracle_per_sent[j]
        sentence_results.append({
            "sentence_idx":        sampled_indices[j],
            "sentence_text":       sampled_sents[j],
            "gt_clips":            sorted(sampled_gt_shots[j]),
            "gt_docs":             sorted(sampled_gt_docs[j]),
            "pred_docs":           sorted(step1_assignments[j]),
            "step1_p":             s1_p_vals[j],
            "step1_r":             s1_r_vals[j],
            "step1_f1":            s1_f1_vals[j],
            "pred_clips":          pm["pred_clips"],
            "n_pred_clips":        pm["n_pred_clips"],
            "final_p":             pm["p"],
            "final_r":             pm["r"],
            "final_f1":            pm["f1"],
            "final_iou":           pm["iou"],
            "final_hit":           pm["hit"],
            "oracle_pred_clips":   om["pred_clips"],
            "oracle_n_pred_clips": om["n_pred_clips"],
            "oracle_p":            om["p"],
            "oracle_r":            om["r"],
            "oracle_f1":           om["f1"],
            "oracle_iou":          om["iou"],
            "oracle_hit":          om["hit"],
        })

    return {
        "video_id":                 video_id,
        "video_name":               meta["video_name"],
        "n_clips":                  n_shots,
        "n_doc_sents":              n_docs,
        "n_sentences_sampled":      n_sample,
        "sampled_sentence_indices": sampled_indices,
        "shuffle_order":            shuffle_order,
        "shuffled_sentence_texts":  shuffled_sents,
        "sampled_gt_docs":          [sorted(sampled_gt_docs[j])  for j in range(n_sample)],
        "sampled_gt_shots":         [sorted(sampled_gt_shots[j]) for j in range(n_sample)],
        "step1_result": {
            "n_input_tokens":             r1["n_input_tokens"],
            "n_output_tokens":            r1["n_output_tokens"],
            "raw_output":                 r1["text"],
            "format_ok":                  s1_parse_info["format_ok"],
            "nonempty_predictions_count": s1_parse_info["nonempty_predictions_count"],
            "failure_reasons":            s1_parse_info["failure_reasons"],
            "parse_ok":                   s1_parse_info["format_ok"] and s1_parse_info["nonempty_predictions_count"] > 0,
            "error":                      r1["error"],
        },
        "step2_result": {
            "skipped":                    s2["skipped"],
            "n_input_tokens":             s2["n_input_tokens"],
            "n_output_tokens":            s2["n_output_tokens"],
            "raw_output":                 s2["raw_output"],
            "format_ok":                  s2["format_ok"],
            "nonempty_predictions_count": s2["nonempty_predictions_count"],
            "failure_reasons":            s2["failure_reasons"],
            "parse_ok":                   s2["parse_ok"],
            "total_frames_submitted":     s2["total_frames_submitted"],
            "error":                      s2["error"],
            "pred_doc_indices":           pred_doc_indices,
        },
        "step2_oracle_result": {
            "skipped":                    s2_oracle["skipped"],
            "n_input_tokens":             s2_oracle["n_input_tokens"],
            "n_output_tokens":            s2_oracle["n_output_tokens"],
            "raw_output":                 s2_oracle["raw_output"],
            "format_ok":                  s2_oracle["format_ok"],
            "nonempty_predictions_count": s2_oracle["nonempty_predictions_count"],
            "failure_reasons":            s2_oracle["failure_reasons"],
            "parse_ok":                   s2_oracle["parse_ok"],
            "total_frames_submitted":     s2_oracle["total_frames_submitted"],
            "error":                      s2_oracle["error"],
            "oracle_doc_indices":         oracle_doc_indices,
        },
        "sentence_results":          sentence_results,
        "step1_avg_p":               step1_avg_p,
        "step1_avg_r":               step1_avg_r,
        "step1_avg_f1":              step1_avg_f1,
        "final_avg_p":               final_avg_p,
        "final_avg_r":               final_avg_r,
        "final_avg_f1":              final_avg_f1,
        "final_avg_iou":             final_avg_iou,
        "final_avg_hit":             final_avg_hit,
        "oracle_final_avg_p":        oracle_avg_p,
        "oracle_final_avg_r":        oracle_avg_r,
        "oracle_final_avg_f1":       oracle_avg_f1,
        "oracle_final_avg_iou":      oracle_avg_iou,
        "oracle_final_avg_hit":      oracle_avg_hit,
        "error":                     r1["error"] or s2["error"],
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
                        help="Path to a text file with one video ID per line. "
                             "If given, only those videos are processed.")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem     = f"hierarchical_{args.model}_{args.dataset}_seed{args.seed}_{timestamp}"
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
          f"({len(video_ids)} videos)\n")

    # Resume from partial checkpoint if exists
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

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    def collect(key):
        return [r[key] for r in all_results if r.get(key) is not None]

    n_success = sum(1 for r in all_results if not r.get("error"))

    avg_s1_p   = _avg(collect("step1_avg_p"))
    avg_s1_r   = _avg(collect("step1_avg_r"))
    avg_s1_f1  = _avg(collect("step1_avg_f1"))
    avg_fn_p   = _avg(collect("final_avg_p"))
    avg_fn_r   = _avg(collect("final_avg_r"))
    avg_fn_f1  = _avg(collect("final_avg_f1"))
    avg_fn_iou = _avg(collect("final_avg_iou"))
    avg_fn_hit = _avg(collect("final_avg_hit"))
    avg_orc_p   = _avg(collect("oracle_final_avg_p"))
    avg_orc_r   = _avg(collect("oracle_final_avg_r"))
    avg_orc_f1  = _avg(collect("oracle_final_avg_f1"))
    avg_orc_iou = _avg(collect("oracle_final_avg_iou"))
    avg_orc_hit = _avg(collect("oracle_final_avg_hit"))

    def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
    print(f"\nDone: {n_success}/{len(all_results)} success")
    print(f"  Step1:  P={fmt(avg_s1_p)}  R={fmt(avg_s1_r)}  F1={fmt(avg_s1_f1)}")
    print(f"  Final:  P={fmt(avg_fn_p)}  R={fmt(avg_fn_r)}  F1={fmt(avg_fn_f1)}  "
          f"IoU={fmt(avg_fn_iou)}  Hit={fmt(avg_fn_hit)}")
    print(f"  Oracle: P={fmt(avg_orc_p)}  R={fmt(avg_orc_r)}  F1={fmt(avg_orc_f1)}  "
          f"IoU={fmt(avg_orc_iou)}  Hit={fmt(avg_orc_hit)}")

    output = {
        "model":                 MODEL_CONFIGS[args.model]["display"],
        "dataset":               args.dataset,
        "mode":                  "hierarchical",
        "timestamp":             timestamp,
        "n_sentences_max":       N_SENTENCES,
        "frames_pct":            FRAMES_PCT,
        "frames_min":            FRAMES_MIN,
        "frames_max":            FRAMES_MAX,
        "seed":                  args.seed,
        "n_success":             n_success,
        "avg_step1_p":           avg_s1_p,
        "avg_step1_r":           avg_s1_r,
        "avg_step1_f1":          avg_s1_f1,
        "avg_final_p":           avg_fn_p,
        "avg_final_r":           avg_fn_r,
        "avg_final_f1":          avg_fn_f1,
        "avg_final_iou":         avg_fn_iou,
        "avg_final_hit":         avg_fn_hit,
        "avg_oracle_final_p":    avg_orc_p,
        "avg_oracle_final_r":    avg_orc_r,
        "avg_oracle_final_f1":   avg_orc_f1,
        "avg_oracle_final_iou":  avg_orc_iou,
        "avg_oracle_final_hit":  avg_orc_hit,
        "video_results":         all_results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if partial.exists():
        partial.unlink()
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
