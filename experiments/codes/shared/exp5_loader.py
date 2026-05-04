"""
Shared data loader for exp5 (hierarchical + direct).

Loads aligned_full_document, aligned_text_summary, and derives GT mappings.
"""

import json
import os

_cache = {}

PROCESSED_DATA = {
    ds: f"/data/MMS_Benchmark/data/processed/{ds}_data.json"
    for ds in ["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"]
}


def _get_video_data(dataset_name, video_id):
    if dataset_name not in _cache:
        path = PROCESSED_DATA[dataset_name]
        if not os.path.exists(path):
            _cache[dataset_name] = {}
        else:
            with open(path) as f:
                _cache[dataset_name] = json.load(f)
    return _cache[dataset_name].get(video_id, {})


def load_exp5_data(dataset_name, video_id):
    """
    Load all data needed for exp5.

    Returns dict with:
        doc_sentences:       list[str], 0-indexed, from aligned_full_document.sentences
        doc_alignments:      list[dict], each {sentence_id, shot_ids, ...}
        summary_sentences:   list[str], 0-indexed, from aligned_text_summary.sentences
        summary_alignments:  list[dict], each {sentence_id, shot_ids, ...}
        gt_doc_per_summary:  list[set[int]], 1-indexed doc sentence indices per summary sentence
                             derived from shot_ids overlap
        gt_shots_per_summary: list[set[int]], shot_ids per summary sentence (from alignments)

    Returns None if doc or summary sentences are unavailable.
    """
    vdata = _get_video_data(dataset_name, video_id)

    doc  = vdata.get("aligned_full_document", {})
    summ = vdata.get("aligned_text_summary",  {})

    doc_sentences     = [s for s in doc.get("sentences", [])  if isinstance(s, str) and s.strip()]
    doc_alignments    = doc.get("alignments", [])
    summary_sentences = [s for s in summ.get("sentences", []) if isinstance(s, str) and s.strip()]
    summary_alignments = summ.get("alignments", [])

    if not doc_sentences or not summary_sentences:
        return None

    # GT: for each summary sentence, which doc sentences (1-indexed) overlap via shot_ids
    gt_doc_per_summary = []
    for s_align in summary_alignments:
        s_shots = set(s_align.get("shot_ids", []))
        matching = set()
        for d_align in doc_alignments:
            if s_shots & set(d_align.get("shot_ids", [])):
                matching.add(d_align["sentence_id"] + 1)  # 1-indexed
        gt_doc_per_summary.append(matching)

    # GT: for each summary sentence, which shot_ids (0-indexed clip IDs)
    gt_shots_per_summary = [
        set(a.get("shot_ids", [])) for a in summary_alignments
    ]

    return {
        "doc_sentences":          doc_sentences,
        "doc_alignments":         doc_alignments,
        "summary_sentences":      summary_sentences,
        "summary_alignments":     summary_alignments,
        "gt_doc_per_summary":     gt_doc_per_summary,
        "gt_shots_per_summary":   gt_shots_per_summary,
    }
