"""
Shared data loader for MMS-Benchmark experiments.
Loads H5 data and frame images for shot-level inference.

H5 field notes:
  - n_frame_per_seg: ORIGINAL frame count per shot (sums to n_frames)
  - picks:           original frame indices of sampled frames (len = n_steps)
  - change_points:   (n_shots, 2) shot boundaries in original frame space
"""

import os
import numpy as np
import h5py
from PIL import Image


DATASET_CONFIG = {
    "summe":    {"h5": "/data/MMS_Benchmark/data/summe/h5/summe.h5",         "frames": "/data/MMS_Benchmark/data/summe/frames"},
    "tvsum":    {"h5": "/data/MMS_Benchmark/data/tvsum/h5/tvsum.h5",         "frames": "/data/MMS_Benchmark/data/tvsum/frames"},
    "ovp":      {"h5": "/data/MMS_Benchmark/data/ovp/h5/ovp.h5",             "frames": "/data/MMS_Benchmark/data/ovp/frames"},
    "youtube":  {"h5": "/data/MMS_Benchmark/data/youtube/h5/youtube.h5",     "frames": "/data/MMS_Benchmark/data/youtube/frames"},
    # VideoXum frames are stored with sequential indices (0, 1, 2, ..., n_steps-1),
    # not original frame indices. Set sequential_frames=True to use position in picks
    # array as the filename index instead of the pick value.
    "videoxum": {"h5": "/data/MMS_Benchmark/data/videoxum/h5/videoxum.h5",   "frames": "/data/MMS_Benchmark/data/videoxum/frames", "sequential_frames": True},
    "mrhisum":  {"h5": "/data/MMS_Benchmark/data/mrhisum/h5/mrhisum.h5",     "frames": "/data/MMS_Benchmark/data/mrhisum/frames"},
}

# Processed data JSONs that contain aligned_text_summary
PROCESSED_DATA = {
    dataset: f"/data/MMS_Benchmark/data/processed/{dataset}_data.json"
    for dataset in ["summe", "tvsum", "ovp", "youtube", "videoxum", "mrhisum"]
}

# In-memory cache to avoid reloading the same large JSON repeatedly
_processed_cache: dict = {}


def get_video_ids(dataset_name):
    """Return sorted list of video IDs in the dataset."""
    cfg = DATASET_CONFIG[dataset_name]
    with h5py.File(cfg["h5"], "r") as f:
        return sorted(f.keys())


def compute_sampled_per_shot(picks, change_points):
    """
    Count how many sampled frames (picks) fall within each shot.

    Args:
        picks:         (n_steps,) original frame indices of sampled frames
        change_points: (n_shots, 2) [start, end] in original frame space

    Returns:
        n_sampled_per_shot: (n_shots,) sampled frame count per shot
    """
    n_sampled = []
    for start, end in change_points:
        count = int(np.sum((picks >= start) & (picks <= end)))
        n_sampled.append(count)
    return np.array(n_sampled)


def load_video_meta(dataset_name, video_id):
    """
    Load metadata for a single video from H5.

    Returns dict with:
        picks:              (n_steps,) original frame indices of sampled frames
        change_points:      (n_shots, 2) shot boundaries in original frame space
        n_frame_per_seg:    (n_shots,) ORIGINAL frame count per shot (for knapsack cost)
        n_sampled_per_shot: (n_shots,) SAMPLED frame count per shot (for indexing picks)
        shot_level_gt:      (n_shots,) binary GT shot selection
        gtscore:            (n_steps,) frame-level importance scores
        video_name:         str
    """
    cfg = DATASET_CONFIG[dataset_name]
    with h5py.File(cfg["h5"], "r") as f:
        v = f[video_id]
        picks         = v["picks"][:]
        change_points = v["change_points"][:]
        meta = {
            "picks":              picks,
            "change_points":      change_points,
            "n_frame_per_seg":    v["n_frame_per_seg"][:],   # original frames per shot
            "n_sampled_per_shot": compute_sampled_per_shot(picks, change_points),
            "shot_level_gt":      v["shot_level_gt"][:],
            "gtscore":            v["gtscore"][:],
            "video_name":         (v["video_name"][()].decode() if isinstance(v["video_name"][()], bytes) else str(v["video_name"][()])) if "video_name" in v else video_id,
        }
    return meta


def get_gt_shot_scores(gtscore, n_sampled_per_shot):
    """
    Compute mean gtscore per shot for Spearman correlation.

    Args:
        gtscore:           (n_steps,) frame-level GT scores
        n_sampled_per_shot: (n_shots,) sampled frame count per shot

    Returns:
        gt_shot_scores: (n_shots,) mean gtscore per shot
    """
    gt_shot_scores = []
    start = 0
    for n in n_sampled_per_shot:
        if n > 0:
            gt_shot_scores.append(float(np.mean(gtscore[start:start + n])))
        else:
            gt_shot_scores.append(0.0)
        start += n
    return np.array(gt_shot_scores)


def load_shot_frames(dataset_name, video_id, shot_idx, meta, max_frames_per_shot=None):
    """
    Load PIL Images for all sampled frames in a given shot.

    Args:
        dataset_name:        e.g. "summe"
        video_id:            e.g. "video_1"
        shot_idx:            int, which shot (0-indexed)
        meta:                dict from load_video_meta()
        max_frames_per_shot: if set, uniformly subsample to this many frames

    Returns:
        frames:        list of PIL.Image
        frame_indices: list of original frame indices (from picks)
    """
    frames_root = DATASET_CONFIG[dataset_name]["frames"]
    video_frames_dir = os.path.join(frames_root, video_id)
    sequential = DATASET_CONFIG[dataset_name].get("sequential_frames", False)

    # Get picks belonging to this shot using n_sampled_per_shot
    start = int(np.sum(meta["n_sampled_per_shot"][:shot_idx]))
    n = int(meta["n_sampled_per_shot"][shot_idx])
    shot_picks = meta["picks"][start:start + n]
    # For sequential datasets, file index = position in global picks array
    shot_file_indices = list(range(start, start + n))

    # Optional uniform subsampling
    if max_frames_per_shot and len(shot_picks) > max_frames_per_shot:
        indices = np.linspace(0, len(shot_picks) - 1, max_frames_per_shot, dtype=int)
        shot_picks = shot_picks[indices]
        shot_file_indices = [shot_file_indices[i] for i in indices]

    frames = []
    frame_indices = []
    for pick, file_idx in zip(shot_picks, shot_file_indices):
        if sequential:
            img_path = os.path.join(video_frames_dir, f"frame_{file_idx:06d}.jpg")
        else:
            img_path = os.path.join(video_frames_dir, f"frame_{pick:05d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(video_frames_dir, f"frame_{pick:06d}.jpg")
        if os.path.exists(img_path):
            frames.append(Image.open(img_path).convert("RGB"))
            frame_indices.append(int(pick))
        else:
            print(f"  [WARN] Missing frame: {img_path}")

    return frames, frame_indices


def load_all_shots(dataset_name, video_id, max_frames_per_shot=None):
    """
    Load all shots for a video.

    Returns:
        meta:  dict from load_video_meta()
        shots: list of dicts, each with:
                 shot_idx, frames, frame_indices, gt, gt_score
    """
    meta = load_video_meta(dataset_name, video_id)
    gt_shot_scores = get_gt_shot_scores(meta["gtscore"], meta["n_sampled_per_shot"])
    n_shots = len(meta["n_frame_per_seg"])

    shots = []
    for i in range(n_shots):
        frames, frame_indices = load_shot_frames(
            dataset_name, video_id, i, meta, max_frames_per_shot
        )
        shots.append({
            "shot_idx":      i,
            "frames":        frames,
            "frame_indices": frame_indices,
            "gt":            int(meta["shot_level_gt"][i]),
            "gt_score":      float(gt_shot_scores[i]),
        })

    return meta, shots


def load_aligned_summary(dataset_name, video_id):
    """
    Load the aligned_text_summary.raw_text for a video from the processed data JSON.

    Data lives in /data/MMS_Benchmark/data/processed/{dataset}_data.json.
    The JSON is cached in memory after first load (large files).

    Returns str if available, None otherwise.
    """
    import json

    if dataset_name not in _processed_cache:
        json_path = PROCESSED_DATA.get(dataset_name)
        if not json_path or not os.path.exists(json_path):
            _processed_cache[dataset_name] = {}
        else:
            try:
                with open(json_path) as f:
                    _processed_cache[dataset_name] = json.load(f)
            except Exception:
                _processed_cache[dataset_name] = {}

    video_data = _processed_cache[dataset_name].get(video_id, {})
    summary = video_data.get("aligned_text_summary", {})

    # Handle string-encoded dict (e.g. "{'raw_text': '...'}")
    if isinstance(summary, str):
        import ast
        try:
            summary = ast.literal_eval(summary)
        except Exception:
            return summary if summary.strip() else None

    if isinstance(summary, dict):
        # Prefer sentences list
        sentences = summary.get("sentences")
        if isinstance(sentences, list) and sentences:
            return " ".join(sentences)
        # Fall back to raw_text
        raw = summary.get("raw_text", "")
        return raw.strip() if raw.strip() else None
    return None


def load_all_frames(dataset_name, video_id):
    """
    Load ALL sampled frames for a video as a flat list (no shot split, no frame limit).

    Returns:
        meta:          dict from load_video_meta()
        frames:        list of PIL.Image in chronological order
        frame_indices: list of original frame indices (from picks)
    """
    meta = load_video_meta(dataset_name, video_id)
    frames_root = DATASET_CONFIG[dataset_name]["frames"]
    video_frames_dir = os.path.join(frames_root, video_id)

    sequential = DATASET_CONFIG[dataset_name].get("sequential_frames", False)
    frames = []
    frame_indices = []
    for file_idx, pick in enumerate(meta["picks"]):
        if sequential:
            img_path = os.path.join(video_frames_dir, f"frame_{file_idx:06d}.jpg")
        else:
            img_path = os.path.join(video_frames_dir, f"frame_{pick:05d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(video_frames_dir, f"frame_{pick:06d}.jpg")
        if os.path.exists(img_path):
            frames.append(Image.open(img_path).convert("RGB"))
            frame_indices.append(int(pick))
        else:
            print(f"  [WARN] Missing frame: {img_path}")

    return meta, frames, frame_indices


def load_aligned_document(dataset_name, video_id):
    """
    Load aligned_full_document sentences as plain text (no alignment info).

    Returns str (sentences joined into a paragraph) if available, None otherwise.
    """
    import json

    if dataset_name not in _processed_cache:
        json_path = PROCESSED_DATA.get(dataset_name)
        if not json_path or not os.path.exists(json_path):
            _processed_cache[dataset_name] = {}
        else:
            try:
                with open(json_path) as f:
                    _processed_cache[dataset_name] = json.load(f)
            except Exception:
                _processed_cache[dataset_name] = {}

    video_data = _processed_cache[dataset_name].get(video_id, {})
    doc = video_data.get("aligned_full_document", {})
    sentences = doc.get("sentences") if isinstance(doc, dict) else None
    if isinstance(sentences, list) and sentences:
        return " ".join(sentences)
    return None


if __name__ == "__main__":
    dataset = "summe"
    vid = get_video_ids(dataset)[0]
    meta, shots = load_all_shots(dataset, vid)
    print(f"Dataset: {dataset} | Video: {vid} ({meta['video_name']})")
    print(f"  Total shots: {len(shots)}")
    print(f"  n_sampled_per_shot[:5]: {meta['n_sampled_per_shot'][:5]}")
    print(f"  n_frame_per_seg[:5]:    {meta['n_frame_per_seg'][:5]}")
    print(f"  Shot 0: {len(shots[0]['frames'])} frames, gt={shots[0]['gt']}, gt_score={shots[0]['gt_score']:.3f}")
    print(f"  Frame indices[:3]: {shots[0]['frame_indices'][:3]}")
