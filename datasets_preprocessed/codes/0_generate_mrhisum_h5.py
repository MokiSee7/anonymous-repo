#!/usr/bin/env python3
"""
Generate / update H5 file for MrHiSum dataset in SumMe-compatible format

MrHiSum H5 already contains: change_points, gt_summary, gtscore (31892 videos).
This script adds the missing fields to align with SumMe/TVSum/VideoXum:
  - features        : CLIP ViT-B/32 visual features  (n_steps, 512)
  - picks           : sampled frame indices           (n_steps,)
  - n_frames        : total frames in original video  (scalar)
  - n_steps         : number of sampled frames        (scalar)
  - video_name      : video key string                (bytes)
  - n_frame_per_seg : frames per KTS segment          (n_segments,)
  - segment_scores  : mean gtscore per segment        (n_segments,)
  - shot_level_gt   : knapsack-selected segments      (n_segments,)
  - gtsummary       : frame-level GT summary          (n_steps,)
  - change_points   : (optionally) regenerated via KTS

It can also regenerate change_points using KTS from extracted features,
replacing the original overlapping-boundary change_points with standard
non-overlapping [start, end] segments used by all other datasets.

Usage:
    # Update existing H5 (add missing fields, use existing change_points)
    python 0_generate_mrhisum_h5.py --test

    # Regenerate change_points with KTS from CLIP features
    python 0_generate_mrhisum_h5.py --regenerate-cp

    # Skip CLIP feature extraction (use random features, for debugging)
    python 0_generate_mrhisum_h5.py --no-extract-features --test
"""

import numpy as np
import h5py
import cv2
import argparse
import torch
import re
import tarfile
import tempfile
import shutil
import traceback
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image

# CLIP for feature extraction
clip = None
try:
    import clip
except ImportError:
    pass

# Import Knapsack algorithm
sys.path.insert(0, str(Path(__file__).parent))
from Knapsack import solve_knapsack


# ==================== CLIP Feature Extraction ====================

def load_clip_model(device='cuda' if torch.cuda.is_available() else 'cpu',
                    model_name='ViT-B/32'):
    """Load CLIP model for feature extraction"""
    if clip is None:
        print("CLIP not installed")
        print("   Install: pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    print(f"Loading CLIP model: {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device)
    print(f"CLIP model loaded")
    return model, preprocess, device


def extract_frames_from_tar(tar_path):
    """
    Extract all frames from tar.gz archive to temporary directory

    Args:
        tar_path: Path to tar.gz file

    Returns:
        Tuple of (sorted list of frame paths, temp_dir path)
        Caller must cleanup temp_dir after use
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="tar_frames_"))

    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        members.sort(key=lambda m: m.name)

        frame_files = [m for m in members if m.isfile() and
                       (m.name.endswith('.jpg') or m.name.endswith('.png'))]

        for member in frame_files:
            tar.extract(member, temp_dir)

    # Get sorted list of extracted frame paths
    frame_paths = sorted(
        list(temp_dir.rglob("*.jpg")) + list(temp_dir.rglob("*.png")),
        key=lambda p: p.name
    )

    return frame_paths, temp_dir


def extract_frame_features(frames_dir, clip_model, preprocess, device):
    """Extract CLIP features from saved frame images"""
    frames_dir = Path(frames_dir)

    # Get all frame files sorted by frame number
    frame_files = sorted(
        list(frames_dir.rglob('frame_*.jpg')) + list(frames_dir.rglob('frame_*.png')),
        key=lambda x: int(re.search(r'frame_(\d+)', x.name).group(1))
    )

    if not frame_files:
        # Fallback: any image files
        frame_files = sorted(
            list(frames_dir.rglob('*.jpg')) + list(frames_dir.rglob('*.png'))
        )

    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    features_list = []

    with torch.no_grad():
        for frame_file in tqdm(frame_files, desc="   Extracting features",
                               leave=False):
            image = Image.open(frame_file)
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True)
            features_list.append(image_features.cpu().numpy()[0])

    features = np.array(features_list, dtype=np.float32)
    return features


def extract_features_from_video(video_path, picks, clip_model, preprocess,
                                device):
    """
    Extract CLIP features directly from video file at specified frame indices

    Args:
        video_path: Path to video file
        picks: Array of frame indices to extract
        clip_model: CLIP model
        preprocess: CLIP preprocessing function
        device: torch device

    Returns:
        features: (n_picks, feature_dim) array
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    features_list = []

    with torch.no_grad():
        for frame_idx in tqdm(picks, desc="   Extracting features from video",
                              leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                # Use zero features for missing frames
                features_list.append(np.zeros(512, dtype=np.float32))
                continue

            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            image_tensor = preprocess(image).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True)
            features_list.append(image_features.cpu().numpy()[0])

    cap.release()

    features = np.array(features_list, dtype=np.float32)
    return features


# ==================== KTS Algorithm ====================

def compute_kernel_matrix(features, sigma=1.0):
    """Compute Gaussian kernel matrix from features"""
    sq_norms = np.sum(features ** 2, axis=1, keepdims=True)
    distances_sq = sq_norms + sq_norms.T - 2 * np.dot(features, features.T)
    K = np.exp(-distances_sq / (2 * sigma ** 2))
    return K


def cpd_auto(K, ncp, vmax, min_seg_len=2, desc_rate=1, **kwargs):
    """Change Point Detection using dynamic programming

    Args:
        K: (n, n) kernel matrix
        ncp: number of change points to detect
        vmax: maximum score
        min_seg_len: minimum segment length constraint (default: 2)
    """
    n = K.shape[0]

    K_cum = np.zeros((n + 1, n + 1))
    K_cum[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)

    def score_segment(start, end):
        if start >= end:
            return 0
        length = end - start
        segment_sum = (K_cum[end, end] - K_cum[start, end] -
                       K_cum[end, start] + K_cum[start, start])
        score = segment_sum / length if length > 0 else 0
        return score

    dp = np.zeros((n + 1, ncp + 1))
    backtrack = np.zeros((n + 1, ncp + 1), dtype=int)

    # Base case: single segment [0, i), only valid if length >= min_seg_len
    for i in range(1, n + 1):
        if i >= min_seg_len:
            dp[i, 0] = score_segment(0, i)
        else:
            dp[i, 0] = -np.inf  # Invalid: too short

    for k in range(1, ncp + 1):
        for i in range(k + 1, n + 1):
            best_score = -np.inf
            best_split = -1

            # j must satisfy:
            # 1) first k segments need at least k * min_seg_len frames
            # 2) new segment [j, i) needs at least min_seg_len frames
            j_min = max(k, k * min_seg_len)
            j_max = i - min_seg_len  # so that (i - j) >= min_seg_len

            for j in range(j_min, j_max + 1):
                if dp[j, k - 1] == -np.inf:
                    continue  # Previous partition invalid
                new_score = dp[j, k - 1] + score_segment(j, i)
                if new_score > best_score:
                    best_score = new_score
                    best_split = j

            dp[i, k] = best_score
            backtrack[i, k] = best_split

    change_points = []
    curr_pos = n
    for k in range(ncp, 0, -1):
        split_pos = backtrack[curr_pos, k]
        change_points.append(split_pos)
        curr_pos = split_pos

    change_points.reverse()
    scores = dp[:, ncp]

    return change_points, scores


def kts_segmentation(features, num_segments=None, sigma=None, min_seg_len=2):
    """Perform KTS segmentation on video features

    Args:
        features: (n_frames, feature_dim) array
        num_segments: target number of segments (None = auto, n_frames//10)
        sigma: kernel bandwidth (None = auto via median heuristic)
        min_seg_len: minimum segment length (default: 2)
    """
    n_frames = features.shape[0]

    if sigma is None:
        sample_size = min(1000, n_frames)
        if sample_size < n_frames:
            indices = np.random.choice(n_frames, sample_size, replace=False)
            sample_features = features[indices]
        else:
            sample_features = features

        sq_norms = np.sum(sample_features ** 2, axis=1, keepdims=True)
        distances_sq = (sq_norms + sq_norms.T
                        - 2 * np.dot(sample_features, sample_features.T))
        distances = np.sqrt(np.maximum(distances_sq, 0))
        sigma = np.median(distances[distances > 0])

    if num_segments is None:
        num_segments = max(1, n_frames // 10)

    # Ensure we can satisfy min_seg_len with requested segment count
    max_possible_segs = n_frames // min_seg_len
    if num_segments > max_possible_segs:
        num_segments = max_possible_segs

    K = compute_kernel_matrix(features, sigma=sigma)

    num_change_points = num_segments - 1

    if num_change_points < 1:
        return np.array([[0, n_frames - 1]])

    change_point_indices, scores = cpd_auto(
        K, num_change_points, vmax=1.0, min_seg_len=min_seg_len)

    segments = []
    prev = 0
    for cp in change_point_indices:
        segments.append([prev, cp - 1])
        prev = cp
    segments.append([prev, n_frames - 1])

    return np.array(segments)


# ==================== Utility Functions ====================

def compute_picks(n_frames_total, n_sampled):
    """
    Compute uniform sampling frame indices

    Matches the convention used in MrHiSum frame extraction:
    np.round(np.linspace(0, n_frames_total-1, n_sampled)).astype(int)

    Args:
        n_frames_total: Total frames in original video
        n_sampled: Number of sampled frames (= len(gtscore))

    Returns:
        picks: (n_sampled,) array of frame indices
    """
    if n_sampled <= 0 or n_frames_total <= 0:
        return np.array([], dtype=np.int64)
    return np.round(np.linspace(0, n_frames_total - 1, n_sampled)).astype(
        np.int64)


def get_video_info(video_path):
    """Get video total frame count using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return {'n_frames': n_frames, 'fps': fps}


# ==================== Main Processing ====================

def generate_mrhisum_h5(
    h5_path,
    video_dir,
    frames_dir,
    test_mode=False,
    target_ratio=10,
    gt_portion=0.15,
    extract_features=True,
    clip_model_name='ViT-B/32',
    regenerate_cp=False,
    checkpoint_interval=100,
    video_list=None
):
    """
    Update MrHiSum H5 file with missing fields

    Pipeline per video:
    1. Read existing gtscore, gt_summary, change_points from H5
    2. Get n_frames from video file (or H5 if already written)
    3. Compute picks from n_frames and len(gtscore)
    4. Extract CLIP features from tar.gz frames or video
    5. (Optional) Regenerate change_points with KTS from features
    6. Compute n_frame_per_seg, segment_scores from change_points + gtscore
    7. Run Knapsack -> shot_level_gt
    8. Generate gtsummary (1 frame per GT shot)
    9. Write new fields to H5

    Args:
        h5_path: Path to existing MrHiSum H5 file
        video_dir: Directory containing video mp4 files
        frames_dir: Directory containing tar.gz frame archives
        test_mode: Only process 3 videos for testing
        target_ratio: Target frame:segment ratio for KTS
        gt_portion: Fraction of frames to select as GT summary
        extract_features: Whether to extract CLIP features
        clip_model_name: CLIP model variant
        regenerate_cp: If True, regenerate change_points with KTS
        checkpoint_interval: Save progress every N videos
    """
    print(f"\n{'=' * 80}")
    print(f"MrHiSum H5 Generation / Update")
    print(f"{'=' * 80}")
    print(f"H5 file: {h5_path}")
    print(f"Video directory: {video_dir}")
    print(f"Frames directory: {frames_dir}")
    print(f"Extract features: {extract_features}")
    if extract_features:
        print(f"CLIP model: {clip_model_name}")
    print(f"Regenerate change_points: {regenerate_cp}")
    print(f"KTS target ratio: {target_ratio}")
    print(f"GT portion: {gt_portion}")
    print(f"Test mode: {test_mode}")
    print(f"{'=' * 80}\n")

    if not h5_path.exists():
        print(f"H5 file not found: {h5_path}")
        return

    # Load CLIP model if needed
    clip_model = None
    preprocess = None
    device = None

    if extract_features:
        clip_model, preprocess, device = load_clip_model(
            model_name=clip_model_name)
        print()

    # Get video list
    if video_list is not None:
        # Use provided list (e.g. from --list)
        video_keys = video_list
        print(f"Using provided video list: {len(video_keys)} videos\n")
    else:
        with h5py.File(h5_path, 'r') as f:
            video_keys = sorted(f.keys(),
                                key=lambda x: int(x.split('_')[1]))

    if test_mode:
        video_keys = video_keys[:3]

    print(f"Processing {len(video_keys)} videos\n")

    # Statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'errors': [],
        'no_frames': [],
        'no_video': []
    }

    for batch_start in range(0, len(video_keys), checkpoint_interval):
        batch_end = min(batch_start + checkpoint_interval, len(video_keys))
        batch_keys = video_keys[batch_start:batch_end]

        # Collect updates for this batch
        updates = {}

        for video_key in tqdm(batch_keys,
                              desc=f"Batch {batch_start // checkpoint_interval + 1}"):
            try:
                # ── 1. Read existing fields from H5 ──
                with h5py.File(h5_path, 'r') as f:
                    group = f[video_key]

                    # Check if already fully processed
                    # When using --list, always reprocess listed videos
                    if video_list is None and \
                            'features' in group and 'gtsummary' in group:
                        stats['skipped'] += 1
                        continue

                    gtscore = group['gtscore'][:]
                    gt_summary = group['gt_summary'][:]
                    existing_cp = group['change_points'][:]

                n_steps = len(gtscore)

                # ── 2. Get n_frames from video ──
                # Try reading from H5 first (may have been written by extraction)
                n_frames = None
                with h5py.File(h5_path, 'r') as f:
                    if 'n_frames' in f[video_key]:
                        n_frames = int(f[video_key]['n_frames'][()])

                if n_frames is None:
                    # Try getting from video file
                    video_path = video_dir / f"{video_key}.mp4"
                    if video_path.exists():
                        info = get_video_info(video_path)
                        n_frames = info['n_frames']
                    else:
                        # Fallback: estimate from gtscore length
                        # (MrHiSum samples at ~1 frame per second for most)
                        tqdm.write(f"  {video_key}: no video file, "
                                   f"using n_frames = len(gtscore) = {n_steps}")
                        n_frames = n_steps

                # ── 3. Compute picks ──
                picks = compute_picks(n_frames, n_steps)

                # ── 4. Extract CLIP features ──
                features = None
                temp_dir_to_cleanup = None

                if extract_features:
                    # Try tar.gz first
                    tar_path = frames_dir / f"{video_key}.tar.gz"
                    frame_dir = frames_dir / video_key

                    if tar_path.exists():
                        frame_paths, temp_dir_to_cleanup = \
                            extract_frames_from_tar(tar_path)
                        if frame_paths:
                            features = extract_frame_features(
                                temp_dir_to_cleanup, clip_model, preprocess,
                                device)
                    elif frame_dir.exists() and \
                            len(list(frame_dir.glob('*.jpg'))) > 0:
                        features = extract_frame_features(
                            frame_dir, clip_model, preprocess, device)
                    else:
                        # Try extracting directly from video
                        video_path = video_dir / f"{video_key}.mp4"
                        if video_path.exists():
                            features = extract_features_from_video(
                                video_path, picks, clip_model, preprocess,
                                device)
                        else:
                            stats['no_frames'].append(video_key)
                            tqdm.write(
                                f"  {video_key}: no frames or video found, "
                                f"skipping")
                            continue

                    # Cleanup temp dir
                    if temp_dir_to_cleanup and temp_dir_to_cleanup.exists():
                        shutil.rmtree(temp_dir_to_cleanup)
                        temp_dir_to_cleanup = None
                else:
                    # Placeholder features for debugging
                    features = np.random.randn(n_steps, 512).astype(
                        np.float32)
                    features = features / np.linalg.norm(features, axis=1,
                                                         keepdims=True)

                # Verify feature count matches gtscore length
                if features.shape[0] != n_steps:
                    tqdm.write(
                        f"  {video_key}: feature count ({features.shape[0]}) "
                        f"!= gtscore length ({n_steps}), adjusting")
                    if features.shape[0] > n_steps:
                        features = features[:n_steps]
                    else:
                        # Pad with zeros
                        pad = np.zeros(
                            (n_steps - features.shape[0], features.shape[1]),
                            dtype=np.float32)
                        features = np.concatenate([features, pad], axis=0)

                # ── 5. Change points ──
                if regenerate_cp:
                    # Regenerate with KTS from features
                    num_segments = max(1, n_steps // target_ratio)
                    change_points = kts_segmentation(
                        features, num_segments=num_segments)
                else:
                    # Use existing change_points from H5
                    # Convert overlapping-boundary format to standard format
                    # MrHiSum original: [[0,1],[1,2],[2,3],...] (overlapping)
                    # Standard:         [[0,0],[1,1],[2,2],...] or KTS-style
                    #
                    # The existing change_points work at the sampled-frame
                    # level (indices into gtscore, not original video frames),
                    # so we keep them as-is for segment_scores computation.
                    change_points = existing_cp

                n_segments = len(change_points)

                # ── 6. Compute n_frame_per_seg and segment_scores ──
                n_frame_per_seg = np.array([
                    change_points[i, 1] - change_points[i, 0] + 1
                    for i in range(n_segments)
                ], dtype=np.int64)

                segment_scores = np.array([
                    np.mean(gtscore[
                        change_points[i, 0]:change_points[i, 1] + 1])
                    for i in range(n_segments)
                ], dtype=np.float64)

                # ── 7. Knapsack -> shot_level_gt ──
                capacity = int(n_steps * gt_portion)
                if capacity < 1:
                    capacity = 1

                shot_level_gt = solve_knapsack(
                    segment_scores, n_frame_per_seg, capacity)

                # ── 8. Generate gtsummary ──
                # Mark 1 frame per GT shot (highest gtscore in that shot)
                gtsummary = np.zeros(n_steps, dtype=np.float32)
                for i in range(n_segments):
                    if shot_level_gt[i] > 0:
                        start = int(change_points[i, 0])
                        end = int(change_points[i, 1])
                        if start <= end and end < n_steps:
                            shot_scores = gtscore[start:end + 1]
                            best_in_shot = np.argmax(shot_scores)
                            gtsummary[start + best_in_shot] = 1.0

                # Convert change_points from sampled-frame indices to original-frame indices
                # (change_points are used in sampled space for computation above;
                #  stored in original-frame space for consistency across datasets)
                change_points_orig = np.array(
                    [[int(picks[s]), int(picks[e])] for s, e in change_points],
                    dtype=np.int64
                )

                # Collect update data
                updates[video_key] = {
                    'video_name': video_key,
                    'n_frames': n_frames,
                    'n_steps': n_steps,
                    'picks': picks,
                    'features': features,
                    'change_points': change_points_orig,  # original-frame space
                    'n_frame_per_seg': n_frame_per_seg,
                    'segment_scores': segment_scores,
                    'shot_level_gt': shot_level_gt,
                    'gtsummary': gtsummary,
                }

                stats['processed'] += 1

                gt_clips = int(shot_level_gt.sum())
                gt_frames = int(gtsummary.sum())
                tqdm.write(
                    f"  {video_key}: {n_steps} frames, "
                    f"{n_segments} segments, "
                    f"{gt_clips} GT clips, "
                    f"{gt_frames} GT frames")

            except Exception as e:
                tqdm.write(f"  {video_key}: error - {e}")
                traceback.print_exc()
                stats['errors'].append((video_key, str(e)))

                # Cleanup on error
                if temp_dir_to_cleanup and Path(temp_dir_to_cleanup).exists():
                    shutil.rmtree(temp_dir_to_cleanup)

        # ── Batch write to H5 ──
        if updates:
            print(f"\nWriting {len(updates)} videos to H5...")
            with h5py.File(h5_path, 'a') as f:
                for vkey, data in updates.items():
                    group = f[vkey]

                    # Write or overwrite each field
                    for field_name, field_data in data.items():
                        if field_name in group:
                            del group[field_name]

                        if field_name == 'video_name':
                            group.create_dataset(
                                field_name,
                                data=field_data.encode('utf-8'))
                        elif field_name == 'n_frames':
                            group.create_dataset(
                                field_name, data=np.int64(field_data))
                        elif field_name == 'n_steps':
                            group.create_dataset(
                                field_name, data=np.int64(field_data))
                        elif field_name == 'picks':
                            group.create_dataset(
                                field_name, data=field_data, dtype='int64')
                        elif field_name == 'features':
                            group.create_dataset(
                                field_name, data=field_data, dtype='float32')
                        elif field_name == 'change_points':
                            group.create_dataset(
                                field_name, data=field_data, dtype='int64')
                        elif field_name == 'n_frame_per_seg':
                            group.create_dataset(
                                field_name, data=field_data, dtype='int64')
                        elif field_name == 'segment_scores':
                            group.create_dataset(
                                field_name, data=field_data, dtype='float64')
                        elif field_name == 'shot_level_gt':
                            group.create_dataset(
                                field_name, data=field_data, dtype='int32')
                        elif field_name == 'gtsummary':
                            group.create_dataset(
                                field_name, data=field_data, dtype='float32')

            print(f"Batch write complete\n")

    # ── Print final statistics ──
    print(f"\n{'=' * 80}")
    print(f"MrHiSum H5 Update Complete")
    print(f"{'=' * 80}")
    print(f"Videos processed: {stats['processed']}")
    print(f"Videos skipped (already done): {stats['skipped']}")

    if stats['no_frames']:
        print(f"\nNo frames/video found ({len(stats['no_frames'])}):")
        for vk in stats['no_frames'][:10]:
            print(f"   - {vk}")
        if len(stats['no_frames']) > 10:
            print(f"   ... and {len(stats['no_frames']) - 10} more")

    if stats['no_video']:
        print(f"\nNo video file ({len(stats['no_video'])}):")
        for vk in stats['no_video'][:10]:
            print(f"   - {vk}")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for vkey, err in stats['errors'][:10]:
            print(f"   - {vkey}: {err}")

    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate / update MrHiSum H5 with missing fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode: process 3 videos
  python 0_generate_mrhisum_h5.py --test

  # Full run with CLIP features from tar.gz frames
  python 0_generate_mrhisum_h5.py

  # Regenerate change_points with KTS (replace original)
  python 0_generate_mrhisum_h5.py --regenerate-cp

  # Skip feature extraction (debugging)
  python 0_generate_mrhisum_h5.py --no-extract-features --test

  # Custom paths
  python 0_generate_mrhisum_h5.py \\
      --h5 /data/MMS_Benchmark/data/mrhisum/h5/mrhisum.h5 \\
      --videos /data/MMS_Benchmark/data/mrhisum/videos \\
      --frames /data/MMS_Benchmark/data/mrhisum/frames

Fields added to H5:
  - video_name      : video key (e.g., 'video_1')
  - n_frames        : total frames in original video
  - n_steps         : number of sampled frames (= len(gtscore))
  - picks           : sampled frame indices in original video
  - features        : CLIP ViT-B/32 features (n_steps, 512)
  - n_frame_per_seg : frames per segment
  - segment_scores  : mean gtscore per segment
  - shot_level_gt   : knapsack-selected GT segments
  - gtsummary       : frame-level GT summary (1 per GT shot)
  - change_points   : (optionally regenerated via KTS)
        """
    )

    parser.add_argument('--test', action='store_true',
                        help='Test mode: only process 3 videos')
    parser.add_argument('--h5', type=str,
                        default='/data/MMS_Benchmark/data/mrhisum/h5/mrhisum.h5',
                        help='Path to MrHiSum H5 file')
    parser.add_argument('--videos', type=str,
                        default='/data/MMS_Benchmark/data/mrhisum/videos',
                        help='Directory containing video mp4 files')
    parser.add_argument('--frames', type=str,
                        default='/data/MMS_Benchmark/data/mrhisum/frames',
                        help='Directory containing tar.gz frame archives')
    parser.add_argument('--ratio', type=int, default=10,
                        help='Target frame:segment ratio for KTS (default: 10)')
    parser.add_argument('--gt-portion', type=float, default=0.15,
                        help='Portion of frames for GT summary (default: 0.15)')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                        help='CLIP model name (default: ViT-B/32)')
    parser.add_argument('--extract-features', action='store_true',
                        default=True,
                        help='Extract CLIP features (default: True)')
    parser.add_argument('--no-extract-features', dest='extract_features',
                        action='store_false',
                        help='Skip CLIP feature extraction')
    parser.add_argument('--regenerate-cp', action='store_true',
                        help='Regenerate change_points with KTS from features')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save to H5 every N videos (default: 100)')
    parser.add_argument('--list', type=str, default=None,
                        help='JSON file with list of video keys to reprocess')

    args = parser.parse_args()

    h5_path = Path(args.h5)
    video_dir = Path(args.videos)
    frames_dir = Path(args.frames)

    if not h5_path.exists():
        print(f"H5 file not found: {h5_path}")
        return

    # Load video list from JSON if provided
    video_list = None
    if args.list:
        import json
        list_path = Path(args.list)
        if not list_path.exists():
            print(f"List file not found: {list_path}")
            return
        with open(list_path, 'r') as f:
            video_list = json.load(f)
        print(f"Loaded {len(video_list)} videos from {list_path}")

    generate_mrhisum_h5(
        h5_path=h5_path,
        video_dir=video_dir,
        frames_dir=frames_dir,
        test_mode=args.test,
        target_ratio=args.ratio,
        gt_portion=args.gt_portion,
        extract_features=args.extract_features,
        clip_model_name=args.clip_model,
        regenerate_cp=args.regenerate_cp,
        checkpoint_interval=args.checkpoint_interval,
        video_list=video_list
    )


if __name__ == "__main__":
    main()
