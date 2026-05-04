#!/usr/bin/env python3
"""
Generate H5 file for VideoXum dataset in SumMe format

This script creates an H5 file for VideoXum dataset with the same structure as SumMe.
It reads video files and JSON annotations to generate all required fields.

Usage:
    python 0_generate_videoxum_h5.py
    python 0_generate_videoxum_h5.py --test  # Process only 3 videos for testing
    python 0_generate_videoxum_h5.py --output custom_name.h5
"""

import numpy as np
import h5py
import json
import cv2
import argparse
import torch
import re
import tarfile
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image

# CLIP for feature extraction
clip = None
try:
    import clip
except ImportError:
    pass  # Will check later if CLIP is actually needed

# Import KTS and Knapsack algorithms
sys.path.insert(0, str(Path(__file__).parent))
from Knapsack import solve_knapsack


# ==================== CLIP Feature Extraction ====================

def load_clip_model(device='cuda' if torch.cuda.is_available() else 'cpu', model_name='ViT-B/32'):
    """Load CLIP model for feature extraction"""
    if clip is None:
        print("❌ CLIP not installed")
        print("   Install: pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)
    
    print(f"Loading CLIP model: {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device)
    print(f"✅ CLIP model loaded")
    return model, preprocess, device


def extract_frames_from_tar(tar_path, frame_indices=None):
    """
    Extract frames from tar.gz archive to temporary directory
    
    Args:
        tar_path: Path to tar.gz file
        frame_indices: List of frame indices to extract (None = all)
    
    Returns:
        Tuple of (sorted list of frame paths, temp_dir path)
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="tar_frames_"))
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        members.sort(key=lambda m: m.name)
        
        frame_files = [m for m in members if m.isfile() and 
                      (m.name.endswith('.jpg') or m.name.endswith('.png'))]
        
        if frame_indices is None:
            # Extract all frames
            for member in frame_files:
                tar.extract(member, temp_dir)
        else:
            # Extract specific frames by index
            for idx in frame_indices:
                if 0 <= idx < len(frame_files):
                    member = frame_files[idx]
                    tar.extract(member, temp_dir)
    
    # Get sorted list of extracted frame paths
    frame_paths = sorted(temp_dir.glob("*.jpg")) + sorted(temp_dir.glob("*.png"))
    
    return frame_paths, temp_dir


def extract_frame_features(frames_dir, clip_model, preprocess, device):
    """Extract CLIP features from saved frame images"""
    frames_dir = Path(frames_dir)

    # Get all frame files sorted by frame number
    frame_files = sorted(frames_dir.glob('frame_*.jpg'),
                        key=lambda x: int(re.search(r'frame_(\d+)\.jpg', x.name).group(1)))
    
    if not frame_files:
        # Try without frame_ prefix or with png extension
        frame_files = sorted(list(frames_dir.glob('*.jpg')) + list(frames_dir.glob('*.png')))

    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    features_list = []

    with torch.no_grad():
        for frame_file in tqdm(frame_files, desc="   Extracting features", leave=False):
            # Load and preprocess image
            image = Image.open(frame_file)
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            # Extract features
            image_features = clip_model.encode_image(image_tensor)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            features_list.append(image_features.cpu().numpy()[0])

    features = np.array(features_list, dtype=np.float32)

    print(f"   ✅ Extracted features: shape={features.shape}")

    return features


# ==================== KTS Algorithm (from 0_2_KTS.py) ====================

def compute_kernel_matrix(features, sigma=1.0):
    """Compute Gaussian kernel matrix from features"""
    n_frames = features.shape[0]
    sq_norms = np.sum(features**2, axis=1, keepdims=True)
    distances_sq = sq_norms + sq_norms.T - 2 * np.dot(features, features.T)
    K = np.exp(-distances_sq / (2 * sigma**2))
    return K


def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Change Point Detection using dynamic programming"""
    n = K.shape[0]

    K_cum = np.zeros((n+1, n+1))
    K_cum[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)

    def score_segment(start, end):
        if start >= end:
            return 0
        length = end - start
        segment_sum = (K_cum[end, end] - K_cum[start, end] -
                      K_cum[end, start] + K_cum[start, start])
        score = segment_sum / (length * length) if length > 0 else 0
        return score

    dp = np.zeros((n+1, ncp+1))
    backtrack = np.zeros((n+1, ncp+1), dtype=int)

    for i in range(1, n+1):
        dp[i, 0] = score_segment(0, i)

    for k in range(1, ncp+1):
        for i in range(k+1, n+1):
            best_score = -np.inf
            best_split = k

            for j in range(k, i):
                new_score = dp[j, k-1] + score_segment(j, i)
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


def kts_segmentation(features, num_segments=None, sigma=None):
    """Perform KTS segmentation on video features"""
    n_frames = features.shape[0]

    if sigma is None:
        sample_size = min(1000, n_frames)
        indices = np.random.choice(n_frames, sample_size, replace=False)
        sample_features = features[indices]
        sq_norms = np.sum(sample_features**2, axis=1, keepdims=True)
        distances_sq = sq_norms + sq_norms.T - 2 * np.dot(sample_features, sample_features.T)
        distances = np.sqrt(np.maximum(distances_sq, 0))
        sigma = np.median(distances[distances > 0])

    if num_segments is None:
        num_segments = max(1, n_frames // 10)

    K = compute_kernel_matrix(features, sigma=sigma)

    num_change_points = num_segments - 1

    if num_change_points < 1:
        return np.array([[0, n_frames - 1]])

    change_point_indices, scores = cpd_auto(K, num_change_points, vmax=1.0)

    segments = []
    prev = 0
    for cp in change_point_indices:
        segments.append([prev, cp - 1])
        prev = cp
    segments.append([prev, n_frames - 1])

    return np.array(segments)


# ==================== Main Processing ====================

def get_video_info(video_path):
    """Get video information using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return {
        'fps': fps,
        'n_frames': n_frames,
        'width': width,
        'height': height
    }


def calculate_sampling_params(fps, n_frames, sampled_frames):
    """
    Calculate sampling parameters for 1 FPS sampling

    VideoXum uses strict 1 FPS sampling: picks[i] = int(i * fps)
    This means frame 0 at 0s, frame fps at 1s, frame 2*fps at 2s, etc.

    Args:
        fps: Original video FPS
        n_frames: Total number of frames
        sampled_frames: Number of sampled frames from JSON annotation

    Returns:
        step: Sampling step (round to nearest integer)
        picks: Array of sampled frame indices (strict 1 FPS positions)
        n_steps: Number of sampled frames (from JSON)
    """
    # Round FPS to nearest integer for step
    step = int(round(fps))

    # Use sampled_frames from JSON to ensure consistency with vsum_onehot
    if sampled_frames == 0:
        return step, np.array([]), 0

    # Generate picks using STRICT 1 FPS sampling
    # picks[i] = int(i * fps) for i in range(sampled_frames)
    # This matches how VideoXum generated their annotations
    picks = np.array([int(i * fps) for i in range(sampled_frames)], dtype=int)

    # Ensure picks don't exceed n_frames
    picks = np.clip(picks, 0, n_frames - 1)

    n_steps = sampled_frames

    return step, picks, n_steps


def load_videoxum_annotations(json_paths):
    """
    Load VideoXum annotations from all JSON files

    Args:
        json_paths: List of paths to train/val/test JSON files

    Returns:
        dict: video_id -> annotation mapping
    """
    annotations = {}

    for json_path in json_paths:
        print(f"Loading {json_path.name}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        for item in data:
            video_id = item['video_id']
            annotations[video_id] = item

    return annotations


def generate_videoxum_h5(
    video_dir,
    json_paths,
    output_h5_path,
    test_mode=False,
    target_ratio=10,
    gt_portion=0.15,
    extract_features=True,
    clip_model_name='ViT-B/32',
    frames_dir=None,
    update_mode=False,
    update_fields=None
):
    """
    Generate H5 file for VideoXum dataset or update existing file

    Args:
        video_dir: Directory containing video files
        json_paths: List of paths to annotation JSON files
        output_h5_path: Output H5 file path
        test_mode: If True, only process first 3 videos
        target_ratio: Target frame:segment ratio for KTS
        gt_portion: Portion of frames to select as GT (for knapsack)
        extract_features: Whether to extract CLIP features (default: True)
        clip_model_name: CLIP model name (default: 'ViT-B/32')
        frames_dir: Directory containing tar.gz files with frames (default: auto)
        update_mode: If True, update existing H5 file instead of creating new one
        update_fields: List of fields to update. If None, updates all features-related fields
    
    When update_mode=True, only specified fields are updated:
    - features: CLIP features
    - n_frame_per_seg: Frames per segment from KTS
    - change_points: KTS segment boundaries
    - num_segments: Number of segments
    - segment_scores: Mean gtscore per segment
    - shot_level_gt: GT selection result per segment
    - gtsummary: Frame-level GT summary
    
    This function expects VideoXum frames to be stored as tar.gz archives in:
    {frames_dir}/{video_id}.tar.gz
    
    Each tar.gz contains extracted frames from videos at 1 FPS sampling rate.
    """
    # Set default update fields
    if update_fields is None:
        update_fields = [
            'features',
            'change_points',
            'n_frame_per_seg',
            'segment_scores',
            'shot_level_gt',
            'gtsummary'
        ]

    print(f"\n{'='*80}")
    if update_mode:
        print(f"VideoXum H5 Update - Update Existing Fields")
        print(f"Fields to update: {', '.join(update_fields)}")
    else:
        print(f"VideoXum H5 Generation with CLIP Features")
    print(f"{'='*80}")
    print(f"Video directory: {video_dir}")
    print(f"Output: {output_h5_path}")
    print(f"Update mode: {update_mode}")
    print(f"Extract features: {extract_features}")
    if extract_features:
        print(f"CLIP model: {clip_model_name}")
    print(f"Segmentation ratio: {target_ratio}")
    print(f"Test mode: {test_mode}")
    if test_mode:
        print("⚠️  Only processing 3 videos for testing")
    print(f"{'='*80}\n")

    # Setup frames directory (where tar.gz files are stored)
    if frames_dir is None:
        # Prefer `frames` directory if it exists, otherwise use `frames_tar`
        candidate_frames = Path('/data/MMS_Benchmark/data/videoxum/frames')
        candidate_frames_tar = Path('/data/MMS_Benchmark/data/videoxum/frames_tar')
        if candidate_frames.exists():
            frames_dir = candidate_frames
        else:
            frames_dir = candidate_frames_tar
    else:
        frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        print(f"⚠️  Frames directory does not exist: {frames_dir}")
        print(f"   Expected tar.gz files there. Creating directory... (empty)")
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"✅ Using frames directory: {frames_dir}")

    # Load CLIP model if feature extraction is enabled
    clip_model = None
    preprocess = None
    device = None

    if extract_features:
        clip_model, preprocess, device = load_clip_model(model_name=clip_model_name)
        print()

    # Load annotations (only in generate mode; in update mode we'll read from H5)
    if not update_mode:
        print("Loading annotations from JSON...")
        annotations = load_videoxum_annotations(json_paths)
        print(f"Loaded annotations for {len(annotations)} videos\n")
    else:
        print("Update mode - annotations will be read from H5 file per video\n")
        annotations = {}  # Not used in update mode

    # Get video list (from file system in generate mode, from H5 in update mode)
    if update_mode:
        # In update mode, get existing video list from H5 file
        if not output_h5_path.exists():
            print(f"❌ H5 file not found for updating: {output_h5_path}")
            return
        
        print(f"Reading existing videos from H5 file...")
        with h5py.File(output_h5_path, 'r') as h5file:
            video_keys = sorted([k for k in h5file.keys() if k.startswith('video_')],
                              key=lambda x: int(x.split('_')[1]))
        
        print(f"Found {len(video_keys)} existing videos in H5 file\n")
        # In update mode, no filtering by annotations - update all existing videos
        video_files_with_anno = video_keys
    else:
        # In generate mode, find video files from directory
        video_files = sorted(list(video_dir.glob('v_*.mp4')))
        print(f"Found {len(video_files)} video files\n")

        # Filter to videos that have annotations
        video_files_with_anno = []
        for vf in video_files:
            video_id = vf.stem  # e.g., v_QOlSCBRmfWY
            if video_id in annotations:
                video_files_with_anno.append(vf)

        print(f"{len(video_files_with_anno)} videos have annotations\n")

    # Limit to 3 videos in test mode
    if test_mode:
        video_files_with_anno = video_files_with_anno[:3]
        print(f"Test mode: Processing {len(video_files_with_anno)} videos\n")

    # Statistics
    stats = {
        'total_videos': 0,
        'total_frames': 0,
        'total_segments': 0,
        'errors': []
    }

    # Create or open H5 file
    file_mode = 'a' if update_mode else 'w'  # 'a' = read/write, create if not exists
    
    with h5py.File(output_h5_path, file_mode) as h5_file:

        for item_idx, video_item in enumerate(tqdm(video_files_with_anno, desc="Processing videos"), 1):

            # Handle both generate mode (Path objects) and update mode (video_key strings)
            if update_mode:
                video_key = video_item
                # In update mode, read metadata from existing H5
                video_id_bytes = h5_file[video_key]['video_name'][()]
                video_id = video_id_bytes.decode('utf-8') if isinstance(video_id_bytes, bytes) else video_id_bytes
                n_frames = int(h5_file[video_key]['n_frames'][()])
                n_steps = int(h5_file[video_key]['n_steps'][()])
                picks = h5_file[video_key]['picks'][()]
            else:
                video_path = video_item
                video_id = video_path.stem
                video_key = f"video_{item_idx}"
                
                # In generate mode, read metadata from video file
                video_info = get_video_info(video_path)
                fps = video_info['fps']
                n_frames = video_info['n_frames']

            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"Processing {video_key}: {video_id}")
            # `n_steps` may not be defined yet in generate mode (computed after reading annotations)
            info_msg = f"  Video info: {n_frames} frames"
            if 'n_steps' in locals():
                info_msg += f", {n_steps} steps"
            tqdm.write(info_msg)

            try:
                # Get annotation
                if video_id in annotations:
                    anno = annotations[video_id]
                    vsum_onehot = np.array(anno['vsum_onehot'])  # (n_users, n_frames)
                    sampled_frames = anno['sampled_frames']  # Number of sampled frames from JSON

                    tqdm.write(f"  Annotation: {vsum_onehot.shape[0]} users, {vsum_onehot.shape[1]} annotated frames")
                    tqdm.write(f"  JSON sampled_frames: {sampled_frames}")

                    # In generate mode, calculate sampling parameters; in update mode we'll use existing picks
                    if not update_mode:
                        # Calculate sampling parameters (1 FPS)
                        step, picks, n_steps = calculate_sampling_params(fps, n_frames, sampled_frames)
                        tqdm.write(f"  Sampling: 1 FPS, step={step}, n_steps={n_steps}")
                else:
                    # Annotation JSON missing for this video
                    if update_mode:
                        # Try to read annotation info from existing H5 group
                        group = h5_file[video_key]
                        # Prefer user_summary (expanded to n_frames)
                        if 'user_summary' in group:
                            user_summary_expanded = group['user_summary'][()]
                            # user_summary_expanded: (n_users, n_frames)
                            n_users = user_summary_expanded.shape[0]
                            # picks and n_steps should already be read earlier from H5
                            vsum_onehot = user_summary_expanded[:, picks]
                            sampled_frames = vsum_onehot.shape[1]
                            tqdm.write(f"  Loaded user_summary from H5: {n_users} users, {sampled_frames} sampled frames")
                        elif 'gtscore' in group:
                            # If only gtscore exists, use it directly
                            gtscore_sampled = group['gtscore'][()]
                            vsum_onehot = None
                            sampled_frames = len(gtscore_sampled)
                            tqdm.write(f"  Loaded gtscore from H5: {sampled_frames} sampled frames")
                        else:
                            tqdm.write(f"  ❌ No annotation JSON and no annotation fields in H5 for {video_id}")
                            stats['errors'].append((video_key, f"No annotations available for {video_id}"))
                            continue
                    else:
                        tqdm.write(f"  ❌ Annotation JSON missing for {video_id}")
                        stats['errors'].append((video_key, f"Annotation JSON missing for {video_id}"))
                        continue

                # Calculate user_summary (expand to original n_frames)
                n_users = vsum_onehot.shape[0]
                user_summary = np.zeros((n_users, n_frames), dtype=np.float32)

                # Map vsum_onehot to original frame indices
                for i, frame_idx in enumerate(picks):
                    if i < vsum_onehot.shape[1]:  # Safety check
                        user_summary[:, frame_idx] = vsum_onehot[:, i]

                tqdm.write(f"  user_summary shape: {user_summary.shape}")

                # Calculate gtscore (average across users)
                gtscore = user_summary.mean(axis=0).astype(np.float32)

                # Sample gtscore to match n_steps (for KTS and knapsack)
                gtscore_sampled = gtscore[picks]

                tqdm.write(f"  gtscore range: [{gtscore_sampled.min():.3f}, {gtscore_sampled.max():.3f}]")

                # Extract frames from tar.gz (or directory if it exists) - only if extract_features is True
                temp_dirs_to_cleanup = []
                
                if extract_features:
                    video_frames_dir = frames_dir / video_key
                    tar_path = frames_dir / f"{video_id}.tar.gz"
                    
                    # Check for tar.gz or directory
                    if tar_path.exists():
                        # Extract from tar.gz
                        tqdm.write(f"   📦 Extracting frames from tar.gz: {tar_path.name}")
                        frame_paths, temp_frames_dir = extract_frames_from_tar(tar_path, frame_indices=None)
                        n_extracted = len(frame_paths)
                        tqdm.write(f"   ✅ Extracted {n_extracted} frames from tar")
                        video_frames_for_features = temp_frames_dir
                        temp_dirs_to_cleanup = [temp_frames_dir]
                    elif video_frames_dir.exists() and len(list(video_frames_dir.glob('*.jpg'))) > 0:
                        # Use existing directory
                        tqdm.write(f"   📁 Using existing frames directory: {video_key}/")
                        video_frames_for_features = video_frames_dir
                        temp_dirs_to_cleanup = []
                    else:
                        tqdm.write(f"   ❌ No frames found (neither tar.gz nor directory): {video_id}")
                        tqdm.write(f"   Expected tar.gz: {tar_path}")
                        tqdm.write(f"   Expected directory: {video_frames_dir}")
                        stats['errors'].append((video_key, f"No frames found for {video_id}"))
                        continue

                    # Extract CLIP features
                    tqdm.write(f"   Extracting CLIP features...")
                    features = extract_frame_features(video_frames_for_features, clip_model, preprocess, device)
                else:
                    tqdm.write(f"   Skipping CLIP feature extraction")
                    features = np.random.randn(n_steps, 1024).astype(np.float32)
                    features = features / np.linalg.norm(features, axis=1, keepdims=True)

                tqdm.write(f"  Features shape: {features.shape}")

                # Run KTS segmentation on sampled frames
                num_segments = max(1, n_steps // target_ratio)
                tqdm.write(f"  Running KTS: {n_steps} frames → ~{num_segments} segments")

                change_points = kts_segmentation(features, num_segments=num_segments)
                n_segments = len(change_points)

                tqdm.write(f"  KTS result: {n_segments} segments")

                # Calculate n_frame_per_seg
                n_frame_per_seg = np.array([
                    change_points[i, 1] - change_points[i, 0] + 1
                    for i in range(n_segments)
                ], dtype=np.int64)

                # Calculate segment scores (average gtscore per segment)
                segment_scores = np.array([
                    np.mean(gtscore_sampled[change_points[i, 0]:change_points[i, 1]+1])
                    for i in range(n_segments)
                ], dtype=np.float64)

                # Use Knapsack to select GT clips
                # Capacity is in sampled-frame units (same units as n_frame_per_seg)
                capacity = int(n_steps * gt_portion)
                tqdm.write(f"  Running Knapsack (target: {gt_portion*100:.0f}%, capacity: {capacity} sampled frames)")

                shot_level_gt = solve_knapsack(segment_scores, n_frame_per_seg, capacity)

                # Fallback: if knapsack selects nothing (capacity too small for
                # any segment), force-select the highest-scoring segment
                if shot_level_gt.sum() == 0 and segment_scores.sum() > 0:
                    best_seg = int(np.argmax(segment_scores))
                    shot_level_gt[best_seg] = 1
                    tqdm.write(f"  Fallback: capacity={capacity} too small, "
                               f"forcing segment {best_seg} "
                               f"(score={segment_scores[best_seg]:.3f})")

                gt_clips = int(shot_level_gt.sum())
                tqdm.write(f"  GT clips: {gt_clips}/{n_segments} segments")

                # Generate gtsummary: highest scoring frame in each GT shot
                gtsummary = np.zeros(n_steps, dtype=np.float32)
                for i in range(n_segments):
                    if shot_level_gt[i] > 0:  # This is a GT shot
                        start, end = int(change_points[i, 0]), int(change_points[i, 1])
                        if start <= end and end < n_steps:
                            # Find the frame with highest gtscore in this shot
                            shot_scores = gtscore_sampled[start:end+1]
                            best_idx_in_shot = np.argmax(shot_scores)
                            best_frame_idx = start + best_idx_in_shot
                            gtsummary[best_frame_idx] = 1.0

                gt_frames_selected = int(gtsummary.sum())
                tqdm.write(f"  gtsummary: {gt_frames_selected} frames (1 per GT shot)")

                # Convert change_points from sampled-frame indices to original-frame indices
                change_points_orig = np.array(
                    [[int(picks[s]), int(picks[e])] for s, e in change_points],
                    dtype=np.int64
                )

                # Create or get group for this video
                if update_mode and video_key in h5_file:
                    # In update mode, get existing group and clear specified fields
                    group = h5_file[video_key]
                    for field in update_fields:
                        if field in group:
                            del group[field]
                else:
                    # Create new group
                    group = h5_file.create_group(video_key)

                # Write all fields (matching SumMe format)
                # Basic metadata (only if not updating)
                if not update_mode:
                    group.create_dataset('video_name', data=video_id.encode('utf-8'))
                    group.create_dataset('n_frames', data=n_frames, dtype='int64')
                    # n_steps: number of sampled frames (= len(picks))
                    group.create_dataset('n_steps', data=len(picks), dtype='int64')
                    group.create_dataset('picks', data=picks, dtype='int64')
                    group.create_dataset('user_summary', data=user_summary, dtype='float32')
                    group.create_dataset('gtscore', data=gtscore_sampled, dtype='float32')

                # Fields to be updated (or created if new)
                if 'features' in update_fields or not update_mode:
                    group.create_dataset('features', data=features, dtype='float32')
                if 'change_points' in update_fields or not update_mode:
                    # change_points stored in original-frame space
                    group.create_dataset('change_points', data=change_points_orig, dtype='int64')
                if 'n_frame_per_seg' in update_fields or not update_mode:
                    group.create_dataset('n_frame_per_seg', data=n_frame_per_seg, dtype='int64')
                if 'segment_scores' in update_fields or not update_mode:
                    group.create_dataset('segment_scores', data=segment_scores, dtype='float64')
                if 'shot_level_gt' in update_fields or not update_mode:
                    group.create_dataset('shot_level_gt', data=shot_level_gt, dtype='int32')
                if 'gtsummary' in update_fields or not update_mode:
                    group.create_dataset('gtsummary', data=gtsummary, dtype='float32')

                # Update statistics
                stats['total_videos'] += 1
                stats['total_frames'] += n_frames
                stats['total_segments'] += n_segments

                tqdm.write(f"  ✅ Successfully processed {video_key}")

            except Exception as e:
                tqdm.write(f"  ❌ Error processing {video_key}: {e}")
                import traceback
                traceback.print_exc()
                stats['errors'].append((video_key, str(e)))
            
            finally:
                # Cleanup temporary directories from tar extraction
                if 'temp_dirs_to_cleanup' in locals():
                    for temp_dir in temp_dirs_to_cleanup:
                        if temp_dir.exists():
                            shutil.rmtree(temp_dir)
                continue

    # Print final statistics
    print(f"\n{'='*80}")
    print(f"💾 H5 file saved to: {output_h5_path}")
    print(f"\n📊 Statistics:")
    print(f"   Videos processed: {stats['total_videos']}")
    print(f"   Total frames: {stats['total_frames']:,}")
    print(f"   Total segments: {stats['total_segments']}")
    print(f"   Average frames per video: {stats['total_frames']/stats['total_videos']:.0f}" if stats['total_videos'] > 0 else "   N/A")
    print(f"   Average segments per video: {stats['total_segments']/stats['total_videos']:.1f}" if stats['total_videos'] > 0 else "   N/A")

    if stats['errors']:
        print(f"\n❌ Errors: {len(stats['errors'])} videos failed")
        for video_key, error in stats['errors'][:5]:
            print(f"   - {video_key}: {error}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate H5 file for VideoXum dataset in SumMe format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos
  python 0_generate_videoxum_h5.py

  # Test mode: process only 3 videos
  python 0_generate_videoxum_h5.py --test

  # Custom output filename
  python 0_generate_videoxum_h5.py --output my_videoxum.h5

  # Custom parameters
  python 0_generate_videoxum_h5.py --ratio 15 --gt-portion 0.20

Output:
  Creates H5 file with fields matching SumMe format:
  - video_name: Video ID (e.g., 'v_QOlSCBRmfWY')
  - n_frames: Total number of frames in original video
  - n_steps: Number of sampled frames (1 FPS sampling)
  - features: Placeholder features (1024-dim, random for now)
  - picks: Sampled frame indices
  - change_points: Segment boundaries from KTS
  - n_frame_per_seg: Frames per segment
  - user_summary: Multi-user annotations (expanded to n_frames)
  - gtscore: Frame importance scores (averaged from users)
  - gtsummary: Binary GT summary from Knapsack
  - shot_level_gt: Binary segment-level GT
        """
    )

    parser.add_argument('--test', action='store_true',
                       help='Test mode: only process 3 videos')
    parser.add_argument('--video-dir', type=str,
                       default='/data/MMS_Benchmark/data/videoxum/videos',
                       help='Directory containing video files')
    parser.add_argument('--json-dir', type=str,
                       default='/data/MMS_Benchmark/data/processed',
                       help='Directory containing annotation JSON files')
    parser.add_argument('--output', type=str,
                       default='videoxum.h5',
                       help='Output H5 filename')
    parser.add_argument('--h5-dir', type=str,
                       default='/data/MMS_Benchmark/data/videoxum/h5',
                       help='Output directory for H5 file')
    parser.add_argument('--ratio', type=int, default=10,
                       help='Target frame:segment ratio for KTS (default: 10)')
    parser.add_argument('--gt-portion', type=float, default=0.15,
                       help='Portion of frames to select as GT (default: 0.15)')
    parser.add_argument('--extract-features', action='store_true', default=True,
                       help='Extract CLIP features (default: True)')
    parser.add_argument('--no-extract-features', dest='extract_features', action='store_false',
                       help='Skip feature extraction')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                       help='CLIP model name (default: ViT-B/32)')
    parser.add_argument('--frames-dir', type=str, default=None,
                       help='Directory to save extracted frames (default: auto)')
    
    # Update mode arguments
    parser.add_argument('--update-only', action='store_true', dest='update_mode',
                       help='Update existing H5 file instead of creating new one')
    parser.add_argument('--update-fields', type=str, nargs='+', default=None,
                       help='Fields to update: features, change_points, n_frame_per_seg, '\
                            'segment_scores, shot_level_gt, gtsummary. '\
                            'Default: all features-related fields')

    args = parser.parse_args()

    # Setup paths
    video_dir = Path(args.video_dir)
    json_dir = Path(args.json_dir)
    h5_dir = Path(args.h5_dir)

    # Workspace root fallback (assume /data/MMS_Benchmark)
    workspace_root = Path('/data/MMS_Benchmark')

    # In update mode, video directory is not needed
    if not args.update_mode:
        if not video_dir.exists():
            print(f"❌ Video directory not found: {video_dir}")
            return

    # If JSON dir missing, try fallback to workspace processed folder
    if not json_dir.exists():
        fallback_json = workspace_root / 'data' / 'processed'
        if fallback_json.exists():
            # In update mode we don't need to announce fallback; in generate mode inform the user
            if not args.update_mode:
                print(f"⚠️ JSON dir {json_dir} not found, using fallback: {fallback_json}")
            json_dir = fallback_json
        else:
            # If JSON directory missing and not update mode, abort
            if not args.update_mode:
                print(f"❌ JSON directory not found: {json_dir}")
                return

    # If h5_dir is missing, try fallback to workspace videoxum/h5
    if not h5_dir.exists():
        fallback_h5 = workspace_root / 'data' / 'videoxum' / 'h5'
        if fallback_h5.exists():
            # In update mode keep silent about fallback selection
            if not args.update_mode:
                print(f"⚠️ H5 dir {h5_dir} not found, using fallback: {fallback_h5}")
            h5_dir = fallback_h5
        else:
            # Create h5_dir if in generate mode; in update mode require existing H5
            if not args.update_mode:
                h5_dir.mkdir(parents=True, exist_ok=True)
            else:
                # In update mode we will proceed; later code will check for the actual H5 file existence
                pass

    # Find JSON files (only required for generate mode, not for update mode)
    if not args.update_mode:
        # In generate mode, look for videoxum_data.json
        json_path = json_dir / 'videoxum_data.json'
        
        if json_path.exists():
            print(f"✅ Using JSON: {json_path}")
            json_paths = [json_path]
        else:
            print(f"❌ JSON file not found: {json_path}")
            return
    else:
        # In update mode, JSON is NOT needed - all annotations are already in H5
        print(f"⚠️  Update mode - annotations will be read from existing H5 file")
        json_paths = []

    # Create output directory
    h5_dir.mkdir(parents=True, exist_ok=True)
    output_h5_path = h5_dir / args.output

    # Generate H5 file
    generate_videoxum_h5(
        video_dir=video_dir,
        json_paths=json_paths,
        output_h5_path=output_h5_path,
        test_mode=args.test,
        target_ratio=args.ratio,
        gt_portion=args.gt_portion,
        extract_features=args.extract_features,
        clip_model_name=args.clip_model,
        frames_dir=args.frames_dir,
        update_mode=args.update_mode,
        update_fields=args.update_fields
    )


if __name__ == "__main__":
    main()
