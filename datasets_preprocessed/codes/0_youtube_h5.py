#!/usr/bin/env python3
"""
Generate YouTube H5 File from Scratch

This script creates the YouTube H5 file by:
1. Sampling video frames at 2 FPS
2. Reading UserSummary annotations (5 users)
3. Computing gtscore for each sampled frame
4. Generating gtsummary (top 1% frames, matching SumMe/TVSum)
5. Extracting CLIP features
6. Running KTS algorithm and Knapsack to generate complete H5

Input:
    - datasets_preprocessed/data/dataset/YouTube/video/*.avi
    - datasets_preprocessed/data/dataset/YouTube/newUserSummary/*

Output:
    - datasets_preprocessed/data/h5/youtube.h5
    - datasets_preprocessed/data/dataset/YouTube/frames/video_*/*.jpg

Usage:
    python 0_youtube_h5.py
    python 0_youtube_h5.py --no-extract-features  # Skip CLIP
    python 0_youtube_h5.py --clip-model ViT-L/14  # Use larger model
"""

import cv2
import h5py
import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import subprocess
import re
import torch
from PIL import Image

# CLIP for feature extraction
try:
    import clip
except ImportError:
    print("❌ CLIP not installed")
    print("   Install: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from Knapsack import solve_knapsack

# Import KTS algorithm from 0_2_KTS.py
try:
    from importlib import import_module
    kts_module = import_module('0_2_KTS')
    kts_segmentation_real = kts_module.kts_segmentation
except Exception as e:
    print(f"Warning: Could not import KTS module: {e}")
    kts_segmentation_real = None


def load_clip_model(device='cuda' if torch.cuda.is_available() else 'cpu', model_name='ViT-B/32'):
    """Load CLIP model for feature extraction"""
    print(f"Loading CLIP model: {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device)
    print(f"✅ CLIP model loaded")
    return model, preprocess, device


def extract_frame_features(frames_dir, clip_model, preprocess, device):
    """Extract CLIP features from saved frame images"""
    frames_dir = Path(frames_dir)

    # Get all frame files sorted by frame number
    frame_files = sorted(frames_dir.glob('frame_*.jpg'),
                        key=lambda x: int(re.search(r'frame_(\d+)\.jpg', x.name).group(1)))

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


def get_video_fps(video_path):
    """Get video FPS using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        fps_str = result.stdout.strip()

        # Parse fraction (e.g., "60000/1001")
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return fps
    except Exception as e:
        print(f"Warning: Could not get FPS for {video_path}, using 30.0: {e}")
        return 30.0


def sample_video_frames(video_path, output_dir, target_fps=2.0):
    """
    Sample video frames at target FPS

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        target_fps: Target sampling rate (default: 2 FPS)

    Returns:
        tuple: (picks, total_frames) where:
            - picks: Array of sampled frame indices
            - total_frames: Original video total frame count
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video FPS
    video_fps = get_video_fps(video_path)

    # Calculate sampling interval
    sample_interval = max(1, int(round(video_fps / target_fps)))

    print(f"   Video FPS: {video_fps:.2f}, sampling every {sample_interval} frames")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices to sample
    picks = []
    frame_idx = 0

    while frame_idx < total_frames:
        picks.append(frame_idx)
        frame_idx += sample_interval

    picks = np.array(picks)

    print(f"   Sampling {len(picks)} frames from {total_frames} total frames")

    # Extract frames
    extracted = 0
    for i, frame_idx in enumerate(tqdm(picks, desc="   Extracting", leave=False)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"   Warning: Could not read frame {frame_idx}")
            continue

        # Save frame with original frame number in filename
        frame_filename = output_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        extracted += 1

    cap.release()

    print(f"   ✅ Extracted {extracted}/{len(picks)} frames")

    return picks, total_frames


def read_user_summary(user_summary_dir, video_id, picks):
    """
    Read UserSummary annotations for a video

    Args:
        user_summary_dir: Path to UserSummary directory
        video_id: Video ID (e.g., "v100")
        picks: Array of sampled frame indices

    Returns:
        gtscore: (n_frames,) array with score for each sampled frame
        user_summary: (n_users, n_picks) binary array
    """
    user_summary_dir = Path(user_summary_dir)
    video_dir = user_summary_dir / video_id

    if not video_dir.exists():
        raise ValueError(f"UserSummary not found: {video_dir}")

    # Find user directories
    user_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('user')])
    n_users = len(user_dirs)
    n_picks = len(picks)

    print(f"   Found {n_users} users")

    # Initialize user_summary matrix
    user_summary = np.zeros((n_users, n_picks), dtype=np.float32)

    # Read each user's selected frames
    for user_idx, user_dir in enumerate(user_dirs):
        # Get selected frame numbers (YouTube format: frame1.jpg, frame25.jpg)
        frame_files = list(user_dir.glob('frame*.jpg'))

        selected_frames = []
        for frame_file in frame_files:
            # Extract frame number from filename (e.g., "frame25.jpg" -> 25)
            # Note: YouTube frames are 1-indexed, so frame1.jpg = frame 0 in video
            match = re.search(r'frame(\d+)\.jpg', frame_file.name)
            if match:
                frame_num = int(match.group(1)) - 1  # Convert to 0-indexed
                selected_frames.append(frame_num)

        # Mark selected frames in user_summary
        for pick_idx, pick_frame in enumerate(picks):
            # Check if this picked frame is close to any selected frame
            tolerance = 15  # Approximately half of typical sample interval

            for selected_frame in selected_frames:
                if abs(pick_frame - selected_frame) <= tolerance:
                    user_summary[user_idx, pick_idx] = 1.0
                    break

    # Calculate gtscore: proportion of users who selected each frame
    gtscore = np.mean(user_summary, axis=0).astype(np.float32)

    return gtscore, user_summary


def compute_gtsummary_from_shots(gtscore, segments, shot_level_gt):
    """
    Compute gtsummary: select the highest scoring frame from each GT shot.

    Args:
        gtscore: (n_sampled_frames,) importance scores
        segments: (n_segments, 2) segment boundaries [start, end]
        shot_level_gt: (n_segments,) binary array indicating GT shots

    Returns:
        gtsummary: (n_sampled_frames,) binary array with 1 at highest scoring frame of each GT shot
    """
    n_sampled_frames = len(gtscore)
    gtsummary = np.zeros(n_sampled_frames, dtype=np.float32)

    for i, (start, end) in enumerate(segments):
        if shot_level_gt[i] > 0:  # This is a GT shot
            start = int(max(0, start))
            end = int(min(end, n_sampled_frames - 1))
            if start <= end:
                # Find the frame with highest gtscore in this shot
                shot_scores = gtscore[start:end+1]
                best_idx_in_shot = np.argmax(shot_scores)
                best_frame_idx = start + best_idx_in_shot
                gtsummary[best_frame_idx] = 1.0

    return gtsummary


def run_kts_on_features(features, num_segments=None, sigma=None):
    """
    Run real KTS algorithm on features

    Args:
        features: (n_frames, feature_dim) array
        num_segments: Target number of segments (auto if None)
        sigma: Gaussian kernel bandwidth (auto if None)

    Returns:
        change_points: (n_segments, 2) array of [start, end] indices
    """
    if kts_segmentation_real is None:
        # Fallback to uniform segmentation
        n_frames = len(features)
        if num_segments is None:
            num_segments = max(1, n_frames // 10)

        segment_size = n_frames // num_segments
        segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size - 1 if i < num_segments - 1 else n_frames - 1
            segments.append([start, end])

        return np.array(segments, dtype=np.int64)

    # Use real KTS algorithm
    try:
        segments = kts_segmentation_real(features, num_segments=num_segments, sigma=sigma)
        return segments
    except Exception as e:
        print(f"   Warning: KTS failed ({e}), using uniform segmentation")
        # Fallback to uniform
        n_frames = len(features)
        if num_segments is None:
            num_segments = max(1, n_frames // 10)

        segment_size = n_frames // num_segments
        segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size - 1 if i < num_segments - 1 else n_frames - 1
            segments.append([start, end])

        return np.array(segments, dtype=np.int64)


def knapsack_shot_selection(gtscore, segments, gt_portion=0.15):
    """
    Select GT clips using Knapsack algorithm

    Args:
        gtscore: (n_sampled_frames,) importance scores
        segments: (n_segments, 2) segment boundaries in sampled-frame space
        gt_portion: Portion of sampled frames to select as GT

    Returns:
        shot_level_gt: (n_segments,) binary array
    """
    n_sampled_frames = len(gtscore)

    # Compute segment scores (average gtscore in segment)
    segment_scores = []
    segment_lengths = []

    for start, end in segments:
        start = int(max(0, start))
        end = int(min(end, n_sampled_frames - 1))
        if start <= end:
            segment_score = np.mean(gtscore[start:end+1])
            segment_length = end - start + 1
        else:
            segment_score = 0.0
            segment_length = 1
        segment_scores.append(segment_score)
        segment_lengths.append(segment_length)

    segment_scores = np.array(segment_scores)
    segment_lengths = np.array(segment_lengths)

    # Knapsack capacity: based on number of sampled frames (consistent units
    # with segment_lengths which are also in sampled-frame space)
    capacity = int(n_sampled_frames * gt_portion)

    # Solve knapsack
    shot_level_gt = solve_knapsack(segment_scores, segment_lengths, capacity)

    return shot_level_gt.astype(np.int32)


def process_youtube_dataset(video_dir, user_summary_dir, frames_dir, output_h5_path, target_fps=2.0,
                            extract_features=True, clip_model_name='ViT-B/32'):
    """
    Process all YouTube videos and create H5 file

    Args:
        video_dir: Directory containing video files
        user_summary_dir: Directory containing UserSummary annotations
        frames_dir: Directory to save extracted frames
        output_h5_path: Path to output H5 file
        target_fps: Target sampling rate
        extract_features: Whether to extract CLIP features (default: True)
        clip_model_name: CLIP model name (default: 'ViT-B/32')
    """
    video_dir = Path(video_dir)
    user_summary_dir = Path(user_summary_dir)
    frames_dir = Path(frames_dir)
    output_h5_path = Path(output_h5_path)

    print("="*80)
    print("YouTube H5 File Generation")
    print("="*80)
    print(f"Video dir: {video_dir}")
    print(f"UserSummary dir: {user_summary_dir}")
    print(f"Frames dir: {frames_dir}")
    print(f"Output H5: {output_h5_path}")
    print(f"Target FPS: {target_fps}")
    print(f"Extract features: {extract_features}")
    if extract_features:
        print(f"CLIP model: {clip_model_name}")
    print("="*80)
    print()

    # Load CLIP model if feature extraction is enabled
    clip_model = None
    preprocess = None
    device = None

    if extract_features:
        clip_model, preprocess, device = load_clip_model(model_name=clip_model_name)
        print()

    # Find all video files
    video_files = sorted(video_dir.glob('v*.avi'))

    if not video_files:
        video_files = sorted(video_dir.glob('v*.mp4'))

    if not video_files:
        print("❌ No video files found")
        return

    print(f"Found {len(video_files)} videos\n")

    stats = {
        'total': len(video_files),
        'success': 0,
        'failed': 0
    }

    # Create H5 file
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5_path, 'w') as h5_file:
        for video_file in tqdm(video_files, desc="Processing videos"):
            # Get video ID (e.g., v100 -> video_100)
            video_num = video_file.stem.replace('v', '')
            video_key = f"video_{video_num}"

            tqdm.write(f"\n🎬 Processing {video_key} ({video_file.name})")

            try:
                # 1. Sample frames
                video_output_dir = frames_dir / video_key

                # Check if already extracted
                if video_output_dir.exists() and len(list(video_output_dir.glob('*.jpg'))) > 0:
                    tqdm.write(f"   ⏭️  Frames already extracted, loading picks...")

                    # Reconstruct picks from existing frames
                    frame_files = sorted(video_output_dir.glob('frame_*.jpg'))
                    picks = []
                    for frame_file in frame_files:
                        # Extract frame number from filename
                        match = re.search(r'frame_(\d+)\.jpg', frame_file.name)
                        if match:
                            picks.append(int(match.group(1)))
                    picks = np.array(picks, dtype=np.int64)

                    # Get original frame count from video
                    cap = cv2.VideoCapture(str(video_file))
                    n_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    tqdm.write(f"   Loaded {len(picks)} picks (original: {n_original_frames} frames)")
                else:
                    picks, n_original_frames = sample_video_frames(video_file, video_output_dir, target_fps)

                # 2. Read user summary and compute gtscore
                video_id = video_file.stem  # e.g., "v100"
                gtscore, user_summary = read_user_summary(user_summary_dir, video_id, picks)

                tqdm.write(f"   gtscore: min={gtscore.min():.3f}, max={gtscore.max():.3f}, mean={gtscore.mean():.3f}")

                # 3. Extract CLIP features (moved before segmentation)
                if extract_features:
                    features = extract_frame_features(video_output_dir, clip_model, preprocess, device)
                else:
                    tqdm.write(f"   ⏭️  Skipping feature extraction")
                    features = None

                # 5. Run KTS segmentation
                if features is not None:
                    # Use real KTS algorithm
                    num_segments = max(1, len(picks) // 10)
                    tqdm.write(f"   Running KTS segmentation (target: {num_segments} segments)...")
                    change_points = run_kts_on_features(features, num_segments=num_segments, sigma=None)
                    tqdm.write(f"   ✅ KTS generated {len(change_points)} segments")
                else:
                    # Fallback to uniform segmentation
                    num_segments = max(1, len(picks) // 10)
                    segment_size = len(picks) // num_segments
                    change_points = []
                    for i in range(num_segments):
                        start = i * segment_size
                        end = start + segment_size - 1 if i < num_segments - 1 else len(picks) - 1
                        change_points.append([start, end])
                    change_points = np.array(change_points, dtype=np.int64)
                    tqdm.write(f"   Using uniform segmentation ({len(change_points)} segments)")

                # 6. Knapsack shot selection (15% of sampled frames)
                # change_points are in sampled-frame space here (used for indexing)
                shot_level_gt = knapsack_shot_selection(gtscore, change_points, gt_portion=0.15)
                n_selected_shots = int(shot_level_gt.sum())
                tqdm.write(f"   shot_level_gt: {n_selected_shots}/{len(change_points)} segments selected")

                # 7. Compute gtsummary: highest scoring frame in each GT shot
                gtsummary = compute_gtsummary_from_shots(gtscore, change_points, shot_level_gt)
                n_gt_frames = int(gtsummary.sum())
                tqdm.write(f"   gtsummary: {n_gt_frames} frames (1 per GT shot)")

                # Convert change_points from sampled-frame indices to original-frame indices
                change_points_orig = np.array(
                    [[int(picks[s]), int(picks[e])] for s, e in change_points],
                    dtype=np.int64
                )

                # 8. Create H5 group
                group = h5_file.create_group(video_key)

                # Save data
                group.create_dataset('picks', data=picks, dtype='int64')
                group.create_dataset('gtscore', data=gtscore, dtype='float32')
                group.create_dataset('gtsummary', data=gtsummary, dtype='float32')
                group.create_dataset('user_summary', data=user_summary, dtype='float32')

                # Save features if extracted
                if features is not None:
                    group.create_dataset('features', data=features, dtype='float32')

                # change_points stored in original-frame space
                group.create_dataset('change_points', data=change_points_orig, dtype='int64')

                # n_frame_per_seg: sampled-frame count per segment
                seg_lengths = change_points[:, 1] - change_points[:, 0] + 1
                group.create_dataset('n_frame_per_seg', data=seg_lengths, dtype='int64')
                group.create_dataset('n_frames', data=n_original_frames, dtype='int64')  # ORIGINAL frame count
                # n_steps: number of sampled frames (= len(picks))
                group.create_dataset('n_steps', data=len(picks), dtype='int64')
                group.create_dataset('shot_level_gt', data=shot_level_gt, dtype='int32')

                tqdm.write(f"   ✅ Successfully processed")
                stats['success'] += 1

            except Exception as e:
                tqdm.write(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                stats['failed'] += 1

    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total videos: {stats['total']}")
    print(f"✅ Success: {stats['success']}")
    print(f"❌ Failed: {stats['failed']}")
    print("="*80)
    print(f"\n💾 H5 file saved: {output_h5_path}")
    print(f"📁 Frames saved: {frames_dir}")

    if stats['success'] > 0:
        if extract_features:
            print("\n✅ Complete H5 file generated!")
            print("   - CLIP features extracted")
            print("   - KTS segmentation completed")
            print("   - Knapsack GT selection done")
            print(f"\n📦 Ready to use: {output_h5_path}")
        else:
            print("\n⚠️  Note: Features not extracted.")
            print("   H5 file created with uniform segmentation only.")
            print("   Re-run with --extract-features for KTS segmentation.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate YouTube H5 file from videos and UserSummary annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Get script directory for computing absolute paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    default_video_dir = project_root / "datasets_preprocessed/data/dataset/YouTube/video"
    default_user_summary_dir = project_root / "datasets_preprocessed/data/dataset/YouTube/newUserSummary"
    default_frames_dir = project_root / "datasets_preprocessed/data/dataset/YouTube/frames"
    default_output_h5 = project_root / "datasets_preprocessed/data/h5/youtube.h5"

    parser.add_argument('--video-dir', type=str, default=str(default_video_dir),
                       help=f'Directory containing YouTube videos (default: {default_video_dir})')
    parser.add_argument('--user-summary-dir', type=str, default=str(default_user_summary_dir),
                       help=f'Directory containing UserSummary annotations (default: {default_user_summary_dir})')
    parser.add_argument('--frames-dir', type=str, default=str(default_frames_dir),
                       help=f'Directory to save extracted frames (default: {default_frames_dir})')
    parser.add_argument('--output', type=str, default=str(default_output_h5),
                       help=f'Output H5 file path (default: {default_output_h5})')
    parser.add_argument('--fps', type=float, default=2.0,
                       help='Target sampling rate in FPS (default: 2.0)')
    parser.add_argument('--extract-features', action='store_true', default=True,
                       help='Extract CLIP features (default: True)')
    parser.add_argument('--no-extract-features', dest='extract_features', action='store_false',
                       help='Skip feature extraction')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                       choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                       help='CLIP model to use (default: ViT-B/32)')

    args = parser.parse_args()

    # Check paths
    video_dir = Path(args.video_dir)
    user_summary_dir = Path(args.user_summary_dir)

    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return

    if not user_summary_dir.exists():
        print(f"❌ UserSummary directory not found: {user_summary_dir}")
        return

    # Process dataset
    process_youtube_dataset(
        video_dir=args.video_dir,
        user_summary_dir=args.user_summary_dir,
        frames_dir=args.frames_dir,
        output_h5_path=args.output,
        target_fps=args.fps,
        extract_features=args.extract_features,
        clip_model_name=args.clip_model
    )


if __name__ == "__main__":
    main()
