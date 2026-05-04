#!/usr/bin/env python3
"""
KTS (Kernel Temporal Segmentation) Algorithm
Automatically segments videos into temporal shots using change point detection.

This script only performs KTS segmentation. For GT clip selection, use 0_3_knapsack.py.

Reference:
- Potapov et al. "Category-specific video summarization" ECCV 2014
- Uses kernel temporal segmentation to find change points in video features

Usage:
    python 0_2_KTS.py --dataset youtube
    python 0_2_KTS.py --dataset youtube --ratio 5
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm


# ==================== KTS Algorithm Implementation ====================

def compute_kernel_matrix(features, sigma=1.0):
    """
    Compute Gaussian kernel matrix from features

    Args:
        features: (n_frames, feature_dim) array
        sigma: Gaussian kernel bandwidth

    Returns:
        K: (n_frames, n_frames) kernel matrix
    """
    n_frames = features.shape[0]

    # Compute pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    sq_norms = np.sum(features**2, axis=1, keepdims=True)
    distances_sq = sq_norms + sq_norms.T - 2 * np.dot(features, features.T)

    # Gaussian kernel: K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2*sigma^2))
    K = np.exp(-distances_sq / (2 * sigma**2))

    return K


def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """
    Change Point Detection using dynamic programming

    Args:
        K: (n_frames, n_frames) kernel matrix
        ncp: number of change points to detect
        vmax: maximum score
        desc_rate: descent rate for score decay

    Returns:
        change_points: list of change point indices
        scores: segmentation scores
    """
    n = K.shape[0]

    # Compute cumulative kernel sums for efficient calculation
    K_cum = np.zeros((n+1, n+1))
    K_cum[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)

    def score_segment(start, end):
        """Calculate kernel change score for segment [start, end)"""
        if start >= end:
            return 0

        length = end - start

        # Sum of kernel values in segment
        segment_sum = (K_cum[end, end] - K_cum[start, end] -
                      K_cum[end, start] + K_cum[start, start])

        # Normalized score (divide by length, not length^2)
        score = segment_sum / length if length > 0 else 0
        return score

    # Dynamic programming for optimal segmentation
    # dp[i][k] = best score for segmenting [0, i) into k+1 segments
    dp = np.zeros((n+1, ncp+1))
    backtrack = np.zeros((n+1, ncp+1), dtype=int)

    # Base case: single segment
    for i in range(1, n+1):
        dp[i, 0] = score_segment(0, i)

    # Fill DP table
    for k in range(1, ncp+1):
        for i in range(k+1, n+1):
            best_score = -np.inf
            best_split = k

            for j in range(k, i):
                # Score = previous segments + new segment [j, i)
                new_score = dp[j, k-1] + score_segment(j, i)

                if new_score > best_score:
                    best_score = new_score
                    best_split = j

            dp[i, k] = best_score
            backtrack[i, k] = best_split

    # Backtrack to find change points
    change_points = []
    curr_pos = n
    for k in range(ncp, 0, -1):
        split_pos = backtrack[curr_pos, k]
        change_points.append(split_pos)
        curr_pos = split_pos

    change_points.reverse()

    # Calculate final scores
    scores = dp[:, ncp]

    return change_points, scores


def kts_segmentation(features, num_segments=None, sigma=None):
    """
    Perform KTS segmentation on video features

    Args:
        features: (n_frames, feature_dim) feature array
        num_segments: target number of segments (if None, auto-detect)
        sigma: Gaussian kernel bandwidth (if None, auto-set)

    Returns:
        change_points: (num_segments, 2) array of [start, end] frame indices
    """
    n_frames = features.shape[0]

    # Auto-set sigma if not provided (median heuristic)
    if sigma is None:
        # Sample for efficiency
        sample_size = min(1000, n_frames)
        indices = np.random.choice(n_frames, sample_size, replace=False)
        sample_features = features[indices]

        # Compute pairwise distances
        sq_norms = np.sum(sample_features**2, axis=1, keepdims=True)
        distances_sq = sq_norms + sq_norms.T - 2 * np.dot(sample_features, sample_features.T)
        distances = np.sqrt(np.maximum(distances_sq, 0))

        # Use median distance as sigma
        sigma = np.median(distances[distances > 0])

    # Auto-set number of segments if not provided (target ~10:1 ratio)
    if num_segments is None:
        num_segments = max(1, n_frames // 10)

    # Compute kernel matrix
    print(f"   Computing kernel matrix (sigma={sigma:.3f})...")
    K = compute_kernel_matrix(features, sigma=sigma)

    # Find change points
    print(f"   Finding {num_segments} change points...")
    num_change_points = num_segments - 1  # n segments need n-1 change points

    if num_change_points < 1:
        # Single segment
        return np.array([[0, n_frames - 1]])

    change_point_indices, scores = cpd_auto(K, num_change_points, vmax=1.0)

    # Convert change points to segments
    segments = []
    prev = 0
    for cp in change_point_indices:
        segments.append([prev, cp - 1])
        prev = cp
    segments.append([prev, n_frames - 1])

    return np.array(segments)


# ==================== Main Processing ====================

def process_h5_with_kts(input_h5_path, output_h5_path, target_ratio=10, sigma=None, force_recompute=False):
    """
    Process H5 file: run KTS on all videos and save segmentation results to new H5 file

    Args:
        input_h5_path: Path to input H5 file with video features
        output_h5_path: Path to save output H5 file with KTS segmentation
        target_ratio: Target frame:clip ratio (default 10:1)
        sigma: Gaussian kernel bandwidth (None for auto)
        force_recompute: Force recompute even if change_points already exist (default False)
    """
    print(f"\n{'='*80}")
    print(f"KTS Segmentation")
    print(f"{'='*80}")
    print(f"Input:  {input_h5_path}")
    print(f"Output: {output_h5_path}")
    print(f"Target frame:clip ratio: {target_ratio}:1")
    if force_recompute:
        print(f"Force recompute: True (will overwrite existing change_points)")
    print(f"{'='*80}\n")

    stats = {
        'total_videos': 0,
        'total_frames': 0,
        'total_segments': 0
    }

    # Read input H5 file and write to output H5 file
    with h5py.File(input_h5_path, 'r') as f_in, h5py.File(output_h5_path, 'w') as f_out:
        video_keys = sorted([k for k in f_in.keys()])
        print(f"Found {len(video_keys)} videos\n")

        for video_key in tqdm(video_keys, desc="Processing videos"):
            # Create group in output file
            group_out = f_out.create_group(video_key)

            # Check if already has segmentation (before copying)
            has_change_points = 'change_points' in f_in[video_key]

            if has_change_points and not force_recompute:
                # Copy all fields and skip
                for field_name in f_in[video_key].keys():
                    f_in.copy(f_in[video_key][field_name], group_out, name=field_name)
                tqdm.write(f"⏭️  {video_key}: Already has change_points, skipping (use --force to recompute)")
                continue

            # Copy all existing fields EXCEPT segmentation-related ones if force_recompute
            segmentation_fields = {'change_points', 'n_frame_per_seg', 'n_frames', 'n_steps'}
            for field_name in f_in[video_key].keys():
                if force_recompute and field_name in segmentation_fields:
                    continue  # Skip old segmentation fields
                f_in.copy(f_in[video_key][field_name], group_out, name=field_name)

            if has_change_points and force_recompute:
                tqdm.write(f"🔄 {video_key}: Recomputing change_points (--force)")

            # Read features for KTS
            if 'features' not in f_in[video_key]:
                tqdm.write(f"⚠️  {video_key}: No features found, skipping KTS")
                continue

            features = f_in[video_key]['features'][:]
            n_frames = features.shape[0]

            # Calculate target number of segments
            num_segments = max(1, n_frames // target_ratio)

            tqdm.write(f"\n📹 {video_key}: {n_frames} frames → ~{num_segments} segments")

            # Run KTS segmentation
            try:
                segments = kts_segmentation(features, num_segments=num_segments, sigma=sigma)

                # Verify segments
                actual_segments = len(segments)
                actual_ratio = n_frames / actual_segments if actual_segments > 0 else 0

                tqdm.write(f"   ✅ Generated {actual_segments} segments (ratio: {actual_ratio:.1f}:1)")

                # Add KTS results to output H5 file (matching SumMe/TVSum format)
                # change_points: (n_segments, 2) array of [start, end] indices
                group_out.create_dataset('change_points', data=segments, dtype='int64')

                # n_frame_per_seg: (n_segments,) array of segment lengths
                seg_lengths = segments[:, 1] - segments[:, 0] + 1
                group_out.create_dataset('n_frame_per_seg', data=seg_lengths, dtype='int64')

                # n_frames: scalar - total number of frames
                group_out.create_dataset('n_frames', data=n_frames, dtype='int64')

                # n_steps: scalar - number of segments
                group_out.create_dataset('n_steps', data=actual_segments, dtype='int64')

                # Update statistics
                stats['total_videos'] += 1
                stats['total_frames'] += n_frames
                stats['total_segments'] += actual_segments

            except Exception as e:
                tqdm.write(f"   ❌ Error processing {video_key}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Print statistics
    avg_ratio = stats['total_frames'] / stats['total_segments'] if stats['total_segments'] > 0 else 0

    print(f"\n{'='*80}")
    print(f"💾 Results saved to: {output_h5_path}")
    print(f"\n📊 Statistics:")
    print(f"   Videos processed: {stats['total_videos']}")
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Total segments: {stats['total_segments']}")
    print(f"   Average frame:clip ratio: {avg_ratio:.1f}:1")
    print(f"{'='*80}\n")


def main():
    """主入口函数 - 所有路径从配置文件读取"""
    parser = argparse.ArgumentParser(
        description='KTS (Kernel Temporal Segmentation) for video shot detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process YouTube dataset with default 10:1 ratio
  python 0_2_KTS.py --dataset youtube

  # Process OVP dataset
  python 0_2_KTS.py --dataset ovp

  # Force recompute even if change_points already exist
  python 0_2_KTS.py --file path/to/ovp.h5 --force

  # Process with custom frame:clip ratio (5:1)
  python 0_2_KTS.py --dataset youtube --ratio 5

  # Use custom H5 file path (for pipeline)
  python 0_2_KTS.py --file path/to/custom.h5

注意：使用 --dataset 时，路径从配置文件读取
        """
    )

    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str,
                       help='Dataset to process (e.g., youtube, ovp, summe)')
    group.add_argument('--file', type=str,
                       help='Custom H5 file path (for pipeline)')

    parser.add_argument('--output', type=str, default=None,
                       help='Output H5 filename (default: input name + _KTS suffix)')
    parser.add_argument('--ratio', type=int, default=10,
                       help='Target frame:clip ratio (default: 10)')
    parser.add_argument('--sigma', type=float, default=None,
                       help='Gaussian kernel bandwidth (default: auto-detect using median heuristic)')
    parser.add_argument('--force', action='store_true',
                       help='Force recompute even if change_points already exist')

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # Determine input file path
    if args.file:
        # Custom file path (for pipeline calls)
        input_h5_path = Path(args.file)
    else:
        # 从配置读取路径（纯配置，无回退）
        dataset = args.dataset.lower()
        import sys
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from utils.config_loader import get_config
        config = get_config()
        ds_paths = config.get_dataset_paths(dataset)

        h5_file = ds_paths.get('h5_file')
        if not h5_file:
            raise ValueError(f"配置文件中未定义 {dataset}.h5_file")
        input_h5_path = h5_file

    if not input_h5_path.exists():
        print(f"❌ Dataset not found: {input_h5_path}")
        return

    # Output directory: same as input h5 file's directory
    output_dir = input_h5_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        # Default: add _KTS suffix
        output_filename = input_h5_path.stem + '_KTS.h5'
        output_h5_path = output_dir / output_filename
    else:
        # Custom output filename (still in output_dir)
        output_h5_path = output_dir / args.output

    # Run KTS segmentation
    process_h5_with_kts(input_h5_path, output_h5_path,
                       target_ratio=args.ratio,
                       sigma=args.sigma,
                       force_recompute=args.force)


if __name__ == "__main__":
    main()
