#!/usr/bin/env python3
"""
Knapsack-based GT Clip Selection

This script selects ground truth clips from segmented videos using the 0/1 knapsack algorithm.
It reads H5 files with shot segmentation and generates shot-level GT annotations.

Input:
    - H5 file with: features, gtscore, change_points, n_frame_per_seg
    - Can be either _KTS.h5 (from KTS) or original with pre-defined segments

Output:
    - H5 file (no suffix) with additional field: shot_level_gt
    - Ensures all datasets have unified format with GT clip information

Usage:
    python 0_3_knapsack.py --dataset youtube
    python 0_3_knapsack.py --dataset SumMe
    python 0_3_knapsack.py --file path/to/custom.h5
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Import Knapsack algorithm
sys.path.insert(0, str(Path(__file__).parent))
from Knapsack import solve_knapsack


def process_h5_with_knapsack(input_h5_path, output_h5_path, gt_portion=0.15):
    """
    Process H5 file: select GT clips using Knapsack algorithm

    Args:
        input_h5_path: Path to input H5 file with segmentation (change_points, n_frame_per_seg)
        output_h5_path: Path to save output H5 file with shot_level_gt
        gt_portion: Portion of frames to select as GT (default 0.15 = 15%)
    """
    import tempfile
    import shutil

    print(f"\n{'='*80}")
    print(f"Knapsack GT Clip Selection")
    print(f"{'='*80}")
    print(f"Input:  {input_h5_path}")
    print(f"Output: {output_h5_path}")
    print(f"GT selection portion: {gt_portion*100:.0f}%")
    print(f"{'='*80}\n")

    stats = {
        'total_videos': 0,
        'total_frames': 0,
        'total_segments': 0,
        'total_gt_frames': 0,
        'total_gt_clips': 0
    }

    # Handle case where input and output are the same file
    same_file = (input_h5_path.resolve() == output_h5_path.resolve())

    if same_file:
        # Read from input, write to temp, then replace
        temp_output = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        temp_output_path = Path(temp_output.name)
        temp_output.close()
        actual_output_path = temp_output_path
    else:
        actual_output_path = output_h5_path

    # Read input H5 file and write to output H5 file
    with h5py.File(input_h5_path, 'r') as f_in, h5py.File(actual_output_path, 'w') as f_out:
        video_keys = sorted([k for k in f_in.keys()])
        print(f"Found {len(video_keys)} videos\n")

        for video_key in tqdm(video_keys, desc="Processing videos"):
            # Create group in output file
            group_out = f_out.create_group(video_key)

            # Copy all existing fields from input to output
            for field_name in f_in[video_key].keys():
                f_in.copy(f_in[video_key][field_name], group_out, name=field_name)

            try:
                # Check if required fields exist
                if 'change_points' not in f_in[video_key]:
                    tqdm.write(f"⚠️  {video_key}: No change_points found, skipping")
                    continue

                if 'gtscore' not in f_in[video_key]:
                    tqdm.write(f"⚠️  {video_key}: No gtscore found, skipping")
                    continue

                # Read required data
                gtscore = f_in[video_key]['gtscore'][:]
                segments = f_in[video_key]['change_points'][:]
                sampled_n_frames = len(gtscore)  # Number of sampled frames

                # Get picks (sampled frame indices in original frame space)
                picks = None
                if 'picks' in f_in[video_key]:
                    picks = f_in[video_key]['picks'][:]

                # Get n_frame_per_seg (original frames per segment) if available
                n_frame_per_seg = None
                if 'n_frame_per_seg' in f_in[video_key]:
                    n_frame_per_seg = f_in[video_key]['n_frame_per_seg'][:]

                # Get ORIGINAL frame count (n_frames) for GT percentage calculation
                if 'n_frames' in f_in[video_key]:
                    original_n_frames = int(f_in[video_key]['n_frames'][()])
                else:
                    original_n_frames = sampled_n_frames

                n_segments = len(segments)

                # Detect if change_points are in original frame space or sampled frame space
                # SumMe/TVSum: change_points are in original frame space (max index >> sampled_n_frames)
                # OVP/YouTube (new): change_points are in sampled frame space (max index < sampled_n_frames)
                max_segment_idx = int(segments.max()) if len(segments) > 0 else 0
                segments_in_original_space = max_segment_idx >= sampled_n_frames

                tqdm.write(f"\n📹 {video_key}: {original_n_frames} original frames, {sampled_n_frames} sampled frames, {n_segments} segments")
                tqdm.write(f"   Segments in {'ORIGINAL' if segments_in_original_space else 'SAMPLED'} frame space")

                # Calculate segment-level scores and lengths
                segment_scores = []
                segment_lengths = []

                if segments_in_original_space and picks is not None:
                    # SumMe/TVSum case: change_points are in original frame space
                    # Need to map to sampled frame indices for score calculation
                    for i, (start_orig, end_orig) in enumerate(segments):
                        start_orig, end_orig = int(start_orig), int(end_orig)

                        # Find sampled frame indices within this segment
                        sampled_indices = np.where((picks >= start_orig) & (picks <= end_orig))[0]

                        if len(sampled_indices) > 0:
                            segment_scores.append(np.mean(gtscore[sampled_indices]))
                        else:
                            segment_scores.append(0.0)

                        # Use n_frame_per_seg for original frame count per segment
                        if n_frame_per_seg is not None and i < len(n_frame_per_seg):
                            segment_lengths.append(int(n_frame_per_seg[i]))
                        else:
                            segment_lengths.append(end_orig - start_orig + 1)
                else:
                    # OVP/YouTube case: change_points are in sampled frame space
                    for start, end in segments:
                        start = int(max(0, start))
                        end = int(min(end, sampled_n_frames - 1))
                        if start <= end:
                            segment_scores.append(np.mean(gtscore[start:end+1]))
                            segment_lengths.append(end - start + 1)
                        else:
                            segment_scores.append(0.0)
                            segment_lengths.append(1)

                segment_scores = np.array(segment_scores)
                segment_lengths = np.array(segment_lengths)

                # Calculate budget based on ORIGINAL frame count
                budget = int(np.floor(gt_portion * original_n_frames))
                tqdm.write(f"   🎯 Selecting GT clips using Knapsack (target: {gt_portion*100:.1f}% of {original_n_frames} original = {budget} frames)...")

                # Use Knapsack to select GT clips
                shot_level_gt = solve_knapsack(segment_scores, segment_lengths, budget)

                # Generate frame-level gtsummary from shot_level_gt
                # gtsummary: highest scoring frame in each GT shot (1 frame per GT shot)
                gtsummary = np.zeros(sampled_n_frames, dtype=np.float32)

                if segments_in_original_space and picks is not None:
                    # SumMe/TVSum case: Map selected segments back to sampled frame indices
                    for i, (start_orig, end_orig) in enumerate(segments):
                        if shot_level_gt[i] > 0:
                            start_orig, end_orig = int(start_orig), int(end_orig)
                            sampled_indices = np.where((picks >= start_orig) & (picks <= end_orig))[0]
                            if len(sampled_indices) > 0:
                                # Find the frame with highest gtscore in this shot
                                shot_scores = gtscore[sampled_indices]
                                best_idx_in_shot = np.argmax(shot_scores)
                                best_frame_idx = sampled_indices[best_idx_in_shot]
                                gtsummary[best_frame_idx] = 1.0
                else:
                    # OVP/YouTube case: Segments are in sampled frame space
                    for i, (start, end) in enumerate(segments):
                        if shot_level_gt[i] > 0:
                            start = int(max(0, start))
                            end = int(min(end, sampled_n_frames - 1))
                            if start <= end:
                                # Find the frame with highest gtscore in this shot
                                shot_scores = gtscore[start:end+1]
                                best_idx_in_shot = np.argmax(shot_scores)
                                best_frame_idx = start + best_idx_in_shot
                                gtsummary[best_frame_idx] = 1.0

                gt_frames_selected = int(gtsummary.sum())
                gt_shot_original_frames = int(np.sum(segment_lengths[shot_level_gt > 0]))
                gt_percentage_original = (gt_shot_original_frames / original_n_frames) * 100 if original_n_frames > 0 else 0
                tqdm.write(f"   ✅ gtsummary: {gt_frames_selected} frames (1 per GT shot)")
                tqdm.write(f"   ✅ GT shots cover {gt_shot_original_frames} original frames ({gt_percentage_original:.1f}% of {original_n_frames})")

                gt_clips = int(shot_level_gt.sum())
                tqdm.write(f"   📊 GT clips: {gt_clips}/{n_segments} segments")

                # Add/Update shot_level_gt field
                if 'shot_level_gt' in group_out:
                    del group_out['shot_level_gt']
                group_out.create_dataset('shot_level_gt', data=shot_level_gt, dtype='int32')

                # Update gtsummary if it exists
                if 'gtsummary' in group_out:
                    del group_out['gtsummary']
                group_out.create_dataset('gtsummary', data=gtsummary, dtype='float32')

                # Update statistics (use original_n_frames for consistent reporting)
                stats['total_videos'] += 1
                stats['total_frames'] += original_n_frames
                stats['total_segments'] += n_segments
                stats['total_gt_frames'] += gt_frames_selected
                stats['total_gt_clips'] += gt_clips

            except Exception as e:
                tqdm.write(f"   ❌ Error processing {video_key}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # If we used a temp file, replace the original
    if same_file:
        shutil.move(str(actual_output_path), str(output_h5_path))

    # Print statistics
    avg_gt_percentage = (stats['total_gt_frames'] / stats['total_frames']) * 100 if stats['total_frames'] > 0 else 0

    print(f"\n{'='*80}")
    print(f"💾 Results saved to: {output_h5_path}")
    print(f"\n📊 Statistics:")
    print(f"   Videos processed: {stats['total_videos']}")
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Total segments: {stats['total_segments']}")
    print(f"   Total GT frames selected: {stats['total_gt_frames']} ({avg_gt_percentage:.1f}%)")
    print(f"   Total GT clips selected: {stats['total_gt_clips']}")
    print(f"{'='*80}\n")


def main():
    """主入口函数 - 所有路径从配置文件读取"""
    parser = argparse.ArgumentParser(
        description='Knapsack-based GT clip selection for video summarization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process YouTube dataset
  python 2_0_knapsack_shot_gt.py --dataset youtube

  # Process SumMe dataset
  python 2_0_knapsack_shot_gt.py --dataset summe

  # Specify custom GT selection portion (10%)
  python 2_0_knapsack_shot_gt.py --dataset youtube --gt-portion 0.10

注意：使用 --dataset 时，路径从配置文件读取
        """
    )

    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str,
                       help='Dataset to process')
    group.add_argument('--file', type=str,
                       help='Custom H5 file path (for pipeline)')

    parser.add_argument('--h5-dir', type=str, default=None,
                       help=argparse.SUPPRESS)  # 保留用于 pipeline 兼容，但忽略
    parser.add_argument('--output', type=str, default=None,
                       help='Output H5 filename (default: same as input without _KTS suffix)')
    parser.add_argument('--gt-portion', type=float, default=0.15,
                       help='Portion of frames to select as GT summary (default: 0.15 = 15%%)')

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    if args.file:
        # Custom file path (for pipeline calls)
        input_h5_path = Path(args.file)
        h5_dir = input_h5_path.parent
    else:
        # 从配置读取路径（纯配置，无回退）
        dataset = args.dataset.lower()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from utils.config_loader import get_config
        config = get_config()
        ds_paths = config.get_dataset_paths(dataset)

        h5_file = ds_paths.get('h5_file')
        if not h5_file:
            raise ValueError(f"配置文件中未定义 {dataset}.h5_file")

        h5_dir = h5_file.parent

        # Try _KTS.h5 first, then original
        kts_file = h5_dir / f"{dataset}_KTS.h5"
        orig_file = h5_file

        if kts_file.exists():
            input_h5_path = kts_file
            print(f"Using KTS output: {kts_file.name}")
        elif orig_file.exists():
            input_h5_path = orig_file
            print(f"Using original file: {orig_file.name}")
        else:
            print(f"❌ No input file found for dataset '{dataset}'")
            print(f"   Tried: {kts_file}")
            print(f"   Tried: {orig_file}")
            return

    if not input_h5_path.exists():
        print(f"❌ Input file not found: {input_h5_path}")
        return

    # Determine output path - save to same h5_dir without suffix
    h5_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        # Extract base name without _KTS suffix
        if args.file:
            base_name = input_h5_path.stem.replace('_KTS', '') + '.h5'
        else:
            base_name = f"{dataset}.h5"
        output_h5_path = h5_dir / base_name
    else:
        output_h5_path = h5_dir / args.output

    # Run Knapsack GT selection
    process_h5_with_knapsack(input_h5_path, output_h5_path, gt_portion=args.gt_portion)


if __name__ == "__main__":
    main()
