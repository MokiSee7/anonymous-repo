#!/usr/bin/env python3
"""
Extract frames from OVP videos based on H5 file

This script:
1. Reads the OVP H5 file to get the number of frames for each video
2. Extracts frames from videos at the same rate as the features (1 FPS typically)
3. Saves frames to datasets_preprocessed/data/dataset/OVP/frames/{video_id}/
"""

import os
import cv2
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


def extract_frames_from_video(video_path, output_dir, target_num_frames, fps=None):
    """
    Extract frames from a video file

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        target_num_frames: Target number of frames to extract (from H5)
        fps: Optional target FPS for extraction

    Returns:
        Number of frames extracted
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return 0

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"   Video FPS: {video_fps:.2f}")
    print(f"   Total frames in video: {total_frames}")
    print(f"   Target frames from H5: {target_num_frames}")

    # Calculate frame indices to extract
    # Sample uniformly across the video to match H5 features
    if target_num_frames >= total_frames:
        # Extract all frames
        frame_indices = list(range(total_frames))
    else:
        # Sample uniformly
        frame_indices = np.linspace(0, total_frames - 1, target_num_frames, dtype=int)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    extracted_count = 0

    for i, frame_idx in enumerate(tqdm(frame_indices, desc="   Extracting frames", unit="frame", leave=False)):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read frame
        ret, frame = cap.read()

        if not ret:
            print(f"   ⚠️  Could not read frame {frame_idx}")
            continue

        # Save frame (use 0-padded index for sorting)
        frame_filename = output_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        extracted_count += 1

    cap.release()

    return extracted_count


def process_ovp_dataset(h5_path, video_dir, output_base_dir):
    """
    Process all OVP videos

    Args:
        h5_path: Path to OVP H5 file
        video_dir: Directory containing video files
        output_base_dir: Base directory for frame output
    """
    h5_path = Path(h5_path)
    video_dir = Path(video_dir)
    output_base_dir = Path(output_base_dir)

    # Check paths
    if not h5_path.exists():
        print(f"❌ H5 file not found: {h5_path}")
        return

    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return

    print("="*80)
    print("OVP Dataset Frame Extraction")
    print("="*80)
    print(f"📂 H5 file: {h5_path}")
    print(f"📂 Video dir: {video_dir}")
    print(f"📂 Output dir: {output_base_dir}")
    print("="*80)
    print()

    # Open H5 file
    with h5py.File(h5_path, 'r') as h5_file:
        video_keys = list(h5_file.keys())
        print(f"Found {len(video_keys)} videos in H5 file\n")

        stats = {
            'total': len(video_keys),
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

        # Process each video
        for video_key in tqdm(video_keys, desc="Processing videos", unit="video"):
            # Get target number of frames from H5
            features = h5_file[video_key]['features']
            target_num_frames = features.shape[0]

            # Find corresponding video file
            # Video naming: video_1 -> v1.mpg or video_1.mpg
            video_num = video_key.replace('video_', '')

            # Try different naming patterns
            possible_names = [
                f"v{video_num}.mpg",
                f"v{video_num}.mp4",
                f"v{video_num}.avi",
                f"{video_key}.mpg",
                f"{video_key}.mp4",
                f"{video_key}.avi"
            ]

            video_path = None
            for name in possible_names:
                candidate = video_dir / name
                if candidate.exists():
                    video_path = candidate
                    break

            if not video_path:
                tqdm.write(f"⚠️  Video file not found for {video_key}, tried: {possible_names}")
                stats['failed'] += 1
                continue

            # Check if already processed
            output_dir = output_base_dir / video_key
            if output_dir.exists() and len(list(output_dir.glob("*.jpg"))) >= target_num_frames * 0.9:
                tqdm.write(f"✅ Skipping {video_key} - already processed ({len(list(output_dir.glob('*.jpg')))} frames)")
                stats['skipped'] += 1
                continue

            # Extract frames
            tqdm.write(f"\n🎬 Processing {video_key}")
            tqdm.write(f"   Video file: {video_path.name}")

            try:
                extracted = extract_frames_from_video(
                    video_path,
                    output_dir,
                    target_num_frames
                )

                if extracted >= target_num_frames * 0.9:  # Allow 10% tolerance
                    tqdm.write(f"   ✅ Extracted {extracted}/{target_num_frames} frames")
                    stats['success'] += 1
                else:
                    tqdm.write(f"   ⚠️  Only extracted {extracted}/{target_num_frames} frames")
                    stats['failed'] += 1

            except Exception as e:
                tqdm.write(f"   ❌ Error: {e}")
                stats['failed'] += 1

    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total videos: {stats['total']}")
    print(f"✅ Successfully processed: {stats['success']}")
    print(f"⏭️  Skipped (already done): {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from OVP videos based on H5 file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames with default paths
  python 0_extract_frames_ovp.py

  # Custom paths
  python 0_extract_frames_ovp.py --video-dir /path/to/videos --output-dir /path/to/output
        """
    )

    # Get script directory for computing absolute paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    default_h5 = project_root / "datasets_preprocessed/data/h5/ovp.h5"
    default_video_dir = project_root / "datasets_preprocessed/data/dataset/OVP/video"
    default_output_dir = project_root / "datasets_preprocessed/data/dataset/OVP/frames"

    parser.add_argument('--h5-path', type=str, default=str(default_h5),
                       help=f'Path to OVP H5 file (default: {default_h5})')
    parser.add_argument('--video-dir', type=str, default=str(default_video_dir),
                       help=f'Directory containing OVP videos (default: {default_video_dir})')
    parser.add_argument('--output-dir', type=str, default=str(default_output_dir),
                       help=f'Output directory for frames (default: {default_output_dir})')

    args = parser.parse_args()

    # Process dataset
    process_ovp_dataset(
        args.h5_path,
        args.video_dir,
        args.output_dir
    )


if __name__ == "__main__":
    main()
