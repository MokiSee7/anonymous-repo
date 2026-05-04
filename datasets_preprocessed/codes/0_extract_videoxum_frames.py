#!/usr/bin/env pythdon3
"""
Extract frames for VideoXum dataset based on H5 file picks

This script reads the VideoXum H5 file and extracts the frames specified
by the 'picks' field from each video.

Usage:
    python 0_extract_videoxum_frames.py
    python 0_extract_videoxum_frames.py --test  # Only process 3 videos
"""

import h5py
import cv2
import numpy as np
import argparse
import traceback
from pathlib import Path
from tqdm import tqdm


def extract_frame_from_video(video_path, frame_idx, output_path):
    """
    Extract a specific frame from video and save it

    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        output_path: Path to save the extracted frame

    Returns:
        bool: True if successful, False otherwise
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False

    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Read frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return False

    # Save frame
    cv2.imwrite(str(output_path), frame)
    return True


def extract_frames_for_videoxum(
    h5_path,
    video_dir,
    output_dir,
    test_mode=False
):
    """
    Extract frames for all videos in VideoXum H5 file

    Args:
        h5_path: Path to VideoXum H5 file
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames
        test_mode: If True, only process first 3 videos
    """
    print(f"\n{'='*80}")
    print(f"VideoXum Frame Extraction")
    print(f"{'='*80}")
    print(f"H5 file: {h5_path}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Test mode: {test_mode}")
    if test_mode:
        print("⚠️  Only processing 3 videos for testing")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'total_videos': 0,
        'total_frames_extracted': 0,
        'failed_videos': [],
        'failed_frames': []
    }

    # Read H5 file
    with h5py.File(h5_path, 'r') as h5_file:
        video_keys = sorted(h5_file.keys())

        if test_mode:
            video_keys = video_keys[:3]

        print(f"Processing {len(video_keys)} videos\n")

        for video_key in tqdm(video_keys, desc="Processing videos"):
            try:
                video = h5_file[video_key]

                # Get video info
                video_name = video['video_name'][()].decode()
                picks = video['picks'][:]
                n_frames_total = video['n_frames'][()]

                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"Processing {video_key}: {video_name}")
                tqdm.write(f"  Total frames: {n_frames_total}")
                tqdm.write(f"  Frames to extract: {len(picks)}")

                # Find video file
                video_path = video_dir / f"{video_name}.mp4"

                if not video_path.exists():
                    tqdm.write(f"  ❌ Video file not found: {video_path}")
                    stats['failed_videos'].append(video_key)
                    continue

                # 统计原始视频总帧数并写入 H5
                cap_info = cv2.VideoCapture(str(video_path))
                if cap_info.isOpened():
                    n_frames_original = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap_info.release()
                    if n_frames_original > 0:
                        stats.setdefault('h5_updates', []).append(
                            (video_key, n_frames_original)
                        )

                # Create output directory for this video
                video_output_dir = output_dir / video_key
                video_output_dir.mkdir(parents=True, exist_ok=True)

                # Check if already extracted
                existing_frames = list(video_output_dir.glob("frame_*.jpg"))
                if len(existing_frames) == len(picks):
                    tqdm.write(f"  ⏭️  Already extracted ({len(existing_frames)} frames), skipping")
                    stats['total_videos'] += 1
                    stats['total_frames_extracted'] += len(existing_frames)
                    continue

                # Extract frames
                success_count = 0
                fail_count = 0

                for i, frame_idx in enumerate(tqdm(picks,
                                                   desc=f"  Extracting frames",
                                                   leave=False)):
                    # Output filename: use actual frame index (6 digits to match pipeline)
                    # e.g., frame_000000.jpg, frame_000025.jpg, frame_000050.jpg (for 25 FPS)
                    output_path = video_output_dir / f"frame_{frame_idx:06d}.jpg"

                    # Skip if already exists
                    if output_path.exists():
                        success_count += 1
                        continue

                    # Extract frame
                    success = extract_frame_from_video(video_path, frame_idx, output_path)

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        stats['failed_frames'].append((video_key, i, frame_idx))

                tqdm.write(f"  ✅ Extracted {success_count}/{len(picks)} frames")
                if fail_count > 0:
                    tqdm.write(f"  ⚠️  Failed: {fail_count} frames")

                stats['total_videos'] += 1
                stats['total_frames_extracted'] += success_count

            except Exception as e:
                tqdm.write(f"  ❌ Error processing {video_key}: {e}")
                import traceback
                traceback.print_exc()
                stats['failed_videos'].append(video_key)
                continue

    # 批量写入 n_frames_original 到 H5
    h5_updates = stats.get('h5_updates', [])
    if h5_updates:
        try:
            with h5py.File(h5_path, 'a') as h5_w:
                for vkey, nf in h5_updates:
                    if vkey in h5_w:
                        vg = h5_w[vkey]
                        if 'n_frames_original' in vg:
                            vg['n_frames_original'][()] = nf
                        else:
                            vg.create_dataset(
                                'n_frames_original', data=np.int64(nf)
                            )
            print(f"\nH5 updated: {len(h5_updates)} videos (n_frames_original)")
        except Exception as e:
            print(f"\nFailed to write H5: {e}")
            traceback.print_exc()

    # Print final statistics
    print(f"\n{'='*80}")
    print(f"Extraction Complete!")
    print(f"{'='*80}")
    print(f"Videos processed: {stats['total_videos']}")
    print(f"Total frames extracted: {stats['total_frames_extracted']:,}")

    if stats['failed_videos']:
        print(f"\nFailed videos ({len(stats['failed_videos'])}):")
        for video_key in stats['failed_videos'][:10]:
            print(f"   - {video_key}")

    if stats['failed_frames']:
        print(f"\nFailed frames ({len(stats['failed_frames'])}):")
        for video_key, frame_i, frame_idx in stats['failed_frames'][:10]:
            print(f"   - {video_key}: frame {frame_i} (idx {frame_idx})")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames for VideoXum dataset from H5 file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames for all videos
  python 0_extract_videoxum_frames.py

  # Test mode: only process 3 videos
  python 0_extract_videoxum_frames.py --test

  # Custom paths
  python 0_extract_videoxum_frames.py \\
      --h5 datasets_preprocessed/data/h5/videoxum.h5 \\
      --videos datasets_preprocessed/data/dataset/VideoXum/videos \\
      --output datasets_preprocessed/data/dataset/VideoXum/frames

Output:
  Frames are saved to: {output_dir}/{video_key}/frame_XXXXXX.jpg (6 digits)
  Frame numbering corresponds to the actual frame index in the original video.
  Example: For 25 FPS video with picks=[0,25,50,...]:
    - frame_000000.jpg (frame 0 of original video)
    - frame_000025.jpg (frame 25 of original video)
    - frame_000050.jpg (frame 50 of original video)

Note:
  - Frames are extracted based on the 'picks' field in H5 file
  - Each frame is saved as JPEG with filename = actual frame index in video
  - Skips already extracted frames automatically
        """
    )

    parser.add_argument('--test', action='store_true',
                       help='Test mode: only process 3 videos')
    parser.add_argument('--h5', type=str,
                       default='datasets_preprocessed/data/h5/videoxum.h5',
                       help='Path to VideoXum H5 file')
    parser.add_argument('--videos', type=str,
                       default='datasets_preprocessed/data/dataset/VideoXum/videos',
                       help='Directory containing video files')
    parser.add_argument('--output', type=str,
                       default='datasets_preprocessed/data/dataset/VideoXum/frames',
                       help='Output directory for extracted frames')

    args = parser.parse_args()

    # Setup paths
    h5_path = Path(args.h5)
    video_dir = Path(args.videos)
    output_dir = Path(args.output)

    # Check inputs
    if not h5_path.exists():
        print(f"❌ H5 file not found: {h5_path}")
        return

    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return

    # Extract frames
    extract_frames_for_videoxum(
        h5_path=h5_path,
        video_dir=video_dir,
        output_dir=output_dir,
        test_mode=args.test
    )


if __name__ == "__main__":
    main()
