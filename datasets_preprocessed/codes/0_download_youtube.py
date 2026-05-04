#!/usr/bin/env python3
"""
Download YouTube Dataset Videos

This script:
1. Reads video IDs from youtube.h5
2. Downloads videos from YouTube using yt-dlp
3. Saves videos to datasets_preprocessed/datasets/YouTube/{video_id}.mp4

Requirements:
    pip install yt-dlp h5py

Usage:
    python 0_0_download_youtube.py
    python 0_0_download_youtube.py --h5-file path/to/custom.h5
    python 0_0_download_youtube.py --output-dir custom/output/dir
    python 0_0_download_youtube.py --skip-existing  # Skip already downloaded videos
"""

import h5py
import subprocess
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import sys


def get_video_ids_from_h5(h5_path):
    """
    Extract video IDs from H5 file

    Args:
        h5_path: Path to H5 file

    Returns:
        list: List of video IDs (e.g., ['video_1', 'video_2', ...])
    """
    with h5py.File(h5_path, 'r') as f:
        video_keys = sorted([k for k in f.keys()])

        # Extract video names if available
        video_info = []
        for video_key in video_keys:
            info = {'key': video_key}

            # Try to get video_name field (YouTube ID)
            if 'video_name' in f[video_key]:
                video_name = f[video_key]['video_name'][()]
                if isinstance(video_name, bytes):
                    video_name = video_name.decode('utf-8')
                info['youtube_id'] = video_name
            else:
                # If no video_name field, use the key itself
                info['youtube_id'] = video_key

            video_info.append(info)

    return video_info


def check_ytdlp_installed():
    """Check if yt-dlp is installed"""
    try:
        result = subprocess.run(['yt-dlp', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_video(youtube_id, output_dir, skip_existing=True):
    """
    Download a single video from YouTube

    Args:
        youtube_id: YouTube video ID
        output_dir: Output directory
        skip_existing: Skip if video already exists

    Returns:
        bool: True if download successful, False otherwise
    """
    output_path = output_dir / f"{youtube_id}.mp4"

    # Skip if already exists
    if skip_existing and output_path.exists():
        print(f"   ⏭️  Already exists: {youtube_id}.mp4")
        return True

    # yt-dlp command
    # Format: best video+audio, merge to mp4
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '-o', str(output_path),
        f'https://www.youtube.com/watch?v={youtube_id}'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode == 0:
            print(f"   ✅ Downloaded: {youtube_id}.mp4")
            return True
        else:
            print(f"   ❌ Failed: {youtube_id}")
            print(f"      Error: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Timeout: {youtube_id} (>5min)")
        return False
    except Exception as e:
        print(f"   ❌ Error: {youtube_id} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube dataset videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all YouTube videos
  python 0_0_download_youtube.py

  # Use custom H5 file
  python 0_0_download_youtube.py --h5-file path/to/custom.h5

  # Custom output directory
  python 0_0_download_youtube.py --output-dir my/videos

  # Skip existing videos
  python 0_0_download_youtube.py --skip-existing

  # Re-download all (overwrite existing)
  python 0_0_download_youtube.py --no-skip-existing

Requirements:
  Install yt-dlp first:
    pip install yt-dlp

  Or using conda:
    conda install -c conda-forge yt-dlp
        """
    )

    parser.add_argument('--h5-file', type=str,
                       default='datasets_preprocessed/data/h5/youtube.h5',
                       help='Path to YouTube dataset H5 file')
    parser.add_argument('--output-dir', type=str,
                       default='datasets_preprocessed/data/dataset/YouTube',
                       help='Output directory for downloaded videos')
    parser.add_argument('--skip-existing', dest='skip_existing', action='store_true',
                       help='Skip already downloaded videos (default)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                       help='Re-download all videos (overwrite existing)')
    parser.set_defaults(skip_existing=True)

    args = parser.parse_args()

    # Check if H5 file exists
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"❌ H5 file not found: {h5_path}")
        print(f"\nPlease ensure the H5 file exists at the specified path.")
        sys.exit(1)

    # Check if yt-dlp is installed
    if not check_ytdlp_installed():
        print("❌ yt-dlp is not installed!")
        print("\nPlease install yt-dlp first:")
        print("  pip install yt-dlp")
        print("\nOr using conda:")
        print("  conda install -c conda-forge yt-dlp")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("YouTube Dataset Video Downloader")
    print("="*80)
    print(f"H5 file:     {h5_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print("="*80 + "\n")

    # Get video IDs
    print("📖 Reading video IDs from H5 file...")
    video_info = get_video_ids_from_h5(h5_path)
    print(f"   Found {len(video_info)} videos\n")

    # Download videos
    print("📥 Starting downloads...\n")

    stats = {
        'total': len(video_info),
        'success': 0,
        'failed': 0,
        'skipped': 0
    }

    for idx, info in enumerate(video_info, 1):
        video_key = info['key']
        youtube_id = info['youtube_id']

        print(f"[{idx}/{len(video_info)}] Processing {video_key} (ID: {youtube_id})")

        # Check if exists
        output_path = output_dir / f"{youtube_id}.mp4"
        if args.skip_existing and output_path.exists():
            print(f"   ⏭️  Already exists: {youtube_id}.mp4")
            stats['skipped'] += 1
            continue

        # Download
        success = download_video(youtube_id, output_dir, skip_existing=False)

        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1

        print()

    # Print summary
    print("="*80)
    print("📊 Download Summary")
    print("="*80)
    print(f"Total videos:      {stats['total']}")
    print(f"Successfully downloaded: {stats['success']}")
    print(f"Failed:            {stats['failed']}")
    print(f"Skipped (existing): {stats['skipped']}")
    print("="*80)

    if stats['failed'] > 0:
        print("\n⚠️  Some videos failed to download. Common reasons:")
        print("   - Video is private or removed")
        print("   - Geographic restrictions")
        print("   - Network issues")
        print("   - Age-restricted content")

    print(f"\n✅ Videos saved to: {output_dir.absolute()}\n")


if __name__ == "__main__":
    main()
