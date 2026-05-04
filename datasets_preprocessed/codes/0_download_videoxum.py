#!/usr/bin/env python3
"""
Download VideoXum Dataset Videos

This script:
1. Reads video IDs from VideoXum JSON files (train/val/test)
2. Downloads videos from YouTube using yt-dlp
3. Saves videos to datasets_preprocessed/data/dataset/VideoXum/videos/

Requirements:
    pip install yt-dlp

Usage:
    # Download all videos
    python 0_download_videoxum.py

    # Download only train set
    python 0_download_videoxum.py --split train

    # Download only val set
    python 0_download_videoxum.py --split val

    # Download only test set
    python 0_download_videoxum.py --split test

    # Skip already downloaded videos
    python 0_download_videoxum.py --skip-existing

    # Limit number of videos to download
    python 0_download_videoxum.py --limit 10
"""

import json
import subprocess
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random


def get_video_ids_from_json(json_path):
    """
    Extract video IDs from VideoXum JSON file

    Args:
        json_path: Path to JSON file

    Returns:
        list: List of video IDs (e.g., ['v_uqiMw7tQ1Cc', ...])
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    video_ids = [item['video_id'] for item in data]
    return video_ids


def download_video(video_id, output_dir, skip_existing=True, cookies_file=None):
    """
    Download a single video from YouTube

    Args:
        video_id: YouTube video ID (e.g., 'v_uqiMw7tQ1Cc')
        output_dir: Directory to save the video
        skip_existing: Skip if video already exists
        cookies_file: Path to cookies file for YouTube authentication (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove 'v_' prefix if present to get YouTube ID
    youtube_id = video_id.replace('v_', '')

    # Output filename
    output_file = output_dir / f"{video_id}.mp4"

    # Check if already exists
    if skip_existing and output_file.exists():
        return True

    # YouTube URL
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Auto-detect cookies file if not provided
    if cookies_file is None:
        # Look for cookies in common locations
        project_root = Path(__file__).resolve().parent.parent.parent
        possible_cookie_paths = [
            project_root / "youtube_cookies.txt",
            Path.home() / "youtube_cookies.txt",
            Path("/path/to/youtube_cookies.txt"),  # Server path
        ]
        for cookie_path in possible_cookie_paths:
            if cookie_path.exists():
                cookies_file = str(cookie_path)
                print(f"      📝 Using cookies from: {cookie_path}")
                break

    try:
        # Add random delay (2-5 seconds) to avoid rate limiting
        time.sleep(random.uniform(2, 5))

        # Download using yt-dlp with enhanced anti-detection
        # Note: If using cookies, don't use android client (they're incompatible)
        cmd = [
            'yt-dlp',
            # Use single stream format to avoid 403 errors from YouTube
            # Don't use bestvideo+bestaudio as it requires merging and triggers stricter restrictions
            '-f', 'best[height<=720]/best',
            '-o', str(output_file),
            '--no-playlist',
            '--quiet',
            '--no-warnings',
            '--retries', '10',           # More retries
            '--fragment-retries', '10',  # More fragment retries
            '--retry-sleep', '3',        # Wait 3s between retries
            '--concurrent-fragments', '1',  # Download one fragment at a time
            # Anti-detection measures
            '--user-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '--referer', 'https://www.youtube.com/',
            '--add-header', 'Accept-Language:en-US,en;q=0.9',
            '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            '--sleep-interval', '2',     # Sleep 2s between downloads
            '--max-sleep-interval', '5', # Max sleep 5s
            url
        ]

        # Add cookies if available (cookies replace need for android client)
        if cookies_file:
            cmd.insert(1, '--cookies')
            cmd.insert(2, str(cookies_file))
        else:
            # Only use android client if no cookies (fallback)
            cmd.insert(1, '--extractor-args')
            cmd.insert(2, 'youtube:player_client=android')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0 and output_file.exists():
            return True
        else:
            # Print detailed error for debugging
            error_msg = result.stderr

            # Filter out Python version warnings
            if error_msg and 'Deprecated Feature' not in error_msg:
                # Print first error line
                error_lines = [line for line in error_msg.split('\n') if line.strip() and 'WARNING' not in line]
                if error_lines:
                    print(f"      Error: {error_lines[0][:200]}")

            # Check if file exists despite error
            if output_file.exists():
                return True

            return False

    except subprocess.TimeoutExpired:
        print(f"      Timeout downloading {video_id}")
        return False
    except Exception as e:
        print(f"      Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download VideoXum dataset videos from YouTube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all videos
  python 0_download_videoxum.py

  # Download only train set
  python 0_download_videoxum.py --split train

  # Download with limit
  python 0_download_videoxum.py --split test --limit 100

  # Skip existing videos
  python 0_download_videoxum.py --skip-existing

  # Parallel download with 5 workers (5x faster!)
  python 0_download_videoxum.py --split test --workers 5 --skip-existing

  # Fast download of all videos with 10 workers
  python 0_download_videoxum.py --workers 10 --skip-existing
        """
    )

    parser.add_argument('--split', type=str, default='all',
                       choices=['all', 'train', 'val', 'test'],
                       help='Which split to download (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: datasets_preprocessed/data/dataset/VideoXum/videos)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip already downloaded videos')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of videos to download')
    parser.add_argument('--save-progress', action='store_true',
                       help='Save download progress to resume later')
    parser.add_argument('--progress-file', type=str, default=None,
                       help='Progress file path (default: datasets_preprocessed/data/dataset/VideoXum/download_progress.json)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel download workers (default: 1, recommended: 3-5)')

    args = parser.parse_args()

    # Get script directory and project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # Define paths
    videoxum_dir = project_root / "datasets_preprocessed/data/dataset/VideoXum"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = videoxum_dir / "videos"

    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: yt-dlp is not installed")
        print("   Please install it with: pip install yt-dlp")
        sys.exit(1)

    # Determine which splits to download
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]

    print(f"\n{'='*80}")
    print(f"VideoXum Video Downloader")
    print(f"{'='*80}")
    print(f"Splits to download: {', '.join(splits)}")
    print(f"Output directory:   {output_dir}")
    print(f"Skip existing:      {args.skip_existing}")
    if args.limit:
        print(f"Limit per split:    {args.limit} videos")
    print(f"{'='*80}\n")

    # Collect all video IDs
    all_video_ids = {}
    total_videos = 0

    for split in splits:
        json_path = videoxum_dir / f"{split}_videoxum.json"

        if not json_path.exists():
            print(f"❌ JSON file not found: {json_path}")
            continue

        video_ids = get_video_ids_from_json(json_path)

        # Apply limit if specified
        if args.limit:
            video_ids = video_ids[:args.limit]

        all_video_ids[split] = video_ids
        total_videos += len(video_ids)

        print(f"📂 {split:5s}: {len(video_ids):5d} videos to download")

    print(f"\n📊 Total: {total_videos} videos\n")

    # Download statistics
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }

    # Download videos
    for split, video_ids in all_video_ids.items():
        print(f"\n{'='*80}")
        print(f"Downloading {split.upper()} split ({len(video_ids)} videos)")
        print(f"{'='*80}\n")

        if args.workers > 1:
            # Parallel download
            print(f"🚀 Using {args.workers} parallel workers\n")

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all download tasks
                future_to_video = {
                    executor.submit(download_video, video_id, output_dir, args.skip_existing): video_id
                    for video_id in video_ids
                }

                # Process results with progress bar
                with tqdm(total=len(video_ids), desc=f"  {split}", unit="video") as pbar:
                    for future in as_completed(future_to_video):
                        video_id = future_to_video[future]
                        stats['total'] += 1

                        try:
                            success = future.result()

                            # Check if skipped
                            output_file = output_dir / f"{video_id}.mp4"
                            if output_file.exists() and args.skip_existing:
                                # This could be pre-existing or just downloaded
                                pass

                            if success:
                                stats['success'] += 1
                            else:
                                stats['failed'] += 1
                                tqdm.write(f"    ❌ Failed ({stats['failed']}): {video_id}")
                        except Exception as e:
                            stats['failed'] += 1
                            tqdm.write(f"    ❌ Exception ({stats['failed']}) for {video_id}: {e}")

                        pbar.update(1)
        else:
            # Sequential download (original behavior)
            for video_id in tqdm(video_ids, desc=f"  {split}", unit="video"):
                stats['total'] += 1

                # Check if already exists
                output_file = output_dir / f"{video_id}.mp4"
                if args.skip_existing and output_file.exists():
                    stats['skipped'] += 1
                    continue

                # Download
                success = download_video(video_id, output_dir, skip_existing=args.skip_existing)

                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    tqdm.write(f"    ❌ Failed ({stats['failed']}): {video_id}")

    # Print summary
    print(f"\n{'='*80}")
    print("Download Summary")
    print(f"{'='*80}")
    print(f"Total videos:       {stats['total']}")
    print(f"✅ Successfully downloaded: {stats['success']}")
    print(f"⏭️  Skipped (existing):     {stats['skipped']}")
    print(f"❌ Failed:                  {stats['failed']}")
    print(f"{'='*80}")
    print(f"\n💾 Videos saved to: {output_dir}\n")


# Export function for use in other scripts (e.g., batch processor)
def download_video_from_youtube(video_id, output_dir, skip_existing=True, cookies_file=None):
    """
    Wrapper function for downloading videos from YouTube.
    Used by 1_0_videoxum_batch_processor.py

    Args:
        video_id: Video ID (e.g., 'v_QOlSCBRmfWY')
        output_dir: Directory to save the video
        skip_existing: Skip if video already exists
        cookies_file: Path to cookies file (optional, will auto-detect if None)

    Returns:
        bool: True if successful, False otherwise
    """
    return download_video(video_id, output_dir, skip_existing, cookies_file)


if __name__ == "__main__":
    main()
