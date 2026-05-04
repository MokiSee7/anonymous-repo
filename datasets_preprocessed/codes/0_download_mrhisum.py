#!/usr/bin/env python3
"""
Download MrHiSum videos from YouTube

This script downloads videos for the MrHiSum dataset using youtube_id from metadata.csv
All paths are read from configs/config.yaml by default.

Uses pytubefix as the primary downloader with yt-dlp as fallback.

Usage:
    python 0_download_mrhisum.py --test        # Download 5 videos for testing
    python 0_download_mrhisum.py               # Download all videos
    python 0_download_mrhisum.py --start 0 --end 100  # Download specific range
    python 0_download_mrhisum.py --remove-failed-from-h5  # Remove failed videos from H5

Note: All paths are read from config files (configs/config.yaml)

Requirements:
  - pytubefix: pip install pytubefix
  - yt-dlp (fallback): pip install yt-dlp
  - ffmpeg: apt install ffmpeg (Linux) or brew install ffmpeg (Mac)
"""

import csv
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import h5py
import tempfile
import shutil

# Check pytubefix availability
try:
    from pytubefix import YouTube
    from pytubefix.exceptions import VideoUnavailable
    HAS_PYTUBEFIX = True
except ImportError:
    HAS_PYTUBEFIX = False

# Check yt-dlp availability
try:
    subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    HAS_YTDLP = True
except (subprocess.CalledProcessError, FileNotFoundError):
    HAS_YTDLP = False


def download_with_pytubefix(youtube_id, output_path, max_retries=3):
    """
    Download a video using pytubefix.

    Returns:
        bool: True if successful, False otherwise
    """
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    for attempt in range(max_retries):
        try:
            yt = YouTube(url)
            # Try to get progressive mp4 stream (video+audio combined)
            stream = yt.streams.filter(
                progressive=True, file_extension='mp4'
            ).order_by('resolution').desc().first()

            if stream is None:
                # Fallback: get highest resolution adaptive video
                stream = yt.streams.filter(
                    file_extension='mp4'
                ).order_by('resolution').desc().first()

            if stream is None:
                tqdm.write(f"    [pytubefix] No suitable stream found")
                return False

            stream.download(
                output_path=str(output_path.parent),
                filename=output_path.name
            )

            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            else:
                if attempt < max_retries - 1:
                    tqdm.write(f"    [pytubefix] Retry {attempt + 1}/{max_retries}")
        except VideoUnavailable:
            tqdm.write(f"    [pytubefix] Video unavailable")
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(f"    [pytubefix] Retry {attempt + 1}/{max_retries}: {str(e)[:100]}")
            else:
                tqdm.write(f"    [pytubefix] Failed: {str(e)[:100]}")

    return False


def download_with_ytdlp(youtube_id, output_path, max_retries=3):
    """
    Download a video using yt-dlp.

    Returns:
        bool: True if successful, False otherwise
    """
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        url
    ]

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0 and output_path.exists():
                return True
            else:
                if attempt < max_retries - 1:
                    tqdm.write(f"    [yt-dlp] Retry {attempt + 1}/{max_retries}")
        except subprocess.TimeoutExpired:
            tqdm.write(f"    [yt-dlp] Timeout, retrying...")
        except Exception as e:
            tqdm.write(f"    [yt-dlp] Error: {e}")

    return False


def download_video(youtube_id, video_id, output_dir, max_retries=3):
    """
    Download a single video from YouTube using pytubefix (primary) with yt-dlp fallback.

    Args:
        youtube_id: YouTube video ID
        video_id: Local video identifier (e.g., video_1)
        output_dir: Directory to save the video
        max_retries: Maximum number of download attempts per tool

    Returns:
        tuple: (success: bool, tool_used: str or None)
    """
    output_path = output_dir / f"{video_id}.mp4"

    # Skip if already exists
    if output_path.exists():
        return True, "skipped"

    # 1) Try pytubefix first
    if HAS_PYTUBEFIX:
        tqdm.write(f"   Trying pytubefix...")
        if download_with_pytubefix(youtube_id, output_path, max_retries):
            return True, "pytubefix"

    # 2) Fallback to yt-dlp
    if HAS_YTDLP:
        tqdm.write(f"   Falling back to yt-dlp...")
        if download_with_ytdlp(youtube_id, output_path, max_retries):
            return True, "yt-dlp"

    return False, None


def load_metadata(csv_path):
    """
    Load metadata from CSV file

    Returns:
        list: List of tuples (video_id, youtube_id, duration)
    """
    metadata = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id']
            youtube_id = row['youtube_id']
            duration = int(row['duration'])
            metadata.append((video_id, youtube_id, duration))

    return metadata


def remove_videos_from_h5(h5_path, video_ids_to_remove):
    """
    Remove failed videos from H5 file

    Args:
        h5_path: Path to H5 file
        video_ids_to_remove: List of video IDs to remove

    Returns:
        int: Number of videos removed
    """
    if not h5_path.exists():
        print(f"⚠️  H5 file not found: {h5_path}")
        return 0

    if not video_ids_to_remove:
        return 0

    print(f"\n{'='*80}")
    print(f"🗑️  Removing {len(video_ids_to_remove)} failed videos from H5 file")
    print(f"{'='*80}")

    # Create backup
    backup_path = h5_path.with_suffix('.h5.backup')
    print(f"📦 Creating backup: {backup_path}")
    shutil.copy2(h5_path, backup_path)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    removed_count = 0
    kept_count = 0

    try:
        # Read from original, write to temp (excluding failed videos)
        with h5py.File(h5_path, 'r') as f_in, h5py.File(temp_path, 'w') as f_out:
            total_videos = len(f_in.keys())

            print(f"📊 Original H5 file has {total_videos} videos")

            for video_key in tqdm(f_in.keys(), desc="Processing H5 file"):
                if video_key in video_ids_to_remove:
                    removed_count += 1
                    tqdm.write(f"  🗑️  Removing {video_key}")
                else:
                    # Copy video group to output
                    f_in.copy(f_in[video_key], f_out, name=video_key)
                    kept_count += 1

        # Replace original with temp
        shutil.move(str(temp_path), str(h5_path))

        print(f"\n✅ H5 file updated:")
        print(f"   Removed: {removed_count} videos")
        print(f"   Kept: {kept_count} videos")
        print(f"   Backup saved: {backup_path}")

        return removed_count

    except Exception as e:
        print(f"\n❌ Error updating H5 file: {e}")
        print(f"   Original file unchanged")
        if temp_path.exists():
            temp_path.unlink()
        return 0


def main():
    # Load config
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.config_loader import get_config
    config = get_config()
    ds_paths = config.get_dataset_paths('mrhisum')
    data_root = config.get_path('paths.data_root')

    # Default paths from config
    default_videos_dir = str(ds_paths.get('videos_dir') or '')
    default_h5 = str(ds_paths.get('h5_file') or '')
    default_csv = str(Path(data_root) / 'mrhisum' / 'metadata.csv') if data_root else ''

    parser = argparse.ArgumentParser(
        description='Download MrHiSum videos from YouTube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 5 videos for testing
  python 0_download_mrhisum.py --test

  # Download all videos
  python 0_download_mrhisum.py

  # Download specific range
  python 0_download_mrhisum.py --start 0 --end 100

  # Download and remove failed videos from H5
  python 0_download_mrhisum.py --remove-failed-from-h5

Requirements:
  - pytubefix (primary): pip install pytubefix
  - yt-dlp (fallback): pip install yt-dlp
  - ffmpeg: apt install ffmpeg (Linux) or brew install ffmpeg (Mac)

Note: All paths are read from config files (configs/config.yaml)
        """
    )

    parser.add_argument('--test', action='store_true',
                       help='Test mode: only download 5 videos')
    parser.add_argument('--start', type=int, default=0,
                       help='Start index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='End index (default: all)')
    parser.add_argument('--csv', type=str,
                       default=default_csv,
                       help='Path to metadata CSV file (default: from config)')
    parser.add_argument('--output', type=str,
                       default=default_videos_dir,
                       help='Output directory for videos (default: from config)')
    parser.add_argument('--h5', type=str,
                       default=default_h5,
                       help='Path to H5 file (default: from config)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum download retries per video (default: 3)')
    parser.add_argument('--remove-failed-from-h5', action='store_true',
                       help='Remove failed videos from H5 file after download')

    args = parser.parse_args()

    # Setup paths
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    h5_path = Path(args.h5)

    if not args.output:
        print("Error: videos_dir not configured for mrhisum in config.yaml")
        sys.exit(1)
    if not args.h5:
        print("Error: h5_file not configured for mrhisum in config.yaml")
        sys.exit(1)
    if not args.csv:
        print("Error: Could not determine metadata CSV path from config")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    if not HAS_PYTUBEFIX and not HAS_YTDLP:
        print("❌ No download tool available!")
        print("   Install at least one:")
        print("   - pytubefix (recommended): pip install pytubefix")
        print("   - yt-dlp (fallback): pip install yt-dlp")
        sys.exit(1)

    print(f"Download tools:")
    print(f"  pytubefix (primary):  {'✅ available' if HAS_PYTUBEFIX else '❌ not installed (pip install pytubefix)'}")
    print(f"  yt-dlp (fallback):    {'✅ available' if HAS_YTDLP else '❌ not installed (pip install yt-dlp)'}")

    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print(f"  ffmpeg:               ✅ available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  ffmpeg:               ⚠️  not found (needed by yt-dlp for merging)")
        print(f"                        Install: apt install ffmpeg (Linux) or brew install ffmpeg (Mac)")

    # Load metadata
    print(f"\n{'='*80}")
    print(f"MrHiSum Video Download")
    print(f"{'='*80}")
    print(f"CSV file: {csv_path}")
    print(f"Output directory: {output_dir}")
    if args.remove_failed_from_h5:
        print(f"H5 file: {h5_path}")
        print(f"⚠️  Failed videos will be REMOVED from H5 file")
    print(f"{'='*80}\n")

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    metadata = load_metadata(csv_path)
    print(f"✅ Loaded {len(metadata)} videos from CSV\n")

    # Determine range
    if args.test:
        start_idx = 0
        end_idx = 5
        print(f"🧪 Test mode: downloading 5 videos\n")
    else:
        start_idx = args.start
        end_idx = args.end if args.end is not None else len(metadata)
        print(f"📥 Downloading videos {start_idx} to {end_idx-1}\n")

    # Download videos
    videos_to_download = metadata[start_idx:end_idx]

    stats = {
        'total': len(videos_to_download),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'by_pytubefix': 0,
        'by_ytdlp': 0,
    }

    failed_videos = []

    for video_id, youtube_id, duration in tqdm(videos_to_download, desc="Downloading videos"):
        output_path = output_dir / f"{video_id}.mp4"

        if output_path.exists():
            stats['skipped'] += 1
            continue

        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"📹 {video_id}: {youtube_id} ({duration}s)")

        success, tool_used = download_video(youtube_id, video_id, output_dir, args.max_retries)

        if success:
            stats['success'] += 1
            if tool_used == "pytubefix":
                stats['by_pytubefix'] += 1
            elif tool_used == "yt-dlp":
                stats['by_ytdlp'] += 1
            tqdm.write(f"   ✅ Downloaded successfully [{tool_used}]")
        else:
            stats['failed'] += 1
            failed_videos.append((video_id, youtube_id))
            tqdm.write(f"   ❌ Failed to download (both tools)")

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 Download Summary")
    print(f"{'='*80}")
    print(f"Total videos: {stats['total']}")
    print(f"Already downloaded (skipped): {stats['skipped']}")
    print(f"Successfully downloaded: {stats['success']}")
    print(f"  - via pytubefix: {stats['by_pytubefix']}")
    print(f"  - via yt-dlp:    {stats['by_ytdlp']}")
    print(f"Failed: {stats['failed']}")

    if failed_videos:
        print(f"\n❌ Failed videos ({len(failed_videos)}):")
        for video_id, youtube_id in failed_videos[:10]:
            print(f"   - {video_id}: {youtube_id}")
        if len(failed_videos) > 10:
            print(f"   ... and {len(failed_videos) - 10} more")

    print(f"{'='*80}\n")

    # Save failed list
    if failed_videos:
        failed_list_path = output_dir / "failed_downloads.txt"
        with open(failed_list_path, 'w') as f:
            for video_id, youtube_id in failed_videos:
                f.write(f"{video_id},{youtube_id}\n")
        print(f"💾 Failed videos list saved to: {failed_list_path}\n")

    # Remove failed videos from H5 file
    if args.remove_failed_from_h5 and failed_videos:
        video_ids_to_remove = [video_id for video_id, _ in failed_videos]
        removed_count = remove_videos_from_h5(h5_path, video_ids_to_remove)

        if removed_count > 0:
            print(f"\n✅ Successfully removed {removed_count} failed videos from H5 file")
            print(f"   Backup created: {h5_path.with_suffix('.h5.backup')}")
        else:
            print(f"\n⚠️  No videos were removed from H5 file")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
