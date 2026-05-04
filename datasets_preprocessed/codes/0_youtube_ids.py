#!/usr/bin/env python3
"""
List YouTube Video IDs from H5 File

Quick script to view all YouTube video IDs in the dataset.

Usage:
    python 0_0_list_youtube_ids.py
    python 0_0_list_youtube_ids.py --h5-file path/to/custom.h5
"""

import h5py
import argparse
from pathlib import Path


def list_youtube_ids(h5_path):
    """List all YouTube video IDs from H5 file"""

    print("="*80)
    print(f"YouTube Video IDs in: {h5_path}")
    print("="*80 + "\n")

    with h5py.File(h5_path, 'r') as f:
        video_keys = sorted([k for k in f.keys()])

        print(f"Total videos: {len(video_keys)}\n")

        for idx, video_key in enumerate(video_keys, 1):
            # Try to get YouTube ID
            if 'video_name' in f[video_key]:
                youtube_id = f[video_key]['video_name'][()]
                if isinstance(youtube_id, bytes):
                    youtube_id = youtube_id.decode('utf-8')
            else:
                youtube_id = video_key

            # Get number of frames
            n_frames = f[video_key]['features'].shape[0] if 'features' in f[video_key] else 'N/A'

            print(f"{idx:3d}. {video_key:15s} → {youtube_id:20s} ({n_frames} frames)")

    print("\n" + "="*80)
    print(f"Total: {len(video_keys)} videos")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='List YouTube video IDs from H5 file')
    parser.add_argument('--h5-file', type=str,
                       default='datasets_preprocessed/data/h5/youtube.h5',
                       help='Path to YouTube dataset H5 file')

    args = parser.parse_args()

    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"❌ H5 file not found: {h5_path}")
        return

    list_youtube_ids(h5_path)


if __name__ == "__main__":
    main()
