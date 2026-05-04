#!/usr/bin/env python3
"""
Generate gtframe and gtsummary fields for H5 datasets

gtframe (per video):
  1. Read user_summary (n_users, n_frames_or_n_steps)
  2. Project to sampled-frame level if needed (via picks)
  3. Find the user with the most frames marked as 1 → k
  4. Select top-k gtscore sampled frames, mark as 1, rest as 0
  5. Write result to gtframe field (n_steps,)

gtsummary (per video):
  1. Read change_points, shot_level_gt, gtscore
  2. For each GT shot (shot_level_gt == 1), find the sampled frame
     with the highest gtscore within that shot
  3. Mark that frame as 1, all other sampled frames as 0
  4. Write result to gtsummary field (n_steps,)

Usage:
    python 0_generate_gtframe.py --h5 /data/MMS_Benchmark/data/summe/h5/summe.h5
    python 0_generate_gtframe.py --all   # Process all datasets
    python 0_generate_gtframe.py --all --force  # Overwrite existing fields
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm


def generate_gtframe_for_h5(h5_path, force=False):
    """
    Generate gtframe field for all videos in an H5 file

    Args:
        h5_path: Path to H5 file
        force: Overwrite existing gtframe fields
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"H5 file not found: {h5_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"Generating gtframe: {h5_path.name}")
    print(f"{'=' * 60}")

    stats = {'processed': 0, 'skipped': 0, 'no_user_summary': 0, 'errors': 0}

    with h5py.File(h5_path, 'a') as f:
        video_keys = sorted(f.keys())
        print(f"Total videos: {len(video_keys)}\n")

        for video_key in tqdm(video_keys, desc="Processing"):
            group = f[video_key]

            # Skip if gtframe already exists and not forcing
            if 'gtframe' in group and not force:
                stats['skipped'] += 1
                continue

            # Check required fields
            if 'user_summary' not in group:
                stats['no_user_summary'] += 1
                continue

            if 'gtscore' not in group:
                tqdm.write(f"  {video_key}: no gtscore, skipping")
                stats['errors'] += 1
                continue

            try:
                user_summary = group['user_summary'][:]
                gtscore = group['gtscore'][:]
                n_steps = len(gtscore)

                # Project user_summary to sampled-frame level if needed
                if user_summary.shape[1] != n_steps:
                    if 'picks' in group:
                        picks = group['picks'][:]
                        # Clip picks to valid range
                        valid = picks < user_summary.shape[1]
                        user_summary_sampled = user_summary[:, picks[valid]]
                        # Pad if some picks were invalid
                        if user_summary_sampled.shape[1] < n_steps:
                            pad = np.zeros(
                                (user_summary.shape[0],
                                 n_steps - user_summary_sampled.shape[1]),
                                dtype=user_summary.dtype)
                            user_summary_sampled = np.concatenate(
                                [user_summary_sampled, pad], axis=1)
                    else:
                        tqdm.write(
                            f"  {video_key}: user_summary size mismatch "
                            f"({user_summary.shape[1]} vs {n_steps}) "
                            f"and no picks, skipping")
                        stats['errors'] += 1
                        continue
                else:
                    user_summary_sampled = user_summary

                # Use fixed 15% proportion to determine k
                k = max(1, round(n_steps * 0.15))

                # Cap k to n_steps
                k = min(k, n_steps)

                # Step 2: Select top-k gtscore frames
                gtframe = np.zeros(n_steps, dtype=np.float32)
                if k > 0:
                    top_k_indices = np.argsort(gtscore)[::-1][:k]
                    gtframe[top_k_indices] = 1.0

                # Write to H5
                if 'gtframe' in group:
                    del group['gtframe']
                group.create_dataset('gtframe', data=gtframe, dtype='float32')

                stats['processed'] += 1

            except Exception as e:
                tqdm.write(f"  {video_key}: error - {e}")
                stats['errors'] += 1

    print(f"\ngtframe done: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, "
          f"{stats['no_user_summary']} no user_summary, "
          f"{stats['errors']} errors")


def generate_gtsummary_for_h5(h5_path, force=False):
    """
    Generate gtsummary field for all videos in an H5 file

    For each GT shot, mark the sampled frame with the highest gtscore as 1.
    change_points may be at original-frame level or sampled-frame level;
    we use picks to map when needed.

    Args:
        h5_path: Path to H5 file
        force: Overwrite existing gtsummary fields
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"H5 file not found: {h5_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"Generating gtsummary: {h5_path.name}")
    print(f"{'=' * 60}")

    stats = {'processed': 0, 'skipped': 0, 'missing_fields': 0, 'errors': 0}

    with h5py.File(h5_path, 'a') as f:
        video_keys = sorted(f.keys())
        print(f"Total videos: {len(video_keys)}\n")

        for video_key in tqdm(video_keys, desc="Processing"):
            group = f[video_key]

            # Skip if gtsummary already exists and not forcing
            if 'gtsummary' in group and not force:
                stats['skipped'] += 1
                continue

            # Check required fields
            required = ['change_points', 'shot_level_gt', 'gtscore']
            missing = [r for r in required if r not in group]
            if missing:
                stats['missing_fields'] += 1
                continue

            try:
                change_points = group['change_points'][:]
                shot_level_gt = group['shot_level_gt'][:]
                gtscore = group['gtscore'][:]
                n_steps = len(gtscore)
                n_segments = len(change_points)

                # Determine if change_points are at original-frame level
                # or sampled-frame level by comparing max value with n_steps
                cp_max = int(change_points.max()) if n_segments > 0 else 0
                cp_at_original_level = cp_max >= n_steps and n_steps > 1

                # If at original-frame level, map to sampled-frame indices
                if cp_at_original_level and 'picks' in group:
                    picks = group['picks'][:]

                    gtsummary = np.zeros(n_steps, dtype=np.float32)
                    for i in range(n_segments):
                        if shot_level_gt[i] <= 0:
                            continue
                        cp_start = int(change_points[i, 0])
                        cp_end = int(change_points[i, 1])

                        # Find sampled frames within this shot
                        in_shot = np.where(
                            (picks >= cp_start) & (picks <= cp_end)
                        )[0]

                        if len(in_shot) == 0:
                            continue

                        # Find the one with highest gtscore
                        best = in_shot[np.argmax(gtscore[in_shot])]
                        gtsummary[best] = 1.0
                else:
                    # change_points already at sampled-frame level
                    gtsummary = np.zeros(n_steps, dtype=np.float32)
                    for i in range(n_segments):
                        if shot_level_gt[i] <= 0:
                            continue
                        start = int(change_points[i, 0])
                        end = int(change_points[i, 1])
                        if start > end or end >= n_steps:
                            continue

                        shot_scores = gtscore[start:end + 1]
                        best = start + int(np.argmax(shot_scores))
                        gtsummary[best] = 1.0

                # Write to H5
                if 'gtsummary' in group:
                    del group['gtsummary']
                group.create_dataset(
                    'gtsummary', data=gtsummary, dtype='float32')

                stats['processed'] += 1

            except Exception as e:
                tqdm.write(f"  {video_key}: error - {e}")
                stats['errors'] += 1

    print(f"\ngtsummary done: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, "
          f"{stats['missing_fields']} missing fields, "
          f"{stats['errors']} errors")


def update_json_from_gtframe(h5_path, json_path):
    """
    Update the is_gt field in a processed JSON file using gtframe from H5.

    The JSON structure is:
        {video_id: {frames: [{frame_id, description, is_gt, status}, ...]}}

    Frames are ordered by picks, so their index in the list corresponds
    directly to the index in the H5 gtframe array.

    Args:
        h5_path: Path to H5 file containing gtframe
        json_path: Path to the processed JSON file to update
    """
    import json

    json_path = Path(json_path)
    if not json_path.exists():
        print(f"  JSON file not found: {json_path}, skipping")
        return

    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"  H5 file not found: {h5_path}, skipping")
        return

    print(f"\n{'=' * 60}")
    print(f"Updating JSON is_gt from gtframe: {json_path.name}")
    print(f"{'=' * 60}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = 0
    skipped = 0

    with h5py.File(h5_path, 'r') as h5file:
        for video_id, video_data in data.items():
            if video_id not in h5file:
                skipped += 1
                continue

            group = h5file[video_id]
            if 'gtframe' not in group:
                skipped += 1
                continue

            gtframe = group['gtframe'][:]
            frames = video_data.get('frames', [])

            for idx, frame in enumerate(frames):
                if idx < len(gtframe):
                    frame['is_gt'] = int(gtframe[idx])

            updated += 1

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  Updated {updated} videos, skipped {skipped}")
    print(f"  Saved: {json_path}")


def main():
    all_datasets = {
        'summe': {
            'h5': '/data/MMS_Benchmark/data/summe/h5/summe.h5',
            'json': '/data/MMS_Benchmark/data/processed/summe_data.json',
        },
        'tvsum': {
            'h5': '/data/MMS_Benchmark/data/tvsum/h5/tvsum.h5',
            'json': '/data/MMS_Benchmark/data/processed/tvsum_data.json',
        },
        'ovp': {
            'h5': '/data/MMS_Benchmark/data/ovp/h5/ovp.h5',
            'json': '/data/MMS_Benchmark/data/processed/ovp_data.json',
        },
        'youtube': {
            'h5': '/data/MMS_Benchmark/data/youtube/h5/youtube.h5',
            'json': '/data/MMS_Benchmark/data/processed/youtube_data.json',
        },
        'videoxum': {
            'h5': '/data/MMS_Benchmark/data/videoxum/h5/videoxum.h5',
            'json': '/data/MMS_Benchmark/data/processed/videoxum_data.json',
        },
        'mrhisum': {
            'h5': '/data/MMS_Benchmark/data/mrhisum/h5/mrhisum.h5',
            'json': '/data/MMS_Benchmark/data/processed/mrhisum_data.json',
        },
    }

    parser = argparse.ArgumentParser(
        description='Generate gtframe and gtsummary fields for H5 datasets')
    for name in all_datasets:
        parser.add_argument(f'--{name}', action='store_true',
                            help=f'Process {name} dataset')
    parser.add_argument('--all', action='store_true',
                        help='Process all datasets')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing fields')

    args = parser.parse_args()

    if args.all:
        selected = list(all_datasets.values())
    else:
        selected = [all_datasets[name] for name in all_datasets
                    if getattr(args, name)]

    if not selected:
        parser.print_help()
        return

    for ds in selected:
        generate_gtframe_for_h5(ds['h5'], force=args.force)
        generate_gtsummary_for_h5(ds['h5'], force=args.force)
        update_json_from_gtframe(ds['h5'], ds['json'])


if __name__ == "__main__":
    main()
