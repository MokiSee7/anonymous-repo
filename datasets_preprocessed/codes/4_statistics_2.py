#!/usr/bin/env python3
"""
Step 4: Statistics for Step 2 Shot-Level Descriptions

This script generates statistics for shot-level descriptions including:
- Average number of shots per video
- Average number of frames per shot (with min/max)
- Average number of GT shots per video
- Average number of frames in GT shots
- Total GT frames percentage
- Average shot description length

Input:  datasets_preprocessed/data/processed/{dataset_name}_data.json
Output: datasets_preprocessed/data/statistics/{dataset_name}_statistic_2.json
"""

import os
import json
import argparse
from collections import defaultdict


def calculate_shot_statistics(data_file, output_file):
    """
    Calculate shot-level statistics for a single dataset

    Args:
        data_file: Path to the input JSON file (e.g., summe_data.json)
        output_file: Path to save the statistics JSON file

    Returns:
        dict: Statistics dictionary
    """
    # Load data
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return None

    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Statistics containers
    total_videos = len(dataset)
    total_shots = 0
    total_gt_shots = 0
    total_frames_in_shots = 0
    total_frames_in_gt_shots = 0
    total_description_length = 0
    total_frames = 0

    all_frames_per_shot = []
    all_gt_frames_per_shot = []

    video_stats = []

    # Process each video
    for video_id, video_data in dataset.items():
        clips = video_data.get("clips", [])
        frames = video_data.get("frames", [])

        if not clips:
            # Skip videos without shot-level data
            continue

        num_shots = len(clips)
        num_gt_shots = sum(1 for clip in clips if clip.get("gt") == 1)

        # Calculate frames per shot
        frames_per_shot = []
        gt_frames_per_shot = []
        frames_in_gt_shots = 0
        description_lengths = []

        for clip in clips:
            clip_frames = clip.get("frames", [])
            num_clip_frames = len(clip_frames)
            frames_per_shot.append(num_clip_frames)

            # Track GT shot frames
            if clip.get("gt") == 1:
                gt_frames_per_shot.append(num_clip_frames)
                frames_in_gt_shots += num_clip_frames

            # Track description length
            summary = clip.get("summary", "")
            if summary and summary != "[CONTENT_FILTERED]":
                description_lengths.append(len(summary))

        # Calculate video-level statistics
        avg_frames_per_shot = sum(frames_per_shot) / len(frames_per_shot) if frames_per_shot else 0
        min_frames_per_shot = min(frames_per_shot) if frames_per_shot else 0
        max_frames_per_shot = max(frames_per_shot) if frames_per_shot else 0

        avg_gt_frames_per_shot = sum(gt_frames_per_shot) / len(gt_frames_per_shot) if gt_frames_per_shot else 0
        avg_description_length = sum(description_lengths) / len(description_lengths) if description_lengths else 0

        total_video_frames = len(frames)
        gt_frames_percentage = (frames_in_gt_shots / total_video_frames * 100) if total_video_frames > 0 else 0

        # Update totals
        total_shots += num_shots
        total_gt_shots += num_gt_shots
        total_frames_in_shots += sum(frames_per_shot)
        total_frames_in_gt_shots += frames_in_gt_shots
        total_description_length += sum(description_lengths)
        total_frames += total_video_frames

        all_frames_per_shot.extend(frames_per_shot)
        all_gt_frames_per_shot.extend(gt_frames_per_shot)

        video_stats.append({
            "video_id": video_id,
            "num_shots": num_shots,
            "num_gt_shots": num_gt_shots,
            "avg_frames_per_shot": round(avg_frames_per_shot, 2),
            "min_frames_per_shot": min_frames_per_shot,
            "max_frames_per_shot": max_frames_per_shot,
            "avg_gt_frames_per_shot": round(avg_gt_frames_per_shot, 2),
            "total_frames_in_gt_shots": frames_in_gt_shots,
            "total_frames": total_video_frames,
            "gt_frames_percentage": round(gt_frames_percentage, 2),
            "avg_description_length": round(avg_description_length, 2)
        })

    # Calculate dataset-level averages
    num_videos_with_shots = len(video_stats)

    avg_shots_per_video = total_shots / num_videos_with_shots if num_videos_with_shots > 0 else 0
    avg_gt_shots_per_video = total_gt_shots / num_videos_with_shots if num_videos_with_shots > 0 else 0
    avg_frames_per_shot = sum(all_frames_per_shot) / len(all_frames_per_shot) if all_frames_per_shot else 0
    min_frames_per_shot = min(all_frames_per_shot) if all_frames_per_shot else 0
    max_frames_per_shot = max(all_frames_per_shot) if all_frames_per_shot else 0
    avg_gt_frames_per_shot = sum(all_gt_frames_per_shot) / len(all_gt_frames_per_shot) if all_gt_frames_per_shot else 0
    avg_total_gt_frames_per_video = total_frames_in_gt_shots / num_videos_with_shots if num_videos_with_shots > 0 else 0
    overall_gt_frames_percentage = (total_frames_in_gt_shots / total_frames * 100) if total_frames > 0 else 0
    avg_description_length = total_description_length / total_shots if total_shots > 0 else 0

    # Compile statistics
    statistics = {
        "dataset_summary": {
            "total_videos": total_videos,
            "videos_with_shots": num_videos_with_shots,
            "total_shots": total_shots,
            "total_gt_shots": total_gt_shots,
            "total_frames": total_frames,
            "total_frames_in_gt_shots": total_frames_in_gt_shots,
            "avg_shots_per_video": round(avg_shots_per_video, 2),
            "avg_gt_shots_per_video": round(avg_gt_shots_per_video, 2),
            "avg_frames_per_shot": round(avg_frames_per_shot, 2),
            "min_frames_per_shot": min_frames_per_shot,
            "max_frames_per_shot": max_frames_per_shot,
            "avg_gt_frames_per_shot": round(avg_gt_frames_per_shot, 2),
            "avg_total_gt_frames_per_video": round(avg_total_gt_frames_per_video, 2),
            "gt_frames_percentage": round(overall_gt_frames_percentage, 2),
            "avg_description_length": round(avg_description_length, 2)
        },
        "per_video_statistics": video_stats
    }

    # Save statistics
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    return statistics


def print_statistics(stats, dataset_name):
    """Print statistics in a readable format"""
    if not stats:
        return

    summary = stats["dataset_summary"]

    print(f"\n{'='*80}")
    print(f"统计结果: {dataset_name} (Shot-Level)")
    print(f"{'='*80}")
    print(f"📊 总体统计:")
    print(f"   - 视频总数:                     {summary['total_videos']}")
    print(f"   - 包含 shot 的视频数:           {summary['videos_with_shots']}")
    print(f"   - Shot 总数:                    {summary['total_shots']}")
    print(f"   - GT Shot 总数:                 {summary['total_gt_shots']}")
    print(f"   - 帧总数:                       {summary['total_frames']}")
    print(f"   - GT Shot 包含的帧总数:         {summary['total_frames_in_gt_shots']}")

    print(f"\n📈 平均统计 (Shot):")
    print(f"   - 平均每视频 shot 数:           {summary['avg_shots_per_video']:.2f}")
    print(f"   - 平均每视频 GT shot 数:        {summary['avg_gt_shots_per_video']:.2f}")
    print(f"   - 平均每 shot 包含帧数:         {summary['avg_frames_per_shot']:.2f}")
    print(f"   - 每 shot 包含帧数 (最小值):    {summary['min_frames_per_shot']}")
    print(f"   - 每 shot 包含帧数 (最大值):    {summary['max_frames_per_shot']}")

    print(f"\n📈 平均统计 (GT Shot):")
    print(f"   - 平均 GT shot 包含帧数:        {summary['avg_gt_frames_per_shot']:.2f}")
    print(f"   - 平均每视频 GT 帧总数:         {summary['avg_total_gt_frames_per_video']:.2f}")
    print(f"   - GT 帧占比:                    {summary['gt_frames_percentage']:.2f}%")

    print(f"\n📝 描述统计:")
    print(f"   - 平均 shot 描述长度 (字符数): {summary['avg_description_length']:.2f}")
    print(f"{'='*80}\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate statistics for Step 2 shot-level descriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate statistics for a specific dataset
  python 4_statistics_2.py --dataset summe

  # Generate statistics for all datasets
  python 4_statistics_2.py --all

  # Specify custom directories
  python 4_statistics_2.py --dataset tvsum --data-dir datasets_preprocessed/data/processed --output-dir datasets_preprocessed/data/statistics
        """
    )

    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name to process (e.g., summe, tvsum, ovp, youtube)')
    parser.add_argument('--all', action='store_true',
                       help='Process all datasets')
    parser.add_argument('--data-dir', type=str, default='datasets_preprocessed/data/processed',
                       help='Directory containing processed JSON files (default: datasets_preprocessed/data/processed)')
    parser.add_argument('--output-dir', type=str, default='datasets_preprocessed/data/statistics',
                       help='Directory for output statistics files (default: datasets_preprocessed/data/statistics)')

    args = parser.parse_args()

    # Validate arguments
    if not args.dataset and not args.all:
        print("❌ 请指定 --dataset 或 --all")
        print("   使用 --help 查看帮助信息")
        return

    print("="*80)
    print("Step 4: Statistics for Step 2 Shot-Level Descriptions")
    print("="*80)
    print(f"\n📂 数据目录: {args.data_dir}")
    print(f"📂 输出目录: {args.output_dir}")

    # Determine which datasets to process
    if args.all:
        # Auto-detect all JSON files
        if not os.path.exists(args.data_dir):
            print(f"❌ 数据目录不存在: {args.data_dir}")
            return

        import re
        pattern = re.compile(r'(.+)_data\.json')
        dataset_list = []

        for file in os.listdir(args.data_dir):
            match = pattern.match(file)
            if match:
                dataset_list.append(match.group(1))

        if not dataset_list:
            print(f"❌ 未检测到任何数据集文件！")
            print(f"   请确保 *_data.json 文件存在于: {args.data_dir}")
            return

        print(f"🎯 检测到数据集: {', '.join(dataset_list)}")
    else:
        # Normalize dataset name to lowercase
        dataset_list = [args.dataset.lower()]
        print(f"🎯 目标数据集: {args.dataset.lower()}")

    print("="*80)

    # Process each dataset
    success_count = 0
    for dataset in dataset_list:
        print(f"\n🎬 处理数据集: {dataset}")
        print("-"*80)

        data_file = os.path.join(args.data_dir, f"{dataset}_data.json")
        output_file = os.path.join(args.output_dir, f"{dataset}_statistic_2.json")

        print(f"📂 输入文件: {data_file}")
        print(f"📂 输出文件: {output_file}")

        stats = calculate_shot_statistics(data_file, output_file)

        if stats:
            print(f"✅ 统计信息已生成: {output_file}")
            print_statistics(stats, dataset)
            success_count += 1
        else:
            print(f"❌ 处理失败: {dataset}")

        print("-"*80)

    print("\n" + "="*80)
    print(f"✅ 完成! 成功处理 {success_count}/{len(dataset_list)} 个数据集")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
