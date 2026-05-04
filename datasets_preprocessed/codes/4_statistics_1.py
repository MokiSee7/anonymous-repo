#!/usr/bin/env python3
"""
Step 4: Statistics for Step 1 Frame-Level Descriptions

This script generates statistics for frame-level descriptions including:
- Average number of frames per video
- Average number of GT (ground truth) frames per video
- Average number of filtered frames per video

Input:  datasets_preprocessed/data/processed/{dataset_name}_data.json
Output: datasets_preprocessed/data/statistics/{dataset_name}_statistic_1.json
"""

import os
import json
import argparse
from collections import defaultdict


def calculate_statistics(data_file, output_file):
    """
    Calculate statistics for a single dataset

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
    total_frames = 0
    total_gt_frames = 0
    total_filtered_frames = 0

    video_stats = []

    # Process each video
    for video_id, video_data in dataset.items():
        frames = video_data.get("frames", [])

        num_frames = len(frames)
        num_gt_frames = sum(1 for frame in frames if frame.get("is_gt") == 1)
        num_filtered_frames = sum(1 for frame in frames if frame.get("status") == 1)

        total_frames += num_frames
        total_gt_frames += num_gt_frames
        total_filtered_frames += num_filtered_frames

        video_stats.append({
            "video_id": video_id,
            "num_frames": num_frames,
            "num_gt_frames": num_gt_frames,
            "num_filtered_frames": num_filtered_frames
        })

    # Calculate averages
    avg_frames_per_video = total_frames / total_videos if total_videos > 0 else 0
    avg_gt_frames_per_video = total_gt_frames / total_videos if total_videos > 0 else 0
    avg_filtered_frames_per_video = total_filtered_frames / total_videos if total_videos > 0 else 0

    # Compile statistics
    statistics = {
        "dataset_summary": {
            "total_videos": total_videos,
            "total_frames": total_frames,
            "total_gt_frames": total_gt_frames,
            "total_filtered_frames": total_filtered_frames,
            "avg_frames_per_video": round(avg_frames_per_video, 2),
            "avg_gt_frames_per_video": round(avg_gt_frames_per_video, 2),
            "avg_filtered_frames_per_video": round(avg_filtered_frames_per_video, 2),
            "filtered_percentage": round((total_filtered_frames / total_frames * 100) if total_frames > 0 else 0, 2)
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
    print(f"统计结果: {dataset_name}")
    print(f"{'='*80}")
    print(f"📊 总体统计:")
    print(f"   - 视频总数:           {summary['total_videos']}")
    print(f"   - 帧总数:             {summary['total_frames']}")
    print(f"   - GT 帧总数:          {summary['total_gt_frames']}")
    print(f"   - 过滤帧总数:         {summary['total_filtered_frames']}")
    print(f"\n📈 平均统计:")
    print(f"   - 平均每视频帧数:     {summary['avg_frames_per_video']:.2f}")
    print(f"   - 平均每视频 GT 帧数: {summary['avg_gt_frames_per_video']:.2f}")
    print(f"   - 平均每视频过滤帧数: {summary['avg_filtered_frames_per_video']:.2f}")
    print(f"   - 过滤帧比例:         {summary['filtered_percentage']:.2f}%")
    print(f"{'='*80}\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate statistics for Step 1 frame-level descriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate statistics for a specific dataset
  python 4_statistics_1.py --dataset summe

  # Generate statistics for all datasets
  python 4_statistics_1.py --all

  # Specify custom directories
  python 4_statistics_1.py --dataset tvsum --data-dir datasets_preprocessed/data/processed --output-dir datasets_preprocessed/data/statistics
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
    print("Step 4: Statistics for Step 1 Frame-Level Descriptions")
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
        output_file = os.path.join(args.output_dir, f"{dataset}_statistic_1.json")

        print(f"📂 输入文件: {data_file}")
        print(f"📂 输出文件: {output_file}")

        stats = calculate_statistics(data_file, output_file)

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
