#!/usr/bin/env python3
"""
Parallel Frame Extraction for VideoXum Dataset
支持多进程并行、边下载边抽帧、抽帧后自动删除视频

Features:
- 多进程并行处理，充分利用CPU资源
- 支持边下载边抽帧（监控视频目录）
- 抽帧成功后自动删除视频文件
- 支持断点续传
- 可选使用ffmpeg加速抽帧（比OpenCV更快）

Usage:
    # 基础用法：多进程并行抽帧
    python 0_extract_videoxum_frames_parallel.py --workers 8

    # 边下载边抽帧模式（监控模式）
    python 0_extract_videoxum_frames_parallel.py --watch --workers 8 --delete-after

    # 使用ffmpeg加速（推荐）
    python 0_extract_videoxum_frames_parallel.py --workers 8 --use-ffmpeg

    # 测试模式
    python 0_extract_videoxum_frames_parallel.py --test --workers 2
"""

import h5py
import cv2
import numpy as np
import argparse
import os
import time
import subprocess
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock, cpu_count
from queue import Queue
from typing import List, Tuple, Optional
import json


class VideoFrameExtractor:
    """视频抽帧器基类"""

    def __init__(self, delete_after: bool = False):
        self.delete_after = delete_after

    def extract_frame_opencv(self, video_path: Path, frame_idx: int, output_path: Path) -> bool:
        """使用OpenCV提取单帧"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        return True

    def extract_frame_ffmpeg(self, video_path: Path, frame_idx: int, output_path: Path) -> bool:
        """使用ffmpeg提取单帧（更快）"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用ffmpeg的select过滤器精确提取指定帧
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'select=eq(n\\,{frame_idx})',
            '-vframes', '1',
            '-y',  # 覆盖已存在的文件
            '-loglevel', 'error',  # 只显示错误
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            return output_path.exists()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def extract_frames_batch_ffmpeg(self, video_path: Path, picks: List[int],
                                   output_dir: Path) -> Tuple[int, int]:
        """使用ffmpeg批量提取帧（最快的方法）"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取视频FPS
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        success_count = 0
        fail_count = 0

        # 按时间戳批量提取帧
        for frame_idx in picks:
            output_path = output_dir / f"frame_{frame_idx:06d}.jpg"

            # 跳过已存在的帧
            if output_path.exists():
                success_count += 1
                continue

            # 计算时间戳
            timestamp = frame_idx / fps

            # 使用ffmpeg的seek功能快速定位
            cmd = [
                'ffmpeg',
                '-ss', str(timestamp),  # 快速seek到指定时间
                '-i', str(video_path),
                '-vframes', '1',
                '-q:v', '2',  # 高质量JPEG
                '-y',
                '-loglevel', 'error',
                str(output_path)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=10)
                if output_path.exists():
                    success_count += 1
                else:
                    fail_count += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                fail_count += 1

        return success_count, fail_count


def process_single_video(args):
    """
    处理单个视频（worker进程调用）

    Args:
        args: (video_key, video_info, video_dir, output_dir, use_ffmpeg, delete_after)

    Returns:
        dict: 处理结果统计
    """
    video_key, video_info, video_dir, output_dir, use_ffmpeg, delete_after = args

    result = {
        'video_key': video_key,
        'success': False,
        'frames_extracted': 0,
        'frames_failed': 0,
        'n_frames_total': 0,
        'deleted': False,
        'error': None
    }

    try:
        video_name = video_info['video_name']
        picks = video_info['picks']

        # 查找视频文件
        video_path = Path(video_dir) / f"{video_name}.mp4"
        if not video_path.exists():
            result['error'] = f"Video not found: {video_path}"
            return result

        # 获取原始视频总帧数
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            result['n_frames_total'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        # 创建输出目录
        video_output_dir = Path(output_dir) / video_key
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否已经完成
        existing_frames = list(video_output_dir.glob("frame_*.jpg"))
        if len(existing_frames) == len(picks):
            result['success'] = True
            result['frames_extracted'] = len(existing_frames)
            # 如果已完成且需要删除，删除视频
            if delete_after and video_path.exists():
                video_path.unlink()
                result['deleted'] = True
            return result

        # 初始化提取器
        extractor = VideoFrameExtractor(delete_after=delete_after)

        # 提取帧
        if use_ffmpeg:
            # 使用ffmpeg批量提取
            success_count, fail_count = extractor.extract_frames_batch_ffmpeg(
                video_path, picks, video_output_dir
            )
        else:
            # 使用OpenCV逐帧提取
            success_count = 0
            fail_count = 0
            for frame_idx in picks:
                output_path = video_output_dir / f"frame_{frame_idx:06d}.jpg"
                if output_path.exists():
                    success_count += 1
                    continue

                if extractor.extract_frame_opencv(video_path, frame_idx, output_path):
                    success_count += 1
                else:
                    fail_count += 1

        result['frames_extracted'] = success_count
        result['frames_failed'] = fail_count
        result['success'] = (success_count > 0)

        # 抽帧成功后删除视频
        if delete_after and video_path.exists() and success_count == len(picks):
            try:
                video_path.unlink()
                result['deleted'] = True
            except Exception as e:
                result['error'] = f"Failed to delete video: {e}"

    except Exception as e:
        result['error'] = str(e)

    return result


class ParallelVideoExtractor:
    """并行视频抽帧器"""

    def __init__(self, h5_path: Path, video_dir: Path, output_dir: Path,
                 workers: int = 4, use_ffmpeg: bool = False,
                 delete_after: bool = False, watch_mode: bool = False):
        self.h5_path = h5_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.workers = workers
        self.use_ffmpeg = use_ffmpeg
        self.delete_after = delete_after
        self.watch_mode = watch_mode

        # 进度文件
        self.progress_file = output_dir / '.extraction_progress.json'
        self.completed_videos = self.load_progress()

    def load_progress(self) -> set:
        """加载已完成的视频列表（断点续传）"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed', []))
            except:
                pass
        return set()

    def save_progress(self):
        """保存进度"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump({
                'completed': list(self.completed_videos),
                'timestamp': time.time()
            }, f)

    def load_video_info_from_h5(self, video_keys: Optional[List[str]] = None) -> dict:
        """从H5文件加载视频信息"""
        video_info_dict = {}

        with h5py.File(self.h5_path, 'r') as h5_file:
            keys = video_keys if video_keys else sorted(h5_file.keys())

            for video_key in keys:
                # 跳过已完成的视频
                if video_key in self.completed_videos:
                    continue

                try:
                    video = h5_file[video_key]
                    video_info_dict[video_key] = {
                        'video_name': video['video_name'][()].decode(),
                        'picks': video['picks'][:].tolist(),
                        'n_frames': int(video['n_frames'][()])
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {video_key}: {e}")

        return video_info_dict

    def _write_n_frames_to_h5(self, results: list):
        """批量写入 n_frames_original（原始视频总帧数）到 H5"""
        to_write = [r for r in results if r['success'] and r['n_frames_total'] > 0]
        if not to_write:
            return

        try:
            with h5py.File(self.h5_path, 'a') as h5_file:
                for result in to_write:
                    video_key = result['video_key']
                    if video_key not in h5_file:
                        continue
                    video_group = h5_file[video_key]
                    if 'n_frames_original' in video_group:
                        video_group['n_frames_original'][()] = result['n_frames_total']
                    else:
                        video_group.create_dataset(
                            'n_frames_original',
                            data=np.int64(result['n_frames_total'])
                        )
            print(f"   H5 updated: {len(to_write)} videos (n_frames_original)")
        except Exception as e:
            print(f"   Failed to write H5: {e}")

    def extract_parallel(self, test_mode: bool = False):
        """并行提取帧"""
        print(f"\n{'='*80}")
        print(f"Parallel VideoXum Frame Extraction")
        print(f"{'='*80}")
        print(f"H5 file: {self.h5_path}")
        print(f"Video directory: {self.video_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Workers: {self.workers}")
        print(f"Use FFmpeg: {self.use_ffmpeg}")
        print(f"Delete after: {self.delete_after}")
        print(f"Watch mode: {self.watch_mode}")
        print(f"Completed videos: {len(self.completed_videos)}")
        if test_mode:
            print("⚠️  Test mode: only processing first 3 videos")
        print(f"{'='*80}\n")

        # 加载视频信息
        video_info_dict = self.load_video_info_from_h5()

        if test_mode:
            video_info_dict = dict(list(video_info_dict.items())[:3])

        if not video_info_dict:
            print("✅ All videos already processed!")
            return

        print(f"Videos to process: {len(video_info_dict)}\n")

        # 准备任务参数
        tasks = [
            (video_key, video_info, self.video_dir, self.output_dir,
             self.use_ffmpeg, self.delete_after)
            for video_key, video_info in video_info_dict.items()
        ]

        # 统计信息
        stats = {
            'total_processed': 0,
            'total_frames': 0,
            'total_failed': 0,
            'videos_deleted': 0,
            'errors': []
        }

        # 收集结果用于批量写入 H5
        batch_results = []
        h5_write_interval = 100

        # 使用进程池并行处理
        with Pool(processes=self.workers) as pool:
            # 使用imap_unordered以便结果一完成就处理
            results = pool.imap_unordered(process_single_video, tasks)

            # 显示进度条
            with tqdm(total=len(tasks), desc="Processing videos") as pbar:
                for result in results:
                    stats['total_processed'] += 1
                    stats['total_frames'] += result['frames_extracted']
                    stats['total_failed'] += result['frames_failed']

                    if result['deleted']:
                        stats['videos_deleted'] += 1

                    if result['error']:
                        stats['errors'].append((result['video_key'], result['error']))

                    if result['success']:
                        self.completed_videos.add(result['video_key'])
                        batch_results.append(result)

                        # 定期保存进度和写入 H5
                        if len(batch_results) % h5_write_interval == 0:
                            self._write_n_frames_to_h5(batch_results)
                            batch_results.clear()
                            self.save_progress()

                    pbar.update(1)
                    pbar.set_postfix({
                        'frames': stats['total_frames'],
                        'deleted': stats['videos_deleted']
                    })

        # 写入剩余的 H5 数据
        if batch_results:
            self._write_n_frames_to_h5(batch_results)

        # 保存最终进度
        self.save_progress()

        # 打印统计信息
        self.print_stats(stats)

    def watch_and_extract(self, check_interval: int = 60):
        """监控模式：持续监控视频目录，发现新视频立即处理"""
        print(f"\n{'='*80}")
        print(f"Watch Mode - Continuous Frame Extraction")
        print(f"{'='*80}")
        print(f"Monitoring directory: {self.video_dir}")
        print(f"Check interval: {check_interval}s")
        print(f"Workers: {self.workers}")
        print(f"Delete after extraction: {self.delete_after}")
        print(f"{'='*80}\n")

        processed_videos = set()

        try:
            while True:
                # 查找视频目录中的所有视频
                available_videos = set()
                if self.video_dir.exists():
                    for video_file in self.video_dir.glob("v_*.mp4"):
                        video_name = video_file.stem
                        available_videos.add(video_name)

                # 找出新下载的视频
                new_videos = available_videos - processed_videos

                if new_videos:
                    print(f"\n🔍 Found {len(new_videos)} new video(s)")

                    # 从H5文件中查找对应的视频信息
                    video_info_dict = self.load_video_info_from_h5()

                    # 筛选出新视频
                    videos_to_process = {
                        k: v for k, v in video_info_dict.items()
                        if v['video_name'] in new_videos
                    }

                    if videos_to_process:
                        print(f"📝 Processing {len(videos_to_process)} video(s)...")

                        # 准备任务
                        tasks = [
                            (video_key, video_info, self.video_dir, self.output_dir,
                             self.use_ffmpeg, self.delete_after)
                            for video_key, video_info in videos_to_process.items()
                        ]

                        # 并行处理
                        with Pool(processes=self.workers) as pool:
                            results = pool.map(process_single_video, tasks)

                        # 统计结果
                        for result in results:
                            if result['success']:
                                processed_videos.add(video_info_dict[result['video_key']]['video_name'])
                                self.completed_videos.add(result['video_key'])
                                print(f"  ✅ {result['video_key']}: "
                                      f"{result['frames_extracted']} frames extracted"
                                      f"{' (deleted)' if result['deleted'] else ''}")
                            else:
                                print(f"  ❌ {result['video_key']}: {result.get('error', 'Unknown error')}")

                        self.save_progress()

                # 等待下一次检查
                print(f"\n⏳ Waiting {check_interval}s for next check... "
                      f"(Processed: {len(processed_videos)} videos)")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\n⚠️  Stopping watch mode...")
            self.save_progress()
            print("✅ Progress saved!")

    def print_stats(self, stats: dict):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"📊 Extraction Complete!")
        print(f"{'='*80}")
        print(f"Videos processed: {stats['total_processed']}")
        print(f"Total frames extracted: {stats['total_frames']:,}")
        print(f"Failed frames: {stats['total_failed']}")

        if self.delete_after:
            print(f"Videos deleted: {stats['videos_deleted']}")

        if stats['errors']:
            print(f"\n❌ Errors ({len(stats['errors'])}):")
            for video_key, error in stats['errors'][:10]:
                print(f"   - {video_key}: {error}")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel frame extraction for VideoXum dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用8个进程并行抽帧
  python 0_extract_videoxum_frames_parallel.py --workers 8

  # 使用ffmpeg加速（推荐）
  python 0_extract_videoxum_frames_parallel.py --workers 8 --use-ffmpeg

  # 边下载边抽帧模式（监控模式）
  python 0_extract_videoxum_frames_parallel.py --watch --workers 8 --delete-after

  # 抽帧后自动删除视频（节省磁盘空间）
  python 0_extract_videoxum_frames_parallel.py --workers 8 --delete-after

  # 测试模式
  python 0_extract_videoxum_frames_parallel.py --test --workers 2

Performance Tips:
  - 推荐worker数 = CPU核心数
  - 使用--use-ffmpeg可以提速2-3倍
  - 使用--delete-after节省磁盘空间
  - 监控模式适合边下载边处理的场景
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
    parser.add_argument('--workers', type=int,
                       default=cpu_count(),
                       help=f'Number of parallel workers (default: {cpu_count()})')
    parser.add_argument('--use-ffmpeg', action='store_true',
                       help='Use ffmpeg for faster extraction (recommended)')
    parser.add_argument('--delete-after', action='store_true',
                       help='Delete video after successful frame extraction')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode: continuously monitor and process new videos')
    parser.add_argument('--check-interval', type=int, default=60,
                       help='Check interval in watch mode (seconds, default: 60)')

    args = parser.parse_args()

    # 验证路径
    h5_path = Path(args.h5)
    video_dir = Path(args.videos)
    output_dir = Path(args.output)

    if not h5_path.exists():
        print(f"❌ H5 file not found: {h5_path}")
        return

    if not video_dir.exists() and not args.watch:
        print(f"❌ Video directory not found: {video_dir}")
        return

    # 创建提取器
    extractor = ParallelVideoExtractor(
        h5_path=h5_path,
        video_dir=video_dir,
        output_dir=output_dir,
        workers=args.workers,
        use_ffmpeg=args.use_ffmpeg,
        delete_after=args.delete_after,
        watch_mode=args.watch
    )

    # 运行提取
    if args.watch:
        extractor.watch_and_extract(check_interval=args.check_interval)
    else:
        extractor.extract_parallel(test_mode=args.test)


if __name__ == "__main__":
    main()
