#!/usr/bin/env python3
"""
Parallel Frame Extraction for MrHiSum Dataset
支持多进程并行、FFmpeg加速、抽帧后删视频、断点续传、tar.gz压缩存储
自动计算 picks 并写入 H5 文件（video_name, picks, n_frames）

Features:
- 多进程并行处理，充分利用CPU资源
- 可选使用FFmpeg加速抽帧（比OpenCV更快）
- 抽帧成功后自动删除视频文件（可选）
- 断点续传（基于 tar.gz 是否存在 + checkpoint 文件）
- 帧压缩为 tar.gz 节省磁盘空间
- 自动从 gtscore 长度推算 picks，写入 H5
- 统计原始视频总帧数，写入 H5 的 n_frames 字段

Usage:
    # 基础用法：多进程并行抽帧
    python 0_extract_mrhisum_frames.py --workers 8

    # 使用FFmpeg加速（推荐）
    python 0_extract_mrhisum_frames.py --workers 8 --use-ffmpeg

    # 抽帧后删除视频
    python 0_extract_mrhisum_frames.py --workers 8 --delete-after

    # 测试模式
    python 0_extract_mrhisum_frames.py --test --workers 2
"""

import h5py
import cv2
import numpy as np
import argparse
import os
import time
import subprocess
import tarfile
import shutil
import json
import traceback
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, Dict


def extract_frames_opencv(video_path: Path, picks: List[int],
                          output_dir: Path) -> Tuple[int, int]:
    """使用OpenCV逐帧提取"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, len(picks)

    success_count = 0
    fail_count = 0

    for frame_idx in picks:
        output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        if output_path.exists():
            success_count += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            success_count += 1
        else:
            fail_count += 1

    cap.release()
    return success_count, fail_count


def extract_frames_ffmpeg(video_path: Path, picks: List[int],
                          output_dir: Path) -> Tuple[int, int]:
    """使用FFmpeg按时间戳提取帧（更快）"""
    # 获取视频FPS
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, len(picks)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        # Fallback to OpenCV
        return extract_frames_opencv(video_path, picks, output_dir)

    success_count = 0
    fail_count = 0

    for frame_idx in picks:
        output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        if output_path.exists():
            success_count += 1
            continue

        timestamp = frame_idx / fps
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-vframes', '1',
            '-q:v', '2',
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


def get_video_info(video_path: Path) -> Tuple[int, float]:
    """
    获取视频的总帧数和FPS

    Returns:
        (total_frames, fps)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps


def compute_picks(n_frames_total: int, n_sampled: int) -> np.ndarray:
    """
    计算均匀采样的帧索引

    Args:
        n_frames_total: 原始视频总帧数
        n_sampled: 需要采样的帧数（= len(gtscore)）

    Returns:
        picks array with original frame indices
    """
    if n_sampled >= n_frames_total:
        return np.arange(n_frames_total)
    return np.round(np.linspace(0, n_frames_total - 1, n_sampled)).astype(int)


def process_single_video(args) -> Dict:
    """
    处理单个视频（worker进程调用）

    Args:
        args: (video_key, video_info, video_dir, output_dir, use_ffmpeg, delete_after)

    Returns:
        dict: 处理结果
    """
    video_key, video_info, video_dir, output_dir, use_ffmpeg, delete_after = args

    result = {
        'video_key': video_key,
        'success': False,
        'frames_extracted': 0,
        'frames_failed': 0,
        'n_frames_total': 0,
        'picks': None,
        'video_name': None,
        'deleted': False,
        'error': None
    }

    try:
        n_gtscore = video_info['n_gtscore']

        # 查找视频文件（video_key 就是文件名，如 video_1000）
        video_path = Path(video_dir) / f"{video_key}.mp4"
        if not video_path.exists():
            result['error'] = f"Video not found: {video_path}"
            return result

        # 获取视频总帧数
        n_frames_total, fps = get_video_info(video_path)
        if n_frames_total <= 0:
            result['error'] = f"Cannot read video info: {video_path}"
            return result

        result['n_frames_total'] = n_frames_total
        result['video_name'] = video_key

        # 计算 picks
        picks = compute_picks(n_frames_total, n_gtscore)
        result['picks'] = picks.tolist()

        # 输出 tar.gz 路径
        tar_path = Path(output_dir) / f"{video_key}.tar.gz"

        # 检查是否已完成
        if tar_path.exists() and tar_path.stat().st_size > 0:
            result['success'] = True
            result['frames_extracted'] = len(picks)
            # 如果已完成且需要删除视频
            if delete_after and video_path.exists():
                video_path.unlink()
                result['deleted'] = True
            return result

        # 创建临时帧目录
        temp_frames_dir = Path(video_dir) / f"{video_key}_frames_temp"
        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)
        temp_frames_dir.mkdir(parents=True, exist_ok=True)

        # 抽帧
        if use_ffmpeg:
            success_count, fail_count = extract_frames_ffmpeg(
                video_path, picks.tolist(), temp_frames_dir
            )
        else:
            success_count, fail_count = extract_frames_opencv(
                video_path, picks.tolist(), temp_frames_dir
            )

        result['frames_extracted'] = success_count
        result['frames_failed'] = fail_count

        # 压缩为 tar.gz
        if success_count > 0:
            with tarfile.open(tar_path, 'w:gz') as tar:
                for frame_file in sorted(temp_frames_dir.glob("frame_*.jpg")):
                    tar.add(frame_file, arcname=frame_file.name)
            result['success'] = True

        # 清理临时帧目录
        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)

        # 抽帧成功后删除视频
        if delete_after and video_path.exists() and success_count == len(picks):
            try:
                video_path.unlink()
                result['deleted'] = True
            except Exception as e:
                result['error'] = f"Failed to delete video: {e}"

    except Exception as e:
        result['error'] = str(e)
        # 清理临时目录
        temp_dir = Path(video_dir) / f"{video_key}_frames_temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return result


class MrHiSumFrameExtractor:
    """MrHiSum 并行帧提取器"""

    def __init__(self, h5_path: Path, video_dir: Path, output_dir: Path,
                 workers: int = 4, use_ffmpeg: bool = False,
                 delete_after: bool = False):
        self.h5_path = h5_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.workers = workers
        self.use_ffmpeg = use_ffmpeg
        self.delete_after = delete_after

        # 进度文件
        self.progress_file = output_dir / '.extraction_progress.json'
        self.completed_videos = self._load_progress()

    def _load_progress(self) -> set:
        """从已存在的 tar.gz 和 checkpoint 文件加载进度"""
        completed = set()

        # 从 tar.gz 文件扫描
        if self.output_dir.exists():
            for tar_file in self.output_dir.glob("*.tar.gz"):
                if tar_file.stat().st_size > 0:
                    # video_key.tar.gz -> video_key
                    video_key = tar_file.name.replace('.tar.gz', '')
                    completed.add(video_key)

        # 从 checkpoint 文件补充
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    completed.update(data.get('completed', []))
            except (json.JSONDecodeError, KeyError):
                pass

        return completed

    def _save_progress(self):
        """保存进度"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump({
                'completed': sorted(self.completed_videos),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_completed': len(self.completed_videos)
            }, f, indent=2)

    def _load_video_info_from_h5(self, test_mode: bool = False,
                                  start_idx: int = None,
                                  end_idx: int = None) -> Dict:
        """
        从H5文件加载视频信息
        MrHiSum H5 字段：change_points, gt_summary, gtscore
        picks 由脚本根据 gtscore 长度和视频总帧数计算
        """
        video_info_dict = {}

        with h5py.File(self.h5_path, 'r') as h5_file:
            all_keys = sorted(h5_file.keys())

            # 确定处理范围
            if test_mode:
                all_keys = all_keys[:5]
            elif start_idx is not None or end_idx is not None:
                start = start_idx if start_idx is not None else 0
                end = end_idx if end_idx is not None else len(all_keys)
                all_keys = all_keys[start:end]

            for video_key in all_keys:
                # 跳过已完成
                if video_key in self.completed_videos:
                    continue

                try:
                    video = h5_file[video_key]
                    n_gtscore = len(video['gtscore'][:])

                    video_info_dict[video_key] = {
                        'n_gtscore': n_gtscore,
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {video_key}: {e}")

        return video_info_dict

    def _write_h5_fields(self, results: List[Dict]):
        """
        批量写入 H5 字段：video_name, picks, n_frames

        Args:
            results: 成功处理的视频结果列表
        """
        successful = [r for r in results if r['success'] and r['picks'] is not None]
        if not successful:
            return

        try:
            with h5py.File(self.h5_path, 'a') as h5_file:
                for result in successful:
                    video_key = result['video_key']
                    if video_key not in h5_file:
                        continue

                    video_group = h5_file[video_key]

                    # 写入 video_name
                    if 'video_name' in video_group:
                        del video_group['video_name']
                    video_group.create_dataset(
                        'video_name',
                        data=result['video_name'].encode('utf-8')
                    )

                    # 写入 picks
                    picks_array = np.array(result['picks'], dtype=np.int64)
                    if 'picks' in video_group:
                        del video_group['picks']
                    video_group.create_dataset('picks', data=picks_array)

                    # 写入 n_frames（原始视频总帧数）
                    if 'n_frames' in video_group:
                        del video_group['n_frames']
                    video_group.create_dataset(
                        'n_frames',
                        data=np.int64(result['n_frames_total'])
                    )

            print(f"   H5 updated: {len(successful)} videos (video_name, picks, n_frames)")

        except Exception as e:
            print(f"   Failed to write H5: {e}")
            traceback.print_exc()

    def extract_parallel(self, test_mode: bool = False,
                         start_idx: int = None, end_idx: int = None):
        """并行提取帧"""
        print(f"\n{'='*80}")
        print(f"MrHiSum Parallel Frame Extraction")
        print(f"{'='*80}")
        print(f"H5 file: {self.h5_path}")
        print(f"Video directory: {self.video_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Workers: {self.workers}")
        print(f"Use FFmpeg: {self.use_ffmpeg}")
        print(f"Delete after: {self.delete_after}")
        print(f"Already completed: {len(self.completed_videos)}")
        if test_mode:
            print("Test mode: only processing 5 videos")
        print(f"{'='*80}\n")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载待处理视频
        video_info_dict = self._load_video_info_from_h5(
            test_mode=test_mode, start_idx=start_idx, end_idx=end_idx
        )

        if not video_info_dict:
            print("All videos already processed!")
            return

        print(f"Videos to process: {len(video_info_dict)}\n")

        # 准备任务参数
        tasks = [
            (video_key, video_info, str(self.video_dir), str(self.output_dir),
             self.use_ffmpeg, self.delete_after)
            for video_key, video_info in video_info_dict.items()
        ]

        # 统计
        stats = {
            'total_processed': 0,
            'total_frames': 0,
            'total_failed': 0,
            'videos_deleted': 0,
            'skipped_no_video': 0,
            'errors': []
        }

        # 收集成功结果用于批量写入 H5
        batch_results = []
        h5_write_interval = 100  # 每100个视频写入一次H5

        # 使用进程池并行处理
        with Pool(processes=self.workers) as pool:
            results = pool.imap_unordered(process_single_video, tasks)

            with tqdm(total=len(tasks), desc="Processing videos") as pbar:
                for result in results:
                    stats['total_processed'] += 1
                    stats['total_frames'] += result['frames_extracted']
                    stats['total_failed'] += result['frames_failed']

                    if result['deleted']:
                        stats['videos_deleted'] += 1

                    if result['error']:
                        if "Video not found" in result['error']:
                            stats['skipped_no_video'] += 1
                        else:
                            stats['errors'].append(
                                (result['video_key'], result['error'])
                            )

                    if result['success']:
                        self.completed_videos.add(result['video_key'])
                        batch_results.append(result)

                        # 定期保存进度和写入 H5
                        if len(batch_results) % h5_write_interval == 0:
                            self._write_h5_fields(batch_results)
                            batch_results.clear()
                            self._save_progress()

                    pbar.update(1)
                    pbar.set_postfix({
                        'frames': stats['total_frames'],
                        'ok': len(self.completed_videos),
                        'skip': stats['skipped_no_video']
                    })

        # 写入剩余的 H5 数据
        if batch_results:
            self._write_h5_fields(batch_results)

        # 保存最终进度
        self._save_progress()

        # 打印统计
        self._print_stats(stats)

    def _print_stats(self, stats: Dict):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"Extraction Complete!")
        print(f"{'='*80}")
        print(f"Videos processed: {stats['total_processed']}")
        print(f"Total frames extracted: {stats['total_frames']:,}")
        print(f"Skipped (no video): {stats['skipped_no_video']}")
        print(f"Failed frames: {stats['total_failed']}")
        print(f"Total completed (cumulative): {len(self.completed_videos)}")

        if self.delete_after:
            print(f"Videos deleted: {stats['videos_deleted']}")

        if stats['errors']:
            print(f"\nErrors ({len(stats['errors'])}):")
            for video_key, error in stats['errors'][:20]:
                print(f"   - {video_key}: {error}")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel frame extraction for MrHiSum dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用8个进程并行抽帧
  python 0_extract_mrhisum_frames.py --workers 8

  # 使用FFmpeg加速（推荐）
  python 0_extract_mrhisum_frames.py --workers 8 --use-ffmpeg

  # 抽帧后自动删除视频（节省磁盘空间）
  python 0_extract_mrhisum_frames.py --workers 8 --delete-after

  # 指定处理范围
  python 0_extract_mrhisum_frames.py --start 0 --end 1000

  # 测试模式
  python 0_extract_mrhisum_frames.py --test --workers 2

  # 自定义路径
  python 0_extract_mrhisum_frames.py \\
      --h5 /data/MMS_Benchmark/data/mrhisum/h5/mrhisum.h5 \\
      --videos /data/MMS_Benchmark/data/mrhisum/videos \\
      --output /data/MMS_Benchmark/data/mrhisum/frames

Output:
  Frames saved as tar.gz: {output_dir}/{video_key}.tar.gz
  Inside each tar.gz: frame_XXXXXX.jpg (original frame index, 6 digits)

H5 fields written:
  - video_name: video filename (= video_key)
  - picks: sampled frame indices (uniform sampling based on gtscore length)
  - n_frames: total frame count of original video

Performance Tips:
  - 推荐 worker 数 = CPU 核心数
  - 使用 --use-ffmpeg 可以提速 2-3 倍
  - 使用 --delete-after 节省磁盘空间
        """
    )

    parser.add_argument('--test', action='store_true',
                       help='Test mode: only process 5 videos')
    parser.add_argument('--start', type=int, default=None,
                       help='Start index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='End index (default: all)')
    parser.add_argument('--h5', type=str,
                       default='/data/MMS_Benchmark/data/mrhisum/h5/mrhisum.h5',
                       help='Path to MrHiSum H5 file')
    parser.add_argument('--videos', type=str,
                       default='/data/MMS_Benchmark/data/mrhisum/videos',
                       help='Directory containing video files')
    parser.add_argument('--output', type=str,
                       default='/data/MMS_Benchmark/data/mrhisum/frames',
                       help='Output directory for extracted frames (tar.gz)')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                       help=f'Number of parallel workers (default: {cpu_count()})')
    parser.add_argument('--use-ffmpeg', action='store_true',
                       help='Use FFmpeg for faster extraction (recommended)')
    parser.add_argument('--delete-after', action='store_true',
                       help='Delete video after successful frame extraction')

    args = parser.parse_args()

    # 验证路径
    h5_path = Path(args.h5)
    video_dir = Path(args.videos)
    output_dir = Path(args.output)

    if not h5_path.exists():
        print(f"H5 file not found: {h5_path}")
        return

    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        print(f"Run download script first.")
        return

    # 创建提取器并运行
    extractor = MrHiSumFrameExtractor(
        h5_path=h5_path,
        video_dir=video_dir,
        output_dir=output_dir,
        workers=args.workers,
        use_ffmpeg=args.use_ffmpeg,
        delete_after=args.delete_after
    )

    extractor.extract_parallel(
        test_mode=args.test,
        start_idx=args.start,
        end_idx=args.end
    )


if __name__ == "__main__":
    main()
