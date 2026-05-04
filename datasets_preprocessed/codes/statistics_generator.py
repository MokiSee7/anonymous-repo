#!/usr/bin/env python3
"""
数据集统计信息生成器

统计内容：
1. 基础信息：视频数、平均帧数、平均shots数、shot平均帧数
2. GT信息：平均gt帧数、平均gt shot数、gt shot的平均帧数
3. 文本信息：帧描述/shot描述/gt帧描述/gt shot描述的平均词数
4. 文档信息：document/summary的平均词数和句子数
5. 其他：帧采样率、gt选中率、词汇量、压缩比
"""

import json
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import argparse


def count_words(text):
    """统计文本词数（英文按空格分词）"""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def count_sentences(text):
    """统计句子数"""
    if not text or not isinstance(text, str):
        return 0
    # 按句号、问号、感叹号分割
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def get_vocabulary(texts):
    """获取词汇表"""
    words = set()
    for text in texts:
        if text and isinstance(text, str):
            words.update(text.lower().split())
    return words


def load_h5_data(h5_path):
    """从H5文件加载数据"""
    data = {}
    with h5py.File(h5_path, 'r') as f:
        for video_key in f.keys():
            video = f[video_key]
            # gtsummary 可能有不同的字段名
            gtsummary = None
            if 'gtsummary' in video:
                gtsummary = video['gtsummary'][:]
            elif 'gt_summary' in video:
                gtsummary = video['gt_summary'][:]

            data[video_key] = {
                'n_frames': int(video['n_frames'][()]) if 'n_frames' in video else None,
                'n_steps': int(video['n_steps'][()]) if 'n_steps' in video else None,
                'change_points': video['change_points'][:] if 'change_points' in video else None,
                'gtscore': video['gtscore'][:] if 'gtscore' in video else None,
                'gtsummary': gtsummary,
                'shot_level_gt': video['shot_level_gt'][:] if 'shot_level_gt' in video else None,
                'n_frame_per_seg': video['n_frame_per_seg'][:] if 'n_frame_per_seg' in video else None,
                'picks': video['picks'][:] if 'picks' in video else None,
                'gtframe': video['gtframe'][:] if 'gtframe' in video else None,
            }
    return data


def load_json_data(json_path):
    """从JSON文件加载数据"""
    if not json_path.exists():
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_statistics(dataset_name, h5_path, json_path=None):
    """计算单个数据集的统计信息"""
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")

    stats = {
        'dataset': dataset_name,
        'num_videos': 0,
        'avg_frames': 0,
        'avg_sampled_frames': 0,
        'avg_shots': 0,
        'avg_frames_per_shot': 0,
        'avg_gt_frames': None,
        'avg_gt_shots': None,
        'avg_frames_per_gt_shot': None,
        'avg_frame_desc_words': None,
        'avg_frame_desc_sentences': None,
        'avg_shot_desc_words': None,
        'avg_shot_desc_sentences': None,
        'avg_gt_frame_desc_words': None,
        'avg_gt_shot_desc_words': None,
        'avg_doc_words': None,
        'avg_doc_sentences': None,
        'avg_summary_words': None,
        'avg_summary_sentences': None,
        'frame_sampling_rate': None,  # 需要视频原始帧数信息
        'gt_frame_ratio': 0,
        'gt_shot_ratio': 0,
        'gt_shot_frame_ratio': 0,
        'vocabulary_size': None,
        'compression_ratio_words': None,
        'compression_ratio_sentences': None,
    }

    # 加载H5数据
    h5_data = load_h5_data(h5_path)
    stats['num_videos'] = len(h5_data)

    # 基础统计
    total_frames = []
    total_sampled_frames = []
    total_shots = []
    gt_frames = []
    gt_shots = []
    gt_sampled_frames_in_gt_shots = []
    gt_frame_ratios = []
    gt_shot_ratios = []
    gt_shot_frame_ratios = []

    for video_key, video in h5_data.items():
        # 获取帧数：优先使用n_frames，否则从gtscore/gtsummary长度推断
        n_frames = video['n_frames']
        if n_frames is None or n_frames == 0:
            if video['gtscore'] is not None:
                n_frames = len(video['gtscore'])
            elif video['gtsummary'] is not None:
                n_frames = len(video['gtsummary'])

        n_shots = len(video['change_points']) if video['change_points'] is not None else 0

        if n_frames:
            total_frames.append(n_frames)
        if video['picks'] is not None:
            total_sampled_frames.append(len(video['picks']))
        if n_shots:
            total_shots.append(n_shots)

        # GT统计：使用 gtframe 字段（长度为采样帧数，1=GT，0=非GT）
        if video['gtframe'] is not None:
            n_gt_frames = np.sum(video['gtframe'] > 0)
            gt_frames.append(n_gt_frames)
            n_sampled = len(video['picks']) if video['picks'] is not None else 0
            if n_sampled:
                gt_frame_ratios.append(n_gt_frames / n_sampled)

        # Shot-level GT: 仅使用 h5 中的 shot_level_gt 字段
        shot_level_gt = video['shot_level_gt']
        if shot_level_gt is not None and len(shot_level_gt) > 0:
            n_gt_shots = np.sum(shot_level_gt > 0)
            gt_shots.append(n_gt_shots)
            if n_shots:
                gt_shot_ratios.append(n_gt_shots / n_shots)

            # 使用 n_frame_per_seg 统计 GT shot 中的帧数（每个GT shot的平均帧数）
            n_frame_per_seg = video.get('n_frame_per_seg')
            if n_frame_per_seg is not None and n_gt_shots > 0:
                gt_indices = np.where(shot_level_gt > 0)[0]
                frames_in_gt = sum(
                    int(n_frame_per_seg[i]) for i in gt_indices
                    if i < len(n_frame_per_seg)
                )
                gt_sampled_frames_in_gt_shots.append(frames_in_gt / n_gt_shots)
                picks = video['picks']
                if picks is not None:
                    n_sampled = len(picks)
                    gt_shot_frame_ratios.append(frames_in_gt / n_sampled if n_sampled else 0)

    # 计算平均值
    if total_frames:
        stats['avg_frames'] = np.mean(total_frames)
    if total_sampled_frames:
        stats['avg_sampled_frames'] = np.mean(total_sampled_frames)
    if total_shots:
        stats['avg_shots'] = np.mean(total_shots)
    if stats['avg_sampled_frames'] and stats['avg_shots']:
        stats['avg_frames_per_shot'] = stats['avg_sampled_frames'] / stats['avg_shots']
    if gt_frames:
        stats['avg_gt_frames'] = np.mean(gt_frames)
    if gt_shots:
        stats['avg_gt_shots'] = np.mean(gt_shots)
    if gt_sampled_frames_in_gt_shots:
        stats['avg_frames_per_gt_shot'] = np.mean(gt_sampled_frames_in_gt_shots)
    if gt_frame_ratios:
        stats['gt_frame_ratio'] = np.mean(gt_frame_ratios)
    if gt_shot_ratios:
        stats['gt_shot_ratio'] = np.mean(gt_shot_ratios)
    if gt_shot_frame_ratios:
        stats['gt_shot_frame_ratio'] = np.mean(gt_shot_frame_ratios)

    # 加载JSON数据（如果有）
    json_data = load_json_data(json_path) if json_path else None

    if json_data:
        frame_desc_words = []
        frame_desc_sentences = []
        shot_desc_words = []
        shot_desc_sentences = []
        gt_frame_desc_words = []
        gt_shot_desc_words = []
        doc_words = []
        doc_sentences = []
        summary_words = []
        summary_sentences = []
        all_texts = []

        for video_key, video in json_data.items():
            # 帧描述统计
            if 'frames' in video:
                for frame in video['frames']:
                    desc = frame.get('description', '')
                    if desc:
                        words = count_words(desc)
                        frame_desc_words.append(words)
                        frame_desc_sentences.append(count_sentences(desc))
                        all_texts.append(desc)
                        if frame.get('is_gt', 0) > 0:
                            gt_frame_desc_words.append(words)

            # Shot/Clip描述统计
            if 'clips' in video and video['clips']:
                for clip in video['clips']:
                    summary = clip.get('summary', '')
                    if summary:
                        words = count_words(summary)
                        shot_desc_words.append(words)
                        shot_desc_sentences.append(count_sentences(summary))
                        all_texts.append(summary)
                        if clip.get('gt', 0) > 0:
                            gt_shot_desc_words.append(words)

            # Document统计
            if 'aligned_full_document' in video:
                doc = video['aligned_full_document']
                raw_text = doc.get('raw_text', '')
                if raw_text:
                    doc_words.append(count_words(raw_text))
                    doc_sentences.append(doc.get('num_sentences', count_sentences(raw_text)))
                    all_texts.append(raw_text)

            # Summary统计
            if 'aligned_text_summary' in video:
                summ = video['aligned_text_summary']
                raw_text = summ.get('raw_text', '')
                if raw_text:
                    summary_words.append(count_words(raw_text))
                    summary_sentences.append(summ.get('num_sentences', count_sentences(raw_text)))
                    all_texts.append(raw_text)

        # 计算文本统计
        if frame_desc_words:
            stats['avg_frame_desc_words'] = np.mean(frame_desc_words)
        if frame_desc_sentences:
            stats['avg_frame_desc_sentences'] = np.mean(frame_desc_sentences)
        if shot_desc_words:
            stats['avg_shot_desc_words'] = np.mean(shot_desc_words)
        if shot_desc_sentences:
            stats['avg_shot_desc_sentences'] = np.mean(shot_desc_sentences)
        # avg_gt_frame_desc_words: 用 h5 gtframe 中 GT 帧总数作为除数
        total_gt_frames_h5 = sum(gt_frames) if gt_frames else 0
        if gt_frame_desc_words and total_gt_frames_h5 > 0:
            stats['avg_gt_frame_desc_words'] = sum(gt_frame_desc_words) / total_gt_frames_h5
        if gt_shot_desc_words:
            stats['avg_gt_shot_desc_words'] = np.mean(gt_shot_desc_words)
        if doc_words:
            stats['avg_doc_words'] = np.mean(doc_words)
        if doc_sentences:
            stats['avg_doc_sentences'] = np.mean(doc_sentences)
        if summary_words:
            stats['avg_summary_words'] = np.mean(summary_words)
        if summary_sentences:
            stats['avg_summary_sentences'] = np.mean(summary_sentences)

        # 词汇量
        vocab = get_vocabulary(all_texts)
        stats['vocabulary_size'] = len(vocab)

        # 压缩比
        if doc_words and summary_words:
            stats['compression_ratio_words'] = np.mean(doc_words) / np.mean(summary_words) if np.mean(summary_words) > 0 else None
        if doc_sentences and summary_sentences:
            stats['compression_ratio_sentences'] = np.mean(doc_sentences) / np.mean(summary_sentences) if np.mean(summary_sentences) > 0 else None

    return stats


def format_value(value, decimals=2):
    """格式化数值"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def print_statistics(all_stats):
    """打印统计表格"""
    print("\n" + "="*100)
    print("数据集统计汇总")
    print("="*100)

    # 基础信息
    print("\n【基础信息】")
    print(f"{'数据集':<12} {'视频数':>10} {'平均帧数':>12} {'平均采样帧数':>14} {'平均shots数':>12} {'shot平均帧数':>14}")
    print("-"*76)
    for stats in all_stats:
        print(f"{stats['dataset']:<12} {stats['num_videos']:>10} {format_value(stats['avg_frames']):>12} {format_value(stats['avg_sampled_frames']):>14} {format_value(stats['avg_shots']):>12} {format_value(stats['avg_frames_per_shot']):>14}")

    # GT信息
    print("\n【GT信息】")
    print(f"{'数据集':<12} {'平均GT帧数':>12} {'平均GT shots':>14} {'GT shot平均帧数':>16} {'GT帧选中率':>12} {'GT shot选中率':>14} {'GT shot帧占比':>14}")
    print("-"*95)
    for stats in all_stats:
        print(f"{stats['dataset']:<12} {format_value(stats['avg_gt_frames']):>12} {format_value(stats['avg_gt_shots']):>14} {format_value(stats['avg_frames_per_gt_shot']):>16} {format_value(stats['gt_frame_ratio']):>12} {format_value(stats['gt_shot_ratio']):>14} {format_value(stats['gt_shot_frame_ratio']):>14}")

    # 文本信息
    print("\n【帧/Shot描述信息】")
    print(f"{'数据集':<12} {'帧描述词数':>12} {'帧描述句数':>12} {'shot描述词数':>14} {'shot描述句数':>14} {'GT帧描述词数':>14} {'GT shot描述词数':>16}")
    print("-"*98)
    for stats in all_stats:
        print(f"{stats['dataset']:<12} {format_value(stats['avg_frame_desc_words']):>12} {format_value(stats['avg_frame_desc_sentences']):>12} {format_value(stats['avg_shot_desc_words']):>14} {format_value(stats['avg_shot_desc_sentences']):>14} {format_value(stats['avg_gt_frame_desc_words']):>14} {format_value(stats['avg_gt_shot_desc_words']):>16}")

    # 文档信息
    print("\n【Document/Summary信息】")
    print(f"{'数据集':<12} {'Doc词数':>12} {'Doc句子数':>12} {'Summary词数':>14} {'Summary句子数':>14} {'词数压缩比':>12} {'句子压缩比':>12} {'词汇量':>10}")
    print("-"*100)
    for stats in all_stats:
        print(f"{stats['dataset']:<12} {format_value(stats['avg_doc_words']):>12} {format_value(stats['avg_doc_sentences']):>12} {format_value(stats['avg_summary_words']):>14} {format_value(stats['avg_summary_sentences']):>14} {format_value(stats['compression_ratio_words']):>12} {format_value(stats['compression_ratio_sentences']):>12} {format_value(stats['vocabulary_size']):>10}")


def save_statistics(all_stats, output_path):
    """保存统计结果到JSON"""
    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_native(all_stats), f, indent=2, ensure_ascii=False)
    print(f"\n统计结果已保存到: {output_path}")


def main():
    base_dir = Path('/data/MMS_Benchmark/data')
    json_dir = base_dir / 'processed'

    # 数据集配置：h5 路径为 /data/MMS_Benchmark/data/[dataset_name]/h5/[dataset_name].h5
    all_datasets = {
        'summe': ('SumMe', base_dir / 'summe' / 'h5' / 'summe.h5', json_dir / 'summe_data.json'),
        'tvsum': ('TVSum', base_dir / 'tvsum' / 'h5' / 'tvsum.h5', json_dir / 'tvsum_data.json'),
        'ovp': ('OVP', base_dir / 'ovp' / 'h5' / 'ovp.h5', json_dir / 'ovp_data.json'),
        'youtube': ('YouTube', base_dir / 'youtube' / 'h5' / 'youtube.h5', json_dir / 'youtube_data.json'),
        'videoxum': ('VideoXum', base_dir / 'videoxum' / 'h5' / 'videoxum.h5', json_dir / 'videoxum_data.json'),
        'mrhisum': ('MrHiSum', base_dir / 'mrhisum' / 'h5' / 'mrhisum.h5', json_dir / 'mrhisum_data.json'),
    }

    parser = argparse.ArgumentParser(description='数据集统计信息生成器')
    parser.add_argument('--datasets', nargs='+', choices=list(all_datasets.keys()),
                        default=None, help='指定要统计的数据集（默认全部）')
    args = parser.parse_args()

    selected_keys = args.datasets if args.datasets else list(all_datasets.keys())
    datasets = [all_datasets[k] for k in selected_keys]

    all_stats = []
    for name, h5_path, json_path in datasets:
        if h5_path.exists():
            stats = compute_statistics(name, h5_path, json_path)
            all_stats.append(stats)
        else:
            print(f"警告: {h5_path} 不存在，跳过 {name}")

    # 打印统计结果
    print_statistics(all_stats)

    # 保存结果
    output_path = base_dir / 'statistics' / 'dataset_statistics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_statistics(all_stats, output_path)


if __name__ == '__main__':
    main()
