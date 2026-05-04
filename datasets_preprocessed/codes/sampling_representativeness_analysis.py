#!/usr/bin/env python3
"""
VideoXum 采样数据集代表性分析

对比三个层级的统计分布:
- 原始论文 (14,001 videos) - 仅汇总统计作为参考
- 本地H5完整集 (11,146 videos) - 完整分布
- 我们的采样子集 (~5,300 videos) - 完整分布

输出:
1. 视频时长分布对比 + 直方图
2. 压缩比分布对比
3. 归一化中心时间戳分布
4. Shots数量分布
5. FPS分布
6. 类别分布 (如ActivityNet注释可用)
7. KS检验 & Chi-squared检验结果

Usage:
    python sampling_representativeness_analysis.py
    python sampling_representativeness_analysis.py --skip-timestamps  # 跳过耗时的时间戳分析
"""

import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
from pathlib import Path
from collections import Counter
import time
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
H5_PATH = Path('/data/MMS_Benchmark/data/videoxum/h5/videoxum.h5')
JSON_PATH = Path('/data/MMS_Benchmark/data/processed/videoxum_data.json')
OUTPUT_DIR = Path('/data/MMS_Benchmark/data/statistics/sampling_analysis')

# ActivityNet category annotation URL (for category distribution)
ACTIVITYNET_URL = 'http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json'
ACTIVITYNET_LOCAL = Path('/data/MMS_Benchmark/data/videoxum/activity_net.v1-3.min.json')

# Paper reference statistics (VideoXum TMM 2023, 14,001 videos)
PAPER_STATS = {
    'n_videos': 14001,
    'train': 8000, 'val': 2001, 'test': 4000,
    'duration_mean': 124.2,
    'duration_median': 121.6,
    'duration_min': 10,
    'duration_max': 755,
    'duration_99_9_pct': 300,
    'compression_mean': 0.136,
    'compression_median': 0.137,
    'compression_max': 0.20,
    'text_summary_words_mean': 49.9,
    'text_summary_98_pct': 128,
    'kappa': 0.49,
    'f1_avg': 36.2,
    'f1_max': 59.5,
    'n_activity_categories': 200,
    'n_top_level_topics': 5,
}


# ============================================================
# Helper Functions
# ============================================================

def extract_center_timestamps(user_summary_row, total_len):
    """从单个annotator的二值摘要中提取归一化中心时间戳

    Args:
        user_summary_row: 1D binary array (0/1), 长度=原始视频总帧数
        total_len: 视频总帧数 (用于归一化)

    Returns:
        list of float: 归一化到[0,1]的中心时间戳
    """
    padded = np.concatenate([[0], user_summary_row, [0]])
    diff = np.diff(padded)
    starts = np.where(diff > 0)[0]
    ends = np.where(diff < 0)[0]

    if len(starts) == 0:
        return []

    centers = ((starts + ends) / 2.0 / total_len).tolist()
    return centers


def load_activitynet_categories():
    """尝试加载 ActivityNet 类别注释

    Returns:
        dict: video_id -> category_name, 或 None (如果不可用)
    """
    # 先检查本地
    if ACTIVITYNET_LOCAL.exists():
        print(f"  从本地加载ActivityNet注释: {ACTIVITYNET_LOCAL}")
        with open(ACTIVITYNET_LOCAL, 'r') as f:
            ann = json.load(f)
    else:
        # 尝试下载
        print(f"  尝试下载ActivityNet注释...")
        try:
            import urllib.request
            urllib.request.urlretrieve(ACTIVITYNET_URL, str(ACTIVITYNET_LOCAL))
            with open(ACTIVITYNET_LOCAL, 'r') as f:
                ann = json.load(f)
            print(f"  下载成功: {ACTIVITYNET_LOCAL}")
        except Exception as e:
            print(f"  下载失败: {e}")
            print(f"  跳过类别分布分析 (可手动下载到 {ACTIVITYNET_LOCAL})")
            return None

    # 解析: video_id -> category
    # ActivityNet格式: {"database": {"video_id": {"annotations": [{"label": "..."}], ...}}}
    video_categories = {}
    db = ann.get('database', {})
    for vid, info in db.items():
        # VideoXum的video_id格式: v_xxxx, ActivityNet的格式也是 v_xxxx
        annotations = info.get('annotations', [])
        if annotations:
            # 取第一个annotation的label作为主类别
            label = annotations[0].get('label', 'Unknown')
            video_categories[f'v_{vid}'] = label
            # 也存一份不带v_前缀的
            video_categories[vid] = label

    print(f"  加载了 {len(video_categories)//2} 个视频的类别信息")
    return video_categories


# ============================================================
# Data Loading & Processing
# ============================================================

def load_and_process_data(skip_timestamps=False):
    """加载H5和JSON数据, 计算各项统计

    Args:
        skip_timestamps: 是否跳过时间戳分析 (节省时间)

    Returns:
        dict with 'full' and 'subset' data
    """
    # 1. 获取子集的video_key集合
    print("[1/3] 读取JSON获取子集视频列表...")
    t0 = time.time()
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # JSON key是YouTube ID (如 v_54Hp_Z-cu-s), video_name是H5 key (如 video_1020)
    subset_keys = set()
    youtube_to_h5 = {}
    for vid_id, info in json_data.items():
        vname = info.get('video_name', '')
        if vname:
            subset_keys.add(vname)
            youtube_to_h5[vid_id] = vname
    print(f"  子集视频数: {len(subset_keys)} ({time.time()-t0:.1f}s)")

    # 2. 处理H5文件
    print("[2/3] 处理H5文件...")
    if skip_timestamps:
        print("  (跳过时间戳分析)")
    t0 = time.time()

    data = {
        'full': {
            'durations': [], 'compressions': [], 'timestamps': [],
            'n_shots': [], 'fps': [], 'video_keys': [], 'video_names': []
        },
        'subset': {
            'durations': [], 'compressions': [], 'timestamps': [],
            'n_shots': [], 'fps': [], 'video_keys': [], 'video_names': []
        }
    }

    with h5py.File(H5_PATH, 'r') as f:
        keys = sorted(f.keys())
        total = len(keys)
        print(f"  H5总视频数: {total}")

        for idx, vk in enumerate(keys):
            if (idx + 1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  已处理 {idx+1}/{total} ({elapsed:.0f}s)")

            video = f[vk]
            is_subset = vk in subset_keys

            # --- Video Name (YouTube ID) ---
            video_name = ''
            if 'video_name' in video:
                vn = video['video_name'][()]
                if isinstance(vn, bytes):
                    video_name = vn.decode('utf-8')
                elif isinstance(vn, np.ndarray):
                    vn0 = vn.flat[0]
                    video_name = vn0.decode('utf-8') if isinstance(vn0, bytes) else str(vn0)
                else:
                    video_name = str(vn)

            # --- Duration ---
            # n_frames 是1FPS采样帧数, 近似等于视频时长(秒)
            # 论文 mean=124.2s 与 H5 mean n_frames≈124.6 吻合
            n_frames = int(video['n_frames'][()])
            duration = n_frames

            # --- FPS ---
            picks = video['picks'][:]
            fps = int(picks[1] - picks[0]) if len(picks) > 1 else 30

            # --- Shots ---
            n_shots = 0
            if 'change_points' in video:
                cp = video['change_points'][:]
                n_shots = len(cp)

            # --- Compression Ratio ---
            # 方法: 用gtscore (1FPS级别, 10个annotator的平均)
            # gtscore[j] = mean(vsum_onehot[i][j] for all annotators)
            # mean(gtscore) ≈ 平均压缩比
            # 注: user_summary展开到原始帧率时每秒只标1帧, 直接算会被fps稀释
            avg_cr = None
            if 'gtscore' in video:
                gtscore = video['gtscore'][:]
                avg_cr = float(np.mean(gtscore))

            # --- Timestamps (from user_summary) ---
            timestamps = []
            if not skip_timestamps and 'user_summary' in video:
                us = video['user_summary'][:]  # (10, total_orig_frames)
                n_annotators, total_orig = us.shape
                for i in range(n_annotators):
                    ts = extract_center_timestamps(us[i], total_orig)
                    timestamps.extend(ts)

            # --- 存储 ---
            data['full']['durations'].append(duration)
            data['full']['n_shots'].append(n_shots)
            data['full']['fps'].append(fps)
            data['full']['video_keys'].append(vk)
            data['full']['video_names'].append(video_name)
            if avg_cr is not None:
                data['full']['compressions'].append(avg_cr)
            if timestamps:
                data['full']['timestamps'].extend(timestamps)

            if is_subset:
                data['subset']['durations'].append(duration)
                data['subset']['n_shots'].append(n_shots)
                data['subset']['fps'].append(fps)
                data['subset']['video_keys'].append(vk)
                data['subset']['video_names'].append(video_name)
                if avg_cr is not None:
                    data['subset']['compressions'].append(avg_cr)
                if timestamps:
                    data['subset']['timestamps'].extend(timestamps)

    elapsed = time.time() - t0
    print(f"  处理完成 ({elapsed:.0f}s)")
    print(f"  Full: {len(data['full']['durations'])} videos, "
          f"{len(data['full']['compressions'])} with compression ratio")
    print(f"  Subset: {len(data['subset']['durations'])} videos, "
          f"{len(data['subset']['compressions'])} with compression ratio")
    if not skip_timestamps:
        print(f"  Timestamps: full={len(data['full']['timestamps'])}, "
              f"subset={len(data['subset']['timestamps'])}")

    # 转换为numpy
    for group in ['full', 'subset']:
        for key in ['durations', 'compressions', 'timestamps', 'n_shots', 'fps']:
            data[group][key] = np.array(data[group][key])

    return data


# ============================================================
# Statistical Tests
# ============================================================

def run_statistical_tests(data, skip_timestamps=False):
    """执行KS检验和Chi-squared检验"""
    print("\n" + "=" * 80)
    print("统计检验结果")
    print("H0: 采样子集与完整数据集来自同一分布 (p > 0.05 表示无显著差异)")
    print("=" * 80)

    results = {}
    n_full = len(data['full']['durations'])
    n_sub = len(data['subset']['durations'])

    # ---- KS Test ----
    tests = [
        ('视频时长 (Duration)', 'durations'),
        ('压缩比 (Compression Ratio)', 'compressions'),
        ('Shots数量 (Number of Shots)', 'n_shots'),
    ]
    if not skip_timestamps:
        tests.append(('中心时间戳 (Center Timestamps)', 'timestamps'))

    print(f"\n--- Two-sample KS Test ---")
    print(f"{'指标':<35} {'KS统计量':>10} {'p-value':>12} {'结论':>12}")
    print("-" * 72)

    for name, key in tests:
        full = data['full'][key]
        sub = data['subset'][key]

        if len(full) == 0 or len(sub) == 0:
            print(f"{name:<35} {'N/A':>10} {'N/A':>12} {'数据不足':>12}")
            continue

        ks_stat, p_val = sp_stats.ks_2samp(full, sub)
        # 大样本下KS检验可能过于敏感, 同时报告效应大小
        conclusion = "无显著差异 ✓" if p_val > 0.05 else "有差异 ✗"
        print(f"{name:<35} {ks_stat:>10.4f} {p_val:>12.6f} {conclusion:>12}")
        results[key + '_ks'] = {
            'statistic': float(ks_stat),
            'p_value': float(p_val),
            'significant': p_val <= 0.05
        }

    # ---- Chi-squared Test (binned) ----
    print(f"\n--- Chi-squared Test (分箱对比) ---")
    print(f"{'指标':<35} {'Chi2':>10} {'df':>6} {'p-value':>12} {'结论':>12}")
    print("-" * 78)

    chi2_configs = [
        ('视频时长 (30s bins)', 'durations', np.arange(0, 800, 30)),
        ('压缩比 (1% bins)', 'compressions', np.arange(0, 0.25, 0.01)),
        ('Shots数量 (5-shot bins)', 'n_shots', np.arange(0, 80, 5)),
    ]

    for name, key, bins in chi2_configs:
        full = data['full'][key]
        sub = data['subset'][key]

        if len(full) == 0 or len(sub) == 0:
            continue

        full_hist, _ = np.histogram(full, bins=bins)
        sub_hist, _ = np.histogram(sub, bins=bins)

        # 期望分布: 按全集分布, 缩放到子集样本量
        scale = len(sub) / len(full)
        expected = full_hist * scale

        # Chi-squared要求期望频数 > 5, 合并小频数的bins
        # 从两端开始合并
        merged_obs = []
        merged_exp = []
        cum_obs = 0
        cum_exp = 0
        for o, e in zip(sub_hist, expected):
            cum_obs += o
            cum_exp += e
            if cum_exp >= 5:
                merged_obs.append(cum_obs)
                merged_exp.append(cum_exp)
                cum_obs = 0
                cum_exp = 0
        # 将剩余的合并到最后一个bin
        if cum_exp > 0:
            if merged_exp:
                merged_obs[-1] += cum_obs
                merged_exp[-1] += cum_exp
            else:
                merged_obs.append(cum_obs)
                merged_exp.append(cum_exp)

        merged_obs = np.array(merged_obs, dtype=float)
        merged_exp = np.array(merged_exp, dtype=float)

        # 归一化使总数精确匹配 (避免浮点误差导致chisquare报错)
        if np.sum(merged_exp) > 0:
            merged_exp = merged_exp * (np.sum(merged_obs) / np.sum(merged_exp))

        if len(merged_obs) > 1:
            chi2, p_chi2 = sp_stats.chisquare(merged_obs, merged_exp)
            df = len(merged_obs) - 1
            conclusion = "无显著差异 ✓" if p_chi2 > 0.05 else "有差异 ✗"
            print(f"{name:<35} {chi2:>10.2f} {df:>6} {p_chi2:>12.6f} {conclusion:>12}")
            results[key + '_chi2'] = {
                'chi2': float(chi2),
                'df': df,
                'p_value': float(p_chi2),
                'significant': p_chi2 <= 0.05
            }

    # ---- 补充: Wasserstein Distance (Earth Mover's Distance) ----
    # 对大样本更稳健的分布差异度量
    print(f"\n--- Wasserstein Distance (效应大小度量) ---")
    print(f"{'指标':<35} {'W-Distance':>12} {'相对误差(%)':>14}")
    print("-" * 64)

    for name, key in [('视频时长', 'durations'), ('压缩比', 'compressions'), ('Shots数量', 'n_shots')]:
        full = data['full'][key]
        sub = data['subset'][key]
        if len(full) == 0 or len(sub) == 0:
            continue

        w_dist = sp_stats.wasserstein_distance(full, sub)
        # 相对误差: W距离 / 全集均值
        rel_err = w_dist / np.mean(full) * 100 if np.mean(full) != 0 else 0
        print(f"{name:<35} {w_dist:>12.4f} {rel_err:>13.2f}%")
        results[key + '_wasserstein'] = {
            'distance': float(w_dist),
            'relative_error_pct': float(rel_err)
        }

    return results


# ============================================================
# Summary Table
# ============================================================

def print_summary_table(data):
    """打印汇总统计对比表: 论文 vs 子集"""
    sd = data['subset']['durations']
    sc = data['subset']['compressions']
    ss = data['subset']['n_shots']

    print("\n" + "=" * 70)
    print("汇总统计对比: 论文原始数据集 vs 我们的采样子集")
    print("=" * 70)

    col1 = f"论文 ({PAPER_STATS['n_videos']})"
    col2 = f"采样子集 ({len(sd)})"

    print(f"\n{'指标':<28} {col1:<20} {col2:<20}")
    print("-" * 68)

    rows = [
        ('视频数', f"{PAPER_STATS['n_videos']}", f"{len(sd)}"),
        ('', '', ''),
        ('--- 视频时长 ---', '', ''),
        ('  Mean (s)', f"{PAPER_STATS['duration_mean']:.1f}",
         f"{np.mean(sd):.1f}"),
        ('  Median (s)', f"{PAPER_STATS['duration_median']:.1f}",
         f"{np.median(sd):.1f}"),
        ('  Std (s)', "N/A",
         f"{np.std(sd):.1f}"),
        ('  Min (s)', f"{PAPER_STATS['duration_min']}",
         f"{int(np.min(sd))}"),
        ('  Max (s)', f"{PAPER_STATS['duration_max']}",
         f"{int(np.max(sd))}"),
        ('  99.9th pct (s)', f"{PAPER_STATS['duration_99_9_pct']}",
         f"{np.percentile(sd, 99.9):.0f}"),
        ('', '', ''),
        ('--- 压缩比 ---', '', ''),
        ('  Mean', f"{PAPER_STATS['compression_mean']:.3f}",
         f"{np.mean(sc):.3f}"),
        ('  Median', f"{PAPER_STATS['compression_median']:.3f}",
         f"{np.median(sc):.3f}"),
        ('  Std', "N/A",
         f"{np.std(sc):.3f}"),
        ('  Max', f"{PAPER_STATS['compression_max']:.3f}",
         f"{np.max(sc):.3f}"),
        ('', '', ''),
        ('--- Shots ---', '', ''),
        ('  Mean', "N/A",
         f"{np.mean(ss):.1f}"),
        ('  Median', "N/A",
         f"{np.median(ss):.1f}"),
        ('  Std', "N/A",
         f"{np.std(ss):.1f}"),
    ]

    for label, paper, subset in rows:
        if label == '':
            print()
        elif label.startswith('---'):
            print(f"  {label}")
        else:
            print(f"  {label:<26} {paper:<20} {subset:<20}")


# ============================================================
# Plotting
# ============================================================

def create_main_plots(data, test_results, skip_timestamps=False):
    """创建主对比图: 子集分布 + 论文参考线"""
    print("\n[3/3] 生成对比图...")

    n_rows = 2
    n_cols = 3 if not skip_timestamps else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle('VideoXum: Sampled Subset vs Original Paper Statistics',
                 fontsize=15, fontweight='bold', y=1.02)

    c_sub = '#4285F4'     # 子集用蓝色
    c_paper = '#E53935'   # 论文参考线用红色
    alpha = 0.7

    n_sub = len(data['subset']['durations'])

    # ---- (a) Video Duration Distribution ----
    ax = axes[0, 0]
    bins_dur = np.arange(0, 780, 15)
    ax.hist(data['subset']['durations'], bins=bins_dur, density=True,
            alpha=alpha, color=c_sub, label=f'Our Subset (n={n_sub})',
            edgecolor='white', linewidth=0.3)
    ax.axvline(PAPER_STATS['duration_mean'], color=c_paper, linestyle='--',
               linewidth=2, label=f'Paper Mean={PAPER_STATS["duration_mean"]}s')
    ax.axvline(PAPER_STATS['duration_median'], color=c_paper, linestyle=':',
               linewidth=2, label=f'Paper Median={PAPER_STATS["duration_median"]}s')
    # 子集均值
    sub_mean = np.mean(data['subset']['durations'])
    ax.axvline(sub_mean, color='#1565C0', linestyle='--',
               linewidth=1.5, label=f'Subset Mean={sub_mean:.1f}s')
    ax.set_xlabel('Video Duration (seconds)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(a) Video Duration Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    _add_ks_annotation(ax, test_results, 'durations_ks')

    # ---- (b) Compression Ratio Distribution ----
    ax = axes[0, 1]
    bins_comp = np.arange(0, 0.35, 0.005)
    ax.hist(data['subset']['compressions'], bins=bins_comp, density=True,
            alpha=alpha, color=c_sub, label='Our Subset',
            edgecolor='white', linewidth=0.3)
    ax.axvline(PAPER_STATS['compression_mean'], color=c_paper, linestyle='--',
               linewidth=2, label=f'Paper Mean={PAPER_STATS["compression_mean"]}')
    ax.axvline(PAPER_STATS['compression_median'], color=c_paper, linestyle=':',
               linewidth=2, label=f'Paper Median={PAPER_STATS["compression_median"]}')
    ax.set_xlabel('Compression Ratio', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(b) Compression Ratio Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    _add_ks_annotation(ax, test_results, 'compressions_ks')

    # ---- (c) Center Timestamp Distribution (if available) ----
    if not skip_timestamps:
        ax = axes[0, 2]
        bins_ts = np.arange(0, 1.02, 0.02)
        if len(data['subset']['timestamps']) > 0:
            ax.hist(data['subset']['timestamps'], bins=bins_ts, density=True,
                    alpha=alpha, color=c_sub, label='Our Subset',
                    edgecolor='white', linewidth=0.3)
        # 论文提到时间戳"generally uniformly distributed with a mild peak at the beginning"
        ax.axhline(1.0, color=c_paper, linestyle='--', linewidth=1.5,
                   alpha=0.5, label='Uniform (Paper reference)')
        ax.set_xlabel('Normalized Center Timestamp', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('(c) Center Timestamp Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        _add_ks_annotation(ax, test_results, 'timestamps_ks')

    # ---- (d) Number of Shots Distribution ----
    ax = axes[1, 0]
    bins_shots = np.arange(0, 80, 2)
    ax.hist(data['subset']['n_shots'], bins=bins_shots, density=True,
            alpha=alpha, color=c_sub, label='Our Subset',
            edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Number of Shots (KTS Segments)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Number of Shots Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    _add_ks_annotation(ax, test_results, 'n_shots_ks')

    # ---- (e) FPS Distribution ----
    ax = axes[1, 1]
    fps_sub = Counter(data['subset']['fps'].astype(int))
    main_fps = sorted([f for f in fps_sub.keys() if fps_sub[f] > 10])

    x = np.arange(len(main_fps))
    sub_pcts = [fps_sub.get(f, 0) / n_sub * 100 for f in main_fps]

    ax.bar(x, sub_pcts, color=c_sub, alpha=0.7, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(main_fps, fontsize=8)
    ax.set_xlabel('FPS (frames per second)', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('(e) FPS Distribution', fontsize=12, fontweight='bold')

    # ---- (f) Summary Table: Paper vs Subset ----
    if not skip_timestamps:
        ax = axes[1, 2]
    else:
        ax = None

    if ax is not None:
        ax.axis('off')

        sd = data['subset']['durations']
        sc = data['subset']['compressions']

        table_data = [
            [f'Paper\n({PAPER_STATS["n_videos"]})',
             f'Our Subset\n({n_sub})'],
            [f'{PAPER_STATS["duration_mean"]:.1f}s',
             f'{np.mean(sd):.1f}s'],
            [f'{PAPER_STATS["duration_median"]:.1f}s',
             f'{np.median(sd):.1f}s'],
            [f'{PAPER_STATS["duration_min"]}–{PAPER_STATS["duration_max"]}s',
             f'{int(np.min(sd))}–{int(np.max(sd))}s'],
            [f'{PAPER_STATS["compression_mean"]:.3f}',
             f'{np.mean(sc):.3f}'],
            [f'{PAPER_STATS["compression_median"]:.3f}',
             f'{np.median(sc):.3f}'],
            ['N/A',
             f'{np.mean(data["subset"]["n_shots"]):.1f}'],
        ]
        row_labels = ['Duration\nMean', 'Duration\nMedian', 'Duration\nRange',
                      'Compress.\nMean', 'Compress.\nMedian', 'Shots\nMean']

        table = ax.table(cellText=table_data[1:],
                         colLabels=table_data[0],
                         rowLabels=row_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        for j in range(2):
            table[0, j].set_facecolor('#E3F2FD')
            table[0, j].set_text_props(fontweight='bold')

        ax.set_title('(f) Paper vs Subset', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'videoxum_sampling_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  主图已保存: {output_path}")
    plt.close()

    return output_path


def _add_ks_annotation(ax, test_results, key):
    """在图上添加KS检验结果标注"""
    if key in test_results:
        ks = test_results[key]
        text = f"KS = {ks['statistic']:.4f}\np = {ks['p_value']:.4f}"
        if not ks['significant']:
            text += "\n(n.s.)"
        ax.text(0.97, 0.95, text,
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))


def create_category_plot(data, video_categories):
    """创建类别分布对比图"""
    if video_categories is None:
        print("  跳过类别分布图 (无ActivityNet类别数据)")
        return None

    print("  生成类别分布图...")

    # 统计full和subset的类别分布
    full_cats = []
    for vname in data['full']['video_names']:
        cat = video_categories.get(vname, None)
        if cat:
            full_cats.append(cat)

    sub_cats = []
    for vname in data['subset']['video_names']:
        cat = video_categories.get(vname, None)
        if cat:
            sub_cats.append(cat)

    if not full_cats or not sub_cats:
        print("  无法匹配类别信息, 跳过")
        return None

    print(f"  匹配到类别: full={len(full_cats)}/{len(data['full']['video_names'])}, "
          f"subset={len(sub_cats)}/{len(data['subset']['video_names'])}")

    # 类别覆盖统计
    full_counter = Counter(full_cats)
    sub_counter = Counter(sub_cats)
    n_cats_full = len(full_counter)
    n_cats_sub = len(sub_counter)
    # 子集未覆盖的类别
    missing_cats = set(full_counter.keys()) - set(sub_counter.keys())

    print(f"\n  --- 类别覆盖统计 ---")
    print(f"  {'指标':<35} {'论文':<15} {'原数据集':<15} {'采样子集':<15}")
    print(f"  {'-'*78}")
    print(f"  {'活动类别总数 (ActivityNet)':<35} {'200':<15} {n_cats_full:<15} {n_cats_sub:<15}")
    print(f"  {'覆盖率':<35} {'100%':<15} "
          f"{n_cats_full/200*100:.1f}%{'':<9} "
          f"{n_cats_sub/200*100:.1f}%")
    if missing_cats:
        print(f"  子集未覆盖的类别 ({len(missing_cats)}个): {', '.join(sorted(missing_cats))}")
    else:
        print(f"  子集完全覆盖了原数据集的所有类别 ✓")

    # ---- Chi-squared + JSD (所有类别) ----
    all_cats = sorted(full_counter.keys())
    obs = np.array([sub_counter.get(c, 0) for c in all_cats], dtype=float)
    exp = np.array([full_counter[c] / len(full_cats) * len(sub_cats) for c in all_cats], dtype=float)
    # 归一化 exp 避免浮点误差
    if exp.sum() > 0:
        exp = exp * (obs.sum() / exp.sum())

    chi2_val, p_chi2 = sp_stats.chisquare(obs, exp)
    df_chi2 = len(all_cats) - 1

    # JSD (Jensen-Shannon Divergence)
    p = obs / obs.sum() if obs.sum() > 0 else obs
    q = exp / exp.sum() if exp.sum() > 0 else exp
    m = 0.5 * (p + q)
    # 避免 log(0)
    eps = 1e-12
    jsd = float(0.5 * np.sum(p * np.log(p / (m + eps) + eps)) +
                0.5 * np.sum(q * np.log(q / (m + eps) + eps)))

    cat_test_results = {
        'n_categories_full': n_cats_full,
        'n_categories_subset': n_cats_sub,
        'coverage_pct': float(n_cats_sub / 200 * 100),
        'chi2': float(chi2_val),
        'df': df_chi2,
        'p_value': float(p_chi2),
        'significant': bool(p_chi2 <= 0.05),
        'jsd': jsd,
    }

    conclusion = "无显著差异 ✓" if p_chi2 > 0.05 else "有差异 ✗"
    print(f"\n  --- 类别分布统计检验 ---")
    print(f"  Chi-squared: χ²={chi2_val:.2f}, df={df_chi2}, p={p_chi2:.6f}  {conclusion}")
    print(f"  Jensen-Shannon Divergence: {jsd:.6f}  {'(高度相似 ✓)' if jsd < 0.05 else '(有差异)'}")

    # 统计Top-20类别
    top_cats = [cat for cat, _ in full_counter.most_common(20)]

    full_pcts = [full_counter[cat] / len(full_cats) * 100 for cat in top_cats]
    sub_pcts = [sub_counter.get(cat, 0) / len(sub_cats) * 100 for cat in top_cats]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(top_cats))
    w = 0.35

    ax.bar(x - w/2, full_pcts, w, color='#4285F4', alpha=0.7,
           label=f'Original Dataset ({n_cats_full} categories)')
    ax.bar(x + w/2, sub_pcts, w, color='#EA4335', alpha=0.7,
           label=f'Our Subset ({n_cats_sub} categories)')

    ax.set_xticks(x)
    ax.set_xticklabels(top_cats, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title(f'Activity Category Distribution (Top 20)\n'
                 f'Coverage: {n_cats_sub}/200  |  χ²={chi2_val:.1f} (p={p_chi2:.3f})  |  JSD={jsd:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'videoxum_category_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  类别分布图已保存: {output_path}")
    plt.close()

    return output_path, n_cats_full, n_cats_sub, missing_cats, cat_test_results


# ============================================================
# Save Results
# ============================================================

def save_results(data, test_results, cat_test_results=None):
    """保存完整结果为JSON: 论文参考 vs 子集"""
    sd = data['subset']['durations']
    sc = data['subset']['compressions']
    ss = data['subset']['n_shots']

    results = {
        'paper_reference': PAPER_STATS,
        'our_subset': {
            'n_videos': int(len(sd)),
            'duration': {
                'mean': float(np.mean(sd)),
                'median': float(np.median(sd)),
                'std': float(np.std(sd)),
                'min': float(np.min(sd)),
                'max': float(np.max(sd)),
                'percentile_25': float(np.percentile(sd, 25)),
                'percentile_75': float(np.percentile(sd, 75)),
                'percentile_99_9': float(np.percentile(sd, 99.9)),
            },
            'compression_ratio': {
                'mean': float(np.mean(sc)),
                'median': float(np.median(sc)),
                'std': float(np.std(sc)),
                'min': float(np.min(sc)),
                'max': float(np.max(sc)),
            },
            'n_shots': {
                'mean': float(np.mean(ss)),
                'median': float(np.median(ss)),
                'std': float(np.std(ss)),
            },
            'n_timestamps': int(len(data['subset']['timestamps'])),
        },
        'statistical_tests': test_results,
    }

    if cat_test_results is not None:
        results['category_tests'] = cat_test_results

    # 递归转换numpy类型为Python原生类型
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    results = convert(results)

    output_path = OUTPUT_DIR / 'videoxum_sampling_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  数据结果已保存: {output_path}")

    return output_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='VideoXum 采样代表性分析')
    parser.add_argument('--skip-timestamps', action='store_true',
                        help='跳过时间戳分析 (可节省大量时间)')
    parser.add_argument('--skip-categories', action='store_true',
                        help='跳过类别分布分析')
    args = parser.parse_args()

    print("=" * 80)
    print("VideoXum 采样数据集代表性分析")
    print("=" * 80)
    print(f"H5文件: {H5_PATH}")
    print(f"JSON文件: {JSON_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载并处理数据
    data = load_and_process_data(skip_timestamps=args.skip_timestamps)

    # 汇总统计对比表
    print_summary_table(data)

    # 统计检验
    test_results = run_statistical_tests(data, skip_timestamps=args.skip_timestamps)

    # 生成主图
    main_plot = create_main_plots(data, test_results, skip_timestamps=args.skip_timestamps)

    # 类别分布 (如果可用)
    cat_plot = None
    n_cats_full, n_cats_sub, missing_cats = 0, 0, set()
    cat_test_results = None
    if not args.skip_categories:
        print("\n--- 类别分布分析 ---")
        video_categories = load_activitynet_categories()
        cat_result = create_category_plot(data, video_categories)
        if cat_result is not None:
            cat_plot, n_cats_full, n_cats_sub, missing_cats, cat_test_results = cat_result

    # 保存结果
    print("\n--- 保存结果 ---")
    json_path = save_results(data, test_results, cat_test_results)

    # 最终总结
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print(f"  主图: {main_plot}")
    if cat_plot:
        print(f"  类别图: {cat_plot}")
    print(f"  数据: {json_path}")
    print()

    # 简要结论
    all_pass = all(
        not v.get('significant', True)
        for k, v in test_results.items()
        if k.endswith('_ks')
    )
    if all_pass:
        print("结论: 所有KS检验均未发现显著差异, 采样子集具有良好的代表性。")
    else:
        failed = [k.replace('_ks', '') for k, v in test_results.items()
                  if k.endswith('_ks') and v.get('significant', False)]
        print(f"注意: 以下指标的KS检验显示显著差异: {', '.join(failed)}")
        print("  但大样本下KS检验可能过于敏感, 请同时参考Wasserstein距离和图形对比。")


if __name__ == '__main__':
    main()
