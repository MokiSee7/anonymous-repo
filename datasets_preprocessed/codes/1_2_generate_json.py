import os
import json
import h5py
import pandas as pd
import re
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def normalize_name(name):
    """把 video1 → video_1"""
    return re.sub(r'video(\d+)', r'video_\1', name)


def find_nearest_valid_description(df, idx):
    """
    邻近替代策略：
    - 如果 status != "OK"，就往前或往后找最近的 OK 描述
    """
    n = len(df)
    offset = 1
    while offset < n:
        # 往前找
        if idx - offset >= 0 and str(df.iloc[idx - offset]["Status"]).upper() == "OK":
            return df.iloc[idx - offset]["Caption"]
        # 往后找
        if idx + offset < n and str(df.iloc[idx + offset]["Status"]).upper() == "OK":
            return df.iloc[idx + offset]["Caption"]
        offset += 1
    return "No description available"


def add_picks_to_h5_if_missing(h5_path, json_data, dataset_name):
    """
    Add 'picks' field to H5 file if it doesn't exist

    This ensures all datasets have consistent H5 structure.
    For datasets like OVP/YouTube that don't originally have picks,
    this adds them based on the frame_id values from JSON.

    Parameters:
    - h5_path: Path to H5 file
    - json_data: Nested dict {video_id: {frames: [{frame_id, ...}]}}
    - dataset_name: Name of dataset for logging
    """
    # Check if we need to add picks
    needs_update = False
    videos_to_update = []

    with h5py.File(h5_path, 'r') as h5file:
        for video_id in h5file.keys():
            if 'picks' not in h5file[video_id]:
                needs_update = True
                videos_to_update.append(video_id)

    if not needs_update:
        return  # All videos already have picks

    print(f"\n📝 添加 'picks' 字段到 H5 文件...")
    print(f"   需要更新 {len(videos_to_update)}/{len(json_data)} 个视频")

    # Open H5 in write mode to add picks
    with h5py.File(h5_path, 'a') as h5file:
        for video_id in videos_to_update:
            if video_id not in json_data:
                continue

            if 'frames' not in json_data[video_id]:
                continue

            # Extract picks from frame_id values
            picks = np.array([frame['frame_id'] for frame in json_data[video_id]['frames']], dtype=np.int64)

            # Add to H5 file
            try:
                h5file[video_id].create_dataset('picks', data=picks, dtype=np.int64)
            except Exception as e:
                print(f"   ⚠️  无法为 {video_id} 添加 picks: {e}")

    print(f"   ✅ 已为 {len(videos_to_update)} 个视频添加 'picks' 字段")


def generate_json(excel_dir, h5_path, dataset_name, output_path):
    """
    Generate JSON from Excel frame descriptions and H5 ground truth data

    Parameters:
    - excel_dir: Directory containing Excel files with frame descriptions
    - h5_path: Path to H5 file containing ground truth annotations
    - dataset_name: Name of the dataset (SumMe or TVSum)
    - output_path: Path to save the output JSON file
    """
    data = []

    with h5py.File(h5_path, "r") as h5file:
        video_keys = list(h5file.keys())
        for video_key in tqdm(video_keys, desc=f"  Processing videos", unit="video"):
            if dataset_name.lower() == "summe":
                # SumMe: Excel 文件名就是 h5 的 video_name 字段
                video_name_bytes = h5file[video_key]["video_name"][()]
                video_name = video_name_bytes.decode() if isinstance(video_name_bytes, bytes) else str(video_name_bytes)
                excel_name = f"{video_name}.xlsx"
            else:
                # TVSum: Excel 文件名用 video_key 映射
                excel_name = video_key + ".xlsx"

            excel_path = os.path.join(excel_dir, excel_name)
            if not os.path.exists(excel_path):
                print(f"⚠️ Excel 文件缺失: {excel_path}")
                continue

            df = pd.read_excel(excel_path)

            picks = h5file[video_key]["picks"][:]

            # Prefer gtframe, fall back to gtsummary
            if "gtframe" in h5file[video_key]:
                gt_label = h5file[video_key]["gtframe"][:]
            elif "gtsummary" in h5file[video_key]:
                gt_label = h5file[video_key]["gtsummary"][:]
            else:
                gt_label = None

            for idx, frame_id in enumerate(picks):
                if idx >= len(df):
                    continue
                description = df.iloc[idx]["Caption"]
                status_str = str(df.iloc[idx]["Status"])
                status = 0 if status_str.upper() == "OK" else 1

                # 邻近替代策略
                if status == 1:
                    description = find_nearest_valid_description(df, idx)

                entry = {
                    "frame_id": int(frame_id),
                    "video_id": video_key,
                    "description": description,
                    "status": status,
                    "is_gt": int(gt_label[idx]) if gt_label is not None else None
                }
                data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON 已生成: {output_path}")


def generate_dataset_json_nested(dataset_name):
    """
    为单个数据集生成嵌套格式的JSON（所有路径从配置读取）
    输出格式: {video_id: {frames: [{frame_id, description, is_gt, status}]}}

    Parameters:
    - dataset_name: 数据集名称（如 summe, tvsum, ovp, youtube）
    """
    from collections import defaultdict
    video_dict = defaultdict(lambda: {"frames": []})

    ds_lower = dataset_name.lower()

    # 从配置读取路径（纯配置，无回退）
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.config_loader import get_config
    config = get_config()
    ds_paths = config.get_dataset_paths(ds_lower)

    excel_dir = ds_paths.get('frame_descriptions_dir')
    h5_path = ds_paths.get('h5_file')
    output_dir = ds_paths.get('processed_dir')

    if not excel_dir:
        raise ValueError(f"配置文件中未定义 {ds_lower}.frame_descriptions_dir")
    if not h5_path:
        raise ValueError(f"配置文件中未定义 {ds_lower}.h5_file")
    if not output_dir:
        raise ValueError(f"配置文件中未定义 paths.processed_dir")

    excel_dir = str(excel_dir)
    h5_path = str(h5_path)
    output_dir = str(output_dir)

    # Output file
    output_path = os.path.join(output_dir, f"{ds_lower}_data.json")

    # Validate paths
    if not os.path.exists(h5_path):
        print(f"❌ H5 文件不存在: {h5_path}")
        return False

    if not os.path.exists(excel_dir):
        print(f"❌ Excel 目录不存在: {excel_dir}")
        print(f"   请先运行 step 1.1 生成 frame descriptions")
        return False

    print(f"📂 Excel 目录: {excel_dir}")
    print(f"📂 H5 文件: {h5_path}")
    print(f"📂 输出文件: {output_path}")

    with h5py.File(h5_path, "r") as h5file:
        video_keys = list(h5file.keys())
        for video_key in tqdm(video_keys, desc=f"  Processing {dataset_name} videos", unit="video"):
            if dataset_name.lower() == "summe":
                video_name_bytes = h5file[video_key]["video_name"][()]
                video_name = video_name_bytes.decode() if isinstance(video_name_bytes, bytes) else str(video_name_bytes)
                excel_name = f"{video_name}.xlsx"
            else:
                excel_name = video_key + ".xlsx"

            excel_path = os.path.join(excel_dir, excel_name)
            if not os.path.exists(excel_path):
                print(f"⚠️ Excel 文件缺失: {excel_path}, 跳过 {video_key}")
                continue

            df = pd.read_excel(excel_path)

            # Check if 'picks' exists (TVSum/SumMe have it, OVP/YouTube don't)
            if "picks" in h5file[video_key]:
                picks = h5file[video_key]["picks"][:]
            else:
                # For datasets without picks (OVP/YouTube):
                # Generate picks with 15-frame intervals to match TVSum/SumMe format
                # This represents absolute frame numbers in the original video (0, 15, 30, 45, ...)
                # Assuming 2fps sampling from ~30fps video = 15:1 ratio
                n_frames = len(df)
                picks = np.array([i * 15 for i in range(n_frames)])

            # Prefer gtframe, fall back to gtsummary
            if "gtframe" in h5file[video_key]:
                gt_label = h5file[video_key]["gtframe"][:]
            elif "gtsummary" in h5file[video_key]:
                gt_label = h5file[video_key]["gtsummary"][:]
            else:
                gt_label = None

            for idx, frame_id in enumerate(picks):
                if idx >= len(df):
                    continue

                description = df.iloc[idx]["Caption"]
                status_str = str(df.iloc[idx]["Status"])
                status = 0 if status_str.upper() == "OK" else 1

                if status == 1:
                    description = find_nearest_valid_description(df, idx)

                frame_entry = {
                    "frame_id": int(frame_id),
                    "description": description,
                    "is_gt": int(gt_label[idx]) if gt_label is not None else None,
                    "status": status
                }
                video_dict[video_key]["frames"].append(frame_entry)

    # Convert to regular dict
    final_dict = dict(video_dict)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ {dataset_name} JSON 已生成: {output_path}")
    print(f"   包含 {len(final_dict)} 个视频")

    # Save per-video independent JSON files
    per_video_dir = os.path.join(output_dir, "step1_frames")
    os.makedirs(per_video_dir, exist_ok=True)
    for video_key, video_data in final_dict.items():
        per_video_path = os.path.join(per_video_dir, f"{video_key}.json")
        with open(per_video_path, "w", encoding="utf-8") as f:
            json.dump({"video_id": video_key, "frames": video_data["frames"]},
                      f, ensure_ascii=False, indent=2)
    print(f"   per-video JSON: {per_video_dir}/ ({len(final_dict)} files)")

    return True




def main():
    """主入口函数 - 所有路径从配置文件读取"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate JSON from Excel frame descriptions and H5 ground truth data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets
  python 1_2_generate_json.py

  # Process a specific dataset
  python 1_2_generate_json.py --dataset summe
  python 1_2_generate_json.py --dataset tvsum

注意：所有路径从配置文件读取，请确保已正确设置 configs/config.yaml
        """
    )
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name to process (e.g., summe, tvsum, ovp, youtube). If not specified, all datasets will be processed.')
    # Suppressed args for pipeline compatibility (paths come from config)
    parser.add_argument('--h5-dir', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--output-dir', type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # 从配置加载
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.config_loader import get_config
    config = get_config()

    print("="*80)
    print("Step 1.2: Generate JSON from Frame Descriptions")
    print("="*80)

    # 确定要处理的数据集
    if args.dataset:
        dataset_list = [args.dataset.lower()]
        print(f"🎯 Target dataset: {args.dataset.lower()}")
    else:
        dataset_list = config.detect_datasets()
        if not dataset_list:
            print("❌ 配置文件中未检测到任何数据集!")
            print("   请确保配置文件中定义了数据集的 h5_file 路径")
            return
        print(f"🎯 检测到数据集: {', '.join(dataset_list)}")

    print("="*80 + "\n")

    # Process each dataset
    success_count = 0
    for dataset in dataset_list:
        print(f"\n🎬 处理数据集: {dataset}")
        print("-"*80)
        result = generate_dataset_json_nested(dataset)
        if result:
            success_count += 1
        print("-"*80)

    print("\n" + "="*80)
    print(f"✅ 完成! 成功处理 {success_count}/{len(dataset_list)} 个数据集")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
