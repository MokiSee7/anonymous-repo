#!/usr/bin/env python3
"""
MMS Benchmark Data Generation Pipeline

This script runs the complete pipeline for generating multi-modal multi-granularity video summaries.
Supports both standard datasets (SumMe, TVSum) and YouTube-based datasets (VideoXum).

Pipeline Steps:
Step 1: Frame-level descriptions
  - Standard datasets (SumMe/TVSum): Videos exist → Extract frames → Generate descriptions → JSON
  - VideoXum: Download from YouTube → Extract frames → Generate descriptions → Compress tar.gz → JSON

Step 2: Shot-level descriptions
  - Standard datasets: Read frames from directory
  - VideoXum: Read frames from tar.gz archives
  - Generate shot-level descriptions with GPT-4V + text fallback (or LLaVA if --use-llava)

Step 3: Aligned documents and summaries
  - All datasets: Generate shot-aligned documents and summaries
  - Default: LLaMA (local model)
  - Alternatives: LLaVA (--use-llava) or Azure OpenAI (--use-gpt)

Usage:
    # Standard datasets (default: LLaMA for Step 3)
    python 5_run_pipeline.py --dataset SumMe --all
    python 5_run_pipeline.py --dataset TVSum --steps 1 2 3

    # Using local LLaVA model for Steps 2 and 3
    python 5_run_pipeline.py --dataset SumMe --all --use-llava

    # Using Azure OpenAI API for Steps 2 and 3
    python 5_run_pipeline.py --dataset SumMe --all --use-gpt

    # VideoXum dataset (with testing limit)
    python 5_run_pipeline.py --dataset videoxum --step 1 --limit 2 --batch-size 1 --workers 1
    python 5_run_pipeline.py --dataset videoxum --all --batch-size 10 --workers 4

    # All datasets
    python 5_run_pipeline.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Directory containing these pipeline scripts (use absolute paths for subprocess calls)
SCRIPT_DIR = Path(__file__).resolve().parent
# Project root directory (parent of codes directory)
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Add project root to sys.path so we can import utils
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def detect_datasets_from_config():
    """
    从配置文件检测可用的数据集

    Returns:
        list: 数据集名称列表（小写）
    """
    from utils.config_loader import get_config
    config = get_config()
    return config.detect_datasets()


class PipelineRunner:
    """Manages the execution of the MMS benchmark pipeline"""

    def __init__(self, dataset, video_limit=None, use_gpt=False, step3_use_llava=False):
        # Normalize dataset name to lowercase for consistency
        self.dataset = dataset.lower()

        # Steps 1/2: vllava by default, GPT if --use-gpt
        self.use_llava = not use_gpt
        # Step 3: llama by default; llava if --use-llava; gpt if --use-gpt
        if use_gpt:
            self.step3_model = 'gpt'
        elif step3_use_llava:
            self.step3_model = 'llava'
        else:
            self.step3_model = 'llama'

        # Load config to get dataset-specific paths (纯配置，无回退)
        from utils.config_loader import get_config
        config = get_config()
        ds_paths = config.get_dataset_paths(self.dataset)

        # H5 file (必须在配置中定义)
        self.h5_file = ds_paths['h5_file']
        if not self.h5_file:
            raise ValueError(f"配置文件中未定义 {self.dataset}.h5_file")
        self.h5_dir = self.h5_file.parent

        # Processed data directory (必须在配置中定义)
        self.data_dir = ds_paths['processed_dir']
        if not self.data_dir:
            raise ValueError(f"配置文件中未定义 paths.processed_dir")
        self.dataset_file = self.data_dir / f"{self.dataset}_data.json"

        # Frames directory (必须在配置中定义)
        self.frames_dir = ds_paths['frames_dir']
        # Frame descriptions directory
        self.frame_descriptions_dir = ds_paths['frame_descriptions_dir']
        # Statistics directory (必须在配置中定义)
        self.statistics_dir = ds_paths['statistics_dir']
        if not self.statistics_dir:
            raise ValueError(f"配置文件中未定义 paths.statistics_dir")
        # Checkpoints directory (必须在配置中定义)
        self.checkpoints_dir = ds_paths['checkpoints_dir']
        if not self.checkpoints_dir:
            raise ValueError(f"配置文件中未定义 paths.checkpoints_dir")

        # Video limit for testing (mainly for VideoXum)
        self.video_limit = video_limit

        # Detect if this is VideoXum dataset (requires YouTube download)
        self.is_videoxum = self.dataset == 'videoxum'

        # Load config for VideoXum-specific paths
        if self.is_videoxum:
            self.videoxum_config = {
                'videos_dir': str(ds_paths['videos_dir']) if ds_paths['videos_dir'] else None,
                'frames_dir': str(ds_paths['frames_dir']) if ds_paths['frames_dir'] else None,
                'h5_file': str(self.h5_file),
                'json_file': str(self.dataset_file),
            }
            if not self.videoxum_config['videos_dir']:
                raise ValueError(f"配置文件中未定义 videoxum.videos_dir")

    def run_step_1_videoxum(self, batch_size=10, workers=4, skip_download=False):
        """Step 1 for VideoXum: Download + Extract + Generate frame descriptions"""
        print("\n" + "="*80)
        if skip_download:
            print("STEP 1: VideoXum - Extract and Generate Frame Descriptions (Skip Download)")
        else:
            print("STEP 1: VideoXum - Download, Extract, and Generate Frame Descriptions")
        print("="*80)

        cmd = [
            sys.executable, str(SCRIPT_DIR / "1_0_videoxum_batch_processor.py"),
            "--h5-path", str(self.videoxum_config['h5_file']),
            "--video-dir", str(self.videoxum_config['videos_dir']),
            "--frames-dir", str(self.videoxum_config['frames_dir']),
            "--json-path", str(self.videoxum_config['json_file']),
            "--batch-size", str(batch_size),
            "--workers", str(workers),
        ]

        # Add limit if specified (for testing)
        if self.video_limit:
            cmd.extend(["--limit", str(self.video_limit)])

        # Add skip-download flag if specified
        if skip_download:
            cmd.append("--skip-download")

        result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))

        if result.returncode != 0:
            print(f"❌ Step 1 (VideoXum) failed with exit code {result.returncode}")
            return False

        print(f"✅ Step 1 (VideoXum) completed successfully")
        return True

    def run_step_1_1(self):
        """Step 1.1: Generate frame-level descriptions with GPT-4o-mini or LLaVA"""
        print("\n" + "="*80)
        if self.use_llava:
            print("STEP 1.1: Generating Frame-Level Descriptions (VLLaVA - Local Model)")
            script_name = "1_1_generate_frame_descriptions_llava.py"
        else:
            print("STEP 1.1: Generating Frame-Level Descriptions (Azure OpenAI API)")
            script_name = "1_1_generate_frame_descriptions.py"
        print("="*80)

        cmd = [sys.executable, str(SCRIPT_DIR / script_name), "--dataset", self.dataset]
        result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))

        if result.returncode != 0:
            print(f"❌ Step 1.1 failed with exit code {result.returncode}")
            return False

        print(f"✅ Step 1.1 completed successfully")
        return True

    def run_step_1_2(self):
        """Step 1.2: Generate frame-level descriptions JSON from Excel + H5"""
        print("\n" + "="*80)
        print("STEP 1.2: Generating Frame-Level Descriptions JSON")
        print("="*80)

        cmd = [
            sys.executable, str(SCRIPT_DIR / "1_2_generate_json.py"),
            "--dataset", self.dataset,
            "--h5-dir", str(self.h5_dir.resolve()),
            "--output-dir", str(self.data_dir.resolve())
        ]
        result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))

        if result.returncode != 0:
            print(f"❌ Step 1.2 failed with exit code {result.returncode}")
            return False

        # Check if output file was created
        if not self.dataset_file.exists():
            print(f"❌ Expected output file not found: {self.dataset_file}")
            return False

        print(f"✅ Step 1.2 completed successfully")

        # Run statistics for Step 1
        if not self.run_statistics_1():
            print(f"⚠️  Warning: Statistics generation failed, but continuing...")

        return True

    def run_kts_segmentation(self, target_ratio=10):
        """Step 2.0a: Check that change_points exist in H5 file (read-only)"""
        print("\n" + "="*80)
        print("STEP 2.0a: Checking for change_points (shot boundaries) in H5 file")
        print("="*80)

        if not self.h5_file.exists():
            print(f"⚠️  H5 file not found: {self.h5_file}")
            return False

        import h5py
        try:
            with h5py.File(self.h5_file, 'r') as f:
                video_keys = list(f.keys())
                if not video_keys:
                    print(f"⚠️  H5 file is empty")
                    return False

                first_video = video_keys[0]
                if 'change_points' in f[first_video]:
                    print(f"✅ change_points already exist in H5 file")
                    return True
                else:
                    print(f"❌ change_points not found in H5 file")
                    print(f"   Please run KTS preprocessing first:")
                    print(f"   python 0_2_KTS.py --dataset {self.dataset}")
                    return False
        except Exception as e:
            print(f"⚠️  Could not check H5 file: {e}")
            return False

    def run_knapsack_gt(self, gt_portion=0.15):
        """Step 2.0b: Check that shot-level GT exists in H5 file (read-only)"""
        print("\n" + "="*80)
        print("STEP 2.0b: Checking for Shot-Level GT in H5 file")
        print("="*80)

        if not self.h5_file.exists():
            print(f"⚠️  H5 file not found: {self.h5_file}")
            print(f"   Step 2 will continue with gt=0 for all clips")
            return False

        import h5py
        try:
            with h5py.File(self.h5_file, 'r') as f:
                video_keys = list(f.keys())
                if not video_keys:
                    print(f"⚠️  H5 file is empty")
                    return False

                first_video = video_keys[0]
                if 'shot_level_gt' in f[first_video]:
                    print(f"✅ Shot-level GT already exists in H5 file")
                    return True
                else:
                    print(f"⚠️  shot_level_gt not found in H5 file")
                    print(f"   Please run Knapsack preprocessing first:")
                    print(f"   python 2_0_knapsack_shot_gt.py --dataset {self.dataset}")
                    print(f"   Step 2 will continue with gt=0 for all clips")
                    return False
        except Exception as e:
            print(f"⚠️  Could not check H5 file: {e}")
            return False

    def run_statistics_1(self):
        """Generate statistics for Step 1"""
        print("\n" + "="*80)
        print("Generating Statistics for Step 1")
        print("="*80)

        statistics_dir = self.statistics_dir

        cmd = [
            sys.executable, str(SCRIPT_DIR / "4_statistics_1.py"),
            "--dataset", self.dataset,
            "--data-dir", str(self.data_dir.resolve()),
            "--output-dir", str(statistics_dir)
        ]
        result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))

        if result.returncode != 0:
            print(f"❌ Statistics generation failed with exit code {result.returncode}")
            return False

        print(f"✅ Statistics generated successfully")
        return True

    def run_statistics_2(self):
        """Generate statistics for Step 2"""
        print("\n" + "="*80)
        print("Generating Statistics for Step 2")
        print("="*80)

        statistics_dir = self.statistics_dir

        cmd = [
            sys.executable, str(SCRIPT_DIR / "4_statistics_2.py"),
            "--dataset", self.dataset,
            "--data-dir", str(self.data_dir.resolve()),
            "--output-dir", str(statistics_dir)
        ]
        result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))

        if result.returncode != 0:
            print(f"❌ Statistics generation failed with exit code {result.returncode}")
            return False

        print(f"✅ Statistics generated successfully")
        return True

    def run_step_2(self, retry_filtered=False, gt_portion=0.15, no_skip=False):
        """Step 2: Generate shot-level descriptions"""
        print("\n" + "="*80)
        if self.use_llava:
            print("STEP 2: Generating Shot-Level Descriptions (VLLaVA - Local Model)")
        else:
            print("STEP 2: Generating Shot-Level Descriptions (Azure OpenAI API)")
        print("="*80)

        # Check if input file exists
        if not self.dataset_file.exists():
            print(f"❌ Input file not found: {self.dataset_file}")
            print(f"   Please run steps 1.1 and 1.2 first")
            return False

        # Step 2.0a: Check and generate change_points using KTS if needed
        if not self.run_kts_segmentation(target_ratio=10):
            print(f"⚠️  Warning: KTS segmentation failed or not needed")
            print(f"   Attempting to continue anyway...")

        # Step 2.0b: Check and generate shot-level GT if needed
        self.run_knapsack_gt(gt_portion=gt_portion)

        # frames_dir: use per-dataset frames dir's parent so the script can find {dataset}/frames/
        # frames_dir 必须在配置中定义
        if not self.frames_dir:
            raise ValueError(f"配置文件中未定义 {self.dataset}.frames_dir")

        if self.use_llava:
            # VLLaVA version: uses config-based paths directly
            cmd = [
                sys.executable, str(SCRIPT_DIR / "2_generate_shot_descriptions_llava.py"),
                "--dataset", self.dataset,
            ]
            if no_skip:
                cmd.append("--no-skip")
        else:
            # Azure API version
            # For per-dataset structure, pass the frames_dir parent (data_root level)
            # The script expects: frames_dir/{dataset}/frames/{video_id}/
            frames_dir_for_script = self.frames_dir.parent
            cmd = [
                sys.executable, str(SCRIPT_DIR / "2_generate_shot_descriptions.py"),
                "--dataset", self.dataset,
                "--h5-dir", str(self.h5_dir.resolve()),
                "--base-dir", str(self.data_dir.resolve()),
                "--frames-dir", str(frames_dir_for_script)
            ]

            if retry_filtered:
                cmd.append("--retry-filtered")

            if no_skip:
                cmd.append("--no-skip")

        result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))

        if result.returncode != 0:
            print(f"❌ Step 2 failed with exit code {result.returncode}")
            return False

        print(f"✅ Step 2 completed successfully")

        # Run statistics for Step 2
        if not self.run_statistics_2():
            print(f"⚠️  Warning: Statistics generation failed, but continuing...")

        return True

    def run_statistics_3(self):
        """Generate statistics for Step 3"""
        print("\n" + "="*80)
        print("Generating Statistics for Step 3")
        print("="*80)

        statistics_dir = self.statistics_dir

        cmd = [
            sys.executable, str(SCRIPT_DIR / "4_statistics_3.py"),
            "--dataset", self.dataset,
            "--data-dir", str(self.data_dir.resolve()),
            "--output-dir", str(statistics_dir)
        ]
        result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))

        if result.returncode != 0:
            print(f"❌ Statistics generation failed with exit code {result.returncode}")
            return False

        print(f"✅ Statistics generated successfully")
        return True

    def run_step_3(self, gt_only=False, deployment=None, no_skip=False):
        """Step 3: Generate shot-aligned documents and summaries"""
        print("\n" + "="*80)
        _label = {'llama': 'LLaMA - Local Model', 'llava': 'VLLaVA - Local Model', 'gpt': 'Azure OpenAI API'}
        print(f"STEP 3: Generating Shot-Aligned Documents and Summaries ({_label[self.step3_model]})")
        print("="*80)

        # Check if input file exists
        if not self.dataset_file.exists():
            print(f"❌ Input file not found: {self.dataset_file}")
            print(f"   Please run steps 1.2 and 2 first")
            return False

        if self.step3_model == 'llama':
            cmd = [
                sys.executable, str(SCRIPT_DIR / "3_generate_aligned_document_llama.py"),
                "--dataset", self.dataset,
            ]
        elif self.step3_model == 'llava':
            cmd = [
                sys.executable, str(SCRIPT_DIR / "3_generate_aligned_document_llava.py"),
                "--dataset", self.dataset,
            ]
        else:
            # Azure API version
            cmd = [
                sys.executable, str(SCRIPT_DIR / "3_generate_aligned_document.py"),
                "--dataset", self.dataset,
                "--data-dir", str(self.data_dir.resolve())
            ]

        if gt_only:
            cmd.append("--gt-only")

        if deployment and self.step3_model == 'gpt':
            # Only pass deployment to Azure version
            cmd.extend(["--deployment", deployment])

        if no_skip:
            cmd.append("--no-skip")

        result = subprocess.run(cmd, capture_output=False, cwd=str(SCRIPT_DIR))

        if result.returncode != 0:
            print(f"❌ Step 3 failed with exit code {result.returncode}")
            return False

        print(f"✅ Step 3 completed successfully")

        # Run statistics for Step 3
        if not self.run_statistics_3():
            print(f"⚠️  Warning: Statistics generation failed, but continuing...")

        return True

    def run_all_steps(self, retry_filtered=False, gt_portion=0.15, gt_only=False, deployment=None, no_skip=False, batch_size=10, workers=4):
        """Run all pipeline steps sequentially"""
        print("\n" + "="*80)
        print(f"Running Complete Pipeline for {self.dataset}")
        print("="*80)

        # Step 1: Different processing based on dataset type
        if self.is_videoxum:
            # VideoXum: Download + Extract + Generate descriptions (all in one step)
            if not self.run_step_1_videoxum(batch_size=batch_size, workers=workers):
                return False
        else:
            # Standard datasets: Step 1.1 + 1.2
            if not self.run_step_1_1():
                return False
            if not self.run_step_1_2():
                return False

        # Step 2 (includes automatic Step 2.0 for GT generation)
        if not self.run_step_2(retry_filtered=retry_filtered, gt_portion=gt_portion, no_skip=no_skip):
            return False

        # Step 3
        if not self.run_step_3(gt_only=gt_only, deployment=deployment, no_skip=no_skip):
            return False

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📂 Final output: {self.dataset_file}")
        print("\nGenerated data includes:")
        print("  - Frame-level descriptions (frames)")
        print("  - Shot-level descriptions (clips)")
        print("  - Aligned full document (aligned_full_document.sentences)")
        print("  - Aligned GT summary (aligned_text_summary.sentences)")
        print("="*80 + "\n")

        return True


def main():
    """主入口函数 - 所有路径从配置文件读取"""

    parser = argparse.ArgumentParser(
        description='Run MMS Benchmark Data Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard datasets (SumMe, TVSum) - videos already exist
  python 5_run_pipeline.py --dataset SumMe --all
  python 5_run_pipeline.py --dataset TVSum --steps 1 2

  # VideoXum dataset - downloads from YouTube
  # Test with 2 videos first
  python 5_run_pipeline.py --dataset videoxum --step 1 --limit 2 --batch-size 1 --workers 1

  # Process all VideoXum videos
  python 5_run_pipeline.py --dataset videoxum --all --batch-size 10 --workers 4

  # Process all datasets
  python 5_run_pipeline.py --all

  # Advanced options
  python 5_run_pipeline.py --dataset TVSum --step 2 --retry-filtered
  python 5_run_pipeline.py --dataset SumMe --step 3 --deployment gpt-4o-risk
  python 5_run_pipeline.py --dataset SumMe --step 2 --gt-portion 0.10

  # Step 3 uses LLaMA by default; use VLLaVA for Step 3 instead
  python 5_run_pipeline.py --dataset SumMe --step 3 --use-llava

  # Use Azure OpenAI API for all steps
  python 5_run_pipeline.py --dataset SumMe --all --use-gpt

Pipeline Steps:
  Step 1: Frame-level descriptions
    - Standard datasets: Videos exist → Extract frames → Generate descriptions → JSON
    - VideoXum: Download YouTube → Extract frames → Generate descriptions → Compress tar.gz → JSON

  Step 2: Shot-level descriptions
    - Standard datasets: Read frames from directory
    - VideoXum: Read frames from tar.gz archives
    - Auto-generate shot boundaries (KTS) and GT (Knapsack) if missing
    - Generate shot descriptions with GPT-4V + text fallback

  Step 3: Aligned documents and summaries
    - All datasets: Generate shot-aligned documents and summaries

注意：所有路径从配置文件读取，请确保已正确设置 configs/config.yaml
        """
    )

    # Note: We don't use choices validation here to allow case-insensitive input
    parser.add_argument('--dataset', type=str, required=False,
                       help='Dataset to process (case-insensitive). If not specified, all datasets will be processed.')

    # Pipeline control
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true',
                       help='Run all pipeline steps (1.2, 2, 3)')
    group.add_argument('--steps', nargs='+', type=int, choices=[1, 2, 3],
                       help='Run specific steps (e.g., --steps 1 2)')
    group.add_argument('--step', type=int, choices=[1, 2, 3],
                       help='Run a single step')

    # Step-specific options
    parser.add_argument('--retry-filtered', action='store_true',
                       help='[Step 2] Retry processing filtered content')
    parser.add_argument('--gt-portion', type=float, default=0.15,
                       help='[Step 2.0] GT selection portion for Knapsack algorithm (default: 0.15 = 15%%)')
    parser.add_argument('--gt-only', action='store_true',
                       help='[Step 3] Generate only GT text summary (no full document)')
    parser.add_argument('--deployment', type=str, default=None,
                       help='[Step 3] Azure deployment name (default: from config)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess all items (disable checkpoint resume for steps 2 and 3)')

    # VideoXum-specific options
    parser.add_argument('--limit', type=int, default=None,
                       help='[VideoXum Step 1] Limit number of videos to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='[VideoXum Step 1] Number of videos per batch (default: 10)')
    parser.add_argument('--workers', type=int, default=4,
                       help='[VideoXum Step 1] Number of parallel workers (default: 4)')
    parser.add_argument('--skip-download', action='store_true',
                       help='[VideoXum Step 1] Skip downloading videos, only process existing ones')

    # Model selection
    # Steps 1/2 default: VLLaVA. Step 3 default: LLaMA.
    parser.add_argument('--use-gpt', action='store_true',
                       help='Use Azure OpenAI API for all steps (Steps 1.1, 2, and 3)')
    parser.add_argument('--use-llava', action='store_true',
                       help='[Step 3 only] Use VLLaVA instead of LLaMA for Step 3')

    args = parser.parse_args()

    # 从配置文件检测可用数据集
    datasets_found = detect_datasets_from_config()

    # 打印检测到的数据集
    if datasets_found:
        print("\n" + "="*80)
        print("配置文件中定义的数据集")
        print("="*80)
        for dataset in datasets_found:
            print(f"  - {dataset}")
        print("="*80 + "\n")

    # 检查是否检测到数据集
    if not datasets_found:
        print("❌ 配置文件中未检测到数据集!")
        print("   请确保配置文件中定义了数据集的 h5_file 路径")
        print("   例如: summe.h5_file: /path/to/summe.h5")
        sys.exit(1)

    # 确定要处理的数据集
    if args.dataset:
        dataset_lower = args.dataset.lower()
        if dataset_lower not in datasets_found:
            print(f"❌ 数据集 '{args.dataset}' 未在配置文件中定义!")
            print(f"   可用数据集: {', '.join(datasets_found)}")
            sys.exit(1)
        datasets_to_process = [dataset_lower]
    else:
        datasets_to_process = datasets_found

    # Determine which steps to run
    if args.all:
        steps_to_run = [1, 2, 3]
    elif args.steps:
        steps_to_run = sorted(args.steps)
    else:
        steps_to_run = [args.step]

    # Process each dataset
    for dataset in datasets_to_process:
        # Initialize pipeline runner for this dataset (所有路径从配置读取)
        runner = PipelineRunner(
            dataset,
            video_limit=args.limit,
            use_gpt=args.use_gpt,
            step3_use_llava=args.use_llava
        )

        print("\n" + "="*80)
        print(f"MMS Benchmark Pipeline - {dataset}")
        print("="*80)
        print(f"Steps to run: {steps_to_run}")
        print(f"H5 file: {runner.h5_file}")
        print(f"Data directory: {runner.data_dir}")
        print(f"Output file: {runner.dataset_file}")
        step12_label = 'Azure OpenAI API' if not runner.use_llava else 'VLLaVA (Local Model)'
        step3_label  = {'llama': 'LLaMA (Local Model)', 'llava': 'VLLaVA (Local Model)', 'gpt': 'Azure OpenAI API'}[runner.step3_model]
        print(f"Steps 1/2 model : {step12_label}")
        print(f"Step 3 model    : {step3_label}")
        print("="*80)

        # Run steps
        success = True
        for step in steps_to_run:
            if step == 1:
                # Step 1: Different processing based on dataset type
                if runner.is_videoxum:
                    # VideoXum: Download + Extract + Generate descriptions
                    batch_size = getattr(args, 'batch_size', 10)
                    workers = getattr(args, 'workers', 4)
                    skip_download = getattr(args, 'skip_download', False)
                    success = runner.run_step_1_videoxum(batch_size=batch_size, workers=workers, skip_download=skip_download)
                else:
                    # Standard datasets: Step 1.1 + 1.2
                    if not runner.run_step_1_1():
                        success = False
                        break
                    if not runner.run_step_1_2():
                        success = False
                        break
            elif step == 2:
                success = runner.run_step_2(retry_filtered=args.retry_filtered, gt_portion=args.gt_portion, no_skip=args.no_skip)
            elif step == 3:
                success = runner.run_step_3(gt_only=args.gt_only, deployment=args.deployment, no_skip=args.no_skip)

            if not success:
                print(f"\n❌ Pipeline failed at step {step} for {dataset}")
                sys.exit(1)

        print("\n" + "="*80)
        print(f"✅ ALL REQUESTED STEPS COMPLETED FOR {dataset}!")
        print("="*80 + "\n")

    print("\n" + "="*80)
    print("✅ ALL DATASETS PROCESSED SUCCESSFULLY!")
    print("="*80 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
