#!/usr/bin/env python3
"""
Step 1: Generate Frame-Level Descriptions using LLaVA (Local Model)

This script:
1. Reads frames from configured frames_dir/{video}/*.jpg
2. Uses locally deployed LLaVA model to generate descriptions for each frame
3. Saves results to configured frame_descriptions_dir/{video}.xlsx

This is the LLaVA alternative to 1_1_generate_frame_descriptions.py (Azure OpenAI version)
"""

import os
import glob
import pandas as pd
from pathlib import Path
import time
import json
import argparse
from tqdm import tqdm
import sys
import torch
from PIL import Image

# ==================== Video-LLaVA Import ====================
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# ======================CONFIG==================
BATCH_SIZE = 16  # Number of frames processed in one GPU forward pass
OUTPUT_FOLDER_NAME = "frame_descriptions"

# Model settings - Video-LLaVA
MODEL_PATH = "LanguageBind/Video-LLaVA-7B-hf"

# Checkpoint directory will be initialized in main() with proper absolute path
CHECKPOINT_ROOT = None

# Global model variables (loaded once)
_model = None
_processor = None
_device = None

# ==================== Device Setup ====================

def get_device():
    """Detect and return the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Warning: Using CPU (will be slow)")
    return device


def load_llava_model():
    """Load Video-LLaVA model (called once at startup)"""
    global _model, _processor, _device

    if _model is not None:
        return _processor, _model, _device

    print("Loading Video-LLaVA model...")
    print(f"   Model: {MODEL_PATH}")
    print("   (This may take a few minutes...)")

    _device = get_device()

    _processor = VideoLlavaProcessor.from_pretrained(MODEL_PATH)
    _model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto" if _device.type == "cuda" else None
    )

    if _device.type != "cuda":
        _model = _model.to(_device)

    _model.eval()

    print("   Model loaded successfully\n")
    return _processor, _model, _device


# ==================== HELPER FUNCTIONS ====================

def generate_frame_descriptions_batch(image_paths: list):
    """
    Generate descriptions for a batch of frames in one GPU forward pass.

    Args:
        image_paths: List of image file paths (up to BATCH_SIZE)

    Returns:
        List of (caption, status) tuples, one per input image
    """
    global _model, _processor, _device

    prompt = "USER: <image>\nDescribe this image in one neutral sentence. ASSISTANT:"

    try:
        images = [Image.open(p).convert('RGB') for p in image_paths]
        prompts = [prompt] * len(images)

        inputs = _processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = _model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=50,
                use_cache=True,
            )

        outputs = _processor.batch_decode(output_ids, skip_special_tokens=True)
        results = []
        for out in outputs:
            text = out.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in out else out.strip()
            results.append((text, "OK"))
        return results

    except Exception as e:
        tqdm.write(f"   Batch error: {e}")
        return [("<ERROR>", "ERROR")] * len(image_paths)


# =====================CHECKPOINT FUNCTIONS=====================
def load_checkpoint(dataset_name):
    path = os.path.join(CHECKPOINT_ROOT, f"processed_videos_{dataset_name}_llava.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"processed_videos": [], "failed_frames": {}}

def save_checkpoint(dataset_name, checkpoint):
    path = os.path.join(CHECKPOINT_ROOT, f"processed_videos_{dataset_name}_llava.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

# ==================== MAIN PROCESSING FUNCTION ====================
def process_video_frames(video_frames_path, output_excel_path, checkpoint, dataset_name):
    """Process a single video folder with LLaVA"""
    # Get all frames
    image_paths = sorted(
        glob.glob(os.path.join(video_frames_path, "*.jpg")) +
        glob.glob(os.path.join(video_frames_path, "*.jpeg")) +
        glob.glob(os.path.join(video_frames_path, "*.png"))
    )
    if not image_paths:
        print(f"   Warning: No frames found in {video_frames_path}")
        return

    results = []
    failed_frames = []

    # Create progress bar for frames
    pbar = tqdm(total=len(image_paths), desc=f"   Processing frames", unit="frame", leave=False)

    # Process frames in batches of BATCH_SIZE
    for batch_start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[batch_start: batch_start + BATCH_SIZE]
        batch_results = generate_frame_descriptions_batch(batch_paths)
        for img_path, (caption, status) in zip(batch_paths, batch_results):
            results.append({"File": os.path.basename(img_path), "Caption": caption, "Status": status})
            if status != "OK":
                failed_frames.append(os.path.basename(img_path))
        pbar.update(len(batch_paths))

    pbar.close()

    # Save Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"   Saved to {output_excel_path}")

    # Update checkpoint
    checkpoint["processed_videos"].append(os.path.basename(video_frames_path))
    if failed_frames:
        checkpoint["failed_frames"][os.path.basename(video_frames_path)] = failed_frames
    save_checkpoint(dataset_name, checkpoint)


def _resolve_dataset_tasks(target_dataset=None):
    """
    Resolve dataset tasks from config file

    Returns:
        list: (dataset_name, frames_root, output_root) tuples
    """
    tasks = []

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.config_loader import get_config
    cfg = get_config()
    datasets_to_check = [target_dataset.lower()] if target_dataset else cfg.detect_datasets()

    for ds_name in datasets_to_check:
        ds_paths = cfg.get_dataset_paths(ds_name)
        frames_dir = ds_paths.get('frames_dir')
        fd_dir = ds_paths.get('frame_descriptions_dir')

        if not frames_dir:
            raise ValueError(f"Config missing: {ds_name}.frames_dir")
        if not fd_dir:
            raise ValueError(f"Config missing: {ds_name}.frame_descriptions_dir")

        tasks.append((ds_name, str(frames_dir), str(fd_dir)))

    return tasks


def process_dataset_frames(target_dataset=None):
    """
    Process all frames in datasets using LLaVA (all paths from config)

    Args:
        target_dataset: Optional dataset name to process (if None, process all)
    """
    dataset_tasks = _resolve_dataset_tasks(target_dataset)

    if not dataset_tasks:
        print(f"No datasets found to process")
        return

    if target_dataset:
        print(f"Processing single dataset: {dataset_tasks[0][0]}")
    else:
        print(f"Processing all datasets: {', '.join(t[0] for t in dataset_tasks)}")

    for dataset, frames_root, output_root in tqdm(dataset_tasks, desc="Processing datasets", unit="dataset"):
        os.makedirs(output_root, exist_ok=True)

        if not os.path.exists(frames_root):
            tqdm.write(f"   Skipping {dataset}, no frames folder found at {frames_root}")
            continue

        # Detect video sources: directories or tar.gz files
        video_dirs = [d for d in os.listdir(frames_root)
                      if os.path.isdir(os.path.join(frames_root, d))]
        tar_files = [f for f in os.listdir(frames_root)
                     if f.endswith('.tar.gz')]

        use_tar = len(tar_files) > 0 and len(video_dirs) == 0
        if use_tar:
            video_names = [f.replace('.tar.gz', '') for f in tar_files]
            tqdm.write(f"   Using tar.gz mode ({len(tar_files)} archives)")
        else:
            video_names = video_dirs

        if not video_names:
            tqdm.write(f"   Skipping {dataset}, no video folders or tar.gz found inside: {frames_root}")
            continue

        checkpoint = load_checkpoint(dataset)
        videos_to_process = [v for v in video_names if v not in checkpoint["processed_videos"]]

        tqdm.write(f"\n   Dataset: {dataset} ({len(videos_to_process)}/{len(video_names)} videos to process)")

        for video in tqdm(videos_to_process, desc=f"   Videos in {dataset}", unit="video", leave=False):
            output_excel_path = os.path.join(output_root, f"{video}.xlsx")

            if use_tar:
                # Extract tar.gz to temp dir, process, then clean up
                import tarfile
                import tempfile
                tar_path = os.path.join(frames_root, f"{video}.tar.gz")
                tmp_dir = tempfile.mkdtemp(prefix=f"llava_{video}_")
                try:
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(tmp_dir)
                    # Find the directory containing frames (may be nested)
                    extracted = os.listdir(tmp_dir)
                    if len(extracted) == 1 and os.path.isdir(os.path.join(tmp_dir, extracted[0])):
                        video_frames_path = os.path.join(tmp_dir, extracted[0])
                    else:
                        video_frames_path = tmp_dir
                    tqdm.write(f"   Processing {video} (from tar.gz)...")
                    process_video_frames(video_frames_path, output_excel_path, checkpoint, dataset)
                except Exception as e:
                    tqdm.write(f"   Error extracting {tar_path}: {e}")
                finally:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                video_frames_path = os.path.join(frames_root, video)
                tqdm.write(f"   Processing {video}...")
                process_video_frames(video_frames_path, output_excel_path, checkpoint, dataset)


# ==================== MAIN ====================

def main():
    """Main entry - all paths from config file"""
    global MODEL_PATH, CHECKPOINT_ROOT

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate frame-level descriptions using LLaVA (local model)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets
  python 1_1_generate_frame_descriptions_llava.py

  # Process a specific dataset
  python 1_1_generate_frame_descriptions_llava.py --dataset summe
  python 1_1_generate_frame_descriptions_llava.py --dataset tvsum

Note: All paths are read from config files (configs/config.yaml)
      Make sure LLaVA is installed (run install_llava.sh)
        """
    )
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name to process (e.g., summe, tvsum, ovp, youtube). If not specified, all datasets will be processed.')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help=f'LLaVA model path (default: {MODEL_PATH})')

    args = parser.parse_args()

    # Normalize dataset name to lowercase for consistency
    target_dataset = args.dataset.lower() if args.dataset else None

    print("="*80)
    print("Step 1.1: Generate Frame-Level Descriptions (LLaVA)")
    print("="*80)

    # Load config for paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.config_loader import get_config
    cfg = get_config()

    # Initialize checkpoint directory
    ckpt_dir = cfg.get_path('paths.checkpoints_dir')
    if not ckpt_dir:
        raise ValueError("Config missing: paths.checkpoints_dir")
    CHECKPOINT_ROOT = str(ckpt_dir)
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

    # Update model path if specified
    if args.model != MODEL_PATH:
        MODEL_PATH = args.model

    print(f"\n   Checkpoint directory: {CHECKPOINT_ROOT}")
    print(f"   Model: {MODEL_PATH}")
    if target_dataset:
        print(f"   Target dataset: {target_dataset}")
    else:
        print(f"   Target: All datasets")
    print("="*80 + "\n")

    # Load LLaVA model
    load_llava_model()

    # Process datasets (all paths from config)
    process_dataset_frames(target_dataset=target_dataset)


if __name__ == "__main__":
    main()
