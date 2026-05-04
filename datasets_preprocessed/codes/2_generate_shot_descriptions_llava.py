#!/usr/bin/env python3
"""
Step 2: Generate Shot-Level Descriptions using Video-LLaVA (Local Model)

This script:
1. Reads shot segmentation from H5 files (change_points)
2. Uses locally deployed Video-LLaVA to generate descriptions for each shot
3. Saves results to configured shot_descriptions_dir/{video}.json

Video-LLaVA is designed for video understanding and natively supports multi-frame input.

Usage:
    python 2_generate_shot_descriptions_llava.py --dataset summe
    python 2_generate_shot_descriptions_llava.py --dataset videoxum
"""

import os
import json
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys
import tarfile
import tempfile
from PIL import Image
from typing import List, Tuple
import argparse

# ==================== Configuration ====================

# Model settings - Video-LLaVA
MODEL_PATH = "LanguageBind/Video-LLaVA-7B-hf"
MAX_FRAMES_PER_SHOT = 8  # Video-LLaVA works well with 8 frames
BATCH_SIZE = 4           # Number of shots processed in one GPU forward pass

# Global model variables (loaded once)
_model = None
_processor = None
_device = None

# ==================== Device Setup ====================

def get_device():
    """Detect and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("   Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("   Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("   Warning: Using CPU (will be slow)")
    return device


def load_video_llava_model():
    """Load Video-LLaVA model (called once at startup)"""
    global _model, _processor, _device

    if _model is not None:
        return _processor, _model, _device

    print("Loading Video-LLaVA model...")
    print(f"   Model: {MODEL_PATH}")
    print("   (This may take a few minutes...)")

    from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

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


# ==================== Helper Functions ====================

def generate_shot_description(images: List[Image.Image]) -> Tuple[str, int]:
    """
    Generate description for a shot using multiple frames with Video-LLaVA

    Args:
        images: List of PIL Image objects for this shot

    Returns:
        Tuple of (description, status) where status is 0 for success, 1 for error
    """
    global _model, _processor, _device

    if not images:
        return "", 1

    try:
        # Convert PIL images to numpy array for Video-LLaVA
        # Video-LLaVA expects: (num_frames, height, width, channels)
        frames = []
        for img in images:
            # Resize to consistent size
            img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
            frame = np.array(img_resized)
            frames.append(frame)

        # Stack frames into video clip array
        video_clip = np.stack(frames, axis=0)  # (num_frames, H, W, C)

        # Prepare prompt - mirrors GPT-4V version in 2_generate_shot_descriptions.py
        num_frames = len(images)
        prompt = (
            f"USER: <video>"
            f"Task: Video Content Analysis for Research Dataset\n\n"
            f"Analyzing {num_frames} sequential frames from a video segment.\n"
            f"Frames are presented in chronological order for academic video summarization research.\n\n"
            f"Provide an objective, factual description (no more than 2 sentences):\n"
            f"- Visual scene composition and setting\n"
            f"- Primary subjects/objects observed\n"
            f"- Temporal progression and scene dynamics\n\n"
            f"Use neutral, descriptive language. Report only observable visual elements.\n\n"
            f"Analysis: ASSISTANT:"
        )

        # Process inputs
        inputs = _processor(
            text=prompt,
            videos=video_clip,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode
        output = _processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Extract only the assistant's response
        if "ASSISTANT:" in output:
            output = output.split("ASSISTANT:")[-1].strip()

        return output, 0

    except Exception as e:
        tqdm.write(f"   Error generating description: {e}")
        return "", 1


def generate_shot_descriptions_batch(batch_images_list: List[List[Image.Image]]) -> List[Tuple[str, int]]:
    """
    Generate descriptions for a batch of shots in one GPU forward pass.

    All shots are padded to MAX_FRAMES_PER_SHOT frames (repeat last frame if shorter)
    so they can be stacked into a uniform batch tensor.

    Args:
        batch_images_list: List of shots, each shot is a list of PIL Images

    Returns:
        List of (description, status) tuples
    """
    global _model, _processor, _device

    if not batch_images_list:
        return []

    try:
        video_clips = []
        prompts = []

        for images in batch_images_list:
            frames = []
            for img in images:
                img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                frames.append(np.array(img_resized))

            # Pad to MAX_FRAMES_PER_SHOT by repeating the last frame
            while len(frames) < MAX_FRAMES_PER_SHOT:
                frames.append(frames[-1])

            video_clips.append(np.stack(frames, axis=0))  # (T, H, W, C)

            num_frames = len(images)  # original frame count for prompt
            prompts.append(
                f"USER: <video>"
                f"Task: Video Content Analysis for Research Dataset\n\n"
                f"Analyzing {num_frames} sequential frames from a video segment.\n"
                f"Frames are presented in chronological order for academic video summarization research.\n\n"
                f"Provide an objective, factual description (no more than 2 sentences):\n"
                f"- Visual scene composition and setting\n"
                f"- Primary subjects/objects observed\n"
                f"- Temporal progression and scene dynamics\n\n"
                f"Use neutral, descriptive language. Report only observable visual elements.\n\n"
                f"Analysis: ASSISTANT:"
            )

        inputs = _processor(
            text=prompts,
            videos=video_clips,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        outputs = _processor.batch_decode(output_ids, skip_special_tokens=True)
        results = []
        for out in outputs:
            text = out.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in out else out.strip()
            results.append((text, 0))
        return results

    except Exception as e:
        tqdm.write(f"   Batch shot error: {e}")
        return [("", 1)] * len(batch_images_list)


def load_frames_from_directory(frames_dir: Path, video_id: str, frame_indices: List[int]) -> List[Image.Image]:
    """
    Load frames from a directory of loose frame files

    Args:
        frames_dir: Base frames directory
        video_id: Video ID (folder name)
        frame_indices: List of frame indices to load

    Returns:
        List of PIL Image objects
    """
    video_frames_dir = frames_dir / video_id
    if not video_frames_dir.exists():
        return []

    images = []
    for idx in frame_indices:
        # Try common frame naming patterns
        for pattern in [f"frame_{idx:06d}.jpg", f"frame_{idx:05d}.jpg", f"frame_{idx:04d}.jpg",
                       f"frame_{idx}.jpg", f"{idx:06d}.jpg", f"{idx:05d}.jpg", f"{idx}.jpg"]:
            frame_path = video_frames_dir / pattern
            if frame_path.exists():
                try:
                    img = Image.open(frame_path).convert('RGB')
                    images.append(img)
                    break
                except Exception as e:
                    tqdm.write(f"   Warning: Failed to load {frame_path}: {e}")

    return images


def load_frames_from_tar(tar_path: Path, frame_indices: List[int]) -> List[Image.Image]:
    """
    Load frames from a tar.gz archive

    Args:
        tar_path: Path to tar.gz file
        frame_indices: List of frame indices to load

    Returns:
        List of PIL Image objects
    """
    if not tar_path.exists():
        return []

    images = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    # Skip macOS metadata
                    if member.name.startswith("._") or "._" in member.name:
                        continue

                    # Check if this is one of our target frames
                    if member.name.startswith("frame_"):
                        try:
                            frame_num = int(member.name.split("_")[1].split(".")[0])
                            if frame_num in frame_indices:
                                tar.extract(member, temp_dir)
                                img = Image.open(temp_dir / member.name).convert('RGB')
                                images.append((frame_num, img))
                        except (ValueError, IndexError):
                            continue

            # Sort by frame number and return just the images
            images.sort(key=lambda x: x[0])
            images = [img for _, img in images]

    except Exception as e:
        tqdm.write(f"   Error loading from tar: {e}")

    return images


def get_shots_from_h5(h5_file, video_key: str) -> List[Tuple[int, int]]:
    """
    Get shot boundaries from H5 file

    Returns:
        List of (start_frame, end_frame) tuples
    """
    if video_key not in h5_file:
        return []

    video_data = h5_file[video_key]

    if 'change_points' not in video_data:
        return []

    change_points = video_data['change_points'][:]

    # change_points is Nx2 array where each row is [start, end]
    shots = [(int(start), int(end)) for start, end in change_points]

    return shots


# ==================== Main Processing ====================

def process_dataset(dataset_name: str, h5_path: Path, frames_dir: Path, output_dir: Path, video_limit: int = None, no_skip: bool = False):
    """
    Process a dataset to generate shot descriptions

    Args:
        dataset_name: Name of dataset
        h5_path: Path to H5 file with shot segmentation
        frames_dir: Path to frames directory
        output_dir: Path to output directory for shot descriptions
        video_limit: Optional limit on number of videos to process
        no_skip: If True, reprocess all videos even if output exists
    """
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"   H5 file: {h5_path}")
    print(f"   Frames dir: {frames_dir}")
    print(f"   Output dir: {output_dir}")

    if not h5_path.exists():
        print(f"   Error: H5 file not found: {h5_path}")
        return

    if not frames_dir.exists():
        print(f"   Error: Frames directory not found: {frames_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if using tar.gz or loose frames
    sample_tar = list(frames_dir.glob("*.tar.gz"))
    use_tar = len(sample_tar) > 0
    if use_tar:
        print(f"   Frame format: tar.gz archives")
    else:
        print(f"   Frame format: loose frame directories")

    # Open H5 file
    with h5py.File(h5_path, 'r') as h5_file:
        video_keys = sorted([k for k in h5_file.keys()])
        print(f"   Found {len(video_keys)} videos in H5\n")

        if video_limit:
            video_keys = video_keys[:video_limit]

        for video_idx, video_key in enumerate(tqdm(video_keys, desc="Processing videos")):
            tqdm.write(f"\n   Video {video_idx+1}/{len(video_keys)}: {video_key}")

            # Check if already processed
            output_file = output_dir / f"{video_key}.json"
            if output_file.exists() and not no_skip:
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    has_empty = any(
                        not s.get('description', '').strip()
                        for s in existing.get('shots', [])
                    )
                    if not has_empty:
                        tqdm.write(f"      ✅ ready, skipping")
                        continue
                    tqdm.write(f"      ⚠️  has empty descriptions, reprocessing")
                except Exception:
                    pass  # Unreadable file — fall through to reprocess

            # Get video ID for frame loading
            if 'video_name' in h5_file[video_key]:
                video_id = h5_file[video_key]['video_name'][()].decode('utf-8')
            else:
                video_id = video_key

            # Check if frames exist for this video (skip if not in subset)
            if use_tar:
                tar_path = frames_dir / f"{video_id}.tar.gz"
                if not tar_path.exists():
                    tqdm.write(f"      not in subset, skipping")
                    continue
            else:
                video_frames_dir = frames_dir / video_id
                if not video_frames_dir.exists():
                    tqdm.write(f"      not in subset, skipping")
                    continue

            # Get shots from H5
            shots = get_shots_from_h5(h5_file, video_key)

            if not shots:
                tqdm.write(f"      No shots found, skipping")
                continue

            tqdm.write(f"      Found {len(shots)} shots")

            # Get picks (sampled frames) if available
            if 'picks' in h5_file[video_key]:
                picks = h5_file[video_key]['picks'][:].tolist()
            else:
                # If no picks, use all frames in shots
                picks = None

            # ---- Phase 1: load all frames, record empty slots ----
            shot_descriptions = [None] * len(shots)
            pending_indices = []   # shot indices that need GPU inference
            pending_images  = []   # corresponding loaded frames

            for shot_idx, (start_frame, end_frame) in enumerate(shots):
                if picks is not None:
                    shot_frame_indices = [p for p in picks if start_frame <= p <= end_frame]
                else:
                    shot_frame_indices = list(range(start_frame, end_frame + 1))

                if not shot_frame_indices:
                    tqdm.write(f"         Shot {shot_idx}: no sampled frames in [{start_frame}-{end_frame}], empty entry")
                    shot_descriptions[shot_idx] = {
                        'shot_id': shot_idx, 'start_frame': start_frame,
                        'end_frame': end_frame, 'description': '', 'num_frames_used': 0
                    }
                    continue

                if len(shot_frame_indices) > MAX_FRAMES_PER_SHOT:
                    indices = torch.linspace(0, len(shot_frame_indices) - 1, MAX_FRAMES_PER_SHOT).long()
                    shot_frame_indices = [shot_frame_indices[i] for i in indices]

                if use_tar:
                    images = load_frames_from_tar(tar_path, shot_frame_indices)
                else:
                    images = load_frames_from_directory(frames_dir, video_id, shot_frame_indices)

                if not images:
                    tqdm.write(f"         Shot {shot_idx}: failed to load frames, empty entry")
                    shot_descriptions[shot_idx] = {
                        'shot_id': shot_idx, 'start_frame': start_frame,
                        'end_frame': end_frame, 'description': '', 'num_frames_used': 0
                    }
                    continue

                pending_indices.append(shot_idx)
                pending_images.append((images, start_frame, end_frame))

            # ---- Phase 2: batch GPU inference over pending shots ----
            pbar_shots = tqdm(total=len(pending_indices), desc=f"      Shots", leave=False)
            for batch_start in range(0, len(pending_indices), BATCH_SIZE):
                batch_slice   = slice(batch_start, batch_start + BATCH_SIZE)
                batch_idx     = pending_indices[batch_slice]
                batch_data    = pending_images[batch_slice]
                batch_img_lists = [d[0] for d in batch_data]

                start_time = time.time()
                results = generate_shot_descriptions_batch(batch_img_lists)
                elapsed = time.time() - start_time

                for i, (shot_idx, (images, start_frame, end_frame), (description, status)) in enumerate(
                        zip(batch_idx, batch_data, results)):
                    if status == 0 and description:
                        tqdm.write(f"         Shot {shot_idx}: ({elapsed/len(batch_idx):.1f}s) {description[:80]}...")
                    else:
                        tqdm.write(f"         Shot {shot_idx}: generation failed, empty entry")
                        description = ''
                    shot_descriptions[shot_idx] = {
                        'shot_id': shot_idx, 'start_frame': start_frame,
                        'end_frame': end_frame, 'description': description,
                        'num_frames_used': len(images)
                    }
                pbar_shots.update(len(batch_idx))
            pbar_shots.close()

            # Fill any remaining None slots (should not happen, safety net)
            for i, entry in enumerate(shot_descriptions):
                if entry is None:
                    s, e = shots[i]
                    shot_descriptions[i] = {
                        'shot_id': i, 'start_frame': s,
                        'end_frame': e, 'description': '', 'num_frames_used': 0
                    }

            # Save results
            if shot_descriptions:
                result = {
                    'video_id': video_id,
                    'video_key': video_key,
                    'total_shots': len(shots),
                    'processed_shots': len(shot_descriptions),
                    'shots': shot_descriptions
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                tqdm.write(f"      Saved {len(shot_descriptions)}/{len(shots)} shots to {output_file.name}")


def main():
    """Main entry - all paths from config file"""
    global MODEL_PATH  # Declare global at the start before any use

    parser = argparse.ArgumentParser(
        description='Generate shot descriptions using Video-LLaVA (local model)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process SumMe dataset
  python 2_generate_shot_descriptions_llava.py --dataset summe

  # Process VideoXum dataset
  python 2_generate_shot_descriptions_llava.py --dataset videoxum

  # Process with limit
  python 2_generate_shot_descriptions_llava.py --dataset summe --limit 5

Note: All paths are read from config files (configs/config.yaml)
      Uses Video-LLaVA which natively supports multi-frame video input
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name to process (e.g., summe, tvsum, ovp, youtube, videoxum)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of videos to process')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help=f'Model path (default: {MODEL_PATH})')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess all videos even if output already exists')

    args = parser.parse_args()

    print("="*80)
    print("  Step 2: Shot Description Generation (Video-LLaVA)")
    print("="*80)

    # Load config
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.config_loader import get_config
    cfg = get_config()

    # Get dataset paths from config
    dataset = args.dataset.lower()
    ds_paths = cfg.get_dataset_paths(dataset)

    h5_file = ds_paths.get('h5_file')
    frames_dir = ds_paths.get('frames_dir')
    shot_desc_dir = ds_paths.get('shot_descriptions_dir')

    if not h5_file:
        raise ValueError(f"Config missing: {dataset}.h5_file")
    if not frames_dir:
        raise ValueError(f"Config missing: {dataset}.frames_dir")
    if not shot_desc_dir:
        raise ValueError(f"Config missing: {dataset}.shot_descriptions_dir")

    # Update model path if specified
    if args.model != MODEL_PATH:
        MODEL_PATH = args.model

    print(f"\n   Dataset: {dataset}")
    print(f"   Model: {MODEL_PATH}")
    if args.limit:
        print(f"   Video limit: {args.limit}")
    print()

    # Load Video-LLaVA model
    load_video_llava_model()

    # Process dataset
    process_dataset(
        dataset_name=dataset,
        h5_path=h5_file,
        frames_dir=frames_dir,
        output_dir=shot_desc_dir,
        video_limit=args.limit,
        no_skip=args.no_skip
    )

    print("\n" + "="*80)
    print("  Processing Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
