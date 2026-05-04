#!/usr/bin/env python3
"""
Step 3: Generate Shot-Aligned Documents using LLaMA (Local Model)

This script:
1. Reads shot descriptions from JSON files
2. Uses locally deployed LLaMA to generate aligned documents
3. Saves results back to the data JSON file

LLaMA is used here for its excellent text understanding and document generation capabilities.

Usage:
    python 3_generate_aligned_document_llama.py --dataset summe
    python 3_generate_aligned_document_llama.py --dataset tvsum --gt-only
"""

import os
import json
import torch
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import time
import sys
import argparse

# ==================== Configuration ====================

# Model settings - LLaMA 3.1 70B Instruct (4-bit quantized)
MODEL_PATH = "meta-llama/Llama-3.1-70B-Instruct"
MAX_TOKENS = 2000  # For document generation
BATCH_SIZE = 4     # Videos processed per inference call (set to 1 to disable batching)
SAVE_INTERVAL = 100  # Save JSON checkpoint every N processed videos

# Global model variables (loaded once)
_model = None
_tokenizer = None
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


def load_llama_model():
    """Load LLaMA model for document generation (called once at startup)"""
    global _model, _tokenizer, _device

    if _model is not None:
        return _tokenizer, _model, _device

    print("Loading LLaMA model for document generation...")
    print(f"   Model: {MODEL_PATH}")
    print("   (This may take a few minutes...)")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    _device = get_device()

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _tokenizer.padding_side = 'left'   # Required for correct batch generation with causal LMs
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    if _device.type == "cuda":
        # 4-bit quantization: fits 70B in ~35GB VRAM (vs 140GB for FP16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # max_memory: cap GPU at 88GiB and allow CPU overflow.
        # This guards against transformers 5.x materializing weights in FP16
        # before quantization is applied, which would otherwise cause OOM.
        max_memory = {0: "88GiB", "cpu": "200GiB"}
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
        )
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
        )
        _model = _model.to(_device)

    _model.eval()

    print("   Model loaded successfully\n")
    return _tokenizer, _model, _device


# ==================== Output Cleanup ====================

def _clean_model_output(text: str) -> str:
    """
    Strip model meta-commentary that should not appear in the final output.
    Removes trailing (Note: ...), Note: ..., [Shot X is missing ...],
    and "Please provide the next shots..." lines.
    Operates paragraph-by-paragraph: keeps only paragraphs that start with
    a valid [Shots X] or [Shot X] tag.
    """
    # Split into double-newline paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    clean = []
    for para in paragraphs:
        # Keep only paragraphs that start with a valid shot tag
        if not re.match(r'\[Shots?\s+\d+', para, re.IGNORECASE):
            continue  # drop Note:, Please provide..., etc.
        # Drop "[Shot X is missing...]" or similar non-content paragraphs
        if re.search(r'is missing|missing from|not (provided|available|given)', para[:80], re.IGNORECASE):
            continue

        # Within a valid paragraph, strip any trailing (Note: ...) or Note: ... suffix
        # Strip inline trailing commentary after the sentence ends
        para = re.sub(r'\s*\(Note[:\s][^)]*\)', '', para, flags=re.IGNORECASE | re.DOTALL)
        para = re.sub(r'\s*\n\s*Note[:\s].*', '', para, flags=re.IGNORECASE | re.DOTALL)
        para = para.strip()
        if para:
            clean.append(para)

    return '\n\n'.join(clean)


# ==================== Document Generation with Alignment ====================

def _build_prompt_text(shots: List[Dict], is_summary: bool) -> str:
    """
    Build the user prompt string for the given (already re-indexed) shots.
    Extracted to allow reuse by both single and batch generation paths.
    """
    input_text = '\n'.join(f"Shot {s['clip_id']}: {s['summary']}" for s in shots)

    if is_summary:
        return f"""You are a professional content writer. You are given highlight shot-level descriptions from a video, each with a shot number. Your task is to create a HIGHLY CONDENSED, coherent narrative summary by aggressively merging shots.

CRITICAL REQUIREMENTS:
1. **Sentence-Shot Alignment**: Each sentence in your summary MUST correspond to one or more consecutive shots
2. **Explicit Shot References**: Start each sentence with [Shots X] or [Shots X-Y] to indicate which shot(s) it covers
3. **Preserve ALL important information**: Include the key visual details and events; omit minor or peripheral details
4. **Chronological order**: Follow the shot sequence exactly
5. **Natural narrative**: Write smooth, readable prose (but keep the shot reference tags)
6. **No meta-references**: Don't use "this video shows", "the clip depicts", etc.
7. **AGGRESSIVE merging strategy**: You MUST combine multiple consecutive shots (typically 4-8 shots) into comprehensive sentences. Look for shots that:
   - Depict the same continuous action or scene
   - Share the same subject, location, or theme
   - Form a logical narrative unit together
   - Can be naturally connected with transitions (e.g., "then", "as", "while", "before")
   IMPORTANT: Only create separate sentences when there's a MAJOR scene change, topic shift, or significant discontinuity.
8. **Compression Ratio Target**: The target number of sentences is 20%-30% of the total number of shots. This means if you have 30 shots, aim for about 6-9 sentences.
9. **Remove redundancy and extract key content**: Actively eliminate repeated descriptions, redundant visual details, and similar scenes that add no new information. Focus on the most distinctive and meaningful moments; merge or drop shots that are repetitive or peripheral.

FORMAT REQUIREMENTS:
- Start each sentence with [Shots X] or [Shots X-Y] where X and Y are shot numbers
- After the tag, write exactly ONE detailed, comprehensive sentence covering all those shots
- Prefer longer shot ranges (3-5+ shots per sentence) over shorter ones
- Each sentence = one alignment tag + one rich, concise sentence capturing the key actions/scenes
- You MUST tag which shots each sentence covers
- **The provided shots are the complete input.** Do NOT mention missing shots, gaps, or assume there are more shots than given
- **Output only narrative sentences with shot tags.** Do NOT add any parenthetical notes, meta-commentary, or explanations about your merging decisions

EXAMPLE OUTPUT FORMAT (notice the aggressive merging and selective focus):
[Shots 0-4] Wide establishing shots reveal an open outdoor setting with varied terrain and architectural elements visible in the background, as the primary subject enters the frame from the right and moves steadily toward the foreground.

[Shots 5-6] A brief close-up frames a specific object at rest in the center of the scene against a softly lit, minimally detailed background.

[Shots 7-11] The perspective broadens across multiple angles as the subject interacts with the surrounding environment, transitioning through several spatial configurations before settling on a final static wide shot of the full scene.

INPUT SHOTS:
{input_text}

Generate a highly condensed summary with explicit shot alignment tags (remember: aim for 20%-30% compression, merge aggressively, remove redundancy, extract key content):"""
    else:
        return f"""You are a professional content writer. You are given shot-level descriptions from a video, each with a shot number. Your task is to create a HIGHLY CONDENSED, coherent narrative document by aggressively merging shots.

CRITICAL REQUIREMENTS:
1. **Sentence-Shot Alignment**: Each sentence in your document MUST correspond to one or more consecutive shots
2. **Explicit Shot References**: Start each sentence with [Shots X] or [Shots X-Y] to indicate which shot(s) it covers
3. **Preserve ALL information**: Include all visual details mentioned in the shots
4. **Chronological order**: Follow the shot sequence exactly
5. **Natural narrative**: Write smooth, readable prose (but keep the shot reference tags)
6. **No meta-references**: Don't use "this video shows", "the clip depicts", etc.
7. **AGGRESSIVE merging strategy**: You MUST combine multiple consecutive shots (typically 4-8 shots) into comprehensive sentences. Look for shots that:
   - Depict the same continuous action or scene
   - Share the same subject, location, or theme
   - Form a logical narrative unit together
   - Can be naturally connected with transitions (e.g., "then", "as", "while", "before")
   IMPORTANT: Only create separate sentences when there's a MAJOR scene change, topic shift, or significant discontinuity.
8. **Compression Ratio Target**: The target number of sentences is 20%-30% of the total number of shots. This means if you have 30 shots, aim for about 6-9 sentences.

FORMAT REQUIREMENTS:
- Start each sentence with [Shots X] or [Shots X-Y] where X and Y are shot numbers
- After the tag, write exactly ONE detailed, comprehensive sentence covering all those shots
- Prefer longer shot ranges (3-5+ shots per sentence) over shorter ones
- Each sentence = one alignment tag + one rich, detailed sentence capturing multiple actions/scenes
- You MUST tag which shots each sentence covers

EXAMPLE OUTPUT FORMAT (notice the aggressive merging):
[Shots 0-4] Wide establishing shots reveal an open outdoor setting with varied terrain and architectural elements visible in the background, as the primary subject enters the frame from the right and moves steadily toward the foreground.

[Shots 5-6] A brief close-up frames a specific object at rest in the center of the scene against a softly lit, minimally detailed background.

[Shots 7-11] The perspective broadens across multiple angles as the subject interacts with the surrounding environment, transitioning through several spatial configurations before settling on a final static wide shot of the full scene.

INPUT SHOTS:
{input_text}

Generate a highly condensed document with explicit shot alignment tags (remember: aim for 20%-30% compression, merge aggressively, 3-5+ shots per sentence typically):"""


def generate_aligned_document_llama(shots: List[Dict], video_id: str,
                                    is_summary: bool = False) -> Dict:
    """
    Generate document where each paragraph aligns to specific shot(s) using LLaMA

    Args:
        shots: List of shot dictionaries with 'clip_id' and 'summary'
        video_id: Video identifier
        is_summary: If True, use the summarization prompt (for GT shots summary);
                    if False, use the full-document prompt (preserve all details)

    Returns:
        Dictionary with document and alignment information
    """
    global _model, _tokenizer, _device

    # Re-index shots to 0, 1, 2, ... to prevent ID-gap hallucination
    original_clip_ids = [shot['clip_id'] for shot in shots]
    reindexed_shots = [dict(shot, clip_id=i) for i, shot in enumerate(shots)]

    user_prompt = _build_prompt_text(reindexed_shots, is_summary)

    try:
        # Build messages in LLaMA chat format
        messages = [{"role": "user", "content": user_prompt}]

        # Apply chat template
        input_ids = _tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(_device)

        # Build attention mask (all 1s; avoids warning when pad==eos token)
        attention_mask = torch.ones_like(input_ids)

        # Generate
        with torch.inference_mode():
            output_ids = _model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=_tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens (exclude the prompt)
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        raw_document = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip model meta-commentary (Note:..., missing shots, etc.)
        raw_document = _clean_model_output(raw_document)

        # Parse the document (uses reindexed shot IDs 0, 1, 2, ...)
        sentences, alignments = parse_aligned_document(raw_document, reindexed_shots)

        # Remap sequential indices back to original clip_ids.
        # Note: raw_document retains re-indexed [Shots X-Y] tags (0, 1, 2, ...);
        # alignments[i]['shot_ids'] stores the true clip_ids for downstream use.
        for alignment in alignments:
            remapped_ids = []
            remapped_summaries = []
            for new_id in alignment['shot_ids']:
                if 0 <= new_id < len(original_clip_ids):
                    remapped_ids.append(original_clip_ids[new_id])
                    remapped_summaries.append(shots[new_id]['summary'])
            alignment['shot_ids'] = remapped_ids
            alignment['shot_summaries'] = remapped_summaries

        return {
            'raw_document': raw_document,
            'sentences': sentences,
            'alignments': alignments,
            'num_sentences': len(sentences)
        }

    except Exception as e:
        tqdm.write(f"WARNING: Error generating document for {video_id}: {e}")
        # Fallback: create one sentence per shot
        sentences = [shot['summary'] for shot in shots]
        alignments = [{'sentence_id': i, 'shot_ids': [shots[i]['clip_id']]}
                     for i in range(len(shots))]
        return {
            'raw_document': '\n\n'.join([f"[Shot {shots[i]['clip_id']}] {p}"
                                        for i, p in enumerate(sentences)]),
            'sentences': sentences,
            'alignments': alignments,
            'num_sentences': len(sentences)
        }


def generate_aligned_document_llama_batch(
        batch: List[Tuple[List[Dict], str, bool]]) -> List[Dict]:
    """
    Batch version: process multiple videos in a single model.generate() call.

    Args:
        batch: List of (shots, video_id, is_summary) tuples

    Returns:
        List of result dicts (same format as generate_aligned_document_llama)
    """
    global _model, _tokenizer, _device

    # Single item — skip batching overhead
    if len(batch) == 1:
        shots, video_id, is_summary = batch[0]
        return [generate_aligned_document_llama(shots, video_id, is_summary)]

    # Prepare: re-index shots and build prompts
    prepared = []
    for shots, video_id, is_summary in batch:
        original_clip_ids = [s['clip_id'] for s in shots]
        reindexed = [dict(s, clip_id=i) for i, s in enumerate(shots)]
        prompt = _build_prompt_text(reindexed, is_summary)
        prepared.append((original_clip_ids, reindexed, shots, video_id, is_summary, prompt))

    try:
        # Tokenize each prompt (apply chat template individually)
        input_ids_list = []
        for *_, prompt in prepared:
            ids = _tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt"
            )[0]
            input_ids_list.append(ids)

        # Left-pad all sequences to the same length
        max_len = max(ids.shape[0] for ids in input_ids_list)
        pad_id  = _tokenizer.eos_token_id
        padded_ids, attention_masks = [], []
        for ids in input_ids_list:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(torch.cat([
                torch.full((pad_len,), pad_id, dtype=ids.dtype), ids
            ]))
            attention_masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(ids.shape[0],  dtype=torch.long),
            ]))

        input_ids      = torch.stack(padded_ids).to(_device)
        attention_mask = torch.stack(attention_masks).to(_device)
        prompt_len     = input_ids.shape[1]

        with torch.inference_mode():
            output_ids = _model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=_tokenizer.eos_token_id,
            )

        # Decode and parse each result
        results = []
        for i, (original_clip_ids, reindexed, shots, video_id, is_summary, _) in enumerate(prepared):
            new_tokens  = output_ids[i][prompt_len:]
            raw_document = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            raw_document = _clean_model_output(raw_document)

            sentences, alignments = parse_aligned_document(raw_document, reindexed)
            for alignment in alignments:
                remapped_ids, remapped_summaries = [], []
                for new_id in alignment['shot_ids']:
                    if 0 <= new_id < len(original_clip_ids):
                        remapped_ids.append(original_clip_ids[new_id])
                        remapped_summaries.append(shots[new_id]['summary'])
                alignment['shot_ids']       = remapped_ids
                alignment['shot_summaries'] = remapped_summaries

            results.append({
                'raw_document': raw_document,
                'sentences':    sentences,
                'alignments':   alignments,
                'num_sentences': len(sentences),
            })

        return results

    except Exception as e:
        tqdm.write(f"WARNING: Batch generation failed ({e}), falling back to individual")
        return [
            generate_aligned_document_llama(shots, vid, is_sum)
            for shots, vid, is_sum in batch
        ]


def parse_aligned_document(document: str, shots: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Parse document with [Shots X] or [Shots X-Y] tags

    Args:
        document: Document text with shot alignment tags
        shots: Original shot list (for validation)

    Returns:
        (sentences, alignments) where:
        - sentences: List of sentence texts (without tags)
        - alignments: List of dicts with 'sentence_id', 'shot_ids', and 'shot_summaries'
    """
    # Split into sentences
    raw_sentences = [p.strip() for p in document.split('\n\n') if p.strip()]

    sentences = []
    alignments = []

    for sent_id, sent in enumerate(raw_sentences):
        # Extract shot tags: [Shots X] or [Shots X-Y] or [Shot X]
        match = re.match(r'\[Shots?\s+(\d+)(?:-(\d+))?\]\s*(.*)', sent, re.IGNORECASE | re.DOTALL)

        if match:
            start_shot = int(match.group(1))
            end_shot = int(match.group(2)) if match.group(2) else start_shot
            sentence_text = match.group(3).strip()

            shot_ids = list(range(start_shot, end_shot + 1))

            # Get original shot summaries for these shot IDs
            shot_summaries = []
            for shot_id in shot_ids:
                matching_shots = [s for s in shots if s['clip_id'] == shot_id]
                if matching_shots:
                    shot_summaries.append(matching_shots[0]['summary'])

            sentences.append(sentence_text)
            alignments.append({
                'sentence_id': sent_id,
                'shot_ids': shot_ids,
                'shot_summaries': shot_summaries
            })
        else:
            # No tag found, treat as unaligned (shouldn't happen with good LLM output)
            sentences.append(sent)
            alignments.append({
                'sentence_id': sent_id,
                'shot_ids': [],
                'shot_summaries': []
            })

    return sentences, alignments


# ==================== Main Processing ====================
def process_dataset(clips_data: Dict, dataset_name: str,
                   gt_only: bool = False,
                   skip_processed: bool = True,
                   batch_size: int = BATCH_SIZE,
                   save_path: str = None) -> Dict:
    """
    Process dataset to generate aligned documents using LLaMA (batched).

    Args:
        clips_data:     Clips description data
        dataset_name:   Dataset name
        gt_only:        If True, generate only GT text summary (no full document)
        skip_processed: If True, skip already processed videos (default: True)
        batch_size:     Number of videos per inference call
        save_path:      If set, periodically save the JSON to this path
    """
    print(f"\n{'='*80}")
    print(f"Generating Shot-Aligned Documents: {dataset_name}")
    print(f"   Using: {MODEL_PATH} (Local Model, 4-bit quant)")
    print(f"   Mode: {'GT shots only' if gt_only else 'All shots + GT summary'}")
    print(f"   Checkpoint resume: {'Enabled' if skip_processed else 'Disabled (--no-skip)'}")
    print(f"   Batch size: {batch_size}")
    print(f"{'='*80}\n")

    stats = {
        "total_videos": 0,
        "processed_videos": 0,
        "total_gt_shots": 0,
        "total_sentences": 0,
    }

    # Support both nested {"dataset": {...}} and flat {"video_id": {...}} structures
    dataset_data = clips_data.get(dataset_name, clips_data)

    # ── Phase 0: Collect items that need processing ────────────────────────────
    doc_queue:     List[Tuple] = []   # (video_id, video_data, valid_shots)
    summary_queue: List[Tuple] = []   # (video_id, video_data, gt_shots)

    for video_id, video_data in dataset_data.items():
        clips = video_data.get("clips", [])
        if not clips:
            continue

        stats["total_videos"] += 1

        if skip_processed:
            if gt_only:
                if "aligned_text_summary" in video_data:
                    stats["processed_videos"] += 1
                    continue
            else:
                if "aligned_full_document" in video_data and "aligned_text_summary" in video_data:
                    stats["processed_videos"] += 1
                    continue

        valid_shots = [c for c in clips
                       if c.get("summary") and c["summary"] != "[CONTENT_FILTERED]"]
        if not valid_shots:
            continue

        gt_shots = [s for s in valid_shots if s.get("gt", 0) == 1]

        if not gt_only:
            doc_queue.append((video_id, video_data, valid_shots))
        if gt_shots:
            summary_queue.append((video_id, video_data, gt_shots))

    # Sort queues by input length to minimise left-padding waste
    doc_queue.sort(key=lambda x: len(x[2]))
    summary_queue.sort(key=lambda x: len(x[2]))

    # ── Periodic checkpoint helper ─────────────────────────────────────────────
    _n_saved = [0]
    def _checkpoint(force: bool = False):
        _n_saved[0] += 1
        if save_path and (force or _n_saved[0] % SAVE_INTERVAL == 0):
            with open(save_path, "w", encoding="utf-8") as sf:
                json.dump(clips_data, sf, ensure_ascii=False, indent=2)
            tqdm.write(f"   ✓ Checkpoint saved ({_n_saved[0]} videos processed)")

    # ── Phase 1: Full document generation ─────────────────────────────────────
    if not gt_only and doc_queue:
        print(f"[Phase 1] Full documents — {len(doc_queue)} videos, batch={batch_size}")
        pbar = tqdm(total=len(doc_queue), desc="Full docs", unit="video")
        for i in range(0, len(doc_queue), batch_size):
            batch = doc_queue[i : i + batch_size]
            results = generate_aligned_document_llama_batch(
                [(vs, vid, False) for vid, vd, vs in batch]
            )
            for (vid, vd, _), res in zip(batch, results):
                vd["aligned_full_document"] = {
                    "raw_text":      res["raw_document"],
                    "sentences":     res["sentences"],
                    "alignments":    res["alignments"],
                    "num_sentences": res["num_sentences"],
                }
                pbar.update(1)
                _checkpoint()
        pbar.close()

    # ── Phase 2: GT summary generation ────────────────────────────────────────
    if summary_queue:
        label = "GT summaries" if not gt_only else "GT summaries (gt-only mode)"
        print(f"[Phase 2] {label} — {len(summary_queue)} videos, batch={batch_size}")
        pbar = tqdm(total=len(summary_queue), desc="GT summaries", unit="video")
        for i in range(0, len(summary_queue), batch_size):
            batch = summary_queue[i : i + batch_size]
            results = generate_aligned_document_llama_batch(
                [(gt, f"{vid}_GT", True) for vid, vd, gt in batch]
            )
            for (vid, vd, gt), res in zip(batch, results):
                vd["aligned_text_summary"] = {
                    "raw_text":      res["raw_document"],
                    "sentences":     res["sentences"],
                    "alignments":    res["alignments"],
                    "num_sentences": res["num_sentences"],
                }
                stats["total_gt_shots"]  += len(gt)
                stats["total_sentences"] += res["num_sentences"]
                stats["processed_videos"] += 1
                pbar.update(1)
                _checkpoint()
        pbar.close()

    # Final save
    _checkpoint(force=True)

    return clips_data, stats


# ==================== Main Function ====================
def main():
    """Main entry point - all paths read from config"""
    global MODEL_PATH  # must be declared before any use within this function
    parser = argparse.ArgumentParser(
        description='Generate shot-aligned full-length documents using LLaMA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full document + GT summary for SumMe using LLaMA
  python 3_generate_aligned_document_llama.py --dataset summe

  # Generate only GT text summary for SumMe using LLaMA
  python 3_generate_aligned_document_llama.py --dataset summe --gt-only

  # Generate for TVSum with checkpoint resume disabled
  python 3_generate_aligned_document_llama.py --dataset tvsum --no-skip

Note: All paths are read from configs/config.yaml. Ensure it is properly configured.
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., summe, tvsum)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help=argparse.SUPPRESS)  # Reserved for pipeline compatibility
    parser.add_argument('--clips-file', type=str, default=None,
                       help='Specific clips description file (default: auto-detect)')
    parser.add_argument('--gt-only', action='store_true',
                       help='Generate only GT text summary (no full document)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess all videos (default: skip already processed)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Videos per inference call (default: {BATCH_SIZE})')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help=f'Model path or HuggingFace ID (default: {MODEL_PATH})')

    args = parser.parse_args()

    # Allow overriding model path via CLI
    MODEL_PATH = args.model

    # Determine skip_processed flag
    skip_processed = not args.no_skip

    # Load configuration (pure config, no fallback)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.config_loader import get_config
    cfg = get_config()

    dataset_lower = args.dataset.lower()
    ds_paths = cfg.get_dataset_paths(dataset_lower)

    data_dir = ds_paths.get('processed_dir')
    if not data_dir:
        raise ValueError(f"Configuration error: paths.processed_dir not defined")

    # Determine data file
    if args.clips_file:
        data_path = data_dir / args.clips_file
    else:
        data_path = data_dir / f"{dataset_lower}_data.json"

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    print(f"\nLoading data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        clips_data = json.load(f)

    # Load LLaMA model once at startup
    load_llama_model()

    # Process dataset (modifies clips_data in-place, with periodic checkpoint saves)
    updated_clips_data, stats = process_dataset(
        clips_data,
        args.dataset,
        args.gt_only,
        skip_processed,
        batch_size=args.batch_size,
        save_path=str(data_path),
    )

    # Save updated clips_data back to the original file
    print(f"\nSaving updated data to: {data_path}")
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(updated_clips_data, f, ensure_ascii=False, indent=2)

    # Save per-video independent JSON files
    dataset_data_for_save = updated_clips_data.get(args.dataset, updated_clips_data)
    per_video_dir = data_path.parent / "step3_aligned"
    per_video_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for video_id, video_data in dataset_data_for_save.items():
        per_video_result = {"video_id": video_id}
        if 'aligned_full_document' in video_data:
            per_video_result['aligned_full_document'] = video_data['aligned_full_document']
        if 'aligned_text_summary' in video_data:
            per_video_result['aligned_text_summary'] = video_data['aligned_text_summary']
        if len(per_video_result) > 1:
            per_video_file = per_video_dir / f"{video_id}.json"
            with open(per_video_file, 'w', encoding='utf-8') as f:
                json.dump(per_video_result, f, ensure_ascii=False, indent=2)
            saved_count += 1
    print(f"   per-video JSON: {per_video_dir}/ ({saved_count} files)")

    print(f"\n{'='*80}")
    print("Shot-Aligned Document Generation Complete!")
    print(f"{'='*80}")
    print(f"Updated file: {data_path}")
    print(f"\nStatistics:")
    print(f"   Total videos: {stats['total_videos']}")
    print(f"   Processed videos: {stats['processed_videos']}")

    if args.gt_only and stats['total_sentences'] > 0:
        print(f"   Total GT shots: {stats['total_gt_shots']}")
        print(f"   Total sentences: {stats['total_sentences']}")
        avg_shots_per_sent = stats['total_gt_shots'] / stats['total_sentences']
        print(f"   Avg GT shots per sentence: {avg_shots_per_sent:.1f}")

    print(f"{'='*80}\n")

    # Show sample output
    dataset_data = updated_clips_data.get(args.dataset, {})
    if dataset_data:
        sample_video_id = list(dataset_data.keys())[0]
        sample_video = dataset_data[sample_video_id]
        print("\nSample Output:")
        print(f"   Video: {sample_video_id}")

        if args.gt_only:
            # Show GT summary sample
            if 'aligned_text_summary' in sample_video:
                print(f"   Sentences: {sample_video['aligned_text_summary']['num_sentences']}")
                print(f"\n   Alignment:")
                for align in sample_video['aligned_text_summary']['alignments'][:3]:
                    shot_range = f"{min(align['shot_ids'])}-{max(align['shot_ids'])}" if len(align['shot_ids']) > 1 else str(align['shot_ids'][0])
                    print(f"      Sentence {align['sentence_id']} → Shots {shot_range}")

                if sample_video['aligned_text_summary']['sentences']:
                    print(f"\n   Sample Sentence (first):")
                    print(f"      {sample_video['aligned_text_summary']['sentences'][0][:200]}...")
        else:
            # Show full document sample
            if 'aligned_full_document' in sample_video:
                print(f"   Sentences: {sample_video['aligned_full_document']['num_sentences']}")
                print(f"\n   Alignment:")
                for align in sample_video['aligned_full_document']['alignments'][:3]:
                    shot_range = f"{min(align['shot_ids'])}-{max(align['shot_ids'])}" if len(align['shot_ids']) > 1 else str(align['shot_ids'][0])
                    print(f"      Sentence {align['sentence_id']} → Shots {shot_range}")

                if sample_video['aligned_full_document']['sentences']:
                    print(f"\n   Sample Sentence (first):")
                    print(f"      {sample_video['aligned_full_document']['sentences'][0][:200]}...")
        print()


if __name__ == "__main__":
    # Point HuggingFace cache to /data/hf-cache where models are pre-downloaded
    os.environ.setdefault("HF_HOME", "/data/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/hf-cache/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data/hf-cache/datasets")
    main()
