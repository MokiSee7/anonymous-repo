"""
Shared frame selection utilities for MMS-Benchmark experiments.

Two strategies for capping the number of submitted frames:
  truncate : take the first max_frames frames (cap auto-computed from context if max_frames=None)
  uniform  : take max_frames evenly-spaced frames (max_frames must be specified)

Per-shot strategy (exp3):
  apply_per_shot_strategy  — cap frames within each shot
  apply_global_shot_strategy — cap total frames across all shots

Dynamic truncation:
  compute_dynamic_max_frames — compute max frames from model context window
"""

import numpy as np


def compute_dynamic_max_frames(model, tokens_per_frame, text_budget=500):
    """
    Compute the maximum number of frames that fit in the model's context window.

    Used for dynamic truncation (frame_strategy='truncate', max_frames not specified).
    The result is constant for a given model + resolution throughout a run.

    Args:
        model           : loaded HuggingFace model (must have config.max_position_embeddings)
        tokens_per_frame: visual tokens per frame = max_pixels // 784
                          (Qwen patch=14×14, 2×2 merge → 784 px/token;
                           e.g. 448×448 → 200704//784 = 256 tokens/frame)
        text_budget     : conservative upper bound on prompt text tokens (default 500).
                          Increase for longer prompts; NOT applicable when --with_doc is used
                          (doc length is variable — specify --max_frames explicitly instead).

    Returns:
        int — maximum safe frame count
    """
    # Qwen2.5-VL / Qwen3-VL store max_position_embeddings in the text sub-config,
    # not the top-level config — check both locations.
    context_len = (
        getattr(model.config, "max_position_embeddings", None)
        or getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None)
        or getattr(getattr(model.config, "language_model", None), "max_position_embeddings", None)
        or 32768
    )
    return max(1, (context_len - text_budget) // tokens_per_frame)


def apply_frame_strategy(frames, frame_indices, max_frames, strategy):
    """
    Cap a flat list of frames using truncate or uniform strategy.

    Args:
        frames        : list[PIL.Image]
        frame_indices : list[int]  — original frame indices
        max_frames    : int | None
        strategy      : 'truncate' | 'uniform'

    Returns (frames, frame_indices) — same length or shorter.
    """
    n = len(frames)
    if max_frames is None or n <= max_frames:
        return frames, frame_indices
    if strategy == "truncate":
        return frames[:max_frames], frame_indices[:max_frames]
    # uniform
    idxs = np.linspace(0, n - 1, max_frames, dtype=int)
    return [frames[i] for i in idxs], [frame_indices[i] for i in idxs]


def apply_per_shot_strategy(shots, max_frames_per_shot, strategy):
    """
    Apply per-shot frame limiting using a given strategy (for exp3).

    Args:
        shots              : list of shot dicts (each has 'frames' and 'frame_indices')
        max_frames_per_shot: int | None
        strategy           : 'truncate' | 'uniform'

    Returns a new list of shot dicts.
    """
    if max_frames_per_shot is None:
        return shots
    new_shots = []
    for shot in shots:
        frames = shot["frames"]
        fidxs  = shot.get("frame_indices", [])
        n = len(frames)
        if n <= max_frames_per_shot:
            new_shots.append(shot)
        elif strategy == "truncate":
            ns = dict(shot)
            ns["frames"]        = frames[:max_frames_per_shot]
            ns["frame_indices"] = fidxs[:max_frames_per_shot]
            new_shots.append(ns)
        else:  # uniform
            idxs = np.linspace(0, n - 1, max_frames_per_shot, dtype=int)
            ns = dict(shot)
            ns["frames"]        = [frames[i] for i in idxs]
            ns["frame_indices"] = [fidxs[i] for i in idxs] if fidxs else []
            new_shots.append(ns)
    return new_shots


def apply_global_shot_strategy(shots, max_total_frames, strategy):
    """
    Apply a global frame budget across all shots (for exp3).

    truncate : fill shots in order greedily; each shot gets at least 1 frame.
    uniform  : distribute budget proportionally; each shot gets at least 1 frame.

    Args:
        shots            : list of shot dicts
        max_total_frames : int | None
        strategy         : 'truncate' | 'uniform'

    Returns a new list of shot dicts with possibly fewer frames per shot.
    """
    if max_total_frames is None:
        return shots
    total = sum(len(s["frames"]) for s in shots)
    if total <= max_total_frames:
        return shots

    if strategy == "truncate":
        new_shots, remaining = [], max_total_frames
        for shot in shots:
            frames = shot["frames"]
            fidxs  = shot.get("frame_indices", [])
            take   = min(len(frames), max(1, remaining))
            ns = dict(shot)
            ns["frames"]        = frames[:take]
            ns["frame_indices"] = fidxs[:take]
            new_shots.append(ns)
            remaining = max(0, remaining - take)
        return new_shots
    else:  # uniform
        new_shots = []
        for shot in shots:
            n = len(shot["frames"])
            if n == 0:
                new_shots.append(shot)
                continue
            alloc = max(1, round(n / total * max_total_frames))
            if alloc >= n:
                new_shots.append(shot)
            else:
                idxs = np.linspace(0, n - 1, alloc, dtype=int)
                fidxs = shot.get("frame_indices", [])
                ns = dict(shot)
                ns["frames"]        = [shot["frames"][i] for i in idxs]
                ns["frame_indices"] = [fidxs[i] for i in idxs] if fidxs else []
                new_shots.append(ns)
        return new_shots
