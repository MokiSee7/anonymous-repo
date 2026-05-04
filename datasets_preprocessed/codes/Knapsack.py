"""
Knapsack algorithm for GT shot selection.

Selects shots (segments) to maximize total importance score
while keeping total frame count within a budget (capacity).
"""

import numpy as np


def solve_knapsack(segment_scores, segment_lengths, capacity):
    """
    Solve 0/1 knapsack problem for shot selection.

    Args:
        segment_scores: Array of importance scores for each segment (n_segments,)
        segment_lengths: Array of frame counts for each segment (n_segments,)
        capacity: Maximum total frames to select (int)

    Returns:
        shot_level_gt: Binary array (n_segments,) indicating selected segments
    """
    n_segments = len(segment_scores)

    if n_segments == 0:
        return np.array([], dtype=np.int32)

    # Convert to integer weights for DP
    weights = np.array(segment_lengths, dtype=np.int32)
    values = np.array(segment_scores, dtype=np.float64)
    capacity = int(capacity)

    # Handle edge case: capacity <= 0
    if capacity <= 0:
        return np.zeros(n_segments, dtype=np.int32)

    # Dynamic programming table
    # dp[i][w] = max value using first i items with capacity w
    dp = np.zeros((n_segments + 1, capacity + 1), dtype=np.float64)

    for i in range(1, n_segments + 1):
        weight = weights[i - 1]
        value = values[i - 1]

        for w in range(capacity + 1):
            # Don't take item i
            dp[i][w] = dp[i - 1][w]

            # Take item i if it fits
            if weight <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weight] + value)

    # Backtrack to find selected items
    shot_level_gt = np.zeros(n_segments, dtype=np.int32)
    w = capacity

    for i in range(n_segments, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            # Item i was selected
            shot_level_gt[i - 1] = 1
            w -= weights[i - 1]

    return shot_level_gt


def solve_knapsack_greedy(segment_scores, segment_lengths, capacity):
    """
    Greedy approximation for knapsack (faster for large inputs).

    Selects segments by value-to-weight ratio (efficiency).

    Args:
        segment_scores: Array of importance scores for each segment
        segment_lengths: Array of frame counts for each segment
        capacity: Maximum total frames to select

    Returns:
        shot_level_gt: Binary array indicating selected segments
    """
    n_segments = len(segment_scores)

    if n_segments == 0:
        return np.array([], dtype=np.int32)

    weights = np.array(segment_lengths, dtype=np.int32)
    values = np.array(segment_scores, dtype=np.float64)
    capacity = int(capacity)

    if capacity <= 0:
        return np.zeros(n_segments, dtype=np.int32)

    # Calculate efficiency (value per unit weight)
    # Add small epsilon to avoid division by zero
    efficiency = values / (weights + 1e-8)

    # Sort by efficiency (descending)
    sorted_indices = np.argsort(efficiency)[::-1]

    # Greedily select segments
    shot_level_gt = np.zeros(n_segments, dtype=np.int32)
    total_weight = 0

    for idx in sorted_indices:
        if total_weight + weights[idx] <= capacity:
            shot_level_gt[idx] = 1
            total_weight += weights[idx]

    return shot_level_gt
