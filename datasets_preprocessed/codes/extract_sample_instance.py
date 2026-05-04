import argparse
import csv
import json
import os
from typing import Any, Dict

import h5py
import numpy as np


DATA_ROOT = "/data/MMS_Benchmark/data"
MRHISUM_ROOT = os.path.join(DATA_ROOT, "mrhisum")
VIDEOXUM_ROOT = os.path.join(DATA_ROOT, "videoxum")
PROCESSED_ROOT = os.path.join(DATA_ROOT, "processed")


def _sample_h5_group(h5_path: str, video_id: str) -> Dict[str, Any]:
    """Sample one value (or small slice) from each dataset under a given video in an h5 file."""
    result: Dict[str, Any] = {"video_id": video_id}

    with h5py.File(h5_path, "r") as f:
        if video_id not in f:
            raise KeyError(f"{video_id} not found in {h5_path}")
        grp = f[video_id]

        datasets_sample: Dict[str, Any] = {}
        for name, ds in grp.items():
            arr = np.array(ds[()])

            if name == "video_name":
                value = arr
                if isinstance(value, (bytes, bytearray)):
                    datasets_sample[name] = value.decode("utf-8")
                else:
                    datasets_sample[name] = str(value)
                continue

            if arr.ndim == 0:
                datasets_sample[name] = arr.item()
            elif arr.ndim == 1:
                datasets_sample[name] = {
                    "sample_values": arr[:8].tolist(),
                    "shape": arr.shape,
                }
            elif arr.ndim == 2:
                datasets_sample[name] = {
                    "sample_row": arr[0, :8].tolist(),
                    "shape": arr.shape,
                }
            else:
                flat = arr.reshape(-1)[:8]
                datasets_sample[name] = {
                    "sample_flat": flat.tolist(),
                    "original_shape": arr.shape,
                }

        result["datasets_sample"] = datasets_sample

    return result


def sample_mrhisum_h5(video_id: str) -> Dict[str, Any]:
    h5_path = os.path.join(MRHISUM_ROOT, "h5", "mrhisum.h5")
    return _sample_h5_group(h5_path, video_id)


def sample_videoxum_h5(video_id: str) -> Dict[str, Any]:
    h5_path = os.path.join(VIDEOXUM_ROOT, "h5", "videoxum.h5")
    return _sample_h5_group(h5_path, video_id)


def sample_metadata_row(video_id: str) -> Dict[str, Any]:
    """Return the row from metadata.csv corresponding to video_id."""
    meta_path = os.path.join(MRHISUM_ROOT, "metadata.csv")
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video_id") == video_id:
                return row
    raise KeyError(f"{video_id} not found in {meta_path}")


def sample_mrhisum_data_json(video_id: str) -> Dict[str, Any]:
    """Sample one frame entry from processed/mrhisum_data.json for the given video."""
    path = os.path.join(PROCESSED_ROOT, "mrhisum_data.json")
    with open(path, "r") as f:
        data = json.load(f)

    if video_id not in data:
        raise KeyError(f"{video_id} not found in {path}")

    video_entry = data[video_id]
    frames = video_entry.get("frames", [])
    frame_example = frames[0] if frames else None

    meta = {k: v for k, v in video_entry.items() if k != "frames"}

    return {
        "video_id": video_id,
        "meta_without_frames": meta,
        "frame_example": frame_example,
        "num_frames": len(frames),
    }


def sample_step1_frames(video_id: str) -> Dict[str, Any]:
    """Sample one frame entry from processed/step1_frames/<video_id>.json."""
    path = os.path.join(PROCESSED_ROOT, "step1_frames", f"{video_id}.json")
    with open(path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    frame_example = frames[0] if frames else None

    return {
        "video_id": data.get("video_id", video_id),
        "frame_example": frame_example,
        "num_frames": len(frames),
    }


def sample_step3_aligned(video_id: str) -> Dict[str, Any]:
    """Sample one sentence and one alignment from processed/step3_aligned/<video_id>.json."""
    path = os.path.join(PROCESSED_ROOT, "step3_aligned", f"{video_id}.json")
    with open(path, "r") as f:
        data = json.load(f)

    aligned = data.get("aligned_full_document", {})
    sentences = aligned.get("sentences", [])
    alignments = aligned.get("alignments", [])

    sentence_example = sentences[0] if sentences else None
    alignment_example = alignments[0] if alignments else None

    return {
        "video_id": data.get("video_id", video_id),
        "sentence_example": sentence_example,
        "num_sentences": len(sentences),
        "alignment_example": alignment_example,
        "num_alignments": len(alignments),
    }


def build_mrhisum_summary(video_id: str) -> Dict[str, Any]:
    """Build a summary of one mrhisum video across all data sources."""
    h5_part = sample_mrhisum_h5(video_id)
    meta_row = sample_metadata_row(video_id)
    mrhisum_part = sample_mrhisum_data_json(video_id)
    step1_part = sample_step1_frames(video_id)
    step3_part = sample_step3_aligned(video_id)

    video_file = os.path.join(MRHISUM_ROOT, "videos", f"{video_id}.mp4")
    frames_archive = os.path.join(MRHISUM_ROOT, "frames", f"{video_id}.tar.gz")
    frame_descriptions_file = os.path.join(
        MRHISUM_ROOT, "frame_descriptions", f"{video_id}.xlsx"
    )
    shot_descriptions_file = os.path.join(
        MRHISUM_ROOT, "shot_descriptions", f"{video_id}.xlsx"
    )

    return {
        "dataset": "mrhisum",
        "video_id": video_id,
        "paths": {
            "video_file": video_file,
            "frames_archive": frames_archive,
            "frame_descriptions_xlsx": frame_descriptions_file,
            "shot_descriptions_xlsx": shot_descriptions_file,
        },
        "metadata_csv_row": meta_row,
        "mrhisum_h5": h5_part,
        "mrhisum_data_json": mrhisum_part,
        "step1_frames": step1_part,
        "step3_aligned": step3_part,
    }


def build_videoxum_summary_by_index(index: int) -> Dict[str, Any]:
    """
    Build a summary of one videoxum video, using the N-th key (0-based index)
    in processed/videoxum_data.json to define which video is the \"N-th\".
    """
    path = os.path.join(PROCESSED_ROOT, "videoxum_data.json")
    with open(path, "r") as f:
        data = json.load(f)

    keys = list(data.keys())
    if not (0 <= index < len(keys)):
        raise IndexError(f"Index {index} out of range for {len(keys)} videoxum videos")

    video_id = keys[index]
    video_entry = data[video_id]
    frames = video_entry.get("frames", [])
    frame_example = frames[0] if frames else None
    meta = {k: v for k, v in video_entry.items() if k != "frames"}

    h5_part = sample_videoxum_h5(video_id)

    video_name = h5_part.get("datasets_sample", {}).get("video_name")
    video_file = os.path.join(VIDEOXUM_ROOT, "videos", f"{video_name}.mp4")
    frames_archive = os.path.join(VIDEOXUM_ROOT, "frames", f"{video_name}.tar.gz")

    return {
        "dataset": "videoxum",
        "index": index,
        "video_id": video_id,
        "paths": {
            "video_file": video_file,
            "frames_archive": frames_archive,
        },
        "videoxum_h5": h5_part,
        "videoxum_data_json": {
            "video_id": video_id,
            "meta_without_frames": meta,
            "frame_example": frame_example,
            "num_frames": len(frames),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a compact, human-readable summary of one video for presentation.\n"
            "Supports mrhisum by video id and videoxum by index."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["mrhisum", "videoxum"],
        default="mrhisum",
        help="Which dataset to sample from. Default: mrhisum.",
    )
    parser.add_argument(
        "--video-id",
        help="For mrhisum: video id to sample (e.g., video_1000).",
    )
    parser.add_argument(
        "--videoxum-index",
        type=int,
        help="For videoxum: 1-based index of the video to sample (e.g., 100 for the 100th video).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to save the summary JSON. If omitted, only prints to stdout.",
    )
    args = parser.parse_args()

    if args.dataset == "mrhisum":
        video_id = args.video_id or "video_1000"
        summary = build_mrhisum_summary(video_id)
    else:
        if args.videoxum_index is None:
            raise SystemExit("For videoxum, please provide --videoxum-index (1-based).")
        index0 = args.videoxum_index - 1
        summary = build_videoxum_summary_by_index(index0)

    # Pretty-print to stdout
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary written to: {args.output}")


if __name__ == "__main__":
    main()

