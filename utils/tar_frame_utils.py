"""
Utility functions for working with tar.gz compressed video frames

Provides helper functions to extract and work with frames stored
in tar.gz archives without unpacking entire archives.
"""

import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from contextlib import contextmanager


class TarFrameExtractor:
    """Extract frames from tar.gz archives efficiently"""

    def __init__(self, tar_path: Path):
        """
        Initialize extractor for a tar.gz file

        Args:
            tar_path: Path to tar.gz file containing frames
        """
        self.tar_path = Path(tar_path)
        if not self.tar_path.exists():
            raise FileNotFoundError(f"Tar file not found: {tar_path}")

    @contextmanager
    def extract_frames(self, frame_indices: Optional[List[int]] = None):
        """
        Context manager to extract frames to temporary directory

        Args:
            frame_indices: List of frame indices to extract (0-based)
                         If None, extracts all frames

        Yields:
            Path to temporary directory containing extracted frames

        Example:
            >>> extractor = TarFrameExtractor("video_1.tar.gz")
            >>> with extractor.extract_frames([0, 5, 10]) as temp_dir:
            >>>     frame_paths = list(temp_dir.glob("*.jpg"))
            >>>     # Use frame_paths...
            >>> # temp_dir automatically deleted after context
        """
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="tar_frames_"))

        try:
            with tarfile.open(self.tar_path, 'r:gz') as tar:
                members = tar.getmembers()

                # Sort members to ensure consistent frame ordering
                members.sort(key=lambda m: m.name)

                if frame_indices is None:
                    # Extract all frames
                    for member in members:
                        if member.isfile() and (member.name.endswith('.jpg') or member.name.endswith('.png')):
                            tar.extract(member, temp_dir)
                else:
                    # Extract specific frames by index
                    # Assumes frame filenames are sorted (frame_000000.jpg, frame_000001.jpg, ...)
                    frame_files = [m for m in members if m.isfile() and
                                  (m.name.endswith('.jpg') or m.name.endswith('.png'))]

                    for idx in frame_indices:
                        if 0 <= idx < len(frame_files):
                            member = frame_files[idx]
                            tar.extract(member, temp_dir)

            yield temp_dir

        finally:
            # Cleanup temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    @contextmanager
    def extract_all_frames(self):
        """
        Extract all frames to temporary directory

        Yields:
            Path to temporary directory containing all extracted frames
        """
        with self.extract_frames(frame_indices=None) as temp_dir:
            yield temp_dir

    def get_frame_count(self) -> int:
        """
        Get number of frames in tar archive

        Returns:
            Number of image files in the archive
        """
        count = 0
        with tarfile.open(self.tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and (member.name.endswith('.jpg') or member.name.endswith('.png')):
                    count += 1
        return count

    def list_frames(self) -> List[str]:
        """
        List all frame filenames in the archive

        Returns:
            Sorted list of frame filenames
        """
        frames = []
        with tarfile.open(self.tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and (member.name.endswith('.jpg') or member.name.endswith('.png')):
                    frames.append(member.name)

        return sorted(frames)


def get_frames_for_clip_from_tar(tar_path: Path,
                                  frame_numbers: List[int],
                                  all_picks: List[int]) -> Tuple[List[Path], Path]:
    """
    Extract frames for a specific clip from tar.gz archive

    This is compatible with the shot description generation workflow.

    Args:
        tar_path: Path to tar.gz file containing frames
        frame_numbers: List of frame numbers (absolute indices) for the clip
        all_picks: List of all picked frame numbers from H5 (for mapping to tar indices)

    Returns:
        Tuple of (list of frame paths, temp_dir path)
        Caller is responsible for cleaning up temp_dir after use

    Example:
        >>> # From H5: picks = [0, 15, 30, 45, ...]
        >>> # Clip frames: [15, 30, 45]
        >>> frame_paths, temp_dir = get_frames_for_clip_from_tar(
        >>>     "video_1.tar.gz", [15, 30, 45], all_picks
        >>> )
        >>> # Use frame_paths...
        >>> shutil.rmtree(temp_dir)  # Cleanup
    """
    # Map absolute frame numbers to tar archive indices
    # Example: If picks = [0, 15, 30], and frame_numbers = [15, 30]
    # Then tar indices are [1, 2] (0-based)
    tar_indices = []
    for frame_num in frame_numbers:
        if frame_num in all_picks:
            idx = all_picks.index(frame_num)
            tar_indices.append(idx)

    # Extract frames
    extractor = TarFrameExtractor(tar_path)

    # Create temp dir and extract (Note: not using context manager here
    # because caller needs to use the frames and cleanup later)
    temp_dir = Path(tempfile.mkdtemp(prefix="clip_frames_"))

    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        members.sort(key=lambda m: m.name)

        frame_files = [m for m in members if m.isfile() and
                      (m.name.endswith('.jpg') or m.name.endswith('.png'))]

        extracted_paths = []
        for idx in tar_indices:
            if 0 <= idx < len(frame_files):
                member = frame_files[idx]
                tar.extract(member, temp_dir)

                # Get full path to extracted file
                extracted_path = temp_dir / member.name
                extracted_paths.append(extracted_path)

    return extracted_paths, temp_dir


def extract_frames_by_pattern(tar_path: Path,
                              frame_pattern: str = "frame_*.jpg") -> Tuple[List[Path], Path]:
    """
    Extract frames matching a pattern from tar.gz

    Args:
        tar_path: Path to tar.gz file
        frame_pattern: Glob pattern for frame filenames

    Returns:
        Tuple of (sorted list of extracted frame paths, temp_dir path)
        Caller must cleanup temp_dir after use
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="extracted_frames_"))

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(temp_dir)

    # Find frames matching pattern
    frame_paths = sorted(temp_dir.glob(frame_pattern))

    return frame_paths, temp_dir


@contextmanager
def temporary_frame_extraction(tar_path: Path,
                               frame_indices: Optional[List[int]] = None):
    """
    Context manager for temporary frame extraction from tar.gz

    Args:
        tar_path: Path to tar.gz file
        frame_indices: Indices of frames to extract (None = all)

    Yields:
        List of paths to extracted frames

    Example:
        >>> with temporary_frame_extraction("video.tar.gz", [0, 5, 10]) as frames:
        >>>     for frame_path in frames:
        >>>         # Process frame...
        >>>         pass
        >>> # Frames automatically deleted
    """
    extractor = TarFrameExtractor(tar_path)

    with extractor.extract_frames(frame_indices) as temp_dir:
        # Get sorted list of frame paths
        frame_paths = sorted(temp_dir.glob("*.jpg")) + sorted(temp_dir.glob("*.png"))
        yield frame_paths
