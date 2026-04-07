"""Save RGB uint8 video with imageio + ffmpeg (pip install imageio[ffmpeg])."""

from __future__ import annotations

from contextlib import contextmanager

import imageio.v2 as imageio
import numpy as np


def save_rgb_mp4(path: str, frames: np.ndarray, fps: float) -> None:
    """Write (T, H, W, 3) uint8 RGB to an MP4 file."""
    arr = np.ascontiguousarray(frames)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3) uint8 RGB, got shape={arr.shape} dtype={arr.dtype}")
    imageio.mimsave(path, arr, fps=float(fps), codec="libx264", macro_block_size=1)


@contextmanager
def rgb_mp4_writer(path: str, fps: float):
    """Streaming writer; append_data(frame) with (H, W, 3) uint8 RGB per frame."""
    w = imageio.get_writer(path, fps=float(fps), codec="libx264", macro_block_size=1)
    try:
        yield w
    finally:
        w.close()
