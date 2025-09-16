"""Lightweight utility functions for the gaming overlay project.

This module intentionally avoids heavy imports (Flask, OpenCV) so it can be
used in unit tests without requiring system packages.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np


def hex_to_bgr(color_hex: str) -> str:
    """Convert a hex color string (e.g. "#00ff00" or "00ff00") to BGR CSV.

    Returns a string like "0,255,0" for green in BGR order. Falls back to a
    safe interpretation by tolerating missing leading '#'.
    """
    if not isinstance(color_hex, str):
        raise TypeError("color_hex must be a string")
    if not color_hex:
        raise ValueError("color_hex must be non-empty")
    if not color_hex.startswith('#'):
        color_hex = '#' + color_hex
    hexdigits = color_hex.lstrip('#')
    if len(hexdigits) != 6 or any(c not in '0123456789abcdefABCDEF' for c in hexdigits):
        raise ValueError(f"Invalid hex color: {color_hex}")
    r, g, b = (int(hexdigits[i:i+2], 16) for i in (0, 2, 4))
    return f"{b},{g},{r}"


def find_candidate_models(search_dirs: List[str]) -> List[str]:
    """Return YOLO model file paths under given directories.

    Considers .engine and .pt files containing "yolo" or "yolov" in the name.
    Ignores directories that cannot be listed.
    """
    candidates: List[str] = []
    exts = (".engine", ".pt")
    for directory in search_dirs:
        try:
            for name in os.listdir(directory):
                lower = name.lower()
                if lower.endswith(exts) and ("yolo" in lower or "yolov" in lower):
                    candidates.append(os.path.join(directory, name))
        except Exception:
            continue
    return candidates


def size_rank_from_name(name: str) -> int:
    """Rank model sizes by common YOLO suffix ordering: n < s < m < l < x.

    Lower rank is better. Unknown sizes receive the lowest priority.
    """
    lower = name.lower()
    order = ["n", "s", "m", "l", "x"]
    for idx, tag in enumerate(order):
        if f"v11{tag}" in lower or f"v{tag}" in lower or lower.endswith(f"{tag}.pt") or lower.endswith(f"{tag}.engine"):
            return idx
    return len(order)


def pick_best_model(paths: List[str], default_dir: Optional[str] = None) -> str:
    """Choose the most efficient candidate, preferring .engine and smaller sizes.

    If no paths are provided, returns a default path: <default_dir>/yolov11n.pt
    where default_dir defaults to the current working directory.
    """
    def score(path: str) -> Tuple[int, int, int]:
        lower = path.lower()
        ext = os.path.splitext(lower)[1]
        ext_priority = 0 if ext == ".engine" else 1
        size_rank = size_rank_from_name(lower)
        return (ext_priority, size_rank, len(lower))

    if not paths:
        base = default_dir or os.getcwd()
        return os.path.join(base, "yolov11n.pt")
    return sorted(paths, key=score)[0]


def alpha_blend_center(frame_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    """Alpha-blend overlay_bgra onto the center of frame_bgr using NumPy only.

    Both arrays must be uint8. The overlay may extend beyond frame bounds; the
    function clips appropriately. Returns the modified frame (same object).
    """
    if frame_bgr.dtype != np.uint8 or overlay_bgra.dtype != np.uint8:
        raise ValueError("frame_bgr and overlay_bgra must be uint8")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape HxWx3")
    if overlay_bgra.ndim != 3 or overlay_bgra.shape[2] != 4:
        raise ValueError("overlay_bgra must have shape HxWx4")

    fh, fw = frame_bgr.shape[:2]
    oh, ow = overlay_bgra.shape[:2]
    x = max(0, (fw - ow) // 2)
    y = max(0, (fh - oh) // 2)
    x2 = min(fw, x + ow)
    y2 = min(fh, y + oh)
    cw_eff = x2 - x
    ch_eff = y2 - y
    if cw_eff <= 0 or ch_eff <= 0:
        return frame_bgr

    roi = frame_bgr[y:y2, x:x2]
    ch_crop = overlay_bgra[0:ch_eff, 0:cw_eff]

    overlay_rgb = ch_crop[:, :, :3].astype(np.float32)
    alpha = (ch_crop[:, :, 3:4].astype(np.float32)) / 255.0
    inv_alpha = 1.0 - alpha
    base_rgb = roi[:, :, :3].astype(np.float32)
    out_rgb = alpha * overlay_rgb + inv_alpha * base_rgb
    roi[:, :, :3] = out_rgb.astype(np.uint8)
    return frame_bgr

