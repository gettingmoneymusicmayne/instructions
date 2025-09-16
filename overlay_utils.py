# Utility functions for color parsing and model selection
from __future__ import annotations

import os
import re
from typing import List


def _clamp_channel(value: int) -> int:
    return max(0, min(255, int(value)))


def hex_to_bgr(color_text: str) -> str:
    """Convert various color text formats to a B,G,R CSV string.

    Accepted inputs:
    - #RRGGBB or RRGGBB
    - #RGB or RGB (shorthand)
    - "r,g,b"
    - "rgb(r,g,b)"

    Invalid inputs fall back to bright green: "0,255,0".
    """
    if not color_text:
        return "0,255,0"

    s = str(color_text).strip()

    # rgb(r,g,b)
    m = re.match(r"^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$", s, re.IGNORECASE)
    if m:
        r, g, b = (_clamp_channel(int(m.group(1))), _clamp_channel(int(m.group(2))), _clamp_channel(int(m.group(3))))
        return f"{b},{g},{r}"

    # r,g,b
    if "," in s:
        parts = s.split(",")
        if len(parts) == 3:
            try:
                r, g, b = (_clamp_channel(int(parts[0])), _clamp_channel(int(parts[1])), _clamp_channel(int(parts[2])))
                return f"{b},{g},{r}"
            except Exception:
                pass

    # Hex forms
    hs = s.lstrip('#')
    if len(hs) == 3 and all(c in "0123456789aAbBcCdDeEfF" for c in hs):
        hs = ''.join(c * 2 for c in hs)
    if len(hs) == 6 and all(c in "0123456789aAbBcCdDeEfF" for c in hs):
        try:
            r = int(hs[0:2], 16)
            g = int(hs[2:4], 16)
            b = int(hs[4:6], 16)
            return f"{b},{g},{r}"
        except Exception:
            pass

    return "0,255,0"


def parse_bgr(csv_text: str) -> tuple[int, int, int]:
    """Parse B,G,R triplet from a CSV string, falling back to (0,255,0)."""
    try:
        b, g, r = map(int, str(csv_text).split(','))
        return (_clamp_channel(b), _clamp_channel(g), _clamp_channel(r))
    except Exception:
        return (0, 255, 0)


def find_candidate_models(search_dirs: List[str]) -> List[str]:
    candidates: List[str] = []
    exts = (".engine", ".pt")
    for d in search_dirs:
        try:
            for name in os.listdir(d):
                lower = name.lower()
                if lower.endswith(exts) and ("yolo" in lower or "yolov" in lower):
                    candidates.append(os.path.join(d, name))
        except Exception:
            continue
    return candidates


def size_rank_from_name(name: str) -> int:
    lower = name.lower()
    order = ["n", "s", "m", "l", "x"]
    for i, tag in enumerate(order):
        if f"v11{tag}" in lower or f"v{tag}" in lower or lower.endswith(f"{tag}.pt") or lower.endswith(f"{tag}.engine"):
            return i
    return len(order)


def pick_best_model(paths: List[str]) -> str:
    def score(p: str):
        lower = p.lower()
        ext = os.path.splitext(lower)[1]
        ext_priority = 0 if ext == ".engine" else 1
        size_rank = size_rank_from_name(lower)
        return (ext_priority, size_rank, len(lower))
    if not paths:
        return os.path.join(os.getcwd(), "yolov11n.pt")
    return sorted(paths, key=score)[0]