"""Utilities for color conversions and camera-specific color mapping for cube stickers.

This module supports saving/loading HSV calibration centers and mapping sampled
BGR/RGB values to the nearest calibrated face color.
"""
import json
from pathlib import Path

import numpy as np
import cv2

CAL_FILE = Path(__file__).parent / "calibration.json"


def save_calibration(centers, path=CAL_FILE):
    """Save calibration centers (dict of label -> [h,s,v]) to JSON."""
    with open(path, "w") as f:
        json.dump(centers, f, indent=2)


def load_calibration(path=CAL_FILE):
    """Load calibration centers or return None if not present."""
    if not Path(path).exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def bgr_to_hsv(bgr):
    """Convert a single BGR tuple to HSV triple (integers)."""
    bgr_arr = np.uint8([[bgr]])
    hsv = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def map_hsv_to_label(hsv, centers):
    """Map an HSV triple to the nearest label from `centers`.

    centers: dict of label -> [h,s,v]
    """
    best = None
    best_d = None
    a = np.array(hsv)
    for label, vals in centers.items():
        d = np.linalg.norm(a - np.array(vals))
        if best_d is None or d < best_d:
            best_d = d
            best = label
    return best


def map_bgr_to_label(bgr, centers=None):
    """Map a BGR tuple (as sampled from OpenCV frame) to a label using loaded calibration.

    If `centers` is None, will attempt to load `calibration.json` from the package.
    Returns the label string or None if no calibration is available.
    """
    if centers is None:
        centers = load_calibration()
        if centers is None:
            return None
    hsv = bgr_to_hsv(bgr)
    return map_hsv_to_label(hsv, centers)

