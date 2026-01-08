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


# confidence thresholds (LAB distance units)
CONFIDENCE_HIGH = 10.0  # distance <= HIGH => good
CONFIDENCE_LOW = 25.0   # distance > LOW => bad


def map_bgr_to_label(bgr, centers=None):
    """Map a BGR tuple (as sampled from OpenCV frame) to a label using loaded calibration.

    Uses perceptual LAB distance for better color matching.
    If `centers` is None, will attempt to load `calibration.json` from the package.
    Returns the label string or None if no calibration is available.
    """
    if centers is None:
        centers = load_calibration()
        if centers is None:
            return None
    # delegate to the distance-aware helper and return only the label
    label, _ = map_bgr_to_label_with_distance(bgr, centers)
    return label


def map_bgr_to_label_with_distance(bgr, centers=None):
    """Map a BGR tuple to (label, distance) using LAB distance to calibrated centers.

    Returns (label, distance) or (None, None) if calibration is missing.
    """
    if centers is None:
        centers = load_calibration()
        if centers is None:
            return None, None
    # convert stored HSV centers to LAB
    lab_centers = {}
    for label, vals in centers.items():
        hsv = np.uint8([[[vals[0], vals[1], vals[2]]]])
        bgr_c = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        lab = cv2.cvtColor(np.uint8([[bgr_c]]), cv2.COLOR_BGR2LAB)[0][0]
        lab_centers[label] = lab.astype(float)

    lab = np.array(bgr_to_lab(bgr), dtype=float)
    # find nearest center
    best_label = None
    best_d = None
    for label, lvals in lab_centers.items():
        d = float(np.linalg.norm(lab - np.array(lvals, dtype=float)))
        if best_d is None or d < best_d:
            best_d = d
            best_label = label
    return best_label, best_d


def bgr_to_lab(bgr):
    """Convert BGR tuple to LAB triple (integers)."""
    bgr_arr = np.uint8([[bgr]])
    lab = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2LAB)[0][0]
    return int(lab[0]), int(lab[1]), int(lab[2])


def map_lab_to_label(lab, lab_centers):
    """Map an LAB triple to the nearest label from `lab_centers`.

    lab_centers: dict of label -> lab numpy array
    """
    best = None
    best_d = None
    a = np.array(lab, dtype=float)
    for label, vals in lab_centers.items():
        d = np.linalg.norm(a - np.array(vals, dtype=float))
        if best_d is None or d < best_d:
            best_d = d
            best = label
    return best


def compute_calibration_from_raw_faces(raw_faces):
    """Compute a calibration dict label->HSV from raw face samples.

    raw_faces: dict label -> list of 9 BGR tuples (row-major order)

    Strategy:
    - Prefer the center cell (index 4) as the canonical color for that face.
    - As fallback/robustness, use median HSV across the 9 samples.
    - Return a dict mapping label->HSV list [h,s,v].
    """
    centers = {}
    for label, samples in raw_faces.items():
        # compute center as HSV of center sticker
        try:
            center_bgr = samples[4]
        except Exception:
            center_bgr = samples[len(samples)//2]
        hsv = bgr_to_hsv(center_bgr)
        # also fallback to median of all HSVs for robustness
        hsvs = [bgr_to_hsv(s) for s in samples]
        med = list(np.median(np.array(hsvs), axis=0).astype(int))
        # choose between center and median by S+V magnitude - heuristic
        sv_center = hsv[1] + hsv[2]
        sv_med = med[1] + med[2]
        chosen = hsv if sv_center >= sv_med * 0.6 else med
        centers[label] = list(chosen)
    return centers

