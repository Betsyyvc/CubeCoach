from cubecoach.vision.colors import map_bgr_to_label_with_distance, save_calibration
from cubecoach.vision.colors import bgr_to_hsv
import numpy as np


def test_map_bgr_to_label_with_distance_exact_match(tmp_path):
    # create a simple calibration where color (0,0,255) maps to label 'R'
    red_bgr = (0, 0, 255)
    hsv = bgr_to_hsv(red_bgr)
    centers = {"R": list(hsv)}

    lbl, dist = map_bgr_to_label_with_distance(red_bgr, centers)
    assert lbl == "R"
    assert dist is not None
    # exact match should be small distance
    assert dist < 5.0


def test_map_bgr_to_label_with_distance_missing_calib():
    # Behavior: if no centers are provided, function will try to load saved calibration.
    # If calibration is absent, it returns (None, None). If calibration exists, it returns (label, dist).
    lbl, dist = map_bgr_to_label_with_distance((10, 20, 30), centers=None)
    assert (lbl is None and dist is None) or (isinstance(lbl, str) and isinstance(dist, float))
