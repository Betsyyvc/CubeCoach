import json
import tempfile
from cubecoach.vision.colors import map_hsv_to_label, save_calibration, load_calibration


def test_map_hsv_to_label_tmpfile():
    centers = {
        'U': [0, 0, 255],
        'R': [0, 255, 255],
        'F': [60, 255, 255],
    }
    # HSV close to 'F'
    label = map_hsv_to_label((58, 240, 250), centers)
    assert label == 'F'


def test_save_and_load(tmp_path):
    centers = {'X': [10, 20, 30], 'Y': [100, 110, 120]}
    p = tmp_path / "cal.json"
    save_calibration(centers, path=p)
    loaded = load_calibration(path=p)
    assert loaded == centers
