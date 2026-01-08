from cubecoach.vision.colors import compute_calibration_from_raw_faces


def test_compute_calibration_from_raw_faces():
    # Create synthetic raw faces where center has distinct HSV-like BGR triples
    raw = {
        'U': [(250, 250, 250)] * 9,
        'R': [(10, 10, 200)] * 9,
        'F': [(10, 200, 10)] * 9,
        'D': [(200, 200, 10)] * 9,
        'L': [(200, 100, 10)] * 9,
        'B': [(10, 10, 200)] * 9,
    }
    calib = compute_calibration_from_raw_faces(raw)
    assert set(calib.keys()) == set(raw.keys())
    for v in calib.values():
        assert len(v) == 3
