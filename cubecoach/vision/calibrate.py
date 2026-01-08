"""Interactive calibration tool for CubeCoach.

Usage:
    python -m cubecoach.vision.calibrate

How it works:
- Opens camera and shows live feed.
- Press one of the color keys to record samples for that color:
  U (u), R (r), F (f), D (d), L (l), B (b)
- Press 's' to save averaged HSV centers to `calibration.json`.
- Press 't' to toggle live test mapping on central patch.
- Press 'q' to quit (prompt to save if you have unsaved samples).
"""
import cv2
import json
import numpy as np
from pathlib import Path

CAL_FILE = Path(__file__).parent / "calibration.json"

COLOR_KEYS = {
    'u': 'U',  # Up / white
    'r': 'R',  # Red
    'f': 'F',  # Front / green
    'd': 'D',  # Down / yellow
    'l': 'L',  # Left / orange
    'b': 'B',  # Back / blue
}


def average_hsv(samples):
    arr = np.array(samples, dtype=float)
    mean = arr.mean(axis=0)
    return [int(mean[0]), int(mean[1]), int(mean[2])]


def save_calibration(centers, path=CAL_FILE):
    with open(path, 'w') as f:
        json.dump(centers, f, indent=2)
    print(f"Saved calibration to {path}")


def load_calibration(path=CAL_FILE):
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def sample_center_hsv(frame, box=30):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    x0, y0 = cx - box, cy - box
    roi = frame[y0: y0 + box * 2, x0: x0 + box * 2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = int(hsv[..., 0].mean())
    s_mean = int(hsv[..., 1].mean())
    v_mean = int(hsv[..., 2].mean())
    return (h_mean, s_mean, v_mean)


def draw_instructions(img, samples):
    txt = "Calibration: press U,R,F,D,L,B to record samples | s=save | t=test | q=quit"
    cv2.putText(img, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y = 40
    for k, label in COLOR_KEYS.items():
        cnt = len(samples.get(label, []))
        cv2.putText(img, f"{label} ({k}): {cnt} samples", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 0), 1)
        y += 20
    cv2.rectangle(img, (img.shape[1]//2 - 30, img.shape[0]//2 - 30), (img.shape[1]//2 + 30, img.shape[0]//2 + 30), (0,255,0), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    samples = {}
    test_mode = False

    print("Calibration starting. Press keys to capture samples. See on-screen instructions.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_instructions(frame, samples)

        if test_mode:
            hsv = sample_center_hsv(frame)
            # show the sampled hsv and mapped label if calibration exists
            calib = load_calibration()
            if calib:
                # simple nearest-center mapping
                best, best_d = None, None
                for label, vals in calib.items():
                    d = np.linalg.norm(np.array(hsv) - np.array(vals))
                    if best_d is None or d < best_d:
                        best_d = d
                        best = label
                cv2.putText(frame, f"Test HSV: {hsv} -> {best}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.putText(frame, f"Test HSV: {hsv} (no calibration loaded)", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("CubeCoach Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # quit
            break
        if key == ord('t'):
            test_mode = not test_mode
        if key == ord('s'):
            # compute averages
            centers = {}
            for label, vals in samples.items():
                centers[label] = average_hsv(vals)
            save_calibration(centers)
        # record sample
        ch = chr(key) if key != 255 else ''
        if ch in COLOR_KEYS:
            label = COLOR_KEYS[ch]
            hsv = sample_center_hsv(frame)
            samples.setdefault(label, []).append(list(hsv))
            print(f"Recorded sample for {label}: {hsv}")

    cap.release()
    cv2.destroyAllWindows()

    # On exit, if some samples exist, prompt to save
    if samples:
        ans = input("Save calibration? (y/N): ")
        if ans.lower().startswith('y'):
            centers = {label: average_hsv(vals) for label, vals in samples.items()}
            save_calibration(centers)


if __name__ == "__main__":
    main()
