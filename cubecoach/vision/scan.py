"""Scan workflow to capture all 6 faces and produce a facelet string."""
import time
import json
from pathlib import Path

import cv2
import numpy as np
from cubecoach.vision.detector import face_from_frame
from cubecoach.vision.colors import map_bgr_to_label, compute_calibration_from_raw_faces, save_calibration

FACE_KEYS = {'u': 'U', 'r': 'R', 'f': 'F', 'd': 'D', 'l': 'L', 'b': 'B'}
FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']

SCANS_DIR = Path(__file__).parent / "scans"
SCANS_DIR.mkdir(exist_ok=True)


def _save_face_image(face_label, warp):
    ts = int(time.time())
    p = SCANS_DIR / f"{face_label}_{ts}.png"
    cv2.imwrite(str(p), warp)
    return p


def _save_face_mapping(face_label, mapped):
    ts = int(time.time())
    p = SCANS_DIR / f"{face_label}_{ts}.json"
    with open(p, "w") as f:
        json.dump({"label": face_label, "mapped": mapped}, f)
    return p


def _save_facelets(facelets):
    ts = int(time.time())
    p = SCANS_DIR / f"facelets_{ts}.txt"
    with open(p, "w") as f:
        f.write(facelets)
    return p


def _draw_mapping_overlay(warp, mapped):
    """Draw mapped labels on the warp image (3x3 grid)."""
    out = warp.copy()
    h, w = out.shape[:2]
    cell_h = h // 3
    cell_w = w // 3
    for i, label in enumerate(mapped):
        r = i // 3
        c = i % 3
        cx = c * cell_w + cell_w // 2
        cy = r * cell_h + cell_h // 2
        cv2.putText(out, str(label), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        # draw cell rectangle
        x = c * cell_w
        y = r * cell_h
        cv2.rectangle(out, (x + 5, y + 5), (x + cell_w - 5, y + cell_h - 5), (0, 255, 0), 1)
    return out


def _draw_color_squares(warp, bgr_colors, labels=None, show_values=False, confidences=None, thresholds=None):
    """Create a preview image showing the 3x3 sampled BGR colors filled per cell.

    - If `show_values` is True, overlay the BGR numeric values on each cell for debugging.
    - `confidences`: optional list of (label, distance) tuples used to draw colored borders.
    - `thresholds`: optional dict with 'high' and 'low' numeric thresholds that determine green/yellow/red coding.

    labels: optional list of 9 labels to draw on top of each cell.
    """
    size = 300
    out = np.zeros((size, size, 3), dtype=np.uint8)
    cell_h = size // 3
    cell_w = size // 3
    # default thresholds
    if thresholds is None:
        from .colors import CONFIDENCE_HIGH, CONFIDENCE_LOW
        thresholds = {"high": CONFIDENCE_HIGH, "low": CONFIDENCE_LOW}

    for i, bgr in enumerate(bgr_colors):
        r = i // 3
        c = i % 3
        x = c * cell_w
        y = r * cell_h
        color = tuple(int(v) for v in bgr)
        # If the sampled color is all zeros, draw a red cross to make it obvious
        if color == (0, 0, 0):
            cv2.rectangle(out, (x + 2, y + 2), (x + cell_w - 2, y + cell_h - 2), (0, 0, 0), -1)
            cv2.line(out, (x + 4, y + 4), (x + cell_w - 4, y + cell_h - 4), (0, 0, 255), 2)
            cv2.line(out, (x + cell_w - 4, y + 4), (x + 4, y + cell_h - 4), (0, 0, 255), 2)
        else:
            cv2.rectangle(out, (x + 2, y + 2), (x + cell_w - 2, y + cell_h - 2), color, -1)

        # draw border colored by confidence if provided
        border_color = (255, 255, 255)
        if confidences and confidences[i] is not None:
            label, dist = confidences[i]
            if dist is None:
                border_color = (0, 0, 255)
            else:
                if dist <= thresholds["high"]:
                    border_color = (0, 255, 0)  # green
                elif dist <= thresholds["low"]:
                    border_color = (0, 255, 255)  # yellow
                else:
                    border_color = (0, 0, 255)  # red
        cv2.rectangle(out, (x + 2, y + 2), (x + cell_w - 2, y + cell_h - 2), border_color, 2)

        if labels:
            label = str(labels[i])
            cv2.putText(out, label, (x + 10, y + cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if show_values:
            txt = f"{color[2]},{color[1]},{color[0]}"  # show as R,G,B for readability
            cv2.putText(out, txt, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        # overlay numeric distance if confidences present
        if confidences and confidences[i] is not None:
            _, d = confidences[i]
            if d is not None:
                cv2.putText(out, f"{d:.1f}", (x + 6, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _average_samples_per_cell(samples_per_cell):
    """Given samples_per_cell: list of 9 lists of BGR tuples, return averaged 9 BGR tuple list."""
    averaged = []
    for cell_samples in samples_per_cell:
        if not cell_samples:
            averaged.append((0, 0, 0))
            continue
        arr = np.array(cell_samples, dtype=float)
        mean = np.mean(arr, axis=0)
        averaged.append(tuple(map(int, mean)))
    return averaged


def prompt_and_capture():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # collect raw BGR samples per face label
    # raw_faces_samples: face_label -> list of 9 lists (each inner list holds repeated samples for that sticker)
    raw_faces_samples = {}
    # raw_faces_final: face_label -> finalized averaged 9 BGR tuples
    raw_faces_final = {}
    try:
        while len(raw_faces_final) < 6:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            cv2.putText(
                display,
                f"Scan faces: finalized {len(raw_faces_final)}/6. Point at a face and press its key (u/r/f/d/l/b).",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("CubeCoach - Scan", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            ch = chr(key) if key != 255 else ""
            if ch.lower() in FACE_KEYS:
                face_label = FACE_KEYS[ch.lower()]
                if face_label in raw_faces_final:
                    print(f"Face {face_label} already finalized. You may re-finalize it by pressing 'u' during the preview.")
                warp, colors = face_from_frame(frame)
                if warp is None:
                    print("No face detected. Try again.")
                    continue

                # compute confidences (label, distance) for each sampled color
                from .colors import map_bgr_to_label_with_distance
                confidences = []
                for s in colors:
                    lbl, dist = map_bgr_to_label_with_distance(s)
                    confidences.append((lbl, dist))

                color_preview = _draw_color_squares(warp, colors, show_values=True, confidences=confidences)
                warp_small = cv2.resize(warp, (300, 300))
                combined = np.hstack([warp_small, color_preview])
                winname = f"Capture Preview - raw colors ({face_label}) (y=store, f=finalize, r=reject, u=unfinalize)"
                cv2.imshow(winname, combined)
                print("Sampled BGR colors:", colors)
                print("Label/distance per cell:", confidences)
                print("Preview controls: 'y' store sample, 'f' finalize face (average stored samples), 'r' reject capture, 'u' unfinalize existing final and start over.")
                while True:
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord('y'):
                        # store this sample for the face
                        if face_label not in raw_faces_samples:
                            raw_faces_samples[face_label] = [[] for _ in range(9)]
                        for i, s in enumerate(colors):
                            raw_faces_samples[face_label][i].append(s)
                        samples_count = len(raw_faces_samples[face_label][0])
                        print(f"Stored sample #{samples_count} for face {face_label}. Press 'f' to finalize (average) or store more samples by scanning again.")
                        cv2.destroyWindow(winname)
                        break
                    elif k == ord('r') or k == ord('n'):
                        print("Capture rejected. Try again.")
                        cv2.destroyWindow(winname)
                        break
                    elif k == ord('f'):
                        # finalize: compute averaged colors from stored samples (or use current colors if none stored)
                        if face_label in raw_faces_samples and len(raw_faces_samples[face_label][0]) > 0:
                            averaged = _average_samples_per_cell(raw_faces_samples[face_label])
                            raw_faces_final[face_label] = averaged
                            print(f"Finalized face {face_label} using {len(raw_faces_samples[face_label][0])} samples per cell.")
                        else:
                            raw_faces_final[face_label] = colors
                            print(f"Finalized face {face_label} using this single capture.")
                        cv2.destroyWindow(winname)
                        break
                    elif k == ord('u'):
                        # unfinalize if previously finalized
                        if face_label in raw_faces_final:
                            print(f"Unfinalized face {face_label}. You can re-scan it now.")
                            del raw_faces_final[face_label]
                        else:
                            print("No finalized sample to unfinalize.")
                        cv2.destroyWindow(winname)
                        break
                    else:
                        # ignore other keys
                        continue

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # validate
    if len(raw_faces_final) != 6:
        raise RuntimeError("Did not finalize all 6 faces")

    # compute calibration from finalized faces and save it
    calib = compute_calibration_from_raw_faces({k: v for k, v in raw_faces_final.items()})
    save_calibration(calib)
    print(f"Computed and saved calibration: {calib}")

    # remap using the new calibration and save images and mappings
    mapped_faces = {}
    for face_label, samples in raw_faces_final.items():
        # remap each sample using the new calibration
        mapped = [map_bgr_to_label(s) for s in samples]
        mapped_faces[face_label] = mapped
        # create a warp-like display by creating a blank image and drawing squares
        blank = 255 * np.ones((300, 300, 3), dtype=np.uint8)
        preview = _draw_color_squares(blank, samples)
        preview2 = _draw_mapping_overlay(preview, mapped)
        img_path = _save_face_image(face_label, preview2)
        map_path = _save_face_mapping(face_label, mapped)
        print(f"Saved face {face_label}: image {img_path}, mapping {map_path}")

    # assemble facelet string in order U R F D L B, each with 9 labels
    facelets = "".join("".join(mapped_faces[f]) for f in FACE_ORDER)
    facelets_path = _save_facelets(facelets)
    print(f"Saved assembled facelets to {facelets_path}")

    # try to solve if solver is available
    try:
        from cubecoach.solver.kociemba_solver import KociembaSolver

        solver = KociembaSolver()
        sol = solver.solve(facelets)
        print("Solution:", sol)
    except Exception as e:
        print("Solver unavailable or failed (ok to ignore for now):", e)

    return facelets


if __name__ == "__main__":
    facelets = prompt_and_capture()
    print(facelets)