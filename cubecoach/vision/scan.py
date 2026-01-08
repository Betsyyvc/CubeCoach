"""Scan workflow to capture all 6 faces and produce a facelet string."""
import time
import json
from pathlib import Path

import cv2
from cubecoach.vision.detector import face_from_frame
from cubecoach.vision.colors import map_bgr_to_label

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


def prompt_and_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    faces = {}
    try:
        while len(faces) < 6:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            cv2.putText(
                display,
                f"Scan faces: collected {len(faces)}/6. Press one of U,R,F,D,L,B when pointing at that face.",
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
                warp, colors = face_from_frame(frame)
                if warp is None:
                    print("No face detected. Try again.")
                    continue
                # map colors to labels using calibration
                mapped = [map_bgr_to_label(c) for c in colors]
                if any(m is None for m in mapped):
                    print("Calibration missing or incomplete. Use --calibrate first.")
                    continue
                # save the warp and mapping
                img_path = _save_face_image(face_label, warp)
                map_path = _save_face_mapping(face_label, mapped)
                faces[face_label] = mapped
                print(f"Captured face {face_label} — image saved to {img_path}, mapping saved to {map_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # validate
    if len(faces) != 6:
        raise RuntimeError("Did not capture all 6 faces")

    # assemble facelet string in order U R F D L B, each with 9 labels
    facelets = "".join("".join(faces[f]) for f in FACE_ORDER)
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