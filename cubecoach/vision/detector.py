"""Cube face detection and sticker grid extraction."""
import cv2
import numpy as np
from typing import List, Tuple


def find_largest_quad(contours):
    """Return the approximated 4-point contour with the largest area, if any."""
    best = None
    best_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best = approx.reshape((4, 2))
    return best


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order quad points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def detect_face_contour(frame: np.ndarray) -> np.ndarray:
    """Detect the largest square/quad in the frame likely to be a cube face.

    Returns the ordered 4x2 float32 polygon or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = find_largest_quad(contours)
    if quad is None:
        return None
    return order_quad(quad)


def warp_face(frame: np.ndarray, quad: np.ndarray, size: int = 300) -> np.ndarray:
    """Warp the quadrilateral to a square of given size (top-down view)."""
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(frame, M, (size, size))
    return warp


def get_sticker_regions(warp: np.ndarray, grid: int = 3) -> List[Tuple[int, int, int, int]]:
    """Return bounding rects (x,y,w,h) for each sticker cell in the warped face image."""
    h, w = warp.shape[:2]
    cell_h = h // grid
    cell_w = w // grid
    regions = []
    pad_h = int(cell_h * 0.15)
    pad_w = int(cell_w * 0.15)
    for r in range(grid):
        for c in range(grid):
            x = c * cell_w + pad_w
            y = r * cell_h + pad_h
            ww = cell_w - 2 * pad_w
            hh = cell_h - 2 * pad_h
            regions.append((x, y, ww, hh))
    return regions


def sample_regions_colors(warp: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int]]:
    """Sample average BGR color from each region and return list in row-major order."""
    colors = []
    for (x, y, w, h) in regions:
        roi = warp[y : y + h, x : x + w]
        if roi.size == 0:
            colors.append((0, 0, 0))
            continue
        avg = cv2.mean(roi)[:3]
        colors.append(tuple(map(int, avg)))
    return colors


def face_from_frame(frame: np.ndarray, size: int = 300) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Detect a face in `frame`, warp it, and return (warp, sampled_colors) or (None, None)."""
    quad = detect_face_contour(frame)
    if quad is None:
        return None, None
    warp = warp_face(frame, quad, size=size)
    regions = get_sticker_regions(warp, grid=3)
    colors = sample_regions_colors(warp, regions)
    return warp, colors


if __name__ == "__main__":
    # quick interactive test: open camera and show detected warp
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        warp, colors = face_from_frame(frame)
        display = frame.copy()
        if warp is not None:
            cv2.imshow("warp", warp)
        cv2.imshow("frame", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
