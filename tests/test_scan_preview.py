import numpy as np
import cv2
from cubecoach.vision.detector import get_sticker_regions, sample_regions_colors
from cubecoach.vision.scan import _draw_color_squares


def make_multicolor_warp():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    colors = [
        (0, 0, 255),   # red
        (0, 255, 0),   # green
        (255, 0, 0),   # blue
        (0, 255, 255), # yellow
        (255, 165, 0), # orange-ish
        (255, 255, 255),# white
        (0, 0, 0),     # black
        (128, 128, 128),# gray
        (0, 128, 255), # coral
    ]
    h = 300 // 3
    w = 300 // 3
    idx = 0
    for r in range(3):
        for c in range(3):
            cv2.rectangle(img, (c * w + 5, r * h + 5), ((c + 1) * w - 5, (r + 1) * h - 5), colors[idx], -1)
            idx += 1
    return img


def test_draw_color_squares_reflects_samples():
    warp = make_multicolor_warp()
    regions = get_sticker_regions(warp, grid=3)
    colors = sample_regions_colors(warp, regions)
    preview = _draw_color_squares(warp, colors, show_values=True)
    # preview should have non-zero content (not all black)
    assert preview.mean() > 5
    # and the preview's right-bottom cell should reflect one of the colors (not be all zeros)
    h, w = preview.shape[:2]
    cell_h = h // 3
    cell_w = w // 3
    bot_right = preview[2 * cell_h + 5 : 3 * cell_h - 5, 2 * cell_w + 5 : 3 * cell_w - 5]
    assert bot_right.mean() > 10
