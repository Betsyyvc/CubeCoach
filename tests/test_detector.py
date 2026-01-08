import numpy as np
import cv2
from cubecoach.vision.detector import warp_face, get_sticker_regions, sample_regions_colors


def make_synthetic_warp():
    # create a 300x300 image with 3x3 colored squares (BGR)
    img = np.zeros((300, 300, 3), dtype=np.uint8) + 50
    colors = [
        (255,255,255), # white
        (0,0,255),     # red
        (0,255,0),     # green
        (0,255,255),   # yellow
        (255,165,0),   # orange
        (255,0,0),     # blue
    ]
    # fill 3x3 grid with first color repeated for test simplicity
    h = 300//3
    w = 300//3
    for r in range(3):
        for c in range(3):
            cv2.rectangle(img, (c*w+5, r*h+5), ((c+1)*w-5, (r+1)*h-5), colors[0], -1)
    return img


def test_get_sticker_regions_and_samples():
    warp = make_synthetic_warp()
    regions = get_sticker_regions(warp, grid=3)
    colors = sample_regions_colors(warp, regions)
    assert len(colors) == 9
    # colors should be close to first color (white)
    for b,g,r in colors:
        assert b > 200 and g > 200 and r > 200
