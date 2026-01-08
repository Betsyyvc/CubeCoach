import pytest

from cubecoach.vision.camera import Camera


def test_camera_init_no_source():
    # This test assumes machine has at least one camera; use invalid source to assert error
    with pytest.raises(RuntimeError):
        Camera(source=9999)
