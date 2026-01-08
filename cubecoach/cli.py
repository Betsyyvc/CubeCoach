"""Simple CLI for CubeCoach."""
import argparse
import sys
from cubecoach.vision.camera import Camera


def demo_camera():
    cam = Camera()
    try:
        cam.run()
    except KeyboardInterrupt:
        print("Exiting demo")
        cam.release()


def calibrate():
    # import lazily to avoid importing cv2 at module load time in non-GUI contexts
    from cubecoach.vision import calibrate as calib

    calib.main()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cubecoach")
    parser.add_argument("--demo", action="store_true", help="Run a webcam demo")
    parser.add_argument("--calibrate", action="store_true", help="Run camera/color calibration tool")
    parser.add_argument("--scan", action="store_true", help="Run guided scan to capture all 6 faces for the solver")
    args = parser.parse_args(argv)

    if args.demo:
        demo_camera()
    elif args.calibrate:
        calibrate()
    elif args.scan:
        from cubecoach.vision.scan import prompt_and_capture
        facelets = prompt_and_capture()
        print("Captured facelets:", facelets)
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
