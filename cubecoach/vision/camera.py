"""Simple webcam wrapper and demo using OpenCV."""
import cv2


class Camera:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")

    def run(self):
        """Start a simple webcam loop and display frames."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imshow("CubeCoach - Press 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        self.release()

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
