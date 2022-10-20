"""Python class to capture webcam frames.
"""
import cv2
import numpy


class Webcam:
    """Returns central crop of webcam frame as Numpy array."""

    def __init__(self, camera=0):
        """Initializes webcam class."""
        self.cam = cv2.VideoCapture(camera)

        if not self.cam.isOpened():
            self.cam.open()

    def _capture(self) -> numpy.ndarray:
        """Captures frame from selected webcam."""
        return_value, frame = self.cam.read()
        assert return_value is True, "Error: No frame captured."
        return frame

    def capture_frame(self):
        """Returns webcam frame.

        Returns: Numpy array of specified size.
        """
        return self._capture()

    def turn_off(self):
        """Turns off webcam."""
        self.cam.release()
