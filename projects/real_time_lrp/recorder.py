"""Class for recording real-time relevance scores.
"""
import cv2
import argparse


class VideoRecorder(object):
    """Class to create video from image stream."""

    def __init__(self, config: argparse.Namespace):
        """Initializes video recorder class."""

        resolution = config.resize
        fps = config.fps
        filename = "output.avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(
            filename, fourcc, fps, (resolution, resolution)
        )

    def record(self, image):
        self.video_writer.write(image)

    def release(self):
        self.video_writer.release()
