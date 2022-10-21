"""Real-time Layer-wise Relevance Propagation

This program uses the frames of a webcam, feeds them into a pre-trained VGG16 or VGG19 network
and performs layer-wise relevance propagation in real time.

"""
import cv2
import time
import argparse

import torch
from torchvision.models import vgg16, VGG16_Weights

from src.lrp import LRPModel
from src.data_processing import DataProcessing

from projects.real_time_lrp.webcam import Webcam
from projects.real_time_lrp.recorder import VideoRecorder


def real_time_lrp(config: argparse.Namespace) -> None:
    """Performs LRP on stream of frames coming from webcam.

    Args:
        config: Argparse Namespace object.

    """
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)

    lrp_model = LRPModel(model=model, top_k=config.top_k)

    webcam = Webcam()
    if config.record_video:
        recorder = VideoRecorder(config=config)
    data_processing = DataProcessing(config=config, device=device)

    is_running = True

    while is_running:
        t0 = time.time()

        frame = webcam.capture_frame()
        frame = data_processing.preprocess(frame)

        relevance_scores = lrp_model(frame)
        relevance_scores = data_processing.postprocess(relevance_scores)

        cv2.imshow("Relevance Scores", relevance_scores)
        if config.record_video:
            recorder.record(relevance_scores)

        t1 = time.time()
        fps = 1.0 / (t1 - t0)
        print(f"{fps:.1f} FPS")

        key = cv2.waitKey(1)
        if key == 27:
            # Exit loop if ESC is pressed.
            is_running = False

    if config.record_video:
        recorder.release()
    webcam.turn_off()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--resize",
        dest="resize",
        help="Resize image before processing.",
        default=640,
        type=int,
    )

    parser.add_argument(
        "-d", "--device", dest="device", help="Device.", default="gpu", type=str
    )

    parser.add_argument(
        "-f",
        "--fps",
        dest="fps",
        help="Frames per second of video.",
        default=20,
        type=int,
    )

    parser.add_argument(
        "-v",
        "--record_video",
        dest="record_video",
        help="Record video.",
        default=True,
        type=bool,
    )

    parser.add_argument(
        "-k",
        "--top-k",
        dest="top_k",
        help="Proportion of relevance scores that are allowed to pass.",
        default=0,
        type=float,
    )

    config = parser.parse_args()

    real_time_lrp(config=config)
