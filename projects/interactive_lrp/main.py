"""Interactive Layer-wise Relevance Propagation."""
import argparse
import time

import cv2
import numpy
import numpy as np
import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms

from src.lrp import LRPModel
from projects.real_time_lrp.data_processing import DataProcessing   # TODO: Move processing to other location.


def interactive_lrp(config: argparse.Namespace):
    """Interactive LRP"""

    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using: {device}\n")

    img = cv2.imread("input/cats/cat_6.jpg")

    window_name_input = "Input Image"
    window_name_scores = "Relevance Scores"

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)
    lrp_model = LRPModel(model=model, top_k=config.top_k)

    data_processing = DataProcessing(config=config, device=device)

    def draw(event, x, y, flags, param, size=64):
        if flags == 1:
            # Solid color
            # cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color=(0, 0, 0), thickness=-1)
            # Noise
            # img[y-size//2:y+size//2, x-size//2:x+size//2, :] = np.random.uniform(0, 255, size=(32, 32, 3))
            # Average
            img[
                y - size // 2 : y + size // 2, x - size // 2 : x + size // 2, :
            ] = np.mean(
                img[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2, :],
                axis=(0, 1),
                keepdims=True,
            )

    cv2.namedWindow(window_name_input)
    cv2.setMouseCallback(window_name_input, draw)

    cv2.namedWindow(window_name_scores)

    is_running = True

    while is_running:
        t0 = time.time()

        cv2.imshow(window_name_input, img)
        relevance_scores = lrp_model(data_processing.preprocess(img))
        cv2.imshow(window_name_scores, data_processing.postprocess(relevance_scores))

        if cv2.waitKey(1) & 0xFF == 27:
            is_running = False

        print("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))

    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--image-path",
        dest="image_path",
        help="Path to image.",
        default="./input/cats/cat1.jpg",
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        help="Device.",
        choices=["gpu", "cpu"],
        default="gpu",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--top-k",
        dest="top_k",
        help="Proportion of relevance scores that are allowed to pass.",
        default=0,
        type=float,
    )
    parser.add_argument(
        "-r",
        "--resize",
        dest="resize",
        help="Resize image before processing.",
        default=0,
        type=int,
    )

    config = parser.parse_args()

    interactive_lrp(config=config)
