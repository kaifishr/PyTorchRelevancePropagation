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
from src.data_processing import DataProcessing
from projects.interactive_lrp.patch import install_patch


def interactive_lrp(config: argparse.Namespace):
    """Interactive LRP"""

    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using: {device}\n")

    image = cv2.imread(config.input_path)

    window_name_input = "Input Image"
    window_name_scores = "Relevance Scores"

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)
    lrp_model = LRPModel(model=model, top_k=config.top_k)

    data_processing = DataProcessing(config=config, device=device)
    apply_patch = install_patch(config=config)

    def draw(event, x, y, flags, img):
        if flags == 1:
            apply_patch(img, x, y, size=config.patch_size)

    cv2.namedWindow(window_name_input)
    cv2.setMouseCallback(window_name_input, draw, image)

    cv2.namedWindow(window_name_scores)

    is_running = True

    while is_running:
        t0 = time.time()

        cv2.imshow(window_name_input, image)
        relevance_scores = lrp_model(data_processing.preprocess(image))
        cv2.imshow(window_name_scores, data_processing.postprocess(relevance_scores))

        if cv2.waitKey(1) & 0xFF == 27:
            is_running = False

        print("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))

    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        dest="input_path",
        help="Path to input image.",
        default="input/cats/cat_8.jpg",
        type=str,
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

    parser.add_argument(
        "-p",
        "--patch-type",
        dest="patch_type",
        choices=["mean", "noise", "black", "white"],
        help="Defines patch type applied to input image.",
        default="noise",
        type=str,
    )

    parser.add_argument(
        "-s",
        "--patch-size",
        dest="patch_size",
        help="Defines patch size used for distortion.",
        default=32,
        type=int,
    )

    config = parser.parse_args()
    interactive_lrp(config=config)
