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


def mean_patch_(img, x, y, size=16):
    """Inplace mean patch to location arround (x, y)."""
    x_ = slice(x - size // 2, x + size // 2)
    y_ = slice(y - size // 2, y + size // 2)
    img[y_, x_, :] = np.mean(img[y_, x_, :], axis=(0, 1), keepdims=True)


def noise_patch_(img, x, y, size=16):
    """Inplace noise patch to location arround (x, y)."""
    x_ = slice(x - size // 2, x + size // 2)
    y_ = slice(y - size // 2, y + size // 2)
    img[y_, x_, :] = np.random.uniform(0, 255, size=(32, 32, 3))


def black_patch(img, x, y, size=16):
    """Inplace black patch to location arround (x, y)."""
    r_0 = (x-size//2, y-size//2)
    r_1 = (x+size//2, y+size//2)
    cv2.rectangle(img, r_0, r_1, color=(0, 0, 0), thickness=-1)


def white_patch(img, x, y, size=16):
    """Inplace white patch to location arround (x, y)."""
    r_0 = (x-size//2, y-size//2)
    r_1 = (x+size//2, y+size//2)
    cv2.rectangle(img, r_0, r_1, color=(255, 255, 255), thickness=-1)


def install_patch_type(config: argparse.Namespace):
    """Installs patch type used for input image distortion."""
    if config.patch_type == "mean":
        return mean_patch_
    elif config.patch_type == "noise":
        return noise_patch_
    elif config.patch_type == "black":
        return black_patch_
    elif config.patch_type == "white":
        return white_patch_


def interactive_lrp(config: argparse.Namespace):
    """Interactive LRP"""

    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using: {device}\n")

    img = cv2.imread("input/cats/cat_1.jpg")

    window_name_input = "Input Image"
    window_name_scores = "Relevance Scores"

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)
    lrp_model = LRPModel(model=model, top_k=config.top_k)

    data_processing = DataProcessing(config=config, device=device)
    apply_patch = install_patch_type(config=config)

    def draw(event, x, y, flags, img):
        if flags == 1:
            apply_patch(img, x, y, size=64)

    cv2.namedWindow(window_name_input)
    cv2.setMouseCallback(window_name_input, draw, img)

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
    parser.add_argument(
        "-p",
        "--patch-type",
        dest="patch_type",
        choices=["mean", "noise", "black", "white"],
        help="Defines patch type applied to input image.",
        default="mean",
        type=str,
    )

    config = parser.parse_args()

    interactive_lrp(config=config)
