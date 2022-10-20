"""Per-image Layer-wise Relevance Propagation

This script uses a pre-trained VGG network from PyTorch's Model Zoo
to perform Layer-wise Relevance Propagation (LRP) on the images
stored in the 'input' folder.

NOTE: LRP supports arbitrary batch size. Plot function does currently support only batch_size=1.

"""
import argparse
import time
import pathlib

import torch
from torchvision.models import vgg16, VGG16_Weights

from src.data import get_data_loader
from src.lrp import LRPModel

from projects.per_image_lrp.visualize import plot_relevance_scores


def per_image_lrp(config: argparse.Namespace) -> None:
    """Test function that plots heatmaps for images placed in the input folder.

    Images have to be placed in their corresponding class folders.

    Args:
        config: Argparse namespace object.

    """
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")

    data_loader = get_data_loader(config)

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)

    lrp_model = LRPModel(model=model, top_k=config.top_k)

    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        # y = y.to(device)  # here not used as method is unsupervised.

        t0 = time.time()
        r = lrp_model.forward(x)
        print("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))

        plot_relevance_scores(x=x, r=r, name=str(i), config=config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Input directory.",
        default="./input/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory.",
        default="./output/",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", help="Batch size.", default=1, type=int
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
        default=0.02,
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

    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    per_image_lrp(config=config)
