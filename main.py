"""Layer-wise Relevance Propagation

This script uses a pre-trained VGG network from PyTorch's Model Zoo
to perform layer-wise relevance propagation.

"""
import time
import torch
import torchvision
import yaml

from src.visualize import plot_relevance_scores
from src.data import get_data_loader
from src.lrp import LRPModel


def run_lrp(config: dict) -> None:
    """Test function that plots heatmaps for images placed in the input folder.

    Images have to be placed in their corresponding class folders.

    Args:
        config: Dictionary holding configuration parameters.

    Returns:
        None

    """
    if config["device"] == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")

    data_loader = get_data_loader(config)

    model = torchvision.models.vgg16(pretrained=True)
    model.to(device)

    lrp_model = LRPModel(model=model)

    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        # y = y.to(device)  # here not used as method is unsupervised.

        t0 = time.time()
        r = lrp_model.forward(x)
        print("{time:.2f} FPS".format(time=(1.0 / (time.time()-t0))))

        plot_relevance_scores(x=x, r=r, name=str(i), config=config)


if __name__ == "__main__":

    with open("config.yml", "r") as stream:
        config = yaml.safe_load(stream)

    run_lrp(config=config)
