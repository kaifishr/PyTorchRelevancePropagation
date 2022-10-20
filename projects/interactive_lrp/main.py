"""Interactive Layer-wise Relevance Propagation."""
import argparse
import time

import cv2
import numpy as np 
import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms

from src.lrp import LRPModel

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# def preprocess(self, frame: numpy.ndarray) -> torch.Tensor:
#     """Preprocesses frame captured by webcam."""
#     return self.transform(frame).to(self.device)[None, ...]


def interactive_lrp(config):
    """Interactive LRP"""

    window_name = "Interactive LRP"
    img = cv2.imread("input/cats/cat_1.jpg")

    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)
    lrp_model = LRPModel(model=model, top_k=config.top_k)

    cv2.namedWindow(window_name) # , cv2.WINDOW_NORMAL)

    def draw(event, x, y, flags, param, size=32):
        if flags == 1:
            # Solid color
            # cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color=(0, 0, 0), thickness=-1)
            # Noise
            # img[y-size//2:y+size//2, x-size//2:x+size//2, :] = np.random.uniform(0, 255, size=(32, 32, 3))
            # Average
            img[y-size//2:y+size//2, x-size//2:x+size//2, :] = np.mean(img[y-size//2:y+size//2, x-size//2:x+size//2, :], axis=(0, 1), keepdims=True)

    cv2.setMouseCallback(window_name, draw)

    is_running = True

    while is_running:
        t0 = time.time()

        cv2.imshow(window_name, img)

        relevance_scores = lrp_model(transform(torch.tensor(img.transpose(2, 0, 1))).to(device)[None, ...])
        relevance_scores = relevance_scores.numpy()
        r_min, r_max = relevance_scores.min(), relevance_scores.max()
        relevance_scores = (relevance_scores - r_min) / (r_max - r_min)

        cv2.imshow("Relevance Scores", relevance_scores)

        if cv2.waitKey(1) & 0xFF == 27:
            is_running = False

        print("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))

    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image-path", dest="image_path", help="Path to image.", default="./input/cats/cat1.jpg")
    parser.add_argument("-d", "--device", dest="device", help="Device.", choices=["gpu", "cpu"], default="gpu", type=str)
    parser.add_argument("-k", "--top-k", dest="top_k", help="Proportion of relevance scores that are allowed to pass.", default=0.01, type=float)

    config = parser.parse_args()

    interactive_lrp(config=config)