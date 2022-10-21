"""Script with method for pre- and post-processing."""
import argparse
import cv2
import numpy

import torch
import torchvision.transforms


class DataProcessing:
    def __init__(self, config: argparse.Namespace, device: torch.device) -> None:
        """Initializes data processing class."""

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transforms = [
            torchvision.transforms.ToPILImage(),
        ]

        if config.resize:

            transforms += [
                torchvision.transforms.Resize(size=int(1.1 * config.resize)),
                torchvision.transforms.CenterCrop(size=config.resize),
            ]

        transforms += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]

        self.transform = torchvision.transforms.Compose(transforms=transforms)
        self.device = device

    def preprocess(self, frame: numpy.ndarray) -> torch.Tensor:
        """Preprocesses frame captured by webcam."""
        return self.transform(frame).to(self.device)[None, ...]

    def postprocess(self, relevance_scores: torch.Tensor):
        """Normalizes relevance scores and applies colormap."""
        relevance_scores = relevance_scores.numpy()
        relevance_scores = cv2.normalize(
            relevance_scores, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
        )
        relevance_scores = cv2.applyColorMap(relevance_scores, cv2.COLORMAP_HOT)
        return relevance_scores
