"""Script holds data loader methods.
"""
import argparse

import torch
import torchvision


def get_data_loader(config: argparse.Namespace) -> torch.utils.data.DataLoader:
    """Creates dataloader for networks from PyTorch's Model Zoo.

    Data loader uses mean and standard deviation for ImageNet.

    Args:
        config: Argparse namespace object.

    Returns:
        Data loader object.

    """
    input_dir = config.input_dir
    batch_size = config.batch_size

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms = []

    if config.resize:

        transforms += [
            torchvision.transforms.Resize(size=int(1.1 * config.resize)),
            torchvision.transforms.CenterCrop(size=config.resize),
        ]

    transforms += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]

    transform = torchvision.transforms.Compose(transforms=transforms)
    dataset = torchvision.datasets.ImageFolder(root=input_dir, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return data_loader
