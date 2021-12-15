"""Script holds data loader methods.
"""
import torch
import torchvision


def get_data_loader(config: dict) -> torch.utils.data.DataLoader:
    """Creates dataloader for networks from PyTorch's Model Zoo.

    Data loader uses mean and standard deviation for ImageNet.

    Args:
        config: Dictionary holding configuration parameters.

    Returns:
        Data loader object.

    """
    input_dir = config["input_dir"]
    batch_size = config["batch_size"]

    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = torchvision.datasets.ImageFolder(root=input_dir, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)

    return data_loader
