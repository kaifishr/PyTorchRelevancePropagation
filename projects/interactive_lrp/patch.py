from typing import Callable
import argparse
import cv2
import numpy


def mean_patch_(img: numpy.ndarray, x: int, y: int, size: int) -> None:
    """Inplace mean patch to location arround (x, y).

    Args:
        img: Input image.
        x: x pixel coordinate.
        y: y pixel coordinate.
        size: Patch size.
    """
    h, w, _ = img.shape
    x_ = slice(max(0, x - size // 2), min(w, x + size // 2))
    y_ = slice(max(0, y - size // 2), min(h, y + size // 2))
    img[y_, x_, :] = numpy.mean(img[y_, x_, :], axis=(0, 1), keepdims=True)


def noise_patch_(img: numpy.ndarray, x: int, y: int, size: int) -> None:
    """Inplace noise patch to location arround (x, y).

    Args:
        img: Input image.
        x: x pixel coordinate.
        y: y pixel coordinate.
        size: Patch size.
    """
    h, w, c = img.shape
    x_ = slice(max(0, x - size // 2), min(w, x + size // 2))
    y_ = slice(max(0, y - size // 2), min(h, y + size // 2))
    size = (y_.stop - y_.start, x_.stop - x_.start, c)
    img[y_, x_, :] = numpy.random.uniform(0, 255, size=size)


def black_patch_(img: numpy.ndarray, x: int, y: int, size: int) -> None:
    """Inplace black patch to location arround (x, y).

    Args:
        img: Input image.
        x: x pixel coordinate.
        y: y pixel coordinate.
        size: Patch size.
    """
    r_0 = (x - size // 2, y - size // 2)
    r_1 = (x + size // 2, y + size // 2)
    cv2.rectangle(img, r_0, r_1, color=(0, 0, 0), thickness=-1)


def white_patch_(img: numpy.ndarray, x: int, y: int, size: int) -> None:
    """Inplace white patch to location arround (x, y).

    Args:
        img: Input image.
        x: x pixel coordinate.
        y: y pixel coordinate.
        size: Patch size.
    """
    r_0 = (x - size // 2, y - size // 2)
    r_1 = (x + size // 2, y + size // 2)
    cv2.rectangle(img, r_0, r_1, color=(255, 255, 255), thickness=-1)


def install_patch(config: argparse.Namespace) -> Callable:
    """Installs patch type used for input image distortion.

    Args:
        config: Argparse namespace holding config.

    """
    if config.patch_type == "mean":
        return mean_patch_
    elif config.patch_type == "noise":
        return noise_patch_
    elif config.patch_type == "black":
        return black_patch_
    elif config.patch_type == "white":
        return white_patch_
