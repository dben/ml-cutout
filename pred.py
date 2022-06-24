from typing import Tuple, Any

import cv2
import numpy as np
import torch

from networks.transforms import trimap_transform, normalise_image


def np_to_torch(x, permute=True):
    """
    Convert numpy array to torch tensor.
    :param x: numpy array
    :param permute: if True, transpose the array to (channels, height, width)
    :return: torch tensor
    """
    if permute:
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()
    else:
        return torch.from_numpy(x)[None, :, :, :].float().cuda()


def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    """
    Scales inputs to multiple of 8.
    :param x: input image.
    :param scale: scale factor.
    :param scale_type: interpolation type.
    :return: scaled image.
    """
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> tuple[Any, Any, Any]:
    """
    Predict alpha, foreground and background.
    :param image_np: the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
    :param trimap_np: two channel trimap, first background then foreground. Dimensions: (h, w, 2)
    :param model: the model to use.
    :return:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    """
    h, w = trimap_np.shape[:2]
    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():
        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(
            trimap_transform(trimap_scale_np), permute=False)
        image_transformed_torch = normalise_image(
            image_torch.clone())

        output = model(
            image_torch,
            trimap_torch,
            image_transformed_torch,
            trimap_transformed_torch)
        output = cv2.resize(
            output[0].cpu().numpy().transpose(
                (1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)

    alpha = output[:, :, 0]
    fg = output[:, :, 1:4]
    bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]

    return fg, bg, alpha
