"""
This file contains utility functions for visualizing image observations in the training pipeline.
These functions can be a useful debugging tool.
"""
import numpy as np

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


def image_tensor_to_numpy(image):
    """
    Converts processed image tensors to numpy so that they can be saved to disk or video.
    A useful utility function for visualizing images in the middle of training.

    Args:
        image (torch.Tensor): images of shape [..., C, H, W]

    Returns:
        image (np.array): converted images of shape [..., H, W, C] and type uint8
    """
    return TensorUtils.to_numpy(
            ObsUtils.unprocess_image(image)
        ).astype(np.uint8)


def image_to_disk(image, fname):
    """
    Writes an image to disk.

    Args:
        image (np.array): image of shape [H, W, 3]
        fname (str): path to save image to
    """
    image = Image.fromarray(image)
    image.save(fname)


def image_tensor_to_disk(image, fname):
    """
    Writes an image tensor to disk. Any leading batch dimensions are indexed out
    with the first element.

    Args:
        image (torch.Tensor): image of shape [..., C, H, W]. All leading dimensions
            will be indexed out with the first element
        fname (str): path to save image to
    """
    # index out all leading dimensions before [C, H, W]
    num_leading_dims = len(image.shape[:-3])
    for _ in range(num_leading_dims):
        image = image[0]
    image = image_tensor_to_numpy(image)
    image_to_disk(image, fname)
