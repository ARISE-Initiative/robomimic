"""
This file contains utility functions for visualizing image observations in the training pipeline.
These functions can be a useful debugging tool.
"""
import numpy as np
import matplotlib.pyplot as plt

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


def visualize_image_randomizer(original_image, randomized_image, randomizer_name=None):
    """
    A function that visualizes the before and after of an image-based input randomizer
    Args:
        original_image: batch of original image shaped [B, H, W, 3]
        randomized_image: randomized image shaped [B, N, H, W, 3]. N is the number of randomization per input sample
        randomizer_name: (Optional) name of the randomizer
    Returns:
        None
    """

    B, N, H, W, C = randomized_image.shape

    # Create a grid of subplots with B rows and N+1 columns (1 for the original image, N for the randomized images)
    fig, axes = plt.subplots(B, N + 1, figsize=(4 * (N + 1), 4 * B))

    for i in range(B):
        # Display the original image in the first column of each row
        axes[i, 0].imshow(original_image[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Display the randomized images in the remaining columns of each row
        for j in range(N):
            axes[i, j + 1].imshow(randomized_image[i, j])
            axes[i, j + 1].axis("off")

    title = randomizer_name if randomizer_name is not None else "Randomized"
    fig.suptitle(title, fontsize=16)

    # Adjust the space between subplots for better visualization
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Show the entire grid of subplots
    plt.show()
