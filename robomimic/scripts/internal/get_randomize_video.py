"""
Short helper script to get a video showing crop randomization.
"""
import numpy as np
import imageio
import torch
from PIL import Image
from tqdm import tqdm

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.vis_utils as VisUtils

IMAGE_PATH = "./test.png"
VIDEO_PATH = "./testttt.mp4"
NUM_CROPS = 25

if __name__ == "__main__":
    # load image into numpy array
    im = Image.open(IMAGE_PATH)
    source_im = np.asarray(im)[..., :3]

    # numpy array to torch Tensor ready for network input
    im = ObsUtils.process_image(source_im)
    im = TensorUtils.to_tensor(im)
    im = TensorUtils.to_float(im)

    # get random crops (90% of image size) and write to video
    crop_h = round(im.shape[-2] * 0.9)
    crop_w = round(im.shape[-1] * 0.9)

    video_writer = imageio.get_writer(VIDEO_PATH, fps=5)
    for _ in tqdm(range(NUM_CROPS)):
        _, crop_inds = ObsUtils.sample_random_image_crops(
            images=im,
            crop_height=crop_h, 
            crop_width=crop_w, 
            num_crops=1,
            pos_enc=False,
        )
        crop_h_start = crop_inds[0, 0]
        crop_w_start = crop_inds[0, 1]
        mask = 0.2 * np.ones_like(source_im)
        mask[crop_h_start : crop_h_start + crop_h, crop_w_start : crop_w_start + crop_w] = 1.
        masked_image = (mask * source_im).astype(np.uint8)
        video_writer.append_data(masked_image)

    video_writer.close()
