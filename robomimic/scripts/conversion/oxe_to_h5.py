from absl import app, flags, logging
import dlimp as dl
import h5py
import os
import tqdm
import tensorflow as tf
from functools import partial
from typing import Callable
import torch

from octo.data.oxe import make_oxe_dataset_kwargs
from octo.data.dataset import make_dataset_from_rlds, apply_frame_transforms
from octo.data import obs_transforms
from octo.data.utils.data_utils import NormalizationType

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16)
model.to('cuda')


FLAGS = flags.FLAGS
flags.DEFINE_string("out_dir", None, "Path of output directory.")
flags.DEFINE_string("dataset", None, "Name of dataset.")

DATA_DIR = "/mnt/fsx/surajnair/datasets/openx_processed"


import tensorflow_graphics.geometry.transformation as tfg

def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)

def quat_to_rmat(quat):
    return tfg.rotation_matrix_3d.from_quaternion(quat)

def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat

def main(_):

    if FLAGS.dataset == "bridge_orig":
        ds = "bridge_dataset"
    # get dataset specific config
    dataset_kwargs = make_oxe_dataset_kwargs(
        FLAGS.dataset,
        DATA_DIR,
        load_camera_views=("primary", "secondary", "wrist"),
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=NormalizationType.BOUNDS,
    )

    # create RLDS dataset
    ds, _ = make_dataset_from_rlds(
        **dataset_kwargs,
        train=True,
    )

    def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
        # observation is chunked -- apply fn along first axis
        frame["observation"] = fn(frame["observation"])
        return frame
    
    def apply_6d_transform(frame: dict) -> dict:
        # observation is chunked -- apply fn along first axis
        if frame["observation"]["proprio"][6] == 0:
            euler_rot = frame["observation"]["proprio"][3:6][:]
            r6 = mat_to_rot6d(euler_to_rmat(euler_rot))
        else:
            quat_rot = frame["observation"]["proprio"][3:7][:]
            mat = quat_to_rmat(quat_rot)
            r6 = mat_to_rot6d(mat)
            euler_rot = tfg.euler.from_rotation_matrix(mat)

        frame["observation"]["proprio_rot_6d"] = r6
        frame["observation"]["proprio"] = tf.concat([frame["observation"]["proprio"][:3], euler_rot, tf.zeros_like(euler_rot)[:1], frame["observation"]["proprio"][7:]], axis=-1) 
        return frame


    # decode + resize images (and depth images)
    ds = ds.frame_map(
        partial(
            apply_obs_transform,
            partial(
                obs_transforms.decode_and_resize,
                resize_size={
                    "primary": (128, 128),
                    "secondary": (128, 128),
                    "wrist": (128, 128),
                },
                depth_resize_size=None,
            ),
        ),
        tf.data.AUTOTUNE,
    )

    ds = ds.frame_map(
        partial(
            apply_6d_transform,
        ),
        tf.data.AUTOTUNE,
    )


    # iterate over dataset and dump to h5:
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    print(f"Converting {FLAGS.dataset}...")
    for i, episode in enumerate(tqdm.tqdm(ds.iterator())):
        with h5py.File(os.path.join(FLAGS.out_dir, f"episode_{i}.h5"), "w") as F:
            F["observation/camera/image/varied_camera_1_left_image"] = episode["observation"]["image_primary"][:-1]
            F["observation/camera/image/varied_camera_2_left_image"] = episode["observation"]["image_secondary"][:-1]
            F["observation/camera/image/hand_camera_left_image"] = episode["observation"]["image_wrist"][:-1]

            F["action/abs_pos"] = episode["observation"]["proprio"][1:, :3]
            # F["action/abs_rot_euler"] = episode["observation"]["proprio"][1:, 3:6]
            F["action/gripper_position"] = episode["observation"]["proprio"][1:, 7:]
            F["action/abs_rot_6d"] = episode["observation"]["proprio_rot_6d"][1:, :]

            F["observation/robot_state/cartesian_position"] = episode["observation"]["proprio"][:-1, :6]
            F["observation/robot_state/gripper_position"] = episode["observation"]["proprio"][:-1, 7]

            H = F["observation/robot_state/gripper_position"].shape[0] 
            l = episode["task"]["language_instruction"][0:1].tolist()[0].decode("utf-8")
            encoded_input = tokenizer([l], return_tensors='pt').to('cuda')
            outputs = model(**encoded_input)
            encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(0).repeat(H, 1)
            F["observation/lang_fixed/language_raw"] = [l] * H
            F["observation/lang_fixed/language_distilbert"] = encoded_lang.cpu().detach().numpy()


if __name__ == "__main__":
    app.run(main)
