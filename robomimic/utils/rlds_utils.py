"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import tensorflow as tf
import torch
import tensorflow_graphics.geometry.transformation as tfg

def filter_success(trajectory: dict[str, any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    T = trajectory["action_dict"]["cartesian_position"][:, :3]
    R = mat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_position"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def robomimic_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "camera/image/varied_camera_1_left_image": 
                tf.cast(trajectory["observation"]["image_primary"], tf.float32) / 255.,
            "camera/image/varied_camera_2_left_image": 
                tf.cast(trajectory["observation"]["image_secondary"], tf.float32) / 255.,
            "raw_language": trajectory["task"]["language_instruction"],
            "robot_state/cartesian_position": trajectory["observation"]["proprio"][..., :6],
            "robot_state/gripper_position": trajectory["observation"]["proprio"][..., -1:],
            "pad_mask": trajectory["observation"]["pad_mask"][..., None],
        },
        "actions": trajectory["action"][1:],
    }

DROID_TO_RLDS_OBS_KEY_MAP = {
    "camera/image/varied_camera_1_left_image": "exterior_image_1_left",
    "camera/image/varied_camera_2_left_image": "exterior_image_2_left"
}

DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP = {
    "robot_state/cartesian_position": "cartesian_position",
    "robot_state/gripper_position": "gripper_position",
}

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)

