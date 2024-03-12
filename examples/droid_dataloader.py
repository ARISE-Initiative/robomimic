import tqdm
import torch
from torch.utils.data import DataLoader
import tensorflow as tf

from robomimic.utils.rlds_utils import droid_dataset_transform, robomimic_transform, DROID_TO_RLDS_OBS_KEY_MAP, DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP, TorchRLDSDataset
import robomimic.utils.action_utils as ActionUtils
from robomimic.utils.dataset import action_stats_to_normalization_stats

from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import combine_dataset_statistics

tf.config.set_visible_devices([], "GPU")

# ------------------------------ Get Dataset Information ------------------------------
DATA_PATH = "/mnt/fsx/ashwinbalakrishna/datasets/rlds_r2d2"
DATASET_NAMES = ["r2_d2", "r2_d2_cmu_toaster"]
sample_weights = [1, 1]

# ------------------------------ Get Observation Information ------------------------------
obs_modalities = ["camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image"]
obs_low_dim_modalities = ["robot_state/cartesian_position", "robot_state/gripper_position"]

# ------------------------------ Get Action Information ------------------------------
action_keys = [
    "action/abs_pos",
    "action/abs_rot_6d",
    "action/gripper_position"]
action_shapes=[
    (1, 3), 
    (1, 6), 
    (1, 1)]

ac_dim = sum([ac_comp[1] for ac_comp in action_shapes])

action_config = {
    "action/cartesian_position":{
        "normalization": "min_max",
    },
    "action/abs_pos":{
        "normalization": "min_max",
    },
    "action/abs_rot_6d":{
        "normalization": "min_max",
        "format": "rot_6d",
        "convert_at_runtime": "rot_euler",
    },
    "action/abs_rot_euler":{
        "normalization": "min_max",
        "format": "rot_euler",
    },
    "action/gripper_position":{
        "normalization": "min_max",
    },
    "action/cartesian_velocity":{
        "normalization": None,
    },
    "action/rel_pos":{
        "normalization": None,
    },
    "action/rel_rot_6d":{
        "format": "rot_6d",
        "normalization": None,
        "convert_at_runtime": "rot_euler",
    },
    "action/rel_rot_euler":{
        "format": "rot_euler",
        "normalization": None,
    },
    "action/gripper_velocity":{
        "normalization": None,
    },
}

is_abs_action = [action_config[k]["normalization"] != None for k in action_config.keys()]

# ------------------------------ Construct Dataset ------------------------------
BASE_DATASET_KWARGS = {
        "data_dir": DATA_PATH,
        "image_obs_keys": {"primary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[0]], "secondary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[1]]},
        "state_obs_keys": [DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP[obs_key] for obs_key in obs_low_dim_modalities],
        "language_key": "language_instruction",
        "norm_skip_keys":  ["proprio"],
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": is_abs_action,
        "action_normalization_mask": is_abs_action,
        "standardize_fn": droid_dataset_transform,
    }

# you can add more datasets here & the sampling weights below if you want to mix
dataset_kwargs_list = [
    {"name": d_name,  **BASE_DATASET_KWARGS} for d_name in DATASET_NAMES
]
# Compute combined normalization stats
combined_dataset_statistics = combine_dataset_statistics(
    [make_dataset_from_rlds(**dataset_kwargs, train=True)[1] for dataset_kwargs in dataset_kwargs_list]
)

dataset = make_interleaved_dataset(
    dataset_kwargs_list,
    sample_weights,
    train=True,
    shuffle_buffer_size=100000,
    batch_size=None,  # batching will be handled in PyTorch Dataloader object
    balance_weights=False,
    dataset_statistics=combined_dataset_statistics,
    traj_transform_kwargs=dict(
        window_size=2,
        future_action_window_size=15,
        subsample_length=100,
        skip_unlabeled=True,    # skip all trajectories without language
    ),
    frame_transform_kwargs=dict(
        image_augment_kwargs=dict(
        ),
        resize_size=dict(
            primary=[128, 128],
            secondary=[128, 128],
        ),
        num_parallel_calls=200,
    ),
    traj_transform_threads=48,
    traj_read_threads=48,
)

rlds_dataset_stats = dataset.dataset_statistics
action_stats = ActionUtils.get_action_stats_dict(rlds_dataset_stats["action"], action_keys, action_shapes)
action_normalization_stats = action_stats_to_normalization_stats(action_stats, action_config)
dataset = dataset.map(robomimic_transform, num_parallel_calls=48)

# ------------------------------ Create Dataloader ------------------------------

pytorch_dataset = TorchRLDSDataset(dataset)
train_loader = DataLoader(
    pytorch_dataset,
    batch_size=128,
    num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
)

for i, sample in tqdm.tqdm(enumerate(train_loader)):
    if i == 5000:
        break