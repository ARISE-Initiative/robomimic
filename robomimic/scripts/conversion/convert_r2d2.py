"""
Add image information to existing r2d2 hdf5 file
"""
import h5py
import os
import numpy as np
import glob
from tqdm import tqdm
import argparse
import shutil
import torch

"""
Follow instructions here to setup zed:
https://www.stereolabs.com/docs/installation/linux/
"""
import pyzed.sl as sl

import robomimic.utils.torch_utils as TorchUtils

from r2d2.camera_utils.wrappers.recorded_multi_camera_wrapper import RecordedMultiCameraWrapper
from r2d2.trajectory_utils.trajectory_reader import TrajectoryReader
from r2d2.camera_utils.info import camera_type_to_string_dict

from r2d2.camera_utils.camera_readers.zed_camera import ZedCamera, standard_params

def get_cam_instrinsics(svo_path):
    """
    utility function to get camera intrinsics
    """
    intrinsics = {}

    return intrinsics

def convert_dataset(path, args):
    recording_folderpath = os.path.join(os.path.dirname(path), "recordings", "MP4")
    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(args.imsize, args.imsize), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(args.imsize, args.imsize), resize_func="cv2"),
    )
    camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    output_path = os.path.join(os.path.dirname(path), "trajectory_im{}.h5".format(args.imsize))
    # if os.path.exists(output_path):
    #     # dataset already exists, skip
    #     f = h5py.File(output_path)
    #     if "observation/camera/image/hand_camera_image" in f.keys():
    #         return
    #     f.close()

    shutil.copyfile(path, output_path)
    f = h5py.File(output_path, "a")

    demo_len = f["action"]["cartesian_position"].shape[0]

    if "camera" not in f["observation"]:
        f["observation"].create_group("camera").create_group("image")
    image_grp = f["observation/camera/image"]

    """
    Extract camera type and keys. Examples of what they should look like:
    camera_type_dict = {
        '17225336': 'hand_camera',
        '24013089': 'varied_camera',
        '25047636': 'varied_camera'
    }
    CAM_NAME_TO_KEY_MAPPING = {
        "hand_camera_left_image": "17225336_left",
        "hand_camera_right_image": "17225336_right",
        "varied_camera_1_left_image": "24013089_left",
        "varied_camera_1_right_image": "24013089_right",
        "varied_camera_2_left_image": "25047636_left",
        "varied_camera_2_right_image": "25047636_right",
    }
    """

    CAM_ID_TO_TYPE = {}
    hand_cam_ids = []
    varied_cam_ids = []
    for k in f["observation"]["camera_type"]:
        cam_type = camera_type_to_string_dict[f["observation"]["camera_type"][k][0]]
        CAM_ID_TO_TYPE[k] = cam_type
        if cam_type == "hand_camera":
            hand_cam_ids.append(k)
        elif cam_type == "varied_camera":
            varied_cam_ids.append(k)
        else:
            raise ValueError

    # sort the camera ids: important to maintain consistency of cams between train and eval!
    hand_cam_ids = sorted(hand_cam_ids)
    varied_cam_ids = sorted(varied_cam_ids)

    IMAGE_NAME_TO_CAM_KEY_MAPPING = {}
    IMAGE_NAME_TO_CAM_KEY_MAPPING["hand_camera_left_image"] = "{}_left".format(hand_cam_ids[0])
    IMAGE_NAME_TO_CAM_KEY_MAPPING["hand_camera_right_image"] = "{}_right".format(hand_cam_ids[0])
    
    # set up mapping for varied cameras
    for i in range(len(varied_cam_ids)):
        for side in ["left", "right"]:
            cam_name = "varied_camera_{}_{}_image".format(i+1, side)
            cam_key = "{}_{}".format(varied_cam_ids[i], side)
            IMAGE_NAME_TO_CAM_KEY_MAPPING[cam_name] = cam_key

    cam_data = {cam_name: [] for cam_name in IMAGE_NAME_TO_CAM_KEY_MAPPING.keys()}
    traj_reader = TrajectoryReader(path, read_images=False)

    for index in range(demo_len):
        
        timestep = traj_reader.read_timestep(index=index)
        timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
        
        timestamp_dict = {}
        camera_obs = camera_reader.read_cameras(
            index=index, camera_type_dict=CAM_ID_TO_TYPE, timestamp_dict=timestamp_dict
        )
        for cam_name in IMAGE_NAME_TO_CAM_KEY_MAPPING.keys():
            if camera_obs is None:
                im = np.zeros((args.imsize, args.imsize, 3))
            else:
                im_key = IMAGE_NAME_TO_CAM_KEY_MAPPING[cam_name]
                im = camera_obs["image"][im_key]

            # perform bgr_to_rgb operation
            im = im[:,:,::-1]
            
            cam_data[cam_name].append(im)

    for cam_name in cam_data.keys():
        cam_data[cam_name] = np.array(cam_data[cam_name]).astype(np.uint8)
        if cam_name in image_grp:
            del image_grp[cam_name]
        image_grp.create_dataset(cam_name, data=cam_data[cam_name], compression="gzip")

    # extract camera extrinsics data
    if "extrinsics" not in f["observation/camera"]:
        f["observation/camera"].create_group("extrinsics")
    extrinsics_grp = f["observation/camera/extrinsics"]    
    for raw_key in f["observation/camera_extrinsics"].keys():
        cam_key = "_".join(raw_key.split("_")[:2])
        # reverse search for image name
        im_name = None
        for (k, v) in IMAGE_NAME_TO_CAM_KEY_MAPPING.items():
            if v == cam_key:
                im_name = k
                break
        if im_name is None: # sometimes the raw_key doesn't correspond to any camera we have images for
            continue
        extr_name = "_".join(im_name.split("_")[:-2] + raw_key.split("_")[1:])
        data = f["observation/camera_extrinsics"][raw_key]
        extrinsics_grp.create_dataset(extr_name, data=data)
    
    svo_path = os.path.join(os.path.dirname(path), "recordings", "SVO")
    cam_reader_svo = RecordedMultiCameraWrapper(svo_path, camera_kwargs)
    if "intrinsics" not in f["observation/camera"]:
        f["observation/camera"].create_group("intrinsics")
    intrinsics_grp = f["observation/camera/intrinsics"]    
    for cam_id, svo_reader in cam_reader_svo.camera_dict.items():
        cam = svo_reader._cam
        calib_params = cam.get_camera_information().camera_configuration.calibration_parameters
        for (posftix, params)in zip(
            ["_left", "_right"],
            [calib_params.left_cam, calib_params.right_cam]
        ):
            # get name to store intrinsics under
            cam_key = cam_id + posftix
            # reverse search for image name
            im_name = None
            for (k, v) in IMAGE_NAME_TO_CAM_KEY_MAPPING.items():
                if v == cam_key:
                    im_name = k
                    break
            if im_name is None: # sometimes the raw_key doesn't correspond to any camera we have images for
                continue
            intr_name = "_".join(im_name.split("_")[:-1])

            if intr_name not in intrinsics_grp:
                intrinsics_grp.create_group(intr_name)
            cam_intr_grp = intrinsics_grp[intr_name]
            
            # these lines are copied from _process_intrinsics function in svo_reader.py
            cam_intrinsics = {
                "camera_matrix": np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]]),
                "dist_coeffs": np.array(list(params.disto)),
            }
            # batchify across trajectory
            for k in cam_intrinsics:
                data = np.repeat(cam_intrinsics[k][None], demo_len, axis=0)
                cam_intr_grp.create_dataset(k, data=data)

    # extract action key data
    action_dict_group = f["action"]
    for in_ac_key in ["cartesian_position", "cartesian_velocity"]:
        in_action = action_dict_group[in_ac_key][:]
        in_pos = in_action[:,:3].astype(np.float64)
        in_rot = in_action[:,3:6].astype(np.float64) # in euler format
        rot_ = torch.from_numpy(in_rot)
        rot_6d = TorchUtils.euler_angles_to_rot_6d(
            rot_, convention="XYZ",
        ) 
        rot_6d = rot_6d.numpy().astype(np.float64)

        if in_ac_key == "cartesian_position":
            prefix = "abs_"
        elif in_ac_key == "cartesian_velocity":
            prefix = "rel_"
        else:
            raise ValueError
        
        this_action_dict = {
            prefix + 'pos': in_pos,
            prefix + 'rot_euler': in_rot,
            prefix + 'rot_6d': rot_6d,
        }
        for key, data in this_action_dict.items():
            if key in action_dict_group:
                del action_dict_group[key]
            action_dict_group.create_dataset(key, data=data)

    # ensure all action keys are batched (ie., are not 0-dimensional)
    for k in action_dict_group:
        if isinstance(action_dict_group[k], h5py.Dataset) and len(action_dict_group[k].shape) == 1:
            reshaped_values = np.reshape(action_dict_group[k][:], (-1, 1))
            del action_dict_group[k]
            action_dict_group.create_dataset(k, data=reshaped_values)

    # post-processing: remove timesteps where robot movement is disabled
    movement_enabled = f["observation/controller_info/movement_enabled"][:]
    timesteps_to_remove = np.where(movement_enabled == False)[0]

    if not args.keep_idle_timesteps:
        remove_timesteps(f, timesteps_to_remove)

    f.close()

def remove_timesteps(f, timesteps_to_remove):
    total_timesteps = f["action/cartesian_position"].shape[0]
    
    def remove_timesteps_for_group(g):
        for k in g:
            if isinstance(g[k], h5py._hl.dataset.Dataset):
                if g[k].shape[0] != total_timesteps:
                    print("skipping {}".format(k))
                    continue
                new_dataset = np.delete(g[k], timesteps_to_remove, axis=0)
                del g[k]
                g.create_dataset(k, data=new_dataset)
            elif isinstance(g[k], h5py._hl.group.Group):
                remove_timesteps_for_group(g[k])
            else:
                raise NotImplementedError

    for k in f:
        remove_timesteps_for_group(f[k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        help="folder containing hdf5's to add camera images to",
        default="~/datasets/r2d2/success"
    )

    parser.add_argument(
        "--imsize",
        type=int,
        default=128,
        help="image size (w and h)",
    )

    parser.add_argument(
        "--keep_idle_timesteps",
        action="store_true",
        help="override the default behavior of truncating idle timesteps",
    )
    
    args = parser.parse_args()

    datasets = []
    for root, dirs, files in os.walk(os.path.expanduser(args.folder)):
        for f in files:
            if f == "trajectory.h5":
                datasets.append(os.path.join(root, f))

    print("converting datasets...")
    for d in tqdm(datasets):
        d = os.path.expanduser(d)
        convert_dataset(d, args)
