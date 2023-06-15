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

from r2d2.camera_utils.wrappers.recorded_multi_camera_wrapper import RecordedMultiCameraWrapper
from r2d2.trajectory_utils.trajectory_reader import TrajectoryReader

def convert_dataset(path, args):
    recording_folderpath = os.path.join(os.path.dirname(path), "recordings", "MP4")
    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(args.imsize, args.imsize), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(args.imsize, args.imsize), resize_func="cv2"),
    )
    camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    output_path = os.path.join(os.path.dirname(path), "trajectory_im{}.h5".format(args.imsize))
    if os.path.exists(output_path):
        # dataset already exists, skip
        f = h5py.File(output_path)
        if "observation/camera/image/hand_camera_image" in f.keys():
            return
        f.close()

    shutil.copyfile(path, output_path)
    f = h5py.File(output_path, "a")

    demo_len = f["action"]["cartesian_position"].shape[0]

    if "camera" not in f["observation"]:
        f["observation"].create_group("camera").create_group("image")
    image_grp = f["observation/camera/image"]

    cam_data = dict(
        hand_camera_image=[],
        varied_camera_left_image=[],
        varied_camera_right_image=[],
    )

    camera_type_dict = {
        '17225336': 'hand_camera',
        '24013089': 'varied_camera',
        '25047636': 'varied_camera'
    }
    CAM_NAME_TO_KEY_MAPPING = {
        "hand_camera_image": "17225336_left",
        "varied_camera_left_image": "25047636_right",
        "varied_camera_right_image": "24013089_left"
    }

    traj_reader = TrajectoryReader(path, read_images=False)

    for index in range(demo_len):
        
        timestep = traj_reader.read_timestep(index=index)
        timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
        
        timestamp_dict = {}
        camera_obs = camera_reader.read_cameras(
            index=index, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
        )
        for cam_name in ["hand_camera_image", "varied_camera_left_image", "varied_camera_right_image"]:
            if camera_obs is None:
                im = np.zeros((args.imsize, args.imsize, 3))
            else:
                im_key = CAM_NAME_TO_KEY_MAPPING[cam_name]
                im = camera_obs["image"][im_key]

            # perform bgr_to_rgb operation
            im = im[:,:,::-1]
            
            cam_data[cam_name].append(im)

    for cam_name in cam_data.keys():
        cam_data[cam_name] = np.array(cam_data[cam_name]).astype(np.uint8)
        if cam_name in image_grp:
            del image_grp[cam_name]
        image_grp.create_dataset(cam_name, data=cam_data[cam_name], compression="gzip")

    f.close()

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
        default=84,
        help="image size (w and h)",
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
