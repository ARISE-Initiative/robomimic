from robomimic.scripts.config_gen.helper import *
import random
import json

# fulldataset = [{"path": p} for p in scan_datasets("/mnt/fsx/surajnair/datasets/r2d2_eval/", postfix="trajectory_im128.h5")]
# import pdb; pdb.set_trace()
# with open("/mnt/fsx/surajnair/datasets/r2d2_full_raw/manifest.json", "r") as f: broaddataset = json.load(f)
# with open("/mnt/fsx/surajnair/datasets/r2d2_eval/manifest.json", "r") as f: evaldataset = json.load(f)
evaldataset = [{"path": p} for p in scan_datasets("/mnt/fsx/surajnair/datasets/r2d2_eval/TRI_2/", postfix="trajectory_im128.h5")]
# with open("/mnt/fsx/surajnair/datasets/r2d2_full_raw/manifest.json", "w") as f: json.dump(fulldataset, f)
# evaldataset = evaldataset[:200]
# broaddataset = broaddataset[:2000]
random.shuffle(evaldataset)
# random.shuffle(broaddataset)
N_EVAL = len(evaldataset)
# N_BROAD = len(broaddataset)

def make_generator_helper(args):
    algo_name_short = "act"
    generator = get_generator(
        algo_name="act",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/act.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"


    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[1000],
    )

    generator.add_param(
        key="train.batch_size",
        name="",
        group=-1,
        values=[2048], # 4096 works for wrist
    )

    generator.add_param(
        key="train.max_grad_norm",
        name="",
        group=-1,
        values=[100.0],
    )

    if args.env == "r2d2":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                # broaddataset + evaldataset,
                # [{"path": p["path"], "weight": (0.5 / N_EVAL if "eval" in p["path"] else 0.5 / N_BROAD)} for p in broaddataset + evaldataset],
                # broaddataset,
                evaldataset,
                ],
	    value_names=[
                # "mixed_unbalanced",
                # "mixed_balanced",
                # "broad",
                "eval",
            ],
        )
        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action/abs_pos",
                    "action/abs_rot_6d",
                    "action/gripper_position",
                ],
            ],
            value_names=[
                "abs",
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.modalities.obs.rgb",
            name="cams",
            group=130,
            values=[
                ["camera/image/hand_camera_left_image"],
                # ["camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image"],
                ["camera/image/hand_camera_left_image", "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image"],
                # [
                #     "camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image",
                #     "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_1_right_image",
                #     "camera/image/varied_camera_2_left_image", "camera/image/varied_camera_2_right_image",
                # ],
            ],
            value_names=[
                "wrist",
                # "wrist-stereo",
                "3cams",
                # "3cams-stereo",
            ]
        )
        generator.add_param(
            key="train.goal_mode",
            name="goal_mode",
            group=5678,
            values = [
                # None, # Change this to "geom" to do goal conditioning
                "geom"
            ]
        )

        generator.add_param(
            key="train.truncated_geom_factor",
            name="truncated_geom_factor",
            group=5555,
            values = [
                # 0.5,
                0.3
            ]
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_class",
            name="obsrand",
            group=130,
            values=[
                # "CropRandomizer", # crop only
                # "ColorRandomizer", # jitter only
                ["ColorRandomizer", "CropRandomizer"], # jitter, followed by crop
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=-1,
            values=[
                # {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False}, # crop only
                # {}, # jitter only
                [{}, {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False}], # jitter, followed by crop
            ],
            hidename=True,
        )

        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="ldkeys",
            group=2498,
            values=[
                # ["robot_state/cartesian_position", "robot_state/gripper_position"],
                [
                    "robot_state/cartesian_position", "robot_state/gripper_position",
                    "camera/extrinsics/hand_camera_left", "camera/intrinsics/hand_camera_left", # "camera/extrinsics/hand_camera_left_gripper_offset", 
                    "camera/extrinsics/hand_camera_right", "camera/intrinsics/hand_camera_right", # "camera/extrinsics/hand_camera_right_gripper_offset",
                    "camera/extrinsics/varied_camera_1_left", "camera/intrinsics/varied_camera_1_left",
                    "camera/extrinsics/varied_camera_1_right", "camera/intrinsics/varied_camera_1_right",
                    "camera/extrinsics/varied_camera_2_left", "camera/intrinsics/varied_camera_2_left",
                    "camera/extrinsics/varied_camera_2_right", "camera/intrinsics/varied_camera_2_right",
                ]
            ],
            value_names=[
                # "proprio",
                "proprio-cam",
            ],
            hidename=False,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained",
            name="",
            group=2498,
            values=[
                # False,
                True,
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="backbone",
            group=1234,
            values=[
                # "ResNet18Conv",
                "ResNet50Conv",
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.feature_dimension",
            name="visdim",
            group=1234,
            values=[
                # 64,
                512,
            ],
        )
    elif args.env == "kitchen":
        raise NotImplementedError
    elif args.env == "square":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [
                    {"path": "TODO.hdf5"}, # replace with your own path
                ],
            ],
            value_names=[
                "square",
            ],
        )

        # update env config to use absolute action control
        generator.add_param(
            key="experiment.env_meta_update_dict",
            name="",
            group=-1,
            values=[
                {"env_kwargs": {"controller_configs": {"control_delta": False}}}
            ],
        )
        
        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action_dict/abs_pos",
                    "action_dict/abs_rot_6d",
                    "action_dict/gripper",
                    # "actions",
                ],
            ],
            value_names=[
                "abs",
            ],
        )


    else:
        raise ValueError
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
