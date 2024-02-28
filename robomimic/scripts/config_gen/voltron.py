from robomimic.scripts.config_gen.helper import *
import random
import json

# fulldataset = [{"path": p} for p in scan_datasets("/mnt/fsx/surajnair/datasets/r2d2_eval/", postfix="trajectory_im128.h5")]
# import pdb; pdb.set_trace()
# with open("/mnt/fsx/surajnair/datasets/r2d2_full_raw/manifest.json", "r") as f: broaddataset = json.load(f)
# # with open("/mnt/fsx/surajnair/datasets/r2d2_eval/manifest.json", "r") as f: evaldataset = json.load(f)
# evaldataset = [{"path": p} for p in scan_datasets("/mnt/fsx/surajnair/datasets/r2d2_eval/TRI_2/", postfix="trajectory_im128.h5")]
# # with open("/mnt/fsx/surajnair/datasets/r2d2_full_raw/manifest.json", "w") as f: json.dump(fulldataset, f)
# # evaldataset = evaldataset[:200]
# # broaddataset = broaddataset[:2000]
# random.shuffle(evaldataset)
# random.shuffle(broaddataset)
# N_EVAL = len(evaldataset)
# N_BROAD = len(broaddataset)

## Getting all language labeled data
with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_lang.json", 'r') as file:
        langs = json.load(file)

## Getting broad data
broaddataset = [{'path': l['path']} for l in langs if '/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads/' in l['path']]
N_BROAD = len(broaddataset)

## Getting multitask eval data
evaldataset_multitask = [{'path': l['path']} for l in langs if 'TRI_bowls' in l['path']]
N_EVAL_MULTI = len(evaldataset_multitask)

## Getting single task eval data (food in bowl)
evaldataset_foodinbowl = [{'path': l['path']} for l in langs if 'TRI_food_in_bowl' in l['path']]
N_EVAL_FOODINBOWL = len(evaldataset_foodinbowl)

## Getting single task eval data (open microwave)
evaldataset_openmicro = [{'path': l['path']} for l in langs if 'TRI_microwave_open' in l['path']]
N_EVAL_OPENMICRO = len(evaldataset_openmicro)

## Getting single task eval data (close microwave)
evaldataset_closemicro = [{'path': l['path']} for l in langs if 'TRI_microwave_close' in l['path']]
N_EVAL_CLOSEMICRO = len(evaldataset_closemicro)

print(f"Broad Datapoints: {N_BROAD} \
      \nMultitask Eval Datapoints: {N_EVAL_MULTI} \
      \nSingle Task Food in Bowl Datapoints: {N_EVAL_FOODINBOWL} \
      \nSingle Task Microwave Open Datapoints: {N_EVAL_OPENMICRO} \
      \nSingle Task Microwave Close Datapoints {N_EVAL_CLOSEMICRO}")


def make_generator_helper(args):
    algo_name_short = "diffusion_policy"

    generator = get_generator(
        algo_name="diffusion_policy",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/diffusion_policy.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[8],
    )

    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[100000],
    )

    generator.add_param(
        key="train.batch_size",
        name="bs",
        group=-1,
        values=[32],
    )

    # use ddim by default
    generator.add_param(
        key="algo.ddim.enabled",
        name="ddim",
        group=1001,
        values=[
            True,
            # False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="algo.ddpm.enabled",
        name="ddpm",
        group=1001,
        values=[
            False,
            # True,
        ],
        hidename=True,
    )

    if args.env == "r2d2":
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
                # ["camera/image/hand_camera_left_image"],
                # ["camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image"],
                ["camera/image/hand_camera_left_image", "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image"],
                # [
                #     "camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image",
                #     "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_1_right_image",
                #     "camera/image/varied_camera_2_left_image", "camera/image/varied_camera_2_right_image",
                # ],
            ],
            value_names=[
                # "wrist",
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
                None, # Change this to "geom" to do goal conditioning
                # "geom"
            ]
        )

        generator.add_param(
            key="train.truncated_geom_factor",
            name="truncated_geom_factor",
            group=5555,
            values = [
                0.3,
                # 0.5
            ]
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_class",
            name="obsrand",
            group=138,
            values=[
                "CropRandomizer", # crop only
                # "ColorRandomizer", # jitter only
                # ["ColorRandomizer", "CropRandomizer"], # jitter, followed by crop
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=-1,
            values=[
                {"crop_height": 224, "crop_width": 224, "num_crops": 1, "pos_enc": False}, # crop only
                # {}, # jitter only
                # [{}, {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False}], # jitter, followed by crop
                # [{}, {"crop_height": 224, "crop_width": 224, "num_crops": 1, "pos_enc": False}], # jitter, followed by crop
            ],
            hidename=True,
        )

        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="ldkeys",
            group=24986,
            values=[
                # ["robot_state/cartesian_position", "robot_state/gripper_position"],
                # [
                #     "robot_state/cartesian_position", "robot_state/gripper_position",
                #     "camera/extrinsics/hand_camera_left", "camera/intrinsics/hand_camera_left", # "camera/extrinsics/hand_camera_left_gripper_offset", 
                #     "camera/extrinsics/hand_camera_right", "camera/intrinsics/hand_camera_right", # "camera/extrinsics/hand_camera_right_gripper_offset",
                #     "camera/extrinsics/varied_camera_1_left", "camera/intrinsics/varied_camera_1_left",
                #     "camera/extrinsics/varied_camera_1_right", "camera/intrinsics/varied_camera_1_right",
                #     "camera/extrinsics/varied_camera_2_left", "camera/intrinsics/varied_camera_2_left",
                #     "camera/extrinsics/varied_camera_2_right", "camera/intrinsics/varied_camera_2_right",
                # ],
                [
                    "robot_state/cartesian_position", "robot_state/gripper_position",
                    "camera/extrinsics/hand_camera_left", "camera/intrinsics/hand_camera_left", # "camera/extrinsics/hand_camera_left_gripper_offset", 
                    "camera/extrinsics/hand_camera_right", "camera/intrinsics/hand_camera_right", # "camera/extrinsics/hand_camera_right_gripper_offset",
                    "camera/extrinsics/varied_camera_1_left", "camera/intrinsics/varied_camera_1_left",
                    "camera/extrinsics/varied_camera_1_right", "camera/intrinsics/varied_camera_1_right",
                    "camera/extrinsics/varied_camera_2_left", "camera/intrinsics/varied_camera_2_left",
                    "camera/extrinsics/varied_camera_2_right", "camera/intrinsics/varied_camera_2_right",
                    "lang_fixed/language_distilbert"
                ],
                [
                    "robot_state/cartesian_position", "robot_state/gripper_position",
                    "camera/extrinsics/hand_camera_left", "camera/intrinsics/hand_camera_left", # "camera/extrinsics/hand_camera_left_gripper_offset", 
                    "camera/extrinsics/hand_camera_right", "camera/intrinsics/hand_camera_right", # "camera/extrinsics/hand_camera_right_gripper_offset",
                    "camera/extrinsics/varied_camera_1_left", "camera/intrinsics/varied_camera_1_left",
                    "camera/extrinsics/varied_camera_1_right", "camera/intrinsics/varied_camera_1_right",
                    "camera/extrinsics/varied_camera_2_left", "camera/intrinsics/varied_camera_2_left",
                    "camera/extrinsics/varied_camera_2_right", "camera/intrinsics/varied_camera_2_right",
                    "lang_fixed/language_distilbert"
                ],
                [
                    "robot_state/cartesian_position", "robot_state/gripper_position",
                    "camera/extrinsics/hand_camera_left", "camera/intrinsics/hand_camera_left", # "camera/extrinsics/hand_camera_left_gripper_offset", 
                    "camera/extrinsics/hand_camera_right", "camera/intrinsics/hand_camera_right", # "camera/extrinsics/hand_camera_right_gripper_offset",
                    "camera/extrinsics/varied_camera_1_left", "camera/intrinsics/varied_camera_1_left",
                    "camera/extrinsics/varied_camera_1_right", "camera/intrinsics/varied_camera_1_right",
                    "camera/extrinsics/varied_camera_2_left", "camera/intrinsics/varied_camera_2_left",
                    "camera/extrinsics/varied_camera_2_right", "camera/intrinsics/varied_camera_2_right",
                    "lang_fixed/language_raw"
                ],
                [
                    "robot_state/cartesian_position", "robot_state/gripper_position",
                    "camera/extrinsics/hand_camera_left", "camera/intrinsics/hand_camera_left", # "camera/extrinsics/hand_camera_left_gripper_offset", 
                    "camera/extrinsics/hand_camera_right", "camera/intrinsics/hand_camera_right", # "camera/extrinsics/hand_camera_right_gripper_offset",
                    "camera/extrinsics/varied_camera_1_left", "camera/intrinsics/varied_camera_1_left",
                    "camera/extrinsics/varied_camera_1_right", "camera/intrinsics/varied_camera_1_right",
                    "camera/extrinsics/varied_camera_2_left", "camera/intrinsics/varied_camera_2_left",
                    "camera/extrinsics/varied_camera_2_right", "camera/intrinsics/varied_camera_2_right",
                    "lang_fixed/language_distilbert"
                ],
                [
                    "robot_state/cartesian_position", "robot_state/gripper_position",
                    "camera/extrinsics/hand_camera_left", "camera/intrinsics/hand_camera_left", # "camera/extrinsics/hand_camera_left_gripper_offset", 
                    "camera/extrinsics/hand_camera_right", "camera/intrinsics/hand_camera_right", # "camera/extrinsics/hand_camera_right_gripper_offset",
                    "camera/extrinsics/varied_camera_1_left", "camera/intrinsics/varied_camera_1_left",
                    "camera/extrinsics/varied_camera_1_right", "camera/intrinsics/varied_camera_1_right",
                    "camera/extrinsics/varied_camera_2_left", "camera/intrinsics/varied_camera_2_left",
                    "camera/extrinsics/varied_camera_2_right", "camera/intrinsics/varied_camera_2_right",
                    "lang_fixed/language_distilbert"
                ],
            ],
            value_names=[
                # "proprio",
                "proprio-cam-lang-bert2",
                "proprio-cam-lang-bert",
                "proprio-cam-lang-raw",
                "proprio-cam-lang-bert3",
                "proprio-cam-lang-bert4",
            ],
            hidename=False,
        )
        ## All of the following are needed for DeFiNe style encoding
        generator.add_param(
            key="observation.encoder.rgb.input_maps",
            name="",
            group=24986,
            values=[
                {
                    "camera/image/hand_camera_left_image": {
                        "image": "camera/image/hand_camera_left_image",
                    },
                    "camera/image/varied_camera_1_left_image": {
                        "image": "camera/image/varied_camera_1_left_image",
                    },
                    "camera/image/varied_camera_2_left_image": {
                        "image": "camera/image/varied_camera_2_left_image",
                    },
                },
                {
                    "camera/image/hand_camera_left_image": {
                        "image": "camera/image/hand_camera_left_image",
                    },
                    "camera/image/varied_camera_1_left_image": {
                        "image": "camera/image/varied_camera_1_left_image",
                    },
                    "camera/image/varied_camera_2_left_image": {
                        "image": "camera/image/varied_camera_2_left_image",
                    },
                },
                {
                    "camera/image/hand_camera_left_image": {
                        "image": "camera/image/hand_camera_left_image",
                        "lang_emb" : "lang_fixed/language_raw"
                    },
                    "camera/image/varied_camera_1_left_image": {
                        "image": "camera/image/varied_camera_1_left_image",
                        "lang_emb" : "lang_fixed/language_raw"
                    },
                    "camera/image/varied_camera_2_left_image": {
                        "image": "camera/image/varied_camera_2_left_image",
                        "lang_emb" : "lang_fixed/language_raw"
                    },
                },
                {
                    "camera/image/hand_camera_left_image": {
                        "image": "camera/image/hand_camera_left_image",
                        "intrinsics": "camera/intrinsics/hand_camera_left",
                        "extrinsics": "camera/extrinsics/hand_camera_left",
                        "lang_emb" : "lang_fixed/language_distilbert"
                    },
                    "camera/image/varied_camera_1_left_image": {
                        "image": "camera/image/varied_camera_1_left_image",
                        "intrinsics": "camera/intrinsics/varied_camera_1_left",
                        "extrinsics": "camera/extrinsics/varied_camera_1_left",
                        "lang_emb" : "lang_fixed/language_distilbert"
                    },
                    "camera/image/varied_camera_2_left_image": {
                        "image": "camera/image/varied_camera_2_left_image",
                        "intrinsics": "camera/intrinsics/varied_camera_2_left",
                        "extrinsics": "camera/extrinsics/varied_camera_2_left",
                        "lang_emb" : "lang_fixed/language_distilbert"
                    },
                },
                {
                    "camera/image/hand_camera_left_image": {
                        "image": "camera/image/hand_camera_left_image",
                        "intrinsics": "camera/intrinsics/hand_camera_left",
                        "extrinsics": "camera/extrinsics/hand_camera_left",
                    },
                    "camera/image/varied_camera_1_left_image": {
                        "image": "camera/image/varied_camera_1_left_image",
                        "intrinsics": "camera/intrinsics/varied_camera_1_left",
                        "extrinsics": "camera/extrinsics/varied_camera_1_left",
                    },
                    "camera/image/varied_camera_2_left_image": {
                        "image": "camera/image/varied_camera_2_left_image",
                        "intrinsics": "camera/intrinsics/varied_camera_2_left",
                        "extrinsics": "camera/extrinsics/varied_camera_2_left",
                    },
                },
            ],
            value_names=[
                "r3m-no-lang",
                "vc1-no-lang",
                "voltron-lang",
                "define-film-lang",
                "define-nofilm-lang",
            ],
            hidename=False,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_class",
            name="visenc",
            group=24986,
            values=["PretrainedVisualCore", "PretrainedVisualCore", "PretrainedVisualCore", "DeFiNeVisualCore", "DeFiNeVisualCore"],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="enc",
            group=24986,
            values=["R3MImageCamEncoder", "VC1ImageCamEncoder", "VoltronImageCamEncoder", "DeFiNeImageCamEncoder", "DeFiNeImageCamEncoder"],
            hidename=False,
        )


        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_kwargs.use_cam",
            name="",
            group=2498,
            values=[
                # False,
                True,
            ],
            hidename=True,
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
            key="observation.encoder.rgb.core_kwargs.feature_dimension",
            name="visdim",
            group=1234,
            values=[
                512,
                # None,
                # None
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.flatten",
            name="flatten",
            group=1234,
            values=[
                True,
                # False,
                # False
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.fuser",
            name="fuser",
            group=1234,
            values=[
                None,
                # "transformer",
                # "perceiver"
            ],
            hidename=False,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_kwargs.downsample",
            name="",
            group=1234,
            values=[
                True,
                # True,
                # False,
            ],
            hidename=False,
        )

        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                broaddataset,
                # evaldataset_multitask,
                # evaldataset_foodinbowl,
                # evaldataset_openmicro,
                # evaldataset_closemicro,
                # [{"path": p["path"], "weight": (0.5 / N_EVAL_MULTI if "eval" in p["path"] else 0.5 / N_BROAD)} for p in broaddataset + evaldataset_multitask],
                # [{"path": p["path"], "weight": (0.5 / N_EVAL_FOODINBOWL if "eval" in p["path"] else 0.5 / N_BROAD)} for p in broaddataset + evaldataset_foodinbowl],
                # [{"path": p["path"], "weight": (0.5 / N_EVAL_OPENMICRO if "eval" in p["path"] else 0.5 / N_BROAD)} for p in broaddataset + evaldataset_openmicro],
                # [{"path": p["path"], "weight": (0.5 / N_EVAL_CLOSEMICRO if "eval" in p["path"] else 0.5 / N_BROAD)} for p in broaddataset + evaldataset_closemicro],
                ],
	    value_names=[
                "broad",
                # "eval_multi", 
                # "eval_foodinbowl", 
                # "eval_openmicro",
                # "eval_closemicro",
                # "balanced_broad_eval_multi", 
                # "balanced_broad_eval_foodinbowl", 
                # "balanced_broad_eval_openmicro",
                # "balanced_broad_eval_closemicro",
            ],
        )


        # generator.add_param(
        #     key="observation.encoder.rgb.core_kwargs.backbone_class",
        #     name="backbone",
        #     group=1234,
        #     values=[
        #         # "ResNet18Conv",
        #         "ResNet50Conv",
        #     ],
        # )
        # generator.add_param(
        #     key="observation.encoder.rgb.core_kwargs.feature_dimension",
        #     name="visdim",
        #     group=1234,
        #     values=[
        #         # 64,
        #         512,
        #     ],
        # )

    elif args.env == "kitchen":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                # [{"path": "~/datasets/kitchen/prior/human_demos/pnp_table_to_cab/bowls/20230816_im84.hdf5", "filter_key": "100_demos"}],
                [{"path": "~/datasets/kitchen/prior/human_demos/pnp_table_to_cab/all/20230806_im84.hdf5", "filter_key": "100_demos"}],
                # [{"path": "~/datasets/kitchen/prior/mimicgen/pnp_table_to_cab/viraj_mg_2023-08-10-20-31-14/demo_im84.hdf5", "filter_key": "100_demos"}],
                # [{"path": "~/datasets/kitchen/prior/mimicgen/pnp_table_to_cab/viraj_mg_2023-08-10-20-31-14/demo_im84.hdf5", "filter_key": "1000_demos"}],
            ],
            value_names=[
                # "bowls-human-100",
                "human-100",
                # "mg-100",
                # "mg-1000",
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
                    "action_dict/base_mode",
                    # "actions",
                ],
            ],
            value_names=[
                "abs",
            ],
            hidename=True,
        )
    elif args.env == "square":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [
                    {"path": "~/datasets/square/ph/square_ph_abs_tmp.hdf5"}, # replace with your own path
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
            "/mnt/fsx/surajnair/expdata/{env}/{mod}/{algo_name_short}".format(
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