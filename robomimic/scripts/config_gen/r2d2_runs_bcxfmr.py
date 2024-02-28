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
broaddataset_full = [{'path': l['path']} for l in langs if '/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads/' in l['path']]
N_BROAD_FULL = len(broaddataset_full)

chips_singletask = [{'path': l['path']} for l in langs if 'TRI_chips_only_1_21' in l['path']]
N_CHIPS = len(chips_singletask)

## Getting all language labeled data
with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_lang_filtered.json", 'r') as file:
        langs = json.load(file)

## Getting broad data
broaddataset = [{'path': l['path']} for l in langs if '/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads/' in l['path']]
N_BROAD = len(broaddataset)

## Getting multitask eval data
# evaldataset_multitask = [{'path': l['path']} for l in langs if 'TRI_1_10_multitask_food_in_bowl' in l['path']]
# N_EVAL_MULTI = len(evaldataset_multitask)

# apple_singletask = [{'path': l['path']} for l in langs if 'TRI_1_10_multitask_food_in_bowl/apple' in l['path']]
# N_APPLE = len(apple_singletask)

## Getting single task eval data (food in bowl)
# evaldataset_foodinbowl = [{'path': l['path']} for l in langs if 'TRI_food_in_bowl' in l['path']]
# evaldataset_foodinbowl = evaldataset_foodinbowl[:20]
# N_EVAL_FOODINBOWL = len(evaldataset_foodinbowl)

## Getting single task eval data (open microwave)
# evaldataset_openmicro = [{'path': l['path']} for l in langs if 'TRI_microwave_open' in l['path']]
# N_EVAL_OPENMICRO = len(evaldataset_openmicro)

## Getting single task eval data (close microwave)
# evaldataset_closemicro = [{'path': l['path']} for l in langs if 'TRI_microwave_close' in l['path']]
# N_EVAL_CLOSEMICRO = len(evaldataset_closemicro)

print(f"Broad Datapoints: {N_BROAD, N_BROAD_FULL} \
      \nSingle Task CHIPS Datapoints: {N_CHIPS} ")
    #  \nMultitask Eval Datapoints: {N_EVAL_MULTI} \
    #   \nSingle Task APPLE Datapoints: {N_APPLE}")
    # #   \nSingle Task Microwave Close Datapoints {N_EVAL_CLOSEMICRO}")


def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc_transformer.json'),
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
        name="bz",
        group=1212111,
        values=[128],
        hidename=False,
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
                ["camera/image/hand_camera_left_image"],
                # ["camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image"],
                ["camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image"],
                # [
                    # "camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image",
                #     "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_1_right_image",
                #     "camera/image/varied_camera_2_left_image", "camera/image/varied_camera_2_right_image",
                # ],
            ],
            value_names=[
                "wrist",
                # "wrist-stereo",
                "2cams",
                # "3cams-stereo",
            ]
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_class",
            name="obsrand",
            group=130,
            values=[
                "ColorRandomizer", # jitter only
                ["ColorRandomizer", "CropRandomizer"], # jitter, followed by crop
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=130,
            values=[
                {}, # jitter only
                [{}, {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False}], # jitter, followed by crop
            ],
            hidename=True,
        )

        ### CONDITIONING
        generator.add_param(
            key="train.goal_mode",
            name="goal_mode",
            group=24986,
            values = [
                "geom",
                None, # Change this to "geom" to do goal conditioning

            ]
        )
        generator.add_param(
            key="train.truncated_geom_factor",
            name="truncated_geom_factor",
            group=5555,
            values = [
                0.3,
            ]
        )
        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="ldkeys",
            group=24986,
            values=[
                ["robot_state/cartesian_position", "robot_state/gripper_position"],
                [
                    "robot_state/cartesian_position", "robot_state/gripper_position",
                    "lang_fixed/language_distilbert"
                ]
            ],
            value_names=[
                # "proprio",
                "proprio",
                "proprio-lang",
            ],
            hidename=False,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_kwargs.use_cam",
            name="",
            group=2498,
            values=[
                False,
                # True,
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
            key="observation.encoder.rgb.core_class",
            name="visenc",
            group=-1,
            values=["VisualCore"],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="",
            group=-1,
            values=["ResNet50Conv"],
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
                False,
            ],
            hidename=False,
        )

        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                chips_singletask,
                [{"path": p["path"], "weight": (0.5 / N_CHIPS if "eval" in p["path"] else 0.5 / N_BROAD)} for p in broaddataset + chips_singletask],
                [{"path": p["path"], "weight": (0.5 / N_CHIPS if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + chips_singletask],
                ],
	    value_names=[
                "eval_chips", 
                "balanced_chips", 
                "full_balanced_chips", 
            ],
        )

        generator.add_param(
            key="algo.transformer.supervise_all_steps",
            name="supallsteps",
            group=1,
            values=[
                True,
            ],
            hidename=True,
        )
        generator.add_param(
            key="algo.transformer.pred_future_acs",
            name="futureacs",
            group=1,
            values=[
                False,
            ],
            hidename=True,
        )
        generator.add_param(
            key="algo.transformer.causal",
            name="causal",
            group=1,
            values=[
                False,
            ],
            hidename=True,
        )
        generator.add_param(
            key="train.frame_stack",
            name="",
            group=-1,
            values=[2],
            hidename=True,
        )
        generator.add_param(
            key="train.seq_length",
            name="H",
            group=-1,
            values=[2],
            hidename=False,
        )
        generator.add_param(
            key="algo.transformer.context_length",
            name="",
            group=-1,
            values=[2],
            hidename=True,
        )
        generator.add_param(
            key="algo.gmm.enabled",
            name="",
            group=-1,
            values=[False],
            hidename=True,
        )


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