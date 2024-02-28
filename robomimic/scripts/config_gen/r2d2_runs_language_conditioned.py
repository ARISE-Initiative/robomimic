from robomimic.scripts.config_gen.helper import *
import random
import json
import numpy as np

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

## Getting all OXE
with open("/mnt/fsx/surajnair/datasets/oxe_hdf5/manifest_oxe.json", 'r') as file:
    oxe = json.load(file)
N_OXE = len(oxe)

## Getting all stanford data
with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_lang_stanford_eval.json", 'r') as file:
    langs = json.load(file)
stanford_singletask_eraser = [{'path': l['path']} for l in langs if 'stanford_eval_0125' in l['path']]
N_STANFORD_ERASER = len(stanford_singletask_eraser)

with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_stanford_evgr_laundry.json", 'r') as file:
    stanford_singletask_laundry = json.load(file)
N_STANFORD_LAUNDRY = len(stanford_singletask_laundry)

with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_stanford_evgr_cooking.json", 'r') as file:
    stanford_singletask_cooking = json.load(file)
N_STANFORD_COOKING = len(stanford_singletask_cooking)


## Getting all TRI data
with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_tri_chips.json", 'r') as file:
    langs = json.load(file)
chips_singletask = [{'path': l['path']} for l in langs if 'TRI_chips_only_1_21' in l['path']]
N_CHIPS = len(chips_singletask)

with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_tri_frenchpress.json", 'r') as file:
    tri_frenchpress = json.load(file)
N_FRENCHPRESS = len(tri_frenchpress)

with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_tri_pot.json", 'r') as file:
    tri_pot = json.load(file)
N_POT = len(tri_pot)

## Getting all language labeled data
with open("/mnt/fsx/surajnair/datasets/r2d2-data/manifest_lang.json", 'r') as file:
    langs = json.load(file)

## Getting broad data
broaddataset_full = [{'path': l['path']} for l in langs if '/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads/' in l['path']]
N_BROAD_FULL = len(broaddataset_full)

with open("/mnt/fsx/surajnair/datasets/r2d2-data/filter_traj_20scene_10k_episodes_success.json", 'r') as file:
    subset = json.load(file)

broaddataset_filtered = [{'path': l['path']} for l in broaddataset_full if "/".join(l['path'].split("/")[7:-1])+"/" in subset]
N_BROAD_FILTERED = len(broaddataset_filtered)

broaddataset_filtered_random = np.random.choice(broaddataset_full, N_BROAD_FILTERED).tolist()
N_BROAD_FILTERED_RANDOM = len(broaddataset_filtered_random)

cmutoast_singletask = [{'path': l['path']} for l in langs if 'r2_d2_toaster3_cmu_rgb' in l['path']]
N_CMU_TOASTER = len(cmutoast_singletask)

cmu_multi = [{'path': l['path']} for l in langs if 'cmu_rgb' in l['path']]
N_CMU_MULTI = len(cmu_multi)


print(f"Broad Datapoints: R2D2 {N_BROAD_FULL, N_BROAD_FILTERED, N_BROAD_FILTERED_RANDOM} OXE {N_OXE} \
      \nSingle Task TRI Datapoints: CHIPS {N_CHIPS} POT {N_POT} FRENCHPRESS {N_FRENCHPRESS} \
      \nStanford Datapoints: Eraser {N_STANFORD_ERASER} Laundry {N_STANFORD_LAUNDRY} Cooking {N_STANFORD_COOKING} \
      \nCMU Datapoints: {N_CMU_MULTI, N_CMU_TOASTER} ")
    #  \nMultitask Eval Datapoints: {N_EVAL_MULTI} \
    #   \nSingle Task APPLE Datapoints: {N_APPLE}")
    # #   \nSingle Task Microwave Close Datapoints {N_EVAL_CLOSEMICRO}")


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
        name="bz",
        group=1212111,
        values=[128],
        hidename=False,
    )

    generator.add_param(
        key="algo.noise_samples",
        name="noise_samples",
        group=1010101,
        values=[8],
        value_names=["8"]
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
                ["camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image"],
                # [
                    # "camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image",
                #     "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_1_right_image",
                #     "camera/image/varied_camera_2_left_image", "camera/image/varied_camera_2_right_image",
                # ],
            ],
            value_names=[
                # "wrist",
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
                # "ColorRandomizer", # jitter only
                ["ColorRandomizer", "CropRandomizer"], # jitter, followed by crop
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=130,
            values=[
                # {}, # jitter only
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
                # "geom",
                None, # Change this to "geom" to do goal conditioning

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
            key="observation.modalities.obs.low_dim",
            name="ldkeys",
            group=24986,
            values=[
                # ["robot_state/cartesian_position", "robot_state/gripper_position"],
                [
                    "robot_state/cartesian_position", "robot_state/gripper_position",
                    "lang_fixed/language_distilbert"
                ]
            ],
            value_names=[
                # "proprio",
                # "proprio",
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
                # stanford_singletask_cooking,
                # stanford_singletask_laundry,
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_COOKING if "eval" in p["path"] else 0.5 / N_BROAD_FILTERED)} for p in broaddataset_filtered + stanford_singletask_cooking],
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_LAUNDRY if "eval" in p["path"] else 0.5 / N_BROAD_FILTERED)} for p in broaddataset_filtered + stanford_singletask_laundry],
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_COOKING if "eval" in p["path"] else 0.5 / N_BROAD_FILTERED_RANDOM)} for p in broaddataset_filtered_random + stanford_singletask_cooking],
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_LAUNDRY if "eval" in p["path"] else 0.5 / N_BROAD_FILTERED_RANDOM)} for p in broaddataset_filtered_random + stanford_singletask_laundry],
                # stanford_singletask_eraser,
                # cmutoast_singletask,
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_ERASER if "eval" in p["path"] else 0.5 / N_BROAD_FILTERED)} for p in broaddataset_filtered + stanford_singletask_eraser],
                # [{"path": p["path"], "weight": (0.5 / N_CMU_TOASTER if "eval" in p["path"] else 0.5 / N_BROAD_FILTERED)} for p in broaddataset_filtered + cmutoast_singletask],
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_ERASER if "eval" in p["path"] else 0.5 / N_BROAD_FILTERED_RANDOM)} for p in broaddataset_filtered_random + stanford_singletask_eraser],
                # [{"path": p["path"], "weight": (0.5 / N_CMU_TOASTER if "eval" in p["path"] else 0.5 / N_BROAD_FILTERED_RANDOM)} for p in broaddataset_filtered_random + cmutoast_singletask],
                # stanford_singletask_cooking,
                # stanford_singletask_laundry,
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_COOKING if "eval" in p["path"] else 0.5 / N_OXE)} for p in oxe + stanford_singletask_cooking],
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_LAUNDRY if "eval" in p["path"] else 0.5 / N_OXE)} for p in oxe + stanford_singletask_laundry],
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_COOKING if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + stanford_singletask_cooking],
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_LAUNDRY if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + stanford_singletask_laundry],
                # tri_pot,
                # tri_frenchpress,
                # [{"path": p["path"], "weight": (0.5 / N_POT if "eval" in p["path"] else 0.5 / N_OXE)} for p in oxe + tri_pot],
                # [{"path": p["path"], "weight": (0.5 / N_FRENCHPRESS if "eval" in p["path"] else 0.5 / N_OXE)} for p in oxe + tri_frenchpress],
                # [{"path": p["path"], "weight": (0.5 / N_POT if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + tri_pot],
                # [{"path": p["path"], "weight": (0.5 / N_FRENCHPRESS if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + tri_frenchpress],
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_ERASER if "eval" in p["path"] else 0.5 / N_OXE)} for p in oxe + stanford_singletask_eraser],
                # [{"path": p["path"], "weight": (0.5 / N_CHIPS if "eval" in p["path"] else 0.5 / N_OXE)} for p in oxe + chips_singletask],
                # [{"path": p["path"], "weight": (0.5 / N_CMU_MULTI if "eval" in p["path"] else 0.5 / N_OXE)} for p in oxe + cmu_multi],
                # [{"path": p["path"], "weight": (0.5 / N_CMU_TOASTER if "eval" in p["path"] else 0.5 / N_OXE)} for p in oxe + cmutoast_singletask],
                # stanford_singletask_eraser,
                # [{"path": p["path"], "weight": (0.5 / N_STANFORD_ERASER if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + stanford_singletask_eraser],
                # cmu_multi,
                # cmutoast_singletask,
                # [{"path": p["path"], "weight": (0.5 / N_CMU_MULTI if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + cmu_multi],
                # [{"path": p["path"], "weight": (0.5 / N_CMU_TOASTER if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + cmutoast_singletask],
                broaddataset_full,
                # chips_singletask,
                # [{"path": p["path"], "weight": (0.5 / N_CHIPS if "eval" in p["path"] else 0.5 / N_BROAD)} for p in broaddataset + chips_singletask],
                # [{"path": p["path"], "weight": (0.5 / N_CHIPS if "eval" in p["path"] else 0.5 / N_BROAD_FULL)} for p in broaddataset_full + chips_singletask],
                ],
	    value_names=[
                # "stanford_cooking", 
                # "stanford_laundry",
                # "filteredr2d2_stanford_cooking", 
                # "filteredr2d2_stanford_laundry",
                # "filteredrandomr2d2_stanford_cooking", 
                # "filteredrandomr2d2_stanford_laundry",
                # "stanford_eraser", 
                # "cmu_toast",
                # "filteredr2d2_stanford_eraser", 
                # "filteredr2d2_cmu_toast",
                # "filteredrandomr2d2_stanford_eraser", 
                # "filteredrandomr2d2_cmu_toast",
                # "stanford_cooking",
                # "stanford_laundry",
                # "oxe_balanced_stanford_cooking",
                # "oxe_balanced_stanford_laundry",
                # "fullr2d2_balanced_stanford_cooking",
                # "fullr2d2_balanced_stanford_laundry",
                # "tri_pot",
                # "tri_frenchpress",
                # "oxe_balanced_tri_pot",
                # "oxe_balanced_tri_frenchpress",
                # "fullr2d2_balanced_tri_pot",
                # "fullr2d2_balanced_tri_frenchpress",
                # "oxe_balanced_stanford_eraser",
                # "oxe_balanced_chips",
                # "oxe_balanced_cmu_multi",
                # "oxe_balanced_cmu_toast",
                # "stanford_eraser", 
                # "fullr2d2_balanced_stanford_eraser"
                # "cmu_multi", 
                # "cmu_toast",
                # "fullr2d2_balanced_cmu_multi", 
                # "fullr2d2_balanced_cmu_toast",
                "broad",
                # "eval_chips", 
                # "balanced_chips", 
                # "full_balanced_chips", 
            ],
        )

        generator.add_param(
            key="train.hdf5_cache_mode",
            name="",
            group=-1,
            values=[None],
            value_names=[""],
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