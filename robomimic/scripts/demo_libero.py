"""
A script to try playing random actions in LIBERO environments.

Example usage:
    python scripts/demo_libero.py 

    python scripts/demo_libero.py --reset --bddl_file_name KITCHEN_SCENE1_D1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.bddl
    python scripts/demo_libero.py --reset --bddl_file_name KITCHEN_SCENE2_D1_open_the_top_drawer_of_the_cabinet.bddl
    python scripts/demo_libero.py --reset --bddl_file_name KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp1_target.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp1_motion_varyTask.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp1_spatial_varyObjSpat.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp1_spatial_varyRecepSpat.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp1_visual_varyCamPose.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp1_visual_varyLighting.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp1_visual_varyObjTex.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp1_visual_varyTableTex.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp2_target_newTask.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp2_target_newCamPose.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp2_big_visualCamPose.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp2_big_visualCamPose5Normal.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp3_big_visualCamPose5Normal.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp3_target_newScene.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp4_small_microwavePullPickPlace.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp4_small_microwavePickPlacePush.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp3_big_motionPickPlaceTopDrawer.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp3_big_motionPullTopDrawer.bddl
    python scripts/demo_libero.py --reset --bddl_file_name exp3_big_motionPushTopDrawer.bddl
    

"""
import os
import json
import argparse
import imageio
import numpy as np

from tqdm import tqdm

import robosuite
from robosuite.controllers import load_controller_config

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import *
from libero.libero.envs.problems.libero_kitchen_tabletop_manipulation import Libero_Kitchen_Tabletop_Manipulation
from termcolor import colored


def get_all_task_bddl_files():
    """
    Collects BDDL file paths (and task names) for each task in each LIBERO benchmark into
    a single dictionary.
    """
    bddl_files_default_path = get_libero_path("bddl_files")

    task_bddl_files = dict(
        libero_object=None,
        libero_goal=None,
        libero_spatial=None,
        libero_10=None,
        libero_90=None,
        # libero_90_d1=None,
        robomimic_v2=None
    )
    for benchmark_name in task_bddl_files:
        # import pdb; pdb.set_trace()
        benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
        num_tasks = benchmark_instance.get_num_tasks()
        task_names = benchmark_instance.get_task_names()
        bddl_files = []
        for task_id in range(num_tasks):
            task_name = task_names[task_id]
            task = benchmark_instance.get_task(task_id)
            bddl_file = os.path.join(
                bddl_files_default_path, task.problem_folder, task.bddl_file
            )
            assert os.path.exists(bddl_file), f"{bddl_file} does not exist!"
            bddl_files.append(bddl_file)

        task_bddl_files[benchmark_name] = dict(
            num_tasks=num_tasks,
            task_names=task_names,
            bddl_files=bddl_files,
        )

    return task_bddl_files


def bddl_file_name_to_task_info(bddl_file_name):
    task_bddl_files = get_all_task_bddl_files()
    for task_benchmark in task_bddl_files:
        for task_id in range(len(task_bddl_files[task_benchmark]["bddl_files"])):
            full_path = task_bddl_files[task_benchmark]["bddl_files"][task_id]
            if os.path.basename(full_path) == bddl_file_name:
                return dict(
                    task_benchmark=task_benchmark,
                    task_name=task_bddl_files[task_benchmark]["task_names"][task_id],
                    task_id=task_id,
                )
    return None


def bddl_file_path_to_task_name(bddl_file_path, verbose=False):
    """
    Gets task name (to pass to robosuite.make) given BDDL file path.
    """

    # BDDL file to problem info
    assert os.path.exists(bddl_file_path)
    problem_info = BDDLUtils.get_problem_info(bddl_file_path)

    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if verbose:
        print("")
        print("BDDL file: {}".format(bddl_file_path))
        print("problem_info")
        print(json.dumps(problem_info, indent=4))
        text = colored(language_instruction, "red", attrs=["bold"])
        print("Goal of the following task: ", text)
    # instruction = colored("Hit any key to proceed to data collection ...", "green", attrs=["reverse", "blink"])
    # print(instruction)
    # input()

    # look up env class object using LIBERO mapping, and then get the class name as a string for use with robosuite registry
    env_class = TASK_MAPPING[problem_name]
    env_name = env_class.__name__
    return env_name


def get_env(task_bddl_files, task_benchmark=None, bddl_file_name=None, task_id=None):
    """
    Get env and env info that will facilitate supporting MimicGen.
    """
    assert (bddl_file_name is not None) or (task_id is not None)
    assert (bddl_file_name is None) or (task_id is None)
    if bddl_file_name is not None:
        if not bddl_file_name.endswith(".bddl"):
            bddl_file_name += ".bddl"
        task_info = bddl_file_name_to_task_info(bddl_file_name)
        task_id = task_info["task_id"]
        task_benchmark = task_info["task_benchmark"]
    else:
        assert task_benchmark is not None

    task_name = task_bddl_files[task_benchmark]["task_names"][task_id]
    bddl_file_path = task_bddl_files[task_benchmark]["bddl_files"][task_id]

    # env name from BDDL file
    env_name = bddl_file_path_to_task_name(bddl_file_path=bddl_file_path, verbose=False)

    # create robosuite env
    controller_config = load_controller_config(default_controller="OSC_POSE")
    config = {
        "env_name": env_name,
        "bddl_file_name": bddl_file_path,
        "robots": ["Panda"],
        # "robots": "Panda", # TODO: this is broken somehow...
        "controller_configs": controller_config,
    }
    env = robosuite.make(
        **config,
        has_renderer=(not args.reset),
        has_offscreen_renderer=(args.reset),
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    obs = env.reset()

    env_info = dict(
        domain=env_name,
        task_benchmark=task_benchmark,
        task_id=task_id,
        language_instruction=" ".join(env.parsed_problem["language_instruction"]),
        objects=list(env.objects_dict.keys()),
        fixtures=list(env.fixtures_dict.keys()),
        objects_of_interest=env.parsed_problem["obj_of_interest"],
        goal_state=env.parsed_problem["goal_state"],
    )

    return env, env_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # below args are only used if reset flag is passed, otherwise random actions and on-screen rendering is used
    parser.add_argument(
        "--reset",
        action='store_true',
        help="set this flag to visualize resets instead of random actions",
    )

    # camera names to use for visualization
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="camera names to use for visualization",
    )

    # number of frames in output video
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="number of frames in output video",
    )

    # path to output video
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="path to output video",
    )

    # path to output video
    parser.add_argument(
        "--task_benchmark",
        type=str,
        default=None,
    )

    # path to output video
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
    )

    # bddl file name
    parser.add_argument(
        "--bddl_file_name",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # # Note: can hardcode BDDL file path here
    # bddl_file_path = "bddl_files/KITCHEN_SCENE9_playdata.bddl"

    # Note: or can choose one of the bddl files from the LIBERO task suite by indexing this dictionary
    task_bddl_files = get_all_task_bddl_files()

    # # pull skill
    # a = bddl_file_name_to_task_info("KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl")
    # # pick and place skill
    # b = bddl_file_name_to_task_info("LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl")
    # # pick and place skill
    # c = bddl_file_name_to_task_info("STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl")
    # # push skill
    # d = bddl_file_name_to_task_info("KITCHEN_SCENE9_turn_on_the_stove.bddl")
    # # push and pick-place skill
    # e = bddl_file_name_to_task_info("KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it.bddl")
    # # pull and pick-place skill
    # f = bddl_file_name_to_task_info("KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.bddl")
    # for testing multimodality
    # g = bddl_file_name_to_task_info("KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl")

    # from IPython import embed; embed()
    # exit()

    # # code for getting env info
    # bddl_file_names = [
    #     # "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    #     # "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
    #     # "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    #     # "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
    #     # "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    #     # "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    #     # "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
    #     # "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
    #     # "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
    #     # "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    #     # "KITCHEN_SCENE6_close_the_microwave",
    #     "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
    #     # "KITCHEN_SCENE8_turn_off_the_stove",
    #     # "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
    #     # "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
    #     # "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    #     # "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    #     # "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
    #     # "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    #     # "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
    #     # "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
    #     # "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
    #     # "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
    #     # "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
    # ]
    # assert not args.reset
    # for bddl_file_name in bddl_file_names:
    #     env, env_info = get_env(
    #         task_bddl_files=task_bddl_files,
    #         task_benchmark=args.task_benchmark,
    #         bddl_file_name=bddl_file_name,
    #         task_id=args.task_id,
    #     )
    #     print("")
    #     print(bddl_file_name)
    #     print(json.dumps(env_info, indent=4))
    # exit()

    # task_id = 40 # note: 40 to 45 correspond to "KITCHEN_SCENE9" eval tasks used in MimicPlay
    env, env_info = get_env(
        task_bddl_files=task_bddl_files,
        task_benchmark=args.task_benchmark,
        bddl_file_name=args.bddl_file_name,
        task_id=args.task_id,
    )

    if args.reset:
        if args.output is None:
            args.output = "reset_{}_{}.mp4".format(env_info["task_benchmark"], env_info["task_id"])
        # write a video
        video_writer = imageio.get_writer(args.output, fps=5)
        for i in tqdm(range(args.frames)):
            obs = env.reset()
            video_img = []
            for cam_name in args.camera_names:
                video_img.append(env.sim.render(height=512, width=512, camera_name=cam_name)[::-1])
            video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
            video_writer.append_data(video_img)
        video_writer.close()
        exit()

    obs = env.reset()
    env.viewer.set_camera(camera_id=0)
    env.render()

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()