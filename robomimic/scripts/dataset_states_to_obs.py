"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # (space saving option) extract 84x84 image observations with compression and without 
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import multiprocessing
import queue
import time

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase


""" 
    These methods: extract_datagen_info_from_trajectory and extract_datagen_info_from_trajectory_real_robot
    are copied over from mimicgen/dataset_states_to_args as importing that file caused environment creation issues 
"""
def extract_datagen_info_from_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
    """
    assert isinstance(env, EnvBase)
    assert len(states) == actions.shape[0]

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    all_datagen_infos = []
    traj_len = len(states)
    for t in range(traj_len):
        # reset to state
        env.reset_to({"states" : states[t]})
        # env.base_env.gym.fetch_results(env.base_env.sim, True)

        # extract datagen info
        datagen_info = env.base_env.get_datagen_info(action=actions[t])
        all_datagen_infos.append(datagen_info)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    all_datagen_infos = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos)
    for k in all_datagen_infos:
        # list to numpy array
        all_datagen_infos[k] = np.array(all_datagen_infos[k])

    return all_datagen_infos


def extract_datagen_info_from_trajectory_real_robot(
    env_meta,
    observations, 
    actions,
):
    """
    Real robot version of helper function to extract datagen info. On the real robot, we assume
    we stored "action-free" datagen-info directly in the observations, and we will use the actions
    to compute the target poses needed for datagen offline.

    Args:
        env_meta (dict): dictionary containing environment metadata from dataset
        observations (list): list of observations for this trajectory (each will be a dict)
        actions (np.array): array of actions
    """
    assert len(observations) == actions.shape[0]
    traj_len = actions.shape[0]

    # use static method of appropriate mimicgen base robot env to convert controller pose in obs + action to target pose
    if EnvUtils.is_real_robot_gprs_env(env_meta=env_meta):
        from mimicgen.envs.real_gprs.base import MG_Real_GPRS_Env
        static_method = MG_Real_GPRS_Env.action_to_pose_target_stateless
        # TODO: remove hardcode of action scaling here
        max_dpos = np.array([0.08, 0.08, 0.08])
        max_drot = np.array([0.5, 0.5, 0.5])
    elif EnvUtils.is_real_robot_env(env_meta=env_meta):
        from mimicgen.envs.real.base import MG_Real_Env
        static_method = MG_Real_Env.action_to_pose_target_stateless
        action_scale = np.array(env_meta["env_kwargs"]["action_scale"]).reshape(-1)
        max_dpos = action_scale[:3]
        max_drot = action_scale[3:6]
    else:
        raise Exception("env meta must be real robot type")

    all_datagen_infos = []
    for t in range(traj_len):
        # first copy action-free datagen info from observation
        datagen_info = dict()
        obs = observations[t]
        for k in obs:
            if k.startswith("datagen_"):
                datagen_info[k[8:]] = np.array(obs[k])

        # use action and controller pose in observation to compute target pose
        datagen_info["target_pos"], datagen_info["target_rot"] = static_method(
            action=actions[t], 
            start_pos=obs["datagen_eef_pos"], 
            start_rot=obs["datagen_eef_rot"], 
            max_dpos=max_dpos, 
            max_drot=max_drot,
        )

        all_datagen_infos.append(datagen_info)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    all_datagen_infos = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos)
    for k in all_datagen_infos:
        # list to numpy array
        all_datagen_infos[k] = np.array(all_datagen_infos[k])

    return all_datagen_infos

""" End of dataset_states_to_args copy over """

def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    actions_abs,
    done_mode,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    if actions_abs is not None:
        traj["actions_abs"] = np.array(actions_abs)
    
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


""" The process that writes over the generated files to memory """
def write_traj_to_file(args, output_path, total_samples, total_run, processes, is_robosuite_env, mul_queue):
    f = h5py.File(args.dataset, "r")
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    start_time = time.time()
    num_processed = 0
    
    try:
        while((total_run.value < (processes)) or not mul_queue.empty()):
            if not mul_queue.empty():
                num_processed = num_processed + 1
                item = mul_queue.get()
                ep = item[0]
                traj = item[1]
                datagen_info = item[2]
                process_num = item[3]
                try:
                    ep_data_grp = data_grp.create_group(ep)
                    ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
                    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                    ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
                    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                    if "actions_abs" in traj:
                        ep_data_grp.create_dataset("actions_abs", data=np.array(traj["actions_abs"]))
                    for k in traj["obs"]:
                        if args.compress:
                            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
                        else:
                            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                        if not args.exclude_next_obs:
                            if args.compress:
                                ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                            else:
                                ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

                    for k in datagen_info:
                        ep_data_grp.create_dataset("datagen_info/{}".format(k), data=np.array(datagen_info[k]))
                    
                    # copy action dict (if applicable)
                    if "data/{}/action_dict".format(ep) in f:
                        action_dict = f["data/{}/action_dict".format(ep)]
                        for k in action_dict:
                            ep_data_grp.create_dataset("action_dict/{}".format(k), data=np.array(action_dict[k][()]))

                    # episode metadata
                    if is_robosuite_env:
                        ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
                    if "ep_info" in f["data/{}".format(ep)].attrs:
                        ep_data_grp.attrs["ep_info"] = f["data/{}".format(ep)].attrs["ep_info"]
                    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
                    
                    total_samples.value += traj["actions"].shape[0]
                except Exception as e:
                    print("++"*50)
                    print(f"Error at Process {process_num} on episode {ep} with \n\n {e}")
                    print("++"*50)
                    raise Exception("Write out to file has failed")
                print("ep {}: wrote {} transitions to group {} at process {} with {} finished".format(num_processed, ep_data_grp.attrs["num_samples"], ep, process_num, total_run.value))
    except KeyboardInterrupt:
        print("Control C pressed. Closing File and ending \n\n\n\n\n\n\n")
        
        
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples.value
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )
    print("total processes end {}".format(total_run.value))
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(total_samples.value, output_path))
    
    f_out.close()
    f.close()
    print("Writing has finished")
    
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    return

# runs multiple trajectory. If there has been an unrecoverable error, the system puts the current work back into the queue and exits
def extract_multiple_trajectories(process_num, current_work_array, work_queue, lock, args2, num_finished, mul_queue):
    try:
        extract_multiple_trajectories_with_error(process_num, current_work_array, work_queue, lock, args2, mul_queue)
    except Exception as e:
        work_queue.put(current_work_array[process_num])
        print("*>*"*50)
        print(e)

    num_finished.value = num_finished.value + 1

    
def retrieve_new_index(process_num, current_work_array, work_queue, lock):
    with lock:
        if work_queue.empty():
            return -1
        try:
            tmp = work_queue.get(False)
            current_work_array[process_num] = tmp
            return tmp
        except queue.Empty:
            return -1

def extract_multiple_trajectories_with_error(process_num, current_work_array, work_queue, lock, args, mul_queue):
    # create environment to use for data processing

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    if args.real:
        is_robosuite_env = False
        is_simpler_env = False
        is_factory_env = False

    else:
        # some operations are env-type-specific
        is_simpler_env = False #EnvUtils.is_simpler_env(env_meta)
        is_factory_env = False #EnvUtils.is_factory_env(env_meta)
    
    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
    while (not work_queue.empty()) and (ind != -1):
        try:
            # print("Running {} index".format(ind))
            ep = demos[ind]

            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

            # extract obs, rewards, dones
            actions = f["data/{}/actions".format(ep)][()]
            if "data/{}/actions_abs".format(ep) in f:
                actions_abs = f["data/{}/actions_abs".format(ep)][()]
            else:
                actions_abs = None
                
                
            traj = extract_trajectory(
                env=env, 
                initial_state=initial_state, 
                states=states, 
                actions=actions,
                actions_abs=actions_abs,
                done_mode=args.done_mode,
            )

            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]
                
                        
            ep_grp = f["data/{}".format(ep)]

            if args.real:
                traj_len = ep_grp["actions"].shape[0]
                obs = []
                obs_grp = ep_grp["obs"]
                for i in range(traj_len):
                    obs.append(
                        { k : np.array(obs_grp[k][i]) for k in obs_grp }
                    )
                datagen_info = extract_datagen_info_from_trajectory_real_robot(
                    env_meta=env_meta,
                    observations=obs,
                    actions=ep_grp["actions"][()],
                )
            else:
                # prepare states to reload from
                if is_simpler_env or is_factory_env:
                    # states are dictionaries - make list of dictionaries
                    traj_len = ep_grp["actions"].shape[0]
                    states = []
                    state_grp = ep_grp["states"]
                    for i in range(traj_len):
                        states.append(
                            { k : np.array(state_grp[k][i]) for k in state_grp }
                        )
                else:
                    states = ep_grp["states"][()]
                initial_state = dict(states=states[0])
                if is_robosuite_env:
                    initial_state["model"] = ep_grp.attrs["model_file"]

                # extract datagen info
                actions = ep_grp["actions"][()]
                datagen_info = extract_datagen_info_from_trajectory(
                    env=env, 
                    initial_state=initial_state, 
                    states=states, 
                    actions=actions,
                )

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            # print("ADD TO QUEUE {} of index {}".format(process_num, ind))
            mul_queue.put([ep, traj, datagen_info, process_num])
            
            ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
        except Exception as e:
            print("_"*50)
            print(process_num)
            print("Error {} {}".format(ind, e))
            print("_"*50)
            env = EnvUtils.create_env_for_data_processing( #when it errors, it like blows up the environment for some reason
                env_meta=env_meta,
                camera_names=args.camera_names, 
                camera_height=args.camera_height, 
                camera_width=args.camera_width, 
                reward_shaping=args.shaped,
            )

    f.close()
    print("Process {} finished".format(process_num))

def dataset_states_to_obs_multiprocessing(args):
    # create environment to use for data processing

    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(args.camera_width)

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)
    
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))
    
    
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[:args.n]

    num_demos = len(demos)
    f.close()

    
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
    num_processes = 8
    
    index = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    total_samples_shared = multiprocessing.Value('i', 0)
    num_finished = multiprocessing.Value('i', 0)
    mul_queue = multiprocessing.Queue()
    work_queue = multiprocessing.Queue()
    for index in range(num_demos):
        work_queue.put(index)
    current_work_array = multiprocessing.Array('i', num_processes)
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=extract_multiple_trajectories, args=(i, current_work_array, work_queue, lock, args, num_finished, mul_queue))
        processes.append(process)
    
    process1 = multiprocessing.Process(target=write_traj_to_file, args=(args, output_path, total_samples_shared, num_finished, num_processes, is_robosuite_env, mul_queue))
    processes.append(process1)
    
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()

    print("Finished Multiprocessing")
    return

def dataset_states_to_obs(args):
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(args.camera_width)
    
    output_path = os.path.join(os.path.dirname(args.dataset), output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind in range(len(demos)):
    # for ind in range(1005):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        if "data/{}/actions_abs".format(ep) in f:
            actions_abs = f["data/{}/actions_abs".format(ep)][()]
        else:
            actions_abs = None
        traj = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            actions_abs=actions_abs,
            done_mode=args.done_mode,
        )

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]
            
        # store transitions

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        if "actions_abs" in traj:
            ep_data_grp.create_dataset("actions_abs", data=np.array(traj["actions_abs"]))
        for k in traj["obs"]:
            if args.compress:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
            else:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            if not args.exclude_next_obs:
                if args.compress:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

        # copy action dict (if applicable)
        if "data/{}/action_dict".format(ep) in f:
            action_dict = f["data/{}/action_dict".format(ep)]
            for k in action_dict:
                ep_data_grp.create_dataset("action_dict/{}".format(k), data=np.array(action_dict[k][()]))

        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
        if "ep_info" in f["data/{}".format(ep)].attrs:
            ep_data_grp.attrs["ep_info"] = f["data/{}".format(ep)].attrs["ep_info"]
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
        total_samples += traj["actions"].shape[0]
        print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))


    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        action='store_true',
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )
    
    # real robot
    parser.add_argument(
        "--real",
        action='store_true',
        help="specify this if using real robot dataset",
    )

    args = parser.parse_args()
    # dataset_states_to_obs(args)
    dataset_states_to_obs_multiprocessing(args)
