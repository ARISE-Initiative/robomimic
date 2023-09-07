import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy
import random
from sklearn.metrics import mean_squared_error
import re
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
import torch
from torch.utils.data import DataLoader

"""
TODO: track rotation magnitude seperately (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.magnitude.html)
"""

# the configs of the models to be plotted
model_config_mapping = {
    # "bottle_less_obs": {
    #     "model":"/home/zehan/expdata/r2d2/im/bc_xfmr/google_bc_baseline/bottle_less_obs/20230815225106/models/model_epoch_60.pth",
    #     'folder':"/home/zehan/expdata/r2d2/im/bc_xfmr/google_bc_baseline/bottle_less_obs/20230815225106/test_inference_figures/",
    #     # "action_names": ['x', 'y', 'z', 'roll', 'pitch', 'yaw', "gripper_action" , 'terminate'],
    #     "action_names": None,
    #     "trajectory_name_regex": r'(\d+_trajectory_im\d+)'
    # },
    "r2d2_wire": {
        # "model": "/home/soroushn/expdata/r2d2/im/diffusion_policy/debug/ds_pen-in-cup_cams_3cams/20230830160945/models/model_epoch_2.pth",
        "model": "/home/soroushn/expdata/r2d2/im/bc_xfmr/debug/ds_pen-in-cup_cams_3cams_predfuture_True_ac_keys_rel/20230830161631/models/model_epoch_2.pth",
        "folder": "/home/soroushn/tmp/model_predictions",
        # "action_names": ['x', 'y', 'z', 'r', 'p', 'y', "gripper_pos"], # use custom names
        "action_names": None, # use default names, see line 71
        "trajectory_name_regex": r'(\w+_\w+_\d{2}_\d{2}:\d{2}:\d{2}_\d{4})' # the name of the figure files need to be custom defined (the part of the names of the trajectories that uniquely identifies them)
    }
}

NUM_SAMPLES = 2

# loop through each model
for model_name in model_config_mapping:
    ckpt_path = model_config_mapping[model_name]['model']
    saving_folder = model_config_mapping[model_name]['folder']
    # can custom-define or using default action_names
    action_names = model_config_mapping[model_name]['action_names']
    trajectory_name_regex = model_config_mapping[model_name]['trajectory_name_regex']
    accuracy_thresholds = np.logspace(-3,-5, num=3).tolist()



    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    config = json.loads(ckpt_dict["config"])
    config["train"]["shuffled_obs_key_groups"] = None
    ckpt_dict["config"] = json.dumps(config)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    shape_meta = ckpt_dict['shape_metadata']
    ext_cfg = json.loads(ckpt_dict["config"])
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    
    
    frame_stack = config.train.frame_stack 
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    # trainset.datasets is a list
    # the trajectories to plot is randomly sampled from the training and validation sets
    training_sampled_data = random.sample(trainset.datasets, NUM_SAMPLES)
    # validation_sampled_data = random.sample(validset.datasets, NUM_SAMPLES)

    inference_datasets_mapping = {"training": training_sampled_data} #, "validation": validation_sampled_data}

    
    if action_names == None:
        # TODO
        action_keys = config.train.action_keys # Need to adjust. For Robomimic datasets, there is no `action_keys`, it is config.train.dataset_keys
        modified_action_keys = [element.replace('action/', '') for element in action_keys]
        action_names = []
        for i, action_key in enumerate(action_keys):
            if isinstance(training_sampled_data[0].__getitem__(0)[action_key][frame_stack-1], np.ndarray):
                action_names.extend([f'{modified_action_keys[i]}_{j+1}' for j in range(len(training_sampled_data[0].__getitem__(0)[action_key][frame_stack-1]))])
            else:
                action_names.append(modified_action_keys[i])

    # loop through training and validation sets
    for inference_key in inference_datasets_mapping:
        mse_training_per_traj = []
        data_name = []
        actual_actions_all_traj = [] # (NxT, D)
        predicted_actions_all_traj = [] # (NxT, D)

        # loop through each trajectory
        for d in inference_datasets_mapping[inference_key]:
            hdf5_path = d.hdf5_path
            mse_for_one_traj = []
            traj_length = len(d)
            action_dim = len(action_names)
            actual_actions = [[] for _ in range(action_dim)] # (T, D)
            predicted_actions = [[] for _ in range(action_dim)] # (T, D)           

            image_keys = [item for item in d.__getitem__(0)['obs'].keys() if "image" in item]
            images = {key: [] for key in image_keys}

            dataloader = DataLoader(
                dataset=d,
                sampler=None,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=True,
            )

            model = policy.policy

            model.reset()

            # loop through each timestep
            for batch in iter(dataloader):
                batch = model.process_batch_for_training(batch)

                for image_key in image_keys:
                    im = batch["obs"][image_key][0][-1]
                    im = TensorUtils.to_numpy(im).astype(np.uint32)
                    images[image_key].append(im)

                batch = model.postprocess_batch_for_training(batch, obs_normalization_stats=None) # ignore obs_normalization for now
                # model_output = model.nets["policy"](batch["obs"])

                model_output = model.get_action(batch["obs"])
                
                actual_action = TensorUtils.to_numpy(
                    batch["actions"][0][0]
                )
                predicted_action = TensorUtils.to_numpy(
                    model_output[0]
                )

                actual_actions_all_traj.append(actual_action)
                predicted_actions_all_traj.append(predicted_action)
            
                for dim in range(action_dim):
                    actual_actions[dim].append(actual_action[dim])
                    predicted_actions[dim].append(predicted_action[dim])

            # Plot
            fig, axs = plt.subplots(len(images) + action_dim, 1, figsize=(30, (len(images) + action_dim) * 3))
            for i, image_key in enumerate(image_keys):
                interval = int(traj_length/15) # plot `5` images
                images[image_key] = images[image_key][::interval]
                combined_images = np.concatenate(images[image_key], axis=1)
                axs[i].imshow(combined_images)
                if i == 0:
                    axs[i].set_title(hdf5_path + '\n' + image_key, fontsize=30)
                else:
                    axs[i].set_title(image_key, fontsize=30)
                axs[i].axis("off")
            for dim in range(action_dim):
                mse = mean_squared_error(actual_actions[dim], predicted_actions[dim])
                mse_for_one_traj.append(mse)
                axs[len(images)+dim].plot(range(traj_length), actual_actions[dim], label='Actual Action', color='blue')
                axs[len(images)+dim].plot(range(traj_length), predicted_actions[dim], label='Predicted Action', color='red')
                # axs[len(images)+dim].set_xlabel('Timestep')
                # axs[len(images)+dim].set_ylabel('Action Dimension {}'.format(dim + 1))
                axs[len(images)+dim].set_title(action_names[dim], fontsize=30)
                axs[len(images)+dim].xaxis.set_tick_params(labelsize=24)
                axs[len(images)+dim].yaxis.set_tick_params(labelsize=24)
                axs[len(images)+dim].legend(fontsize=20)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.6)

            # Save inference figures
            save_path = saving_folder + inference_key+"/" #remember to add / at the end
            # data_content = re.search(trajectory_name_regex, hdf5_path).group(1)
            data_content = "test"
            filename = "comparison_figure_"+data_content +".png"    
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(save_path + filename)
            # Save the figure with the specified path and filename
            plt.savefig(save_path + filename) 
            mse_training_per_traj.append(mse_for_one_traj)
            data_name.append(hdf5_path)

        # log MSE information
        accuracy_thresholds = np.logspace(-3,-5, num=3).tolist()
        mse = torch.nn.functional.mse_loss(torch.tensor(predicted_actions_all_traj), torch.tensor(actual_actions_all_traj), reduction='none') # (NxT, D)
        step_log = {}
        step_log[f'{inference_key}_action_mse_error'] = mse.mean().item() # average MSE across all timesteps averaged across all action dimensions (D,)
        
        # compute percentage of timesteps that have MSE less than the accuracy thresholds
        for accuracy_threshold in accuracy_thresholds:
            step_log[f'{inference_key}_action_accuracy@{accuracy_threshold}'] = (torch.less(mse,accuracy_threshold).float().mean().item())
        

        average_mse_per_dimension = np.mean(mse_training_per_traj, axis=0) # (D,)
        txt_path = saving_folder+inference_key+"/" +"output.txt"
        list_str = '\n'.join(['{} {}'.format(desc, ' '.join(map(str, sublist))) for desc, sublist in zip(data_name, mse_training_per_traj)])
        
        # save MSE information
        with open(txt_path, "w+") as txt_file:
            txt_file.write(f"MSE per trajectory:\n{list_str}\n")
            txt_file.write("\n")
            txt_file.write(f"Average MSE across trajectories per dimension: {average_mse_per_dimension}\n")
            txt_file.write("\n")
            txt_file.write(f"MSE log: {step_log}\n")


