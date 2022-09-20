"""
WARNING: This script is only for instructive purposes, and is missing several useful 
         components used during training such as logging and rollout evaluation. 

Example script for demonstrating how the SequenceDataset class and a training loop
can interact. This is meant to help others who would like to use our provided
datasets and dataset class in other applications.
"""
import numpy as np

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.dataset import SequenceDataset

from robomimic.config import config_factory
from robomimic.algo import algo_factory


def get_data_loader(dataset_path):
    """
    Get a data loader to sample batches of data.

    Args:
        dataset_path (str): path to the dataset hdf5
    """
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions", 
            "rewards", 
            "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=10,                  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,       # can optionally provide a filter key here
    )
    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader


def get_example_model(dataset_path, device):
    """
    Use a default config to construct a BC model.
    """

    # default BC config
    config = config_factory(algo_name="bc")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # read dataset to get some metadata for constructing model
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, 
        all_obs_keys=sorted((
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        )),
    )

    # make BC model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    return model


def print_batch_info(batch):
    print("\n============= Batch Info =============")
    for k in batch:
        if k in ["obs", "next_obs"]:
            print("key {}".format(k))
            for obs_key in batch[k]:
                print("    obs key {} with shape {}".format(obs_key, batch[k][obs_key].shape))
        else:
            print("key {} with shape {}".format(k, batch[k].shape))
    print("")


def run_train_loop(model, data_loader):
    """
    Note: this is a stripped down version of @TrainUtils.run_epoch and the train loop
    in the train function in train.py. Logging and evaluation rollouts were removed.

    Args:
        model (Algo instance): instance of Algo class to use for training
        data_loader (torch.utils.data.DataLoader instance): torch DataLoader for
            sampling batches
    """
    num_epochs = 50
    gradient_steps_per_epoch = 100
    has_printed_batch_info = False

    # ensure model is in train mode
    model.set_train()

    for epoch in range(1, num_epochs + 1): # epoch numbers start at 1

        # iterator for data_loader - it yields batches
        data_loader_iter = iter(data_loader)

        # record losses
        losses = []

        for _ in range(gradient_steps_per_epoch):

            # load next batch from data loader
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                # data loader ran out of batches - reset and yield first batch
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)

            if not has_printed_batch_info:
                has_printed_batch_info = True
                print_batch_info(batch)

            # process batch for training
            input_batch = model.process_batch_for_training(batch)
            input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

            # forward and backward pass
            info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=False)

            # record loss
            step_log = model.log_info(info)
            losses.append(step_log["Loss"])

        # do anything model needs to after finishing epoch
        model.on_epoch_end(epoch)

        print("Train Epoch {}: Loss {}".format(epoch, np.mean(losses)))


if __name__ == "__main__":

    # small dataset with a handful of trajectories
    dataset_path = TestUtils.example_dataset_path()

    # set torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # get model
    model = get_example_model(dataset_path=dataset_path, device=device)

    # get dataset loader
    data_loader = get_data_loader(dataset_path=dataset_path)

    # run train loop
    run_train_loop(model=model, data_loader=data_loader)
