"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.
"""
import os
import h5py
import json
import time
import urllib.request
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import torch

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.algo import RolloutPolicy


def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    """
    Creates a new hdf5 filter key in hdf5 file @hdf5_path with
    name @key_name that corresponds to the demonstrations
    @demo_keys. Filter keys are generally useful to create
    named subsets of the demonstrations in an hdf5, making it
    easy to train, test, or report statistics on a subset of
    the trajectories in a file.

    Returns the list of episode lengths that correspond to the filtering.

    Args:
        hdf5_path (str): path to hdf5 file
        demo_keys ([str]): list of demonstration keys which should
            correspond to this filter key. For example, ["demo_0", 
            "demo_1"].
        key_name (str): name of filter key to create

    Returns:
        ep_lengths ([int]): list of episode lengths that corresponds to
            each demonstration in the new filter key
    """
    f = h5py.File(hdf5_path, "a")  
    demos = sorted(list(f["data"].keys()))

    # collect episode lengths for the keys of interest
    ep_lengths = []
    for ep in demos:
        ep_data_grp = f["data/{}".format(ep)]
        if ep in demo_keys:
            ep_lengths.append(ep_data_grp.attrs["num_samples"])

    # store list of filtered keys under mask group
    k = "mask/{}".format(key_name)
    if k in f:
        del f[k]
    f[k] = np.array(demo_keys, dtype='S')

    f.close()
    return ep_lengths


def get_env_metadata_from_dataset(dataset_path):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    f.close()
    return env_meta


def get_shape_metadata_from_dataset(dataset_path, all_modalities=None, verbose=False):
    """
    Retrieves shape metadata from dataset.

    Args:
        dataset_path (str): path to dataset
        all_modalities (list): list of all modalities used by the model. If not provided, all modalities
            present in the file are used.
        verbose (bool): if True, include print statements

    Returns:
        shape_meta (dict): shape metadata. Contains the following keys:

            :`'ac_dim'`: action space dimension
            :`'all_shapes'`: dictionary that maps observation modality string to modality shape
            :`'all_modalities'`: list of all observation modalities used
            :`'use_images'`: bool, whether or not image modalities are present
    """

    shape_meta = {}

    # read demo file for some metadata
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    demo_id = list(f["data"].keys())[0]
    demo = f["data/{}".format(demo_id)]

    # action dimension
    shape_meta['ac_dim'] = f["data/{}/actions".format(demo_id)].shape[1]

    # observation dimensions
    all_shapes = OrderedDict()

    if all_modalities is None:
        # use all modalities present in the file
        all_modalities = [k for k in demo["obs"]]

    for k in sorted(all_modalities):
        all_shapes[k] = demo["obs/{}".format(k)].shape[1:]
        if verbose:
            print("obs modality {} with shape {}".format(k, all_shapes[k]))

    for k in all_shapes:
        if ObsUtils.key_is_image(k):
            all_shapes[k] = ObsUtils.process_image_shape(all_shapes[k])

    f.close()

    shape_meta['all_shapes'] = all_shapes
    shape_meta['all_modalities'] = all_modalities
    shape_meta['use_images'] = ObsUtils.has_image(all_modalities)

    return shape_meta


def load_dict_from_checkpoint(ckpt_path):
    """
    Load checkpoint dictionary from a checkpoint file.
    
    Args:
        ckpt_path (str): Path to checkpoint file.

    Returns:
        ckpt_dict (dict): Loaded checkpoint dictionary.
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    if not torch.cuda.is_available():
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    else:
        ckpt_dict = torch.load(ckpt_path)
    return ckpt_dict


def maybe_dict_from_checkpoint(ckpt_path=None, ckpt_dict=None):
    """
    Utility function for the common use case where either an ckpt path
    or a ckpt_dict is provided. This is a no-op if ckpt_dict is not
    None, otherwise it loads the model dict from the ckpt path.

    Args:
        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

    Returns:
        ckpt_dict (dict): Loaded checkpoint dictionary.
    """
    assert (ckpt_path is not None) or (ckpt_dict is not None)
    if ckpt_dict is None:
        ckpt_dict = load_dict_from_checkpoint(ckpt_path)
    return ckpt_dict


def algo_name_from_checkpoint(ckpt_path=None, ckpt_dict=None):
    """
    Return algorithm name that was used to train a checkpoint or
    loaded model dictionary.

    Args:
        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

    Returns:
        algo_name (str): algorithm name

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)
    algo_name = ckpt_dict["algo_name"]
    return algo_name, ckpt_dict


def config_from_checkpoint(algo_name=None, ckpt_path=None, ckpt_dict=None, verbose=False):
    """
    Helper function to restore config from a checkpoint file or loaded model dictionary.

    Args:
        algo_name (str): Algorithm name.

        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        verbose (bool): if True, include print statements

    Returns:
        config (dict): Raw loaded configuration, without properties replaced.

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)
    if algo_name is None:
        algo_name, _ = algo_name_from_checkpoint(ckpt_dict=ckpt_dict)

    if verbose:
        print("============= Loaded Config =============")
        print(ckpt_dict['config'])

    # restore config from loaded model dictionary
    config_json = ckpt_dict['config']
    config = config_factory(algo_name, dic=json.loads(config_json))

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    return config, ckpt_dict


def policy_from_checkpoint(device=None, ckpt_path=None, ckpt_dict=None, verbose=False):
    """
    This function restores a trained policy from a checkpoint file or
    loaded model dictionary.

    Args:
        device (torch.device): if provided, put model on this device

        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        verbose (bool): if True, include print statements

    Returns:
        model (RolloutPolicy): instance of Algo that has the saved weights from
            the checkpoint file, and also acts as a policy that can easily
            interact with an environment in a training loop

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)

    # algo name and config from model dict
    algo_name, _ = algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=verbose)

    # read config to set up metadata for observation types (e.g. detecting image observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # env meta from model dict to get info needed to create model
    env_meta = ckpt_dict["env_metadata"]
    shape_meta = ckpt_dict["shape_metadata"]

    # maybe restore observation normalization stats
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])

    if device is None:
        # get torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # create model and load weights
    model = algo_factory(
        algo_name,
        config,
        modality_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
    if verbose:
        print("============= Loaded Policy =============")
        print(model)
    return model, ckpt_dict


def env_from_checkpoint(ckpt_path=None, ckpt_dict=None, env_name=None, render=False, render_offscreen=False, verbose=False):
    """
    Creates an environment using the metadata saved in a checkpoint.

    Args:
        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        env_name (str): if provided, override environment name saved in checkpoint

        render (bool): if True, environment supports on-screen rendering

        render_offscreen (bool): if True, environment supports off-screen rendering. This
            is forced to be True if saved model uses image observations.

    Returns:
        env (EnvBase instance): environment created using checkpoint

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)

    # metadata from model dict to get info needed to create environment
    env_meta = ckpt_dict["env_metadata"]
    shape_meta = ckpt_dict["shape_metadata"]

    # create env from saved metadata
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, 
        render=render, 
        render_offscreen=render_offscreen,
        use_image_obs=shape_meta["use_images"],
    )
    if verbose:
        print("============= Loaded Environment =============")
        print(env)
    return env, ckpt_dict


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    From https://gist.github.com/dehowell/884204.

    Args:
        url (str): url string

    Returns:
        is_alive (bool): True if url is reachable, False otherwise
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_url(url, download_dir):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.

    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.

    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
    """

    # check if url is reachable. We need the sleep to make sure server doesn't reject subsequent requests
    assert url_is_alive(url), "@download_url got unreachable url: {}".format(url)
    time.sleep(0.5)

    # infer filename from url link
    fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=fname) as t:
        urllib.request.urlretrieve(url, filename=file_to_write, reporthook=t.update_to)
