"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.
"""
import os
import shutil
import tempfile
import h5py
import json
import time
import urllib.request
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import torch

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.lang_utils as LangUtils
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


def get_demos_for_filter_key(hdf5_path, filter_key):
    """
    Gets demo keys that correspond to a particular filter key.

    Args:
        hdf5_path (str): path to hdf5 file
        filter_key (str): name of filter key

    Returns:
        demo_keys ([str]): list of demonstration keys that
            correspond to this filter key. For example, ["demo_0", 
            "demo_1"].
    """
    f = h5py.File(hdf5_path, "r")
    demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
    f.close()
    return demo_keys


def get_env_metadata_from_dataset(dataset_path, set_env_specific_obs_processors=True):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

        set_env_specific_obs_processors (bool): environment might have custom rules for how to process
            observations - if this flag is true, make sure ObsUtils will use these custom settings. This
            is a good place to do this operation to make sure it happens before loading data, running a 
            trained model, etc.

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    if "env_lang" in env_meta["env_kwargs"]: del env_meta["env_kwargs"]["env_lang"]

    f.close()
    if set_env_specific_obs_processors:
        # handle env-specific custom observation processing logic
        EnvUtils.set_env_specific_obs_processing(env_meta=env_meta)
    return env_meta


def get_shape_metadata_from_dataset(dataset_config, action_keys, all_obs_keys=None, verbose=False):
    """
    Retrieves shape metadata from dataset.

    Args:
        dataset_config (str): config for dataset
        action_keys (list): list of all action key strings
        all_obs_keys (list): list of all modalities used by the model. If not provided, all modalities
            present in the file are used.
        verbose (bool): if True, include print statements

    Returns:
        shape_meta (dict): shape metadata. Contains the following keys:

            :`'ac_dim'`: action space dimension
            :`'all_shapes'`: dictionary that maps observation key string to shape
            :`'all_obs_keys'`: list of all observation modalities used
            :`'use_images'`: bool, whether or not image modalities are present
            :`'use_depths'`: bool, whether or not depth modalities are present
    """

    shape_meta = {}

    # read demo file for some metadata
    dataset_path = os.path.expanduser(dataset_config["path"])
    f = h5py.File(dataset_path, "r")
    
    demo_id = list(f["data"].keys())[0]
    demo = f["data/{}".format(demo_id)]
    
    for key in action_keys:
        assert len(demo[key].shape) == 2 # shape should be (B, D)
    action_dim = sum([demo[key].shape[1] for key in action_keys])
    shape_meta["ac_dim"] = action_dim

    # observation dimensions
    all_shapes = OrderedDict()

    if all_obs_keys is None:
        # use all modalities present in the file
        all_obs_keys = [k for k in demo["obs"]]

    for k in sorted(all_obs_keys):
        if k == LangUtils.LANG_EMB_OBS_KEY:
            # NOTE: currently supporting fixed language embedding per dataset
            ## that is fetched from dataset config and not from file
            assert "lang" in dataset_config, "Expected 'lang' key in dataset config."
            initial_shape = LangUtils.get_lang_emb_shape()
        else:
            initial_shape = demo["obs/{}".format(k)].shape[1:]
        if verbose:
            print("obs key {} with shape {}".format(k, initial_shape))
        # Store processed shape for each obs key
        all_shapes[k] = ObsUtils.get_processed_shape(
            obs_modality=ObsUtils.OBS_KEYS_TO_MODALITIES[k],
            input_shape=initial_shape,
        )

    f.close()

    shape_meta['all_shapes'] = all_shapes
    shape_meta['all_obs_keys'] = all_obs_keys
    shape_meta['use_images'] = ObsUtils.has_modality("rgb", all_obs_keys)
    shape_meta['use_depths'] = ObsUtils.has_modality("depth", all_obs_keys)

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
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
    else:
        ckpt_dict = torch.load(ckpt_path, weights_only=False)
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


def update_config(cfg):
    """
    Updates the config for backwards-compatibility if it uses outdated configurations.

    See https://github.com/ARISE-Initiative/robomimic/releases/tag/v0.2.0 for more info.

    Args:
        cfg (dict): Raw dictionary of config values
    """
    # Check if image modality is defined -- this means we're using an outdated config
    # Note: There may be a nested hierarchy, so we possibly check all the nested obs cfgs which can include
    # e.g. a planner and actor for HBC

    def find_obs_dicts_recursively(dic):
        dics = []
        if "modalities" in dic:
            dics.append(dic)
        else:
            for child_dic in dic.values():
                dics += find_obs_dicts_recursively(child_dic)
        return dics

    obs_cfgs = find_obs_dicts_recursively(cfg["observation"])
    for obs_cfg in obs_cfgs:
        modalities = obs_cfg["modalities"]

        found_img = False
        for modality_group in ("obs", "subgoal", "goal"):
            if modality_group in modalities:
                img_modality = modalities[modality_group].pop("image", None)
                if img_modality is not None:
                    found_img = True
                    modalities[modality_group]["rgb"] = img_modality

        if found_img:
            # Also need to map encoder kwargs correctly
            old_encoder_cfg = obs_cfg.pop("encoder")

            # Create new encoder entry for RGB
            rgb_encoder_cfg = {
                "core_class": "VisualCore",
                "core_kwargs": {
                    "backbone_kwargs": dict(),
                    "pool_kwargs": dict(),
                },
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": dict(),
            }

            if "visual_feature_dimension" in old_encoder_cfg:
                rgb_encoder_cfg["core_kwargs"]["feature_dimension"] = old_encoder_cfg["visual_feature_dimension"]

            if "visual_core" in old_encoder_cfg:
                rgb_encoder_cfg["core_kwargs"]["backbone_class"] = old_encoder_cfg["visual_core"]

            for kwarg in ("pretrained", "input_coord_conv"):
                if "visual_core_kwargs" in old_encoder_cfg and kwarg in old_encoder_cfg["visual_core_kwargs"]:
                    rgb_encoder_cfg["core_kwargs"]["backbone_kwargs"][kwarg] = old_encoder_cfg["visual_core_kwargs"][kwarg]

            # Optionally add pooling info too
            if old_encoder_cfg.get("use_spatial_softmax", True):
                rgb_encoder_cfg["core_kwargs"]["pool_class"] = "SpatialSoftmax"

            for kwarg in ("num_kp", "learnable_temperature", "temperature", "noise_std"):
                if "spatial_softmax_kwargs" in old_encoder_cfg and kwarg in old_encoder_cfg["spatial_softmax_kwargs"]:
                    rgb_encoder_cfg["core_kwargs"]["pool_kwargs"][kwarg] = old_encoder_cfg["spatial_softmax_kwargs"][kwarg]

            # Update obs randomizer as well
            for kwarg in ("obs_randomizer_class", "obs_randomizer_kwargs"):
                if kwarg in old_encoder_cfg:
                    rgb_encoder_cfg[kwarg] = old_encoder_cfg[kwarg]

            # Store rgb config
            obs_cfg["encoder"] = {"rgb": rgb_encoder_cfg}

            # Also add defaults for low dim
            obs_cfg["encoder"]["low_dim"] = {
                "core_class": None,
                "core_kwargs": {
                    "backbone_kwargs": dict(),
                    "pool_kwargs": dict(),
                },
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": dict(),
            }


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

    # restore config from loaded model dictionary
    config_dict = json.loads(ckpt_dict['config'])
    update_config(cfg=config_dict)

    if verbose:
        print("============= Loaded Config =============")
        print(json.dumps(config_dict, indent=4))

    config = config_factory(algo_name, dic=config_dict)

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

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # shape meta from model dict to get info needed to create model
    shape_meta = ckpt_dict["shape_metadata"]

    # maybe restore observation normalization stats
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])

    # maybe restore action normalization stats
    action_normalization_stats = ckpt_dict.get("action_normalization_stats", None)
    if action_normalization_stats is not None:
        for m in action_normalization_stats:
            for k in action_normalization_stats[m]:
                action_normalization_stats[m][k] = np.array(action_normalization_stats[m][k])

    if device is None:
        # get torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # create model and load weights
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    model = RolloutPolicy(
        model,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats
    )
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
        env_name=env_name, 
        render=render, 
        render_offscreen=render_offscreen,
        use_image_obs=shape_meta.get("use_images", False),
        use_depth_obs=shape_meta.get("use_depths", False),
    )
    config, _ = config_from_checkpoint(algo_name=ckpt_dict["algo_name"], ckpt_dict=ckpt_dict, verbose=False)
    env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment wrapper, if applicable
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


def download_url(url, download_dir, check_overwrite=True):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.

    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.

    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """

    # check if url is reachable. We need the sleep to make sure server doesn't reject subsequent requests
    assert url_is_alive(url), "@download_url got unreachable url: {}".format(url)
    time.sleep(0.5)

    # infer filename from url link
    fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    # If we're checking overwrite and the path already exists,
    # we ask the user to verify that they want to overwrite the file
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(f"Warning: file {file_to_write} already exists. Overwrite? y/n\n")
        assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=fname) as t:
        urllib.request.urlretrieve(url, filename=file_to_write, reporthook=t.update_to)


def download_file_from_hf(repo_id, filename, download_dir, check_overwrite=True):
    """
    Downloads a file from Hugging Face.
    Reference: https://huggingface.co/docs/huggingface_hub/main/en/guides/download
    Example usage:
        repo_id = "amandlek/mimicgen_datasets"
        filename = "core/coffee_d0.hdf5"
        download_dir = "/tmp"
        download_file_from_hf(repo_id, filename, download_dir, check_overwrite=True)
    Args:
        repo_id (str): Hugging Face repo ID
        filename (str): path to file in repo
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """
    with tempfile.TemporaryDirectory() as td:
        # first check if file exists
        file_to_write = os.path.join(download_dir, os.path.basename(filename))
        if check_overwrite and os.path.exists(file_to_write):
            user_response = input(f"Warning: file {file_to_write} already exists. Overwrite? y/n\n")
            assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

        # note: fpath is a pointer, so we need to look up the actual path on disk and then move it
        fpath = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=td)
        shutil.move(os.path.realpath(fpath), file_to_write)
