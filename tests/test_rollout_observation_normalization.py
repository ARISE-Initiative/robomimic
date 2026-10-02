"""Train and rollout preprocess visual observations before normalization."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import torch
from robomimic.utils.dataset import SequenceDataset

# Load the complete native base module without eager registration of unrelated
# algorithms. These preprocessing tests require no simulator or actor registry.
_spec = importlib.util.spec_from_file_location(
    "robomimic_base_preprocessing",
    Path(robomimic.__file__).parent / "algo" / "algo.py",
)
_algo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_algo)


class _PreprocessingPolicy(_algo.PolicyAlgo):
    def _create_networks(self):
        self.nets["identity"] = torch.nn.Identity()


@pytest.fixture(autouse=True)
def observation_modalities():
    old_keys = ObsUtils.OBS_KEYS_TO_MODALITIES
    old_modalities = ObsUtils.OBS_MODALITIES_TO_KEYS
    ObsUtils.initialize_obs_modality_mapping_from_dict({
        "rgb": ["image"], "depth": ["depth"], "low_dim": ["state", "unused"],
    })
    yield
    ObsUtils.OBS_KEYS_TO_MODALITIES = old_keys
    ObsUtils.OBS_MODALITIES_TO_KEYS = old_modalities


def make_policy(keys):
    return _PreprocessingPolicy(
        algo_config=SimpleNamespace(optim_params={}),
        obs_config=SimpleNamespace(modalities={}),
        global_config=SimpleNamespace(all_obs_keys=keys),
        obs_key_shapes={}, ac_dim=1, device=torch.device("cpu"),
    )


def observation_data():
    return {
        "image": np.arange(240, dtype=np.uint8).reshape(4, 4, 5, 3),
        "depth": np.linspace(0.1, 0.9, 80, dtype=np.float32).reshape(4, 4, 5, 1),
        "state": np.arange(8, dtype=np.float32).reshape(4, 2),
    }


def dataset_stats(path, data, keys):
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.attrs["num_samples"] = 4
        for key in keys:
            demo.create_dataset(f"obs/{key}", data=data[key])
        demo.create_dataset("actions", data=np.zeros((4, 1), dtype=np.float32))
    dataset = SequenceDataset(
        hdf5_path=str(path), obs_keys=keys, action_keys=["actions"], dataset_keys=[],
        action_config={"actions": {"normalization": None}},
        hdf5_normalize_obs=True, load_next_obs=False,
    )
    try:
        return dataset.get_obs_normalization_stats()
    finally:
        dataset.close_and_delete_hdf5_handle()


@pytest.mark.parametrize("keys", [["image"], ["depth"], ["image", "depth", "state"]])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_rollout_matches_native_training_preprocessing(tmp_path, keys, batched, normalize):
    data = observation_data()
    policy = make_policy(keys)
    stats = dataset_stats(tmp_path / "visual.hdf5", data, keys) if normalize else None
    rollout = _algo.RolloutPolicy(policy, obs_normalization_stats=stats)
    raw_batch = {key: torch.from_numpy(data[key][1:3] if batched else data[key][2:3]) for key in keys}
    expected = policy.postprocess_batch_for_training({"obs": raw_batch}, stats)["obs"]
    raw_observation = {key: data[key][1:3] if batched else data[key][2] for key in keys}
    if normalize:
        raw_observation["unused"] = np.ones((2, 2) if batched else (2,), dtype=np.float32)
    actual = rollout._prepare_observation(raw_observation, batched_ob=batched)
    assert set(actual) == set(expected)
    for key in keys:
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize("keys", [["image"], ["depth"], ["image", "depth", "state"]])
@pytest.mark.parametrize("batched", [False, True])
def test_already_processed_visual_input_is_normalized_without_reprocessing(tmp_path, keys, batched):
    data = observation_data()
    policy = make_policy(keys)
    stats = dataset_stats(tmp_path / "processed.hdf5", data, keys)
    rollout = _algo.RolloutPolicy(policy, obs_normalization_stats=stats)
    raw_batch = {key: torch.from_numpy(data[key][1:3] if batched else data[key][2:3]) for key in keys}
    expected = policy.postprocess_batch_for_training({"obs": raw_batch}, stats)["obs"]
    raw_observation = {key: data[key][1:3] if batched else data[key][2] for key in keys}
    processed = ObsUtils.process_obs_dict(raw_observation)
    actual = rollout._prepare_observation(processed, batched_ob=batched, postprocess_visual_obs=False)
    for key in keys:
        torch.testing.assert_close(actual[key], expected[key])
