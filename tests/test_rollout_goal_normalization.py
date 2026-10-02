"""Observation and goal groups need not contain the same normalized keys."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import torch
from robomimic.utils.dataset import SequenceDataset


def native_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_root = Path(robomimic.__file__).parent
_algo = native_module("robomimic_goal_base", _root / "algo" / "algo.py")
Config = native_module("robomimic_goal_config", _root / "config" / "config.py").Config


class _GoalPolicy(_algo.PolicyAlgo):
    def _create_networks(self):
        self.nets["head"] = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.nets["head"].weight.copy_(torch.eye(3))

    @torch.no_grad()
    def get_action(self, obs_dict, goal_dict=None):
        features = torch.cat((obs_dict["state"], goal_dict["target"]), dim=-1)
        return self.nets["head"](features)


@pytest.fixture(autouse=True)
def low_dim_observations():
    old_keys = ObsUtils.OBS_KEYS_TO_MODALITIES
    old_modalities = ObsUtils.OBS_MODALITIES_TO_KEYS
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["state", "target", "unused"]})
    yield
    ObsUtils.OBS_KEYS_TO_MODALITIES = old_keys
    ObsUtils.OBS_MODALITIES_TO_KEYS = old_modalities


def make_policy():
    return _GoalPolicy(
        algo_config=Config({"optim_params": {}}),
        obs_config=Config({"modalities": {
            "obs": {"low_dim": ["state"]}, "goal": {"low_dim": ["target"]},
        }}),
        global_config=Config({"all_obs_keys": ["state", "target"]}),
        obs_key_shapes={"state": (2,), "target": (1,)}, ac_dim=3,
        device=torch.device("cpu"),
    )


def observation_data():
    return {
        "state": np.array([[2, 4], [3, 6], [4, 8], [5, 10]], dtype=np.float32),
        "target": np.array([[10], [12], [14], [16]], dtype=np.float32),
    }


def normalization_stats(path, data):
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.attrs["num_samples"] = 4
        for key, values in data.items():
            demo.create_dataset(f"obs/{key}", data=values)
        demo.create_dataset("actions", data=np.zeros((4, 1), dtype=np.float32))
    dataset = SequenceDataset(
        hdf5_path=str(path), obs_keys=list(data), action_keys=["actions"], dataset_keys=[],
        action_config={"actions": {"normalization": None}},
        hdf5_normalize_obs=True, load_next_obs=False,
    )
    try:
        return dataset.get_obs_normalization_stats()
    finally:
        dataset.close_and_delete_hdf5_handle()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("observation_includes_goal_key", [False, True])
@pytest.mark.parametrize("goal_includes_observation_key", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_goal_specific_key_groups_match_training_and_rollout(
    tmp_path, batched, observation_includes_goal_key, goal_includes_observation_key, normalize,
):
    data = observation_data()
    policy = make_policy()
    assert list(policy.obs_shapes) == ["state"]
    assert list(policy.goal_shapes) == ["target"]
    stats = normalization_stats(tmp_path / "goal.hdf5", data) if normalize else None
    rollout = _algo.RolloutPolicy(policy, obs_normalization_stats=stats)
    obs_batch = torch.from_numpy(data["state"][1:3] if batched else data["state"][2:3])
    goal_batch = torch.from_numpy(data["target"][2:4] if batched else data["target"][3:4])
    training = policy.postprocess_batch_for_training({
        "obs": {"state": obs_batch}, "goal_obs": {"target": goal_batch},
    }, stats)
    expected = torch.cat((training["obs"]["state"], training["goal_obs"]["target"]), dim=-1).numpy()
    observation = {"state": data["state"][1:3] if batched else data["state"][2]}
    if observation_includes_goal_key:
        observation["target"] = data["target"][1:3] if batched else data["target"][2]
    goal = {"target": data["target"][2:4] if batched else data["target"][3]}
    if goal_includes_observation_key:
        goal["state"] = data["state"][2:4] if batched else data["state"][3]
    if normalize:
        observation["unused"] = np.ones((2, 1) if batched else (1,), dtype=np.float32)
        goal["unused"] = np.ones((2, 1) if batched else (1,), dtype=np.float32)
    action = rollout(observation, goal, batched_ob=batched)
    assert action.shape == ((2, 3) if batched else (3,))
    np.testing.assert_allclose(action, expected if batched else expected[0], atol=1e-6)
