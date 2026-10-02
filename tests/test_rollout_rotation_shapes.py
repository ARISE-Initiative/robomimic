"""Rotation conversion must retain native batch and action-chunk dimensions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

_spec = importlib.util.spec_from_file_location(
    "robomimic_base_rotation",
    Path(robomimic.__file__).parent / "algo" / "algo.py",
)
_algo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_algo)


class _LinearActionPolicy(_algo.PolicyAlgo):
    def _create_networks(self):
        self.nets["head"] = torch.nn.Linear(1, 10)
        with torch.no_grad():
            self.nets["head"].weight.zero_()
            rotation = TorchUtils.axis_angle_to_rot_6d(torch.tensor([[0.2, -0.1, 0.3]]))[0]
            self.nets["head"].bias.copy_(torch.cat((torch.tensor([0.1, 0.2, 0.3]), rotation, torch.tensor([0.4]))))
        self.chunk_length = None

    @torch.no_grad()
    def get_action(self, obs_dict, goal_dict=None):
        action = self.nets["head"](obs_dict["state"])
        if self.chunk_length is not None:
            action = action[:, None, :].expand(-1, self.chunk_length, -1)
        return action


@pytest.fixture(autouse=True)
def low_dim_observations():
    old_keys = ObsUtils.OBS_KEYS_TO_MODALITIES
    old_modalities = ObsUtils.OBS_MODALITIES_TO_KEYS
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["state"]})
    yield
    ObsUtils.OBS_KEYS_TO_MODALITIES = old_keys
    ObsUtils.OBS_MODALITIES_TO_KEYS = old_modalities


def make_rollout(conversion, chunk_length):
    config = SimpleNamespace(
        all_obs_keys=["state"],
        train=SimpleNamespace(
            action_keys=["translation", "rotation", "gripper"],
            action_config={
                "translation": {}, "gripper": {},
                "rotation": {"format": "rot_6d", "convert_at_runtime": conversion},
            },
        ),
    )
    policy = _LinearActionPolicy(
        algo_config=SimpleNamespace(optim_params={}),
        obs_config=SimpleNamespace(modalities={}), global_config=config,
        obs_key_shapes={}, ac_dim=10, device=torch.device("cpu"),
    )
    policy.chunk_length = chunk_length
    stats = {
        "translation": {"offset": np.array([[1.0, 2.0, 3.0]], dtype=np.float32), "scale": np.full((1, 3), 2.0, dtype=np.float32)},
        "rotation": {"offset": np.zeros((1, 6), dtype=np.float32), "scale": np.ones((1, 6), dtype=np.float32)},
        "gripper": {"offset": np.ones((1, 1), dtype=np.float32), "scale": np.full((1, 1), 0.5, dtype=np.float32)},
    }
    return _algo.RolloutPolicy(policy, action_normalization_stats=stats)


@pytest.mark.parametrize("conversion", ["rot_axis_angle", "rot_euler"])
@pytest.mark.parametrize(("batched", "batch_size", "chunk_length", "expected_shape"), [
    (False, 1, None, (7,)),
    (True, 1, None, (1, 7)),
    (True, 3, None, (3, 7)),
    (False, 1, 1, (1, 7)),
    (False, 1, 3, (3, 7)),
])
def test_native_rollout_preserves_action_leading_dimensions(
    conversion, batched, batch_size, chunk_length, expected_shape,
):
    rollout = make_rollout(conversion, chunk_length)
    state = np.ones((batch_size, 1) if batched else (1,), dtype=np.float32)
    action = rollout({"state": state}, batched_ob=batched)
    assert action.shape == expected_shape
    axis_angle = torch.tensor([0.2, -0.1, 0.3])
    if conversion == "rot_axis_angle":
        expected_rotation = axis_angle.numpy()
    else:
        expected_rotation = TorchUtils.matrix_to_euler_angles(
            TorchUtils.axis_angle_to_matrix(axis_angle), convention="XYZ",
        ).numpy()
    expected = np.concatenate(([1.2, 2.4, 3.6], expected_rotation, [1.2]))
    np.testing.assert_allclose(action, np.broadcast_to(expected, expected_shape), atol=1e-6)
