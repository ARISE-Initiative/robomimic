"""Terminal goal observations must not depend on the sequence padding policy."""
import h5py
import numpy as np
import pytest
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset


@pytest.fixture(autouse=True)
def low_dim_observations():
    old_keys = ObsUtils.OBS_KEYS_TO_MODALITIES
    old_modalities = ObsUtils.OBS_MODALITIES_TO_KEYS
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["state"]})
    yield
    ObsUtils.OBS_KEYS_TO_MODALITIES = old_keys
    ObsUtils.OBS_MODALITIES_TO_KEYS = old_modalities


@pytest.fixture(params=[None, "low_dim", "all"])
def cache_mode(request):
    return request.param


def write_dataset(path):
    with h5py.File(path, "w") as file:
        for demo_index, length in enumerate([7, 9]):
            demo = file.create_group(f"data/demo_{demo_index}")
            demo.attrs["num_samples"] = length
            values = (100 * demo_index + np.arange(length, dtype=np.float32))[:, None]
            demo.create_dataset("obs/state", data=values)
            demo.create_dataset("next_obs/state", data=values + 0.5)
            demo.create_dataset("actions", data=values)


def open_dataset(path, cache_mode, frame_stack, seq_length, pad_seq_length, goal_mode):
    return SequenceDataset(
        hdf5_path=str(path),
        obs_keys=["state"],
        action_keys=["actions"],
        dataset_keys=[],
        action_config={"actions": {"normalization": None}},
        frame_stack=frame_stack,
        seq_length=seq_length,
        pad_frame_stack=False,
        pad_seq_length=pad_seq_length,
        get_pad_mask=True,
        goal_mode=goal_mode,
        load_next_obs=True,
        hdf5_cache_mode=cache_mode,
    )


@pytest.mark.parametrize("frame_stack", [1, 3])
@pytest.mark.parametrize("seq_length", [1, 3])
@pytest.mark.parametrize("pad_seq_length", [False, True])
def test_last_goal_is_terminal_next_observation_for_every_window(
    tmp_path, cache_mode, frame_stack, seq_length, pad_seq_length,
):
    path = tmp_path / "goals.hdf5"
    write_dataset(path)
    dataset = open_dataset(path, cache_mode, frame_stack, seq_length, pad_seq_length, "last")
    try:
        expected_count = sum(
            length - (frame_stack - 1) - (0 if pad_seq_length else seq_length - 1)
            for length in [7, 9]
        )
        assert len(dataset) == expected_count
        for index in range(len(dataset)):
            item = dataset[index]
            demo_id = dataset._index_to_demo_id[index]
            demo_index = int(demo_id.split("_")[-1])
            length = [7, 9][demo_index]
            expected_goal = np.array([100 * demo_index + length - 1 + 0.5], dtype=np.float32)
            np.testing.assert_array_equal(item["goal_obs"]["state"], expected_goal)
            np.testing.assert_array_equal(item["goal_obs"]["pad_mask"], [True])
            assert item["obs"]["state"].shape[0] == frame_stack - 1 + seq_length
    finally:
        dataset.close_and_delete_hdf5_handle()


def test_disabled_goal_mode_does_not_add_goals_to_windows(tmp_path, cache_mode):
    path = tmp_path / "no_goals.hdf5"
    write_dataset(path)
    dataset = open_dataset(path, cache_mode, 3, 3, False, None)
    try:
        for index in range(len(dataset)):
            item = dataset[index]
            assert "goal_obs" not in item
            assert item["obs"]["state"].shape == (5, 1)
    finally:
        dataset.close_and_delete_hdf5_handle()
