"""Real HDF5 regressions for frame-stack and sequence boundaries."""

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


def write_dataset(path, lengths):
    with h5py.File(path, "w") as file:
        for demo_index, length in enumerate(lengths):
            demo = file.create_group(f"data/demo_{demo_index}")
            demo.attrs["num_samples"] = length
            values = (100 * demo_index + np.arange(length, dtype=np.float32))[:, None]
            demo.create_dataset("obs/state", data=values)
            demo.create_dataset("next_obs/state", data=values + 0.5)
            demo.create_dataset("actions", data=values)
    return path


def open_dataset(path, cache_mode, pad_frame_stack, pad_seq_length):
    return SequenceDataset(
        hdf5_path=str(path),
        obs_keys=["state"],
        action_keys=["actions"],
        dataset_keys=[],
        action_config={"actions": {"normalization": None}},
        frame_stack=3,
        seq_length=2,
        pad_frame_stack=pad_frame_stack,
        pad_seq_length=pad_seq_length,
        get_pad_mask=True,
        load_next_obs=True,
        hdf5_cache_mode=cache_mode,
    )


@pytest.mark.parametrize(
    ("pad_frame_stack", "pad_seq_length", "length", "first_anchor", "count"),
    [
        (True, True, 1, 0, 1),
        (True, True, 2, 0, 2),
        (False, True, 3, 2, 1),
        (False, True, 4, 2, 2),
        (True, False, 2, 0, 1),
        (True, False, 3, 0, 2),
        (False, False, 4, 2, 1),
        (False, False, 5, 2, 2),
    ],
)
def test_boundary_windows_and_padding_masks(
    tmp_path, cache_mode, pad_frame_stack, pad_seq_length, length, first_anchor, count
):
    path = write_dataset(tmp_path / "boundary.hdf5", [length])
    dataset = open_dataset(path, cache_mode, pad_frame_stack, pad_seq_length)
    try:
        assert len(dataset) == count
        for index in range(len(dataset)):
            anchor = first_anchor + index
            frame_ids = np.arange(anchor - 2, anchor + 2)
            expected = np.clip(frame_ids, 0, length - 1)[:, None].astype(np.float32)
            mask = ((frame_ids >= 0) & (frame_ids < length))[:, None]
            item = dataset[index]
            np.testing.assert_array_equal(item["actions"], expected)
            np.testing.assert_array_equal(item["obs"]["state"], expected)
            np.testing.assert_array_equal(item["next_obs"]["state"], expected + 0.5)
            np.testing.assert_array_equal(item["pad_mask"], mask)
            np.testing.assert_array_equal(item["obs"]["pad_mask"], mask)
            assert item["index"] == index
    finally:
        dataset.close_and_delete_hdf5_handle()


@pytest.mark.parametrize(
    ("pad_frame_stack", "pad_seq_length", "length"),
    [(False, True, 1), (False, True, 2), (True, False, 1), (False, False, 3)],
)
def test_unsampleable_demonstration_is_rejected_during_construction(
    tmp_path, cache_mode, pad_frame_stack, pad_seq_length, length
):
    path = write_dataset(tmp_path / "too_short.hdf5", [length])
    with pytest.raises(ValueError, match=f"Demo demo_0 has {length} samples"):
        open_dataset(path, cache_mode, pad_frame_stack, pad_seq_length)


def test_demo_boundaries_do_not_leak_between_global_indices(tmp_path, cache_mode):
    path = write_dataset(tmp_path / "two_demos.hdf5", [3, 4])
    dataset = open_dataset(path, cache_mode, False, True)
    try:
        assert len(dataset) == 3
        expected_windows = [[0, 1, 2, 2], [100, 101, 102, 103], [101, 102, 103, 103]]
        for index, values in enumerate(expected_windows):
            item = dataset[index]
            expected = np.array(values, dtype=np.float32)[:, None]
            np.testing.assert_array_equal(item["obs"]["state"], expected)
            np.testing.assert_array_equal(item["actions"], expected)
    finally:
        dataset.close_and_delete_hdf5_handle()
