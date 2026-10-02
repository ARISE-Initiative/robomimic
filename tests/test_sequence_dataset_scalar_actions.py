"""Scalar action statistics and sampled features use the same channel shape."""
import h5py
import numpy as np
import pytest
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.dataset import MetaDataset, SequenceDataset


@pytest.fixture(autouse=True)
def low_dim_observations():
    old_keys = ObsUtils.OBS_KEYS_TO_MODALITIES
    old_modalities = ObsUtils.OBS_MODALITIES_TO_KEYS
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["state"]})
    yield
    ObsUtils.OBS_KEYS_TO_MODALITIES = old_keys
    ObsUtils.OBS_MODALITIES_TO_KEYS = old_modalities


def write_dataset(path, values, scalar):
    values = np.asarray(values, dtype=np.float32)
    with h5py.File(path, "w") as file:
        for demo_index in range(2):
            demo = file.create_group(f"data/demo_{demo_index}")
            demo.attrs["num_samples"] = len(values)
            demo.create_dataset("obs/state", data=values[:, None])
            demo.create_dataset("translation", data=np.stack((10 + values, 20 + values), axis=1))
            demo.create_dataset("gripper", data=values if scalar else values[:, None])


def open_dataset(path, cache_mode, normalization):
    return SequenceDataset(
        hdf5_path=str(path),
        obs_keys=["state"],
        action_keys=["translation", "gripper"],
        dataset_keys=[],
        action_config={
            "translation": {"normalization": None},
            "gripper": {"normalization": normalization},
        },
        frame_stack=1,
        seq_length=1,
        load_next_obs=False,
        hdf5_cache_mode=cache_mode,
    )


def expected_gripper(values, method):
    values = np.asarray(values, dtype=np.float32)
    if method is None:
        return values
    if method == "min_max":
        if np.ptp(values) < 1e-4:
            return values - values.min()
        return ((values - values.min()) / np.ptp(values) * 2 - 1) * 0.999999
    if values.std() < 1e-6:
        return values - values.mean()
    return (values - values.mean()) / values.std()


@pytest.mark.parametrize("cache_mode", [None, "low_dim", "all"])
@pytest.mark.parametrize("normalization", [None, "min_max", "gaussian"])
@pytest.mark.parametrize("values", [[0.0, 1.0, 2.0], [0.5, 0.5, 0.5]])
def test_scalar_and_column_storage_have_equivalent_normalized_actions(
    tmp_path, cache_mode, normalization, values,
):
    paths = [tmp_path / "scalar.hdf5", tmp_path / "column.hdf5"]
    for path, scalar in zip(paths, [True, False]):
        write_dataset(path, values, scalar)
    datasets = []
    try:
        for path in paths:
            datasets.append(open_dataset(path, cache_mode, normalization))
        assert len(datasets[0]) == len(datasets[1]) == 6
        np.testing.assert_array_equal(datasets[0][0]["actions"], datasets[1][0]["actions"])
        for key in ("offset", "scale"):
            scalar_stats = datasets[0].get_action_normalization_stats()["gripper"][key]
            column_stats = datasets[1].get_action_normalization_stats()["gripper"][key]
            assert scalar_stats.shape == column_stats.shape == (1, 1)
            np.testing.assert_array_equal(scalar_stats, column_stats)
        expected = expected_gripper(values, normalization)
        for index in range(6):
            scalar_actions = datasets[0][index]["actions"]
            column_actions = datasets[1][index]["actions"]
            assert scalar_actions.shape == column_actions.shape == (1, 3)
            np.testing.assert_array_equal(scalar_actions, column_actions)
            source_index = index % 3
            np.testing.assert_allclose(
                scalar_actions[0],
                [10 + values[source_index], 20 + values[source_index], expected[source_index]],
                atol=1e-6,
            )
    finally:
        for dataset in datasets:
            dataset.close_and_delete_hdf5_handle()


@pytest.mark.parametrize("cache_mode", [None, "low_dim"])
@pytest.mark.parametrize("normalization", [None, "min_max", "gaussian"])
def test_meta_dataset_aggregates_scalar_and_column_action_channels(
    tmp_path, cache_mode, normalization,
):
    paths = [tmp_path / "scalar.hdf5", tmp_path / "column.hdf5"]
    for path, values, scalar in zip(paths, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [True, False]):
        write_dataset(path, values, scalar)
    datasets = []
    try:
        for path in paths:
            datasets.append(open_dataset(path, cache_mode, normalization))
        combined = MetaDataset(datasets, ds_weights=[1.0, 1.0])
        expected = expected_gripper(np.arange(6), normalization)
        assert len(combined) == 12
        for index in range(12):
            value = index % 3 + (0 if index < 6 else 3)
            action = combined[index]["actions"]
            assert action.shape == (1, 3)
            np.testing.assert_allclose(action[0], [10 + value, 20 + value, expected[value]], atol=1e-6)
    finally:
        for dataset in datasets:
            dataset.close_and_delete_hdf5_handle()
