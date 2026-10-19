"""Unit tests for full-trajectory access to demonstration datasets."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset


class TestTrajectoryAccess(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "demonstrations.hdf5"
        self.values = np.arange(5, dtype=np.float32)[:, None]
        self.addCleanup(setattr, ObsUtils, "OBS_KEYS_TO_MODALITIES", ObsUtils.OBS_KEYS_TO_MODALITIES)
        self.addCleanup(setattr, ObsUtils, "OBS_MODALITIES_TO_KEYS", ObsUtils.OBS_MODALITIES_TO_KEYS)
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["state"]}})
        with h5py.File(self.path, "w") as file:
            demo = file.create_group("data/demo_0")
            demo.attrs["num_samples"] = len(self.values)
            demo.create_dataset("actions", data=self.values)
            demo.create_dataset("rewards", data=self.values[:, 0] + 10)
            demo.create_dataset("obs/state", data=self.values)
            demo.create_dataset("next_obs/state", data=self.values + 1)

    def make_dataset(self, **kwargs):
        dataset = SequenceDataset(
            hdf5_path=str(self.path),
            obs_keys=("state",),
            action_keys=("actions",),
            dataset_keys=("actions", "rewards"),
            action_config={"actions": {}},
            seq_length=2,
            get_pad_mask=True,
            **kwargs,
        )
        self.addCleanup(dataset.close_and_delete_hdf5_handle)
        return dataset

    def test_full_trajectory_is_aligned_without_training_frame_stack(self):
        """Trajectory arrays contain each recorded timestep once, in temporal order."""
        for cache in (None, "low_dim"):
            for frame_stack in (1, 3):
                for pad in (False, True):
                    for load_next_obs in (False, True):
                        with self.subTest(
                            cache=cache,
                            frame_stack=frame_stack,
                            pad=pad,
                            load_next_obs=load_next_obs,
                        ):
                            dataset = self.make_dataset(
                                hdf5_cache_mode=cache,
                                frame_stack=frame_stack,
                                pad_frame_stack=pad,
                                load_next_obs=load_next_obs,
                            )
                            trajectory = dataset.get_trajectory_at_index(0)
                            np.testing.assert_array_equal(trajectory["actions"], self.values)
                            np.testing.assert_array_equal(trajectory["rewards"], self.values[:, 0] + 10)
                            np.testing.assert_array_equal(trajectory["obs"]["state"], self.values)
                            np.testing.assert_array_equal(
                                trajectory["pad_mask"], np.ones((len(self.values), 1), dtype=bool)
                            )
                            np.testing.assert_array_equal(
                                trajectory["obs"]["pad_mask"], trajectory["pad_mask"]
                            )
                            if load_next_obs:
                                np.testing.assert_array_equal(
                                    trajectory["next_obs"]["state"], self.values + 1
                                )
                                np.testing.assert_array_equal(
                                    trajectory["next_obs"]["pad_mask"], trajectory["pad_mask"]
                                )
                            else:
                                self.assertNotIn("next_obs", trajectory)
                            self.assertEqual(trajectory["ep"], "demo_0")

    def test_training_sample_still_stacks_frames(self):
        """Full-trajectory access does not change frame stacking in training samples."""
        for pad in (False, True):
            with self.subTest(pad=pad):
                dataset = self.make_dataset(frame_stack=3, pad_frame_stack=pad)
                trajectory = dataset.get_trajectory_at_index(0)
                sample = dataset[0]
                expected = np.array([0, 0, 0, 1] if pad else [0, 1, 2, 3], dtype=np.float32)[:, None]
                np.testing.assert_array_equal(sample["actions"], expected)
                np.testing.assert_array_equal(sample["obs"]["state"], expected)
                self.assertEqual(len(trajectory["actions"]), len(self.values))


if __name__ == "__main__":
    unittest.main()
