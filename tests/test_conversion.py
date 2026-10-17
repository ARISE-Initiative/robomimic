"""
Tests for the dataset conversion scripts (delta <-> absolute actions).
Excludes stdout output by default (pass --verbose to see stdout output).
"""
import argparse
import shutil
import traceback
import os
import h5py
import numpy as np
from termcolor import colored

import robomimic
import robomimic.utils.test_utils as TestUtils
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr


# tolerance for the delta -> absolute -> delta round trip. Converting between
# rotation matrices and axis-angle representations, and float32 storage in the
# dataset, introduces small numerical errors, so the round trip is not exact.
ROUNDTRIP_ATOL = 1e-4


def test_delta_abs_roundtrip(silence=True):
    """
    Test that converting delta actions to absolute actions
    (robosuite_add_absolute_actions.py) and then converting the result back
    to delta actions (robosuite_add_delta_actions.py) recovers the original
    delta actions.
    """
    context = silence_stdout() if silence else dummy_context_mgr()
    with context:

        try:
            from robomimic.scripts.conversion.robosuite_add_absolute_actions import add_absolute_actions_to_dataset
            from robomimic.scripts.conversion.robosuite_add_delta_actions import add_delta_actions_to_dataset

            # copy the small test dataset, since the conversion scripts
            # modify the dataset in place
            dataset_path = TestUtils.temp_dataset_path()
            shutil.copyfile(TestUtils.example_dataset_path(), dataset_path)

            # delta actions -> absolute actions -> delta actions
            add_absolute_actions_to_dataset(dataset=dataset_path, num_workers=1)
            add_delta_actions_to_dataset(dataset=dataset_path, num_workers=1)

            # compare recovered delta actions with the original ones
            with h5py.File(dataset_path, "r") as f:
                for ep in f["data"]:
                    original_actions = f["data/{}/actions".format(ep)][:]
                    recovered_actions = f["data/{}/actions_delta".format(ep)][:]
                    assert original_actions.shape == recovered_actions.shape
                    max_err = np.abs(original_actions - recovered_actions).max()
                    assert max_err < ROUNDTRIP_ATOL, (
                        "round trip error {} exceeds tolerance {} for demo {}".format(
                            max_err, ROUNDTRIP_ATOL, ep
                        )
                    )

            os.remove(dataset_path)
            ret = colored("passed!", "green")
        except Exception as e:
            ret = colored("failed with error:\n{}\n\n{}".format(e, traceback.format_exc()), "red")

        print("\nround trip (delta -> abs -> delta) conversion test: {}".format(ret))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    test_delta_abs_roundtrip(silence=not args.verbose)
