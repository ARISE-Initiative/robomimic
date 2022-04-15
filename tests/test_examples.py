"""
Tests for the provided examples in the repository. Excludes stdout output 
by default (pass --verbose to see stdout output).
"""
import argparse
import traceback
import os
import subprocess
import time
import h5py
import numpy as np
import torch
from collections import OrderedDict
from termcolor import colored

import robomimic
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr


def test_example_script(script_name, args_string, test_name, silence=True):
    """
    Helper function to run an example script with filename @script_name and
    with test name @test_name (which will be printed to terminal with
    the stderr output of the example script).
    """

    # run example script
    stdout = subprocess.DEVNULL if silence else None
    path_to_script = os.path.join(robomimic.__path__[0], "../examples/{}".format(script_name))
    example_job = subprocess.Popen("python {} {}".format(path_to_script, args_string), 
        shell=True, stdout=stdout, stderr=subprocess.PIPE)
    example_job.wait()

    # get stderr output
    out, err = example_job.communicate()
    err = err.decode("utf-8")
    if len(err) > 0:
        ret = "maybe failed - stderr output below (if it's only from tqdm, the test passed)\n{}".format(err)
        ret = colored(ret, "red")
    else:
        ret = colored("passed", "green")
    print("{}: {}".format(test_name, ret))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    test_example_script(
        script_name="simple_config.py", 
        args_string="",
        test_name="simple-config-example", 
        silence=(not args.verbose),
    )
    test_example_script(
        script_name="simple_obs_nets.py", 
        args_string="",
        test_name="simple-obs-nets-example", 
        silence=(not args.verbose),
    )
    test_example_script(
        script_name="simple_train_loop.py", 
        args_string="",
        test_name="simple-train-loop-example", 
        silence=(not args.verbose),
    )
    # clear tmp model dir before running script
    TestUtils.maybe_remove_dir(TestUtils.temp_model_dir_path())
    test_example_script(
        script_name="train_bc_rnn.py", 
        args_string="--debug",
        test_name="train-bc-rnn-example", 
        silence=(not args.verbose),
    )
    # cleanup
    TestUtils.maybe_remove_dir(TestUtils.temp_model_dir_path())
