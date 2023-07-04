"""
This file contains utility classes and functions for logging to stdout, stderr,
and to tensorboard.
"""
import os
import sys
import numpy as np
from datetime import datetime
from contextlib import contextmanager
import textwrap
import time
from tqdm import tqdm
from termcolor import colored

import robomimic

# global list of warning messages can be populated with @log_warning and flushed with @flush_warnings
WARNINGS_BUFFER = []


class PrintLogger(object):
    """
    This class redirects print statements to both console and a file.
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class DataLogger(object):
    """
    Logging class to log metrics to tensorboard and/or retrieve running statistics about logged data.
    """
    def __init__(self, log_dir, config, log_tb=True, log_wandb=False):
        """
        Args:
            log_dir (str): base path to store logs
            log_tb (bool): whether to use tensorboard logging
        """
        self._tb_logger = None
        self._wandb_logger = None
        self._data = dict() # store all the scalar data logged so far

        if log_tb:
            from tensorboardX import SummaryWriter
            self._tb_logger = SummaryWriter(os.path.join(log_dir, 'tb'))

        if log_wandb:
            import wandb
            import robomimic.macros as Macros
            
            # set up wandb api key if specified in macros
            if Macros.WANDB_API_KEY is not None:
                os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY

            assert Macros.WANDB_ENTITY is not None, "WANDB_ENTITY macro is set to None." \
                    "\nSet this macro in {base_path}/macros_private.py" \
                    "\nIf this file does not exist, first run python {base_path}/scripts/setup_macros.py".format(base_path=robomimic.__path__[0])
            
            # attempt to set up wandb 10 times. If unsuccessful after these trials, don't use wandb
            num_attempts = 10
            for attempt in range(num_attempts):
                try:
                    # set up wandb
                    self._wandb_logger = wandb

                    self._wandb_logger.init(
                        entity=Macros.WANDB_ENTITY,
                        project=config.experiment.logging.wandb_proj_name,
                        name=config.experiment.name,
                        dir=log_dir,
                        mode=("offline" if attempt == num_attempts - 1 else "online"),
                    )

                    # set up info for identifying experiment
                    wandb_config = {k: v for (k, v) in config.meta.items() if k not in ["hp_keys", "hp_values"]}
                    for (k, v) in zip(config.meta["hp_keys"], config.meta["hp_values"]):
                        wandb_config[k] = v
                    if "algo" not in wandb_config:
                        wandb_config["algo"] = config.algo_name
                    self._wandb_logger.config.update(wandb_config)

                    break
                except Exception as e:
                    log_warning("wandb initialization error (attempt #{}): {}".format(attempt + 1, e))
                    self._wandb_logger = None
                    time.sleep(30)

    def record(self, k, v, epoch, data_type='scalar', log_stats=False):
        """
        Record data with logger.
        Args:
            k (str): key string
            v (float or image): value to store
            epoch: current epoch number
            data_type (str): the type of data. either 'scalar' or 'image'
            log_stats (bool): whether to store the mean/max/min/std for all data logged so far with key k
        """

        assert data_type in ['scalar', 'image']

        if data_type == 'scalar':
            # maybe update internal cache if logging stats for this key
            if log_stats or k in self._data: # any key that we're logging or previously logged
                if k not in self._data:
                    self._data[k] = []
                self._data[k].append(v)

        # maybe log to tensorboard
        if self._tb_logger is not None:
            if data_type == 'scalar':
                self._tb_logger.add_scalar(k, v, epoch)
                if log_stats:
                    stats = self.get_stats(k)
                    for (stat_k, stat_v) in stats.items():
                        stat_k_name = '{}-{}'.format(k, stat_k)
                        self._tb_logger.add_scalar(stat_k_name, stat_v, epoch)
            elif data_type == 'image':
                self._tb_logger.add_images(k, img_tensor=v, global_step=epoch, dataformats="NHWC")

        if self._wandb_logger is not None:
            try:
                if data_type == 'scalar':
                    self._wandb_logger.log({k: v}, step=epoch)
                    if log_stats:
                        stats = self.get_stats(k)
                        for (stat_k, stat_v) in stats.items():
                            self._wandb_logger.log({"{}/{}".format(k, stat_k): stat_v}, step=epoch)
                elif data_type == 'image':
                    raise NotImplementedError
            except Exception as e:
                log_warning("wandb logging: {}".format(e))

    def get_stats(self, k):
        """
        Computes running statistics for a particular key.
        Args:
            k (str): key string
        Returns:
            stats (dict): dictionary of statistics
        """
        stats = dict()
        stats['mean'] = np.mean(self._data[k])
        stats['std'] = np.std(self._data[k])
        stats['min'] = np.min(self._data[k])
        stats['max'] = np.max(self._data[k])
        return stats

    def close(self):
        """
        Run before terminating to make sure all logs are flushed
        """
        if self._tb_logger is not None:
            self._tb_logger.close()

        if self._wandb_logger is not None:
            self._wandb_logger.finish()


class custom_tqdm(tqdm):
    """
    Small extension to tqdm to make a few changes from default behavior.
    By default tqdm writes to stderr. Instead, we change it to write
    to stdout.
    """
    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super(custom_tqdm, self).__init__(*args, file=sys.stdout, **kwargs)


@contextmanager
def silence_stdout():
    """
    This contextmanager will redirect stdout so that nothing is printed
    to the terminal. Taken from the link below:

    https://stackoverflow.com/questions/6735917/redirecting-stdout-to-nothing-in-python
    """
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target


def log_warning(message, color="yellow", print_now=True):
    """
    This function logs a warning message by recording it in a global warning buffer.
    The global registry will be maintained until @flush_warnings is called, at
    which point the warnings will get printed to the terminal.

    Args:
        message (str): warning message to display
        color (str): color of message - defaults to "yellow"
        print_now (bool): if True (default), will print to terminal immediately, in
            addition to adding it to the global warning buffer
    """
    global WARNINGS_BUFFER
    buffer_message = colored("ROBOMIMIC WARNING(\n{}\n)".format(textwrap.indent(message, "    ")), color)
    WARNINGS_BUFFER.append(buffer_message)
    if print_now:
        print(buffer_message)


def flush_warnings():
    """
    This function flushes all warnings from the global warning buffer to the terminal and
    clears the global registry.
    """
    global WARNINGS_BUFFER
    for msg in WARNINGS_BUFFER:
        print(msg)
    WARNINGS_BUFFER = []
