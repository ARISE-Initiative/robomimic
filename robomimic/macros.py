"""
Set of global variables shared across robomimic
"""
# Sets debugging mode. Should be set at top-level script so that internal
# debugging functionalities are made active
DEBUG = False

# Whether to visualize the before & after of an observation randomizer
VISUALIZE_RANDOMIZER = False

# wandb entity (eg. username or team name)
WANDB_ENTITY = None

# wandb api key (obtain from https://wandb.ai/authorize)
# alternatively, set up wandb from terminal with `wandb login`
WANDB_API_KEY = None


### Slack Notifications ###

# Token for sending slack notifications
SLACK_TOKEN = None

# User ID for user that should receive slack notifications
SLACK_USER_ID = None


### Local Sync Settings ###

# By specifying this path, you can sync the most important results of training back to this folder
RESULTS_SYNC_PATH = None

# This will be automatically populated.
RESULTS_SYNC_PATH_ABS = None


### MagLev and NGC Cluster Settings ###

# Whether training is happening on MagLev / NGC (should set this on repos hosted in MagLev / NGC scratch space or in Docker)
USE_MAGLEV = False
USE_NGC = False

# When using MagLev / NGC, sync the most important results of training back to this directory in scratch space. 
# This path should be relative to the base scratch space directory (for MagLev) or an absolute path (for NGC)
MAGLEV_SCRATCH_SYNC_PATH = None
NGC_SCRATCH_SYNC_PATH = None


try:
    from robomimic.macros_private import *
except ImportError:
    from robomimic.utils.log_utils import log_warning
    import robomimic
    log_warning(
        "No private macro file found!"\
        "\nIt is recommended to use a private macro file"\
        "\nTo setup, run: python {}/scripts/setup_macros.py".format(robomimic.__path__[0])
    )
