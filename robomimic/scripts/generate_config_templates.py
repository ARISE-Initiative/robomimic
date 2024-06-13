"""
Helpful script to generate example config files for each algorithm. These should be re-generated
when new config options are added, or when default settings in the config classes are modified.
"""

import os
import json

import robomimic
from robomimic.config import get_all_registered_configs


def main():
    # store template config jsons in this directory
    target_dir = os.path.join(robomimic.__path__[0], "exps/templates/")

    # iterate through registered algorithm config classes
    all_configs = get_all_registered_configs()
    for algo_name in all_configs:
        # make config class for this algorithm
        c = all_configs[algo_name]()
        assert algo_name == c.algo_name
        # dump to json
        json_path = os.path.join(target_dir, "{}.json".format(algo_name))
        c.dump(filename=json_path)


if __name__ == "__main__":
    main()
