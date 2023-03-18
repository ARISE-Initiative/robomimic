"""
A collection of utility functions and classes for generating config jsons for hyperparameter sweeps.
"""
import argparse
import os
import json
import re
import itertools

from collections import OrderedDict
from copy import deepcopy


class ConfigGenerator(object):
    """
    Useful class to keep track of hyperparameters to sweep, and to generate
    the json configs for each experiment run.
    """
    def __init__(self, base_config_file, wandb_proj_name="debug", script_file=None, generated_config_dir=None):
        """
        Args:
            base_config_file (str): path to a base json config to use as a starting point
                for the parameter sweep.

            script_file (str): script filename to write as output
        """
        assert isinstance(base_config_file, str)
        self.base_config_file = base_config_file
        assert generated_config_dir is None or isinstance(generated_config_dir, str)
        if generated_config_dir is not None:
            generated_config_dir = os.path.expanduser(generated_config_dir)
        self.generated_config_dir = generated_config_dir
        assert script_file is None or isinstance(script_file, str)
        if script_file is None:
            self.script_file = os.path.join('~', 'tmp/tmpp.sh')
        else:
            self.script_file = script_file
        self.script_file = os.path.expanduser(self.script_file)
        self.parameters = OrderedDict()

        assert isinstance(wandb_proj_name, str)
        self.wandb_proj_name = wandb_proj_name

    def add_param(self, key, name, group, values, value_names=None):
        """
        Add parameter to the hyperparameter sweep.

        Args:
            key (str): location of parameter in the config, using hierarchical key format
                (ex. train/data = config.train.data)

            name (str): name, as it will appear in the experiment name

            group (int): group id - parameters with the same ID have their values swept
                together

            values (list): list of values to sweep over for this parameter

            value_names ([str]): if provided, strings to use in experiment name for
                each value, instead of the parameter value. This is helpful for parameters
                that may have long or large values (for example, dataset path).
        """
        if value_names is not None:
            assert len(values) == len(value_names)
        self.parameters[key] = argparse.Namespace(
            key=key, 
            name=name, 
            group=group, 
            values=values, 
            value_names=value_names,
        )

    def generate(self):
        """
        Generates json configs for the hyperparameter sweep using attributes
        @self.parameters, @self.base_config_file, and @self.script_file,
        all of which should have first been set externally by calling
        @add_param, @set_base_config_file, and @set_script_file.
        """
        assert len(self.parameters) > 0, "must add parameters using add_param first!"
        generated_json_paths = self._generate_jsons()
        self._script_from_jsons(generated_json_paths)

    def _name_for_experiment(self, base_name, parameter_values, parameter_value_names):
        """
        This function generates the name for an experiment, given one specific
        parameter setting.

        Args:
            base_name (str): base experiment name
            parameter_values (OrderedDict): dictionary that maps parameter name to
                the parameter value for this experiment run
            parameter_value_names (dict): dictionary that maps parameter name to
                the name to use for its value in the experiment name

        Returns:
            name (str): generated experiment name
        """
        name = base_name
        for k in parameter_values:
            # append parameter name and value to end of base name
            if len(self.parameters[k].name) == 0:
                # empty string indicates that naming should be skipped
                continue
            if parameter_value_names[k] is not None:
                # take name from passed dictionary
                val_str = parameter_value_names[k]
            else:
                val_str = parameter_values[k]
                if isinstance(parameter_values[k], list) or isinstance(parameter_values[k], tuple):
                    # convert list to string to avoid weird spaces and naming problems
                    val_str = "_".join([str(x) for x in parameter_values[k]])
            val_str = str(val_str)
            name += '_{}'.format(self.parameters[k].name)
            if len(val_str) > 0:
                name += '_{}'.format(val_str)
        return name

    def _get_parameter_ranges(self):
        """
        Extract parameter ranges from base json file. Also takes all possible
        combinations of the parameter ranges to generate an expanded set of values.

        Returns:
            parameter_ranges (dict): dictionary that maps the parameter to a list
                of all values it should take for each generated config. The length 
                of the list will be the total number of configs that will be
                generated from this scan.

            parameter_names (dict): dictionary that maps the parameter to a list
                of all name strings that should contribute to each invididual
                experiment's name. The length of the list will be the total 
                number of configs that will be generated from this scan.
        """

        # mapping from group id to list of indices to grab from each parameter's list 
        # of values in the parameter group
        parameter_group_indices = OrderedDict()
        for k in self.parameters:
            group_id = self.parameters[k].group
            assert isinstance(self.parameters[k].values, list)
            num_param_values = len(self.parameters[k].values)
            if group_id not in parameter_group_indices:
                parameter_group_indices[group_id] = list(range(num_param_values))
            else:
                assert len(parameter_group_indices[group_id]) == num_param_values, \
                    "error: inconsistent number of parameter values in group with id {}".format(group_id)

        keys = list(parameter_group_indices.keys())
        inds = list(parameter_group_indices.values())
        new_parameter_group_indices = OrderedDict(
            { k : [] for k in keys }
        )
        # get all combinations of the different parameter group indices
        # and then use these indices to determine the new parameter ranges
        # per member of each parameter group.
        #
        # e.g. with two parameter groups, one with two values, and another with three values
        # we have [0, 1] x [0, 1, 2] = [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
        # so the corresponding parameter group indices are [0, 0, 0, 1, 1, 1] and 
        # [0, 1, 2, 0, 1, 2], and all parameters in each parameter group are indexed
        # together using these indices, to get each parameter range.
        for comb in itertools.product(*inds):
            for i in range(len(comb)):
                new_parameter_group_indices[keys[i]].append(comb[i])
        parameter_group_indices = new_parameter_group_indices

        # use the indices to gather the parameter values to sweep per parameter
        parameter_ranges = OrderedDict()
        parameter_names = OrderedDict()
        for k in self.parameters:
            parameter_values = self.parameters[k].values
            group_id = self.parameters[k].group
            inds = parameter_group_indices[group_id]
            parameter_ranges[k] = [parameter_values[ind] for ind in inds]

            # add in parameter names if supplied
            parameter_names[k] = None
            if self.parameters[k].value_names is not None:
                par_names = self.parameters[k].value_names
                assert isinstance(par_names, list)
                assert len(par_names) == len(parameter_values)
                parameter_names[k] = [par_names[ind] for ind in inds]

        # ensure that the number of parameter settings is the same per parameter
        first_key = list(parameter_ranges.keys())[0]
        num_settings = len(parameter_ranges[first_key])
        for k in parameter_ranges:
            assert len(parameter_ranges[k]) == num_settings, "inconsistent number of values"

        return parameter_ranges, parameter_names

    def _generate_jsons(self):
        """
        Generates json configs for the hyperparameter sweep, using @self.parameters and
        @self.base_config_file.

        Returns:
            json_paths (list): list of paths to created json files, one per experiment
        """

        # base directory for saving jsons
        if self.generated_config_dir:
            base_dir = self.generated_config_dir
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
        else:
            base_dir = os.path.abspath(os.path.dirname(self.base_config_file))

        # read base json
        base_config = load_json(self.base_config_file, verbose=False)

        # base exp name from this base config
        base_exp_name = base_config['experiment']['name']

        # use base json to determine the parameter ranges
        parameter_ranges, parameter_names = self._get_parameter_ranges()

        # iterate through each parameter setting to create each json
        first_key = list(parameter_ranges.keys())[0]
        num_settings = len(parameter_ranges[first_key])

        # keep track of path to generated jsons
        json_paths = []

        for i in range(num_settings):
            # the specific parameter setting for this experiment
            setting = { k : parameter_ranges[k][i] for k in parameter_ranges }
            maybe_parameter_names = OrderedDict()
            for k in parameter_names:
                maybe_parameter_names[k] = None
                if parameter_names[k] is not None:
                    maybe_parameter_names[k] = parameter_names[k][i]

            # experiment name from setting
            exp_name = self._name_for_experiment(
                base_name=base_exp_name, 
                parameter_values=setting, 
                parameter_value_names=maybe_parameter_names,
            )

            # copy old json, but override name, and parameter values
            json_dict = deepcopy(base_config)
            json_dict['experiment']['name'] = exp_name
            for k in parameter_ranges:
                set_value_for_key(json_dict, k, v=parameter_ranges[k][i])

            # populate list of identifying meta for logger;
            # see meta_config method in base_config.py for more info
            json_dict["experiment"]["logging"]["wandb_proj_name"] = self.wandb_proj_name
            if "meta" not in json_dict:
                json_dict["meta"] = dict()
            json_dict["meta"].update(
                hp_base_config_file=self.base_config_file,
                hp_keys=list(),
                hp_values=list(),
            )
            # logging: keep track of hyp param names and values as meta info
            for k in parameter_ranges.keys():
                key_name = self.parameters[k].name
                if key_name is not None and len(key_name) > 0:
                    if maybe_parameter_names[k] is not None:
                        value_name = maybe_parameter_names[k]
                    else:
                        value_name = setting[k]
            
                    json_dict["meta"]["hp_keys"].append(key_name)
                    json_dict["meta"]["hp_values"].append(value_name)

            # save file in same directory as old json
            json_path = os.path.join(base_dir, "{}.json".format(exp_name))
            save_json(json_dict, json_path)
            json_paths.append(json_path)

        print("Num exps:", len(json_paths))

        return json_paths

    def _script_from_jsons(self, json_paths):
        """
        Generates a bash script to run the experiments that correspond to
        the input jsons.
        """
        with open(self.script_file, 'w') as f:
            f.write("#!/bin/bash\n\n")
            for path in json_paths:
                # write python command to file
                cmd = "python train.py --config {}\n".format(path)
                
                print()
                print(cmd)
                f.write(cmd)


def load_json(json_file, verbose=True):
    """
    Simple utility function to load a json file as a dict.

    Args:
        json_file (str): path to json file to load
        verbose (bool): if True, pretty print the loaded json dictionary

    Returns:
        config (dict): json dictionary
    """
    with open(json_file, 'r') as f:
        config = json.load(f)
    if verbose:
        print('loading external config: =================')
        print(json.dumps(config, indent=4))
        print('==========================================')
    return config


def save_json(config, json_file):
    """
    Simple utility function to save a dictionary to a json file on disk.

    Args:
        config (dict): dictionary to save
        json_file (str): path to json file to write
    """
    with open(json_file, 'w') as f:
        # preserve original key ordering
        json.dump(config, f, sort_keys=False, indent=4)


def get_value_for_key(dic, k):
    """
    Get value for nested dictionary with levels denoted by "/" or ".".
    For example, if @k is "a/b", then this function returns
    @dic["a"]["b"].

    Args:
        dic (dict): a nested dictionary
        k (str): a single string meant to index several levels down into
            the nested dictionary, where levels can be denoted by "/" or
            by ".".
    Returns:
        val: the nested dictionary value for the provided key
    """
    val = dic
    subkeys = re.split('/|\.', k)
    for s in subkeys[:-1]:
        val = val[s]
    return val[subkeys[-1]]


def set_value_for_key(dic, k, v):
    """
    Set value for hierarchical dictionary with levels denoted by "/" or ".".

    Args:
        dic (dict): a nested dictionary
        k (str): a single string meant to index several levels down into
            the nested dictionary, where levels can be denoted by "/" or
            by ".".
        v: the value to set at the provided key
    """
    val = dic
    subkeys = re.split('/|\.', k) #k.split('/')
    for s in subkeys[:-1]:
        val = val[s]
    val[subkeys[-1]] = v
