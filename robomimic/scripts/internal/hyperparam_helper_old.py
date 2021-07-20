"""
A useful script for generating json files and shell scripts for conducting parameter scans.
The script takes a path to a base json file as an argument and a shell file name.
It generates a set of new json files in the same folder as the base json file, and 
a shell file script that contains commands to run for each experiment.

Instructions:

(1) Decide on what json parameters you would like to sweep over, and fill those in as 
    keys in @PARAMETERS below, taking note of the hierarchical key
    formatting using "/". Fill in corresponding values for each - these will
    be used in creating the experiment names, and for determining the range
    of values to sweep. Make sure that the list of values for each parameter is 
    equal in size (unless you want to use the @combinations functionality).

(2) Start with a base json that specifies a complete set of parameters for a single 
    run (including those that you don't want to change).

(3) That's it! Run the script with the following arguments. The new experiment jsons
    will be put into the same directory as your base json, and the script to run them
    will be created at the path you specify below.
    
    Args:
        algo (str): desired algorithm (e.g. bc, bcq, etc.)

        config (str): path to the base json you created above

        script (str): path to generated script to run the experiments

        combinations (bool): if True, generate experiments for all possible
            parameter combinations.
"""

import argparse
import os
import json
import itertools

from collections import OrderedDict
from copy import deepcopy

# Parameter Dictionary that should map from parameters (specified in hierarchical key format)
# to tuple values that should specify:
#     (1) base name for each parameter variation when naming each experiment
#     (2) list of parameter values to sweep
#     (3) parameter group identifier - only used when @combinations argument is set. 
#         Assign parameters the same identifier to sweep over them together. Combinations
#         are only taken between parameter groups.


# BC-RNN tuning
PARAMETERS = OrderedDict({
    "train/data": 
        (
            "ds", 
            [
                # "~/Desktop/lift_suboptimal/states1.hdf5",
                # "~/Desktop/roboturk_v1/RoboTurkPilot/bins-Can/states.hdf5", # "fastest_225 filter key"
                # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states.hdf5",
                # "~/Desktop/d4rl_manip/lift_v1_subopt_careless/states.hdf5",
                # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_1.hdf5",

                # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_2_succ.hdf5",

                # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_images_done_2.hdf5",

                # "~/Desktop/robosuite_v1_demos/panda_lift/states_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/sawyer_lift/states_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_done_2.hdf5",

                # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_done_2_combined.hdf5",
                # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_done_2_combined.hdf5",

                # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_v1_2_1/states_done_2_all.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/ajay/states_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/josiah/states_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/roberto/states_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/yuke/states_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/states_done_2.hdf5",

                "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/ajay/states_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/states_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/roberto/states_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/yuke/states_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/states_done_2.hdf5",

                # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can_paired2/states_done_2.hdf5",

                # "~/Desktop/robosuite_v1_demos/panda_lift/states_images_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/sawyer_lift/states_images_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_images_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_images_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_images_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_images_done_2.hdf5",

                # "~/Desktop/d4rl/converted/antmaze_umaze_diverse.hdf5",
                # "~/Desktop/d4rl/converted/antmaze_medium_diverse.hdf5",

                # "~/Desktop/d4rl/converted/pen-v0_demos_clipped.hdf5",
                # "~/Desktop/d4rl/converted/door-v0_demos_clipped.hdf5",
                # "~/Desktop/d4rl/converted/hammer-v0_demos_clipped.hdf5",
                # "~/Desktop/d4rl/converted/relocate-v0_demos_clipped.hdf5",
                # "~/Desktop/d4rl/converted/pen-demos-v0-bc-combined.hdf5",

                # "~/Desktop/d4rl/converted/kitchen_complete_v0.hdf5",
                # "~/Desktop/d4rl/converted/kitchen_partial_v0.hdf5",
                # "~/Desktop/d4rl/converted/kitchen_mixed_v0.hdf5",
            ],
            0,
            [
                # "lift_subopt",
                # "cans_top_225",
                # "lift_subopt_paired1",
                # "lift_subopt_careless",
                # "lift_subopt_paired2",

                # "lift_subopt_paired2_done_2_succ",

                # "lift_subopt_paired2_image",

                # "v1_ds_panda_lift",
                # "v1_ds_sawyer_lift",
                # "v1_ds_panda_pick_place_can",
                # "v1_ds_sawyer_pick_place_can",
                # "v1_ds_panda_nut_assembly_square",
                # "v1_ds_sawyer_nut_assembly_square",

                # "v1_ds_panda_nut_assembly_square_200",
                # "v1_ds_sawyer_nut_assembly_square_200",

                # "v1_2_panda_can",
                "multi_panda_can_ajay",
                "multi_panda_can_josiah",
                "multi_panda_can_roberto",
                "multi_panda_can_yuke",
                "multi_panda_can_all",

                "multi_panda_square_ajay",
                "multi_panda_square_josiah",
                "multi_panda_square_roberto",
                "multi_panda_square_yuke",
                "multi_panda_square_all",

                # "v1_ds_sawyer_pick_place_can_paired2",

                # "v1_ds_panda_lift_image",
                # "v1_ds_sawyer_lift_image",
                # "v1_ds_panda_pick_place_can_image",
                # "v1_ds_sawyer_pick_place_can_image",
                # "v1_ds_panda_nut_assembly_square_image",
                # "v1_ds_sawyer_nut_assembly_square_image",

                # "antmaze_umaze_diverse",
                # "antmaze_medium_diverse",

                # "pen_human",
                # "door_human",
                # "hammer_human",
                # "relocate_human",
                # "pen_cloned",

                # "kitchen_complete",
                # "kitchen_partial",
                # "kitchen_mixed",
            ],
        ),
    # "train/hdf5_filter": 
    #     (
    #         "", 
    #         [
    #             None, None, None,
    #             # "fastest_225",
    #         ],
    #         0,
    #     ),
    # "train/num_epochs": 
    #     (
    #         "nepoch", 
    #         [1001, 501, 501],
    #         0,
    #     ),
    # "experiment/render_video": 
    #     (
    #         "", 
    #         [False, False, False],
    #         0,
    #     ),
    # "experiment/rollout/horizon": 
    #     (
    #         "horz", 
    #         [281, 281, 281],
    #         0,
    #     ),
    # "experiment/rollout/rate": 
    #     (
    #         "", 
    #         [50, 25, 25],
    #         0,
    #     ),
    # "experiment/save_every_n_epochs": 
    #     (
    #         "", 
    #         [50, 25, 25],
    #         0,
    #     ),
    "algo/optim_params/policy/learning_rate/initial": 
        (
            "plr", 
            [1e-3, 1e-4],
            # [3e-4, 7e-4],
            # [1e-4],
            # [1e-3],
            1,
        ),
    "train/seq_length": 
        (
            "seq", 
            [10],
            # [5, 10, 50, 100],
            # [1],
            2,
        ),
    "train/subgoal_horizons": 
        (
            "", 
            [[10]],
            # [[5], [10], [50], [100]],
            # [[1]],
            2,
        ),
    "algo/rnn/horizon": 
        (
            "", 
            [10],
            # [5, 10, 50, 100],
            # [1],
            2,
        ),
    "algo/actor_layer_dims":
        (
            "mlp", 
            # [[], [300, 400]],
            [[]],
            # [[300, 400], [256, 256, 256]],
            # [[300, 400]],
            # [[300, 400], [1024, 1024]],
            # [[1024, 1024]],
            3,
        ),
    "train/num_epochs": 
        (
            "nepoch", 
            # [2201],
            [1501],
            # [2001],
            # [1001, 1001],
            # [501, 501],
            4,
        ),
    "algo/rnn/hidden_dim": 
        (
            "rnnd", 
            # [128],
            # [100, 400],
            # [400, 1000],
            # [400],
            [1000],
            # [100, 400, 1000],
            4,
        ),
    # "algo/rnn/num_layers": 
    #     (
    #         "rnnnl", 
    #         [8],
    #         4,
    #     ),
    # "train/num_epochs": 
    #     (
    #         "nepoch", 
    #         # [1001],
    #         [501, 501],
    #         # [201],
    #         # [2001, 2001],
    #         # [1001, 1001, 1001],
    #         # [30, 30],
    #         4,
    #     ),

    # "algo/rnn/enabled": 
    #     (
    #         "rnn", 
    #         [False, False],
    #         # [False],
    #         5,
    #     ),

    "algo/gmm/enabled": 
        (
            "gmm", 
            [True, False],
            # [True],
            # [False],
            5,
        ),

    # "algo/rnn/open_loop": 
    #     (
    #         "open_loop", 
    #         [True],
    #         # [True],
    #         # [False],
    #         6,
    #     ),

    # "algo/gmm/num_modes": 
    #     (
    #         "nm", 
    #         [5, 5, 5, 100, 100],
    #         6,
    #     ),
    # "algo/gmm/min_std": 
    #     (
    #         "min", 
    #         [0.01, 1e-4, 1e-4, 0.01, 0.01],
    #         6,
    #     ),
    # "algo/gmm/std_activation": 
    #     (
    #         "act", 
    #         ["exp", "softplus", "exp", "softplus", "exp"],
    #         6,
    #     ),

    # "algo/modalities/observation": 
    #     (
    #         "mod", 
    #         # [["agentview_image"], ["agentview_image", "proprio"]],
    #         [["agentview_image", "proprio"]],
    #         6,
    #         # ["im", "im_prop"],
    #         ["im_prop"],
    #     ),

    # "train/seed": 
    #     (
    #         "seed", 
    #         [0, 1],
    #         7,
    #     ),
    # "test/seed": 
    #     (
    #         "", 
    #         [0, 1],
    #         7,
    #     ),

    # "train/hdf5_in_memory": 
    #     (
    #         "in_mem", 
    #         [True],
    #         6,
    #     ),

    # "algo/vae/enabled": 
    #     (
    #         "vae", 
    #         [True],
    #         5,
    #     ),
    # "algo/vae/kl_weight": 
    #     (
    #         "kl", 
    #         # [0.5, 0.05, 5e-3, 5e-4],
    #         # [0.05],
    #         # [0.05, 5e-4],
    #         [0.5, 0.05],
    #         6,
    #     ),
    # "algo/vae/decoder_layer_dims": 
    #     (
    #         # "dec_mlp",
    #         "mlp", 
    #         # [[], [300, 400]],
    #         [[]],
    #         # [[300, 400]],
    #         7,
    #     ),
    # "algo/vae/encoder_layer_dims": 
    #     (
    #         # "enc_mlp", 
    #         "",
    #         # [[], [300, 400]],
    #         [[]],
    #         # [[300, 400]],
    #         7,
    #     ),

    # # # gmm
    # # "algo/vae/prior/learn": 
    # #     (
    # #         "prior_gmm", 
    # #         # [False, True],
    # #         [True],
    # #         8,
    # #     ),
    # # "algo/vae/prior/use_gmm": 
    # #     (
    # #         "", 
    # #         # [False, True],
    # #         [True],
    # #         8,
    # #     ),
    # # "algo/vae/prior/gmm_learn_weights": 
    # #     (
    # #         "", 
    # #         # [False, True],
    # #         [True],
    # #         8,
    # #     ),
    # # "algo/vae/prior/is_conditioned": 
    # #     (
    # #         "", 
    # #         # [False, True],
    # #         [True],
    # #         8,
    # #     ),
    # # "algo/vae/prior/gmm_weights_are_conditioned": 
    # #     (
    # #         "", 
    # #         # [False, True],
    # #         [True],
    # #         8,
    # #     ),
    # # "algo/vae/prior_layer_dims": 
    # #     (
    # #         "pld", 
    # #         # [[], [300, 400]],
    # #         [[300, 400]],
    # #         8,
    # #     ),
    # # # "algo/vae/prior/gmm_low_noise_eval": 
    # # #     (
    # # #         "lne", 
    # # #         [True],
    # # #         8,
    # # #     ),

    # # # categ
    # # "algo/vae/prior/use_categorical": 
    # #     (
    # #         "categ", 
    # #         [True],
    # #         8,
    # #     ),
    # # "algo/vae/latent_dim": 
    # #     (
    # #         # "ld",
    # #         "", 
    # #         [1],
    # #         8,
    # #     ),
    # # "algo/vae/prior/categorical_dim": 
    # #     (
    # #         "cd", 
    # #         [10],
    # #         8,
    # #     ),
    # # "algo/vae/prior/categorical_min_temp": 
    # #     (
    # #         # "mtemp", 
    # #         "",
    # #         [0.5],
    # #         8,
    # #     ),
    # # "algo/vae/prior/categorical_gumbel_softmax_hard": 
    # #     (
    # #         # "hard",
    # #         "", 
    # #         # [True, False],
    # #         [False],
    # #         9,
    # #     ),
    # # "algo/vae/prior/learn": 
    # #     (
    # #         "", 
    # #         # [False, True],
    # #         [False],
    # #         10,
    # #     ),
    # # "algo/vae/prior/is_conditioned": 
    # #     (
    # #         "", 
    # #         # [False, True],
    # #         [False],
    # #         10,
    # #     ),


    # "algo/vae/rnn/encoder_is_rnn": 
    #     (
    #         # "enc_rnn", 
    #         "ed_rnn",
    #         # [False],
    #         [True],
    #         11,
    #     ),
    # # "algo/rnn/kwargs/bidirectional": 
    # #     (
    # #         "bidir", 
    # #         [True],
    # #         11,
    # #     ),
    # "algo/vae/rnn/decoder_is_rnn": 
    #     (
    #         "", 
    #         [True],
    #         11,
    #     ),
    # # "algo/vae/rnn/encoder_use_subgoals": 
    # #     (
    # #         "enc_sg", 
    # #         [True],
    # #         12,
    # #     ),
    # # "algo/vae/rnn/encoder_replace_actions": 
    # #     (
    # #         "", 
    # #         [True],
    # #         12,
    # #     ),
    # # "algo/vae/rnn/decoder_use_subgoals": 
    # #     (
    # #         "dec_sg", 
    # #         [True],
    # #         13,
    # #     ),

    # # "algo/vae/decoder/is_conditioned": 
    # #     (
    # #         "dec_cond",
    # #         [False],
    # #         14,
    # #     ),

    # # "algo/rnn/open_loop": 
    # #     (
    # #         "open_loop", 
    # #         [True],
    # #         # [False],
    # #         15,
    # #     ),

    # "algo/flow/enabled": 
    #     (
    #         "flow", 
    #         # [True],
    #         [True, True, True],
    #         5,
    #     ),
    # "algo/rnn/enabled": 
    #     (
    #         "", 
    #         # [False],
    #         [False, False, False],
    #         5,
    #     ),
    # "algo/flow/layer_dims": 
    #     (
    #         "ldims", 
    #         # [[256, 256]],
    #         # [[1024, 1024, 1024]],
    #         [[1024, 1024], [1024, 1024, 1024], [1024, 1024]],
    #         5,
    #     ),
    # "algo/flow/obs_layer_dims": 
    #     (
    #         "oldims", 
    #         # [[256, 256, 256]],
    #         # [[1024, 1024, 256], [1024, 1024, 8], [256, 256, 8]],
    #         # [[256, 256, 8]],
    #         [[1024, 1024, 256], [1024, 1024, 256], [256, 256, 256]],
    #         5,
    #     ),
    # "algo/flow/num_flows": 
    #     (
    #         "nf", 
    #         # [2, 3],
    #         # [3, 4],
    #         # [6, 8],
    #         [6, 4, 3],
    #         5,
    #     ),
    # "algo/flow/activation": 
    #     (
    #         "act", 
    #         ["relu", "tanh"],
    #         # ["relu"],
    #         7,
    #     ),
})

# # BC-SPIRL tuning
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 # "~/Desktop/lift_suboptimal/states1.hdf5",
#                 # "~/Desktop/roboturk_v1/RoboTurkPilot/bins-Can/states.hdf5", # "fastest_225 filter key"
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_careless/states.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_1.hdf5",

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_2_succ.hdf5",

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_images_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_lift/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_lift/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_done_2.hdf5",

#                 "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_done_2_combined.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_done_2_combined.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can_paired2/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_lift/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_lift/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_images_done_2.hdf5",

#                 # "~/Desktop/d4rl/converted/antmaze_umaze_diverse.hdf5",
#                 # "~/Desktop/d4rl/converted/antmaze_medium_diverse.hdf5",

#                 # "~/Desktop/d4rl/converted/pen-v0_demos_clipped.hdf5",
#                 # "~/Desktop/d4rl/converted/door-v0_demos_clipped.hdf5",
#                 # "~/Desktop/d4rl/converted/hammer-v0_demos_clipped.hdf5",
#                 # "~/Desktop/d4rl/converted/relocate-v0_demos_clipped.hdf5",
#                 # "~/Desktop/d4rl/converted/pen-demos-v0-bc-combined.hdf5",

#                 # "~/Desktop/d4rl/converted/kitchen_complete_v0.hdf5",
#                 # "~/Desktop/d4rl/converted/kitchen_partial_v0.hdf5",
#                 # "~/Desktop/d4rl/converted/kitchen_mixed_v0.hdf5",
#             ],
#             0,
#             [
#                 # "lift_subopt",
#                 # "cans_top_225",
#                 # "lift_subopt_paired1",
#                 # "lift_subopt_careless",
#                 # "lift_subopt_paired2",

#                 # "lift_subopt_paired2_done_2_succ",

#                 # "lift_subopt_paired2_image",

#                 # "v1_ds_panda_lift",
#                 # "v1_ds_sawyer_lift",
#                 # "v1_ds_panda_pick_place_can",
#                 # "v1_ds_sawyer_pick_place_can",
#                 # "v1_ds_panda_nut_assembly_square",
#                 # "v1_ds_sawyer_nut_assembly_square",

#                 "v1_ds_panda_nut_assembly_square_200",
#                 # "v1_ds_sawyer_nut_assembly_square_200",

#                 # "v1_ds_sawyer_pick_place_can_paired2",

#                 # "v1_ds_panda_lift_image",
#                 # "v1_ds_sawyer_lift_image",
#                 # "v1_ds_panda_pick_place_can_image",
#                 # "v1_ds_sawyer_pick_place_can_image",
#                 # "v1_ds_panda_nut_assembly_square_image",
#                 # "v1_ds_sawyer_nut_assembly_square_image",

#                 # "antmaze_umaze_diverse",
#                 # "antmaze_medium_diverse",

#                 # "pen_human",
#                 # "door_human",
#                 # "hammer_human",
#                 # "relocate_human",
#                 # "pen_cloned",

#                 # "kitchen_complete",
#                 # "kitchen_partial",
#                 # "kitchen_mixed",
#             ],
#         ),
#     # "train/hdf5_filter": 
#     #     (
#     #         "", 
#     #         [
#     #             None, None, None,
#     #             # "fastest_225",
#     #         ],
#     #         0,
#     #     ),
#     "train/num_epochs": 
#         (
#             "nepoch", 
#             [1501],
#             1,
#         ),

#     "algo/spirl/enabled":
#         (
#             "spirl", 
#             [True],
#             2,
#             [""],
#         ),
#     "algo/vae/enabled": 
#         (
#             "", 
#             [True],
#             2,
#         ),

#     "algo/optim_params/policy/learning_rate/initial": 
#         (
#             "plr", 
#             # [1e-3, 1e-3, 1e-4],
#             [1e-3, 1e-4],
#             # [1e-3],
#             3,
#         ),
#     "algo/optim_params/skill_prior/learning_rate/initial": 
#         (
#             "slr", 
#             # [1e-3, 1e-4, 1e-4],
#             [1e-4, 1e-4],
#             3,
#         ),
#     "train/seq_length": 
#         (
#             "seq", 
#             [10],
#             # [5, 10, 50, 100],
#             # [1],
#             4,
#         ),
#     "train/subgoal_horizons": 
#         (
#             "", 
#             [[10]],
#             # [[5], [10], [50], [100]],
#             # [[1]],
#             4,
#         ),
#     "algo/rnn/horizon": 
#         (
#             "", 
#             [10],
#             # [5, 10, 50, 100],
#             # [1],
#             4,
#         ),
#     "algo/spirl/layer_dims":
#         (
#             "sld", 
#             [(300, 400), (128, 128, 128, 128, 128), (1024, 1024)],
#             # [(128, 128, 128, 128, 128)],
#             # [(1024, 1024)],
#             5,
#         ),
#     "algo/vae/encoder_layer_dims": 
#         (
#             "enc_ld", 
#             [[]],
#             # [(300, 400)],
#             6,
#         ),
#     "algo/vae/decoder_layer_dims": 
#         (
#             "dec_ld", 
#             [[]],
#             # [(300, 400)],
#             6,
#         ),
#     "algo/rnn/hidden_dim": 
#         (
#             "rnnd", 
#             # [100],
#             # [100, 400],
#             # [400, 1000],
#             [400],
#             # [1000],
#             # [100, 400, 1000],
#             6,
#         ),

#     "algo/gaussian/enabled": 
#         (
#             "gaussian", 
#             # [True, False],
#             # [True],
#             [False],
#             7,
#         ),
#     "algo/gmm/enabled": 
#         (
#             "gmm", 
#             # [False, True],
#             [True],
#             # [False],
#             7,
#         ),

#     "algo/gaussian/low_noise_eval": 
#         (
#             "lne", 
#             # [True, False],
#             [True],
#             8,
#         ),
#     "algo/gmm/low_noise_eval": 
#         (
#             "", 
#             # [True, False],
#             [True],
#             8,
#         ),

#     # "algo/rnn/kwargs/bidirectional": 
#     #     (
#     #         "bidir", 
#     #         [True],
#     #         9,
#     #     ),

#     "algo/vae/kl_weight": 
#         (
#             "kl", 
#             # [0.05],
#             # [0.5, 0.005],
#             # [5e-3],
#             [5e-6],
#             10,
#         ),

#     "algo/gmm/min_std": 
#         (
#             "min", 
#             # [0.01, 1e-4],
#             [1e-4],
#             11,
#         ),
# })

# # OPAL tuning
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 "~/Desktop/d4rl/converted/antmaze_medium_diverse_v0.hdf5",
#                 # "~/Desktop/d4rl/converted/antmaze_large_diverse_v0.hdf5",
#             ],
#             0,
#             [
#                 "antmaze_medium_diverse",
#                 # "antmaze_large_diverse",
#             ],
#         ),
#     "train/num_epochs": 
#         (
#             "nepoch", 
#             [201],
#             1,
#         ),
#     "algo/opal/pretrain_epochs": 
#         (
#             "pretrain", 
#             [100],
#             0,
#         ),
#     "algo/low_level/optim_params/policy/learning_rate/epoch_schedule": 
#         (
#             "", 
#             [[100]],
#             0,
#         ),
#     "experiment/rollout/warmstart": 
#         (
#             "", 
#             [100],
#             0,
#         ),

#     # learning rates
#     "algo/high_level/optim_params/critic/learning_rate/initial": 
#         (
#             "clr", 
#             [3e-4],
#             3,
#         ),

#     "algo/high_level/optim_params/actor/learning_rate/initial": 
#         (
#             "alr", 
#             # [3e-5],
#             [1e-4],
#             3,
#         ),

#     "algo/low_level/optim_params/policy/learning_rate/initial": 
#         (
#             "plr", 
#             [1e-3],
#             3,
#         ),

#     # OPAL params
#     "algo/opal/use_n_step_return": 
#         (
#             "ns", 
#             [True],
#             # [False],
#             3,
#         ),
#     "algo/opal/use_single_step_return": 
#         (
#             "ss", 
#             [False],
#             # [True],
#             3,
#         ),
#     "algo/high_level/discount": 
#         (
#             "gamma", 
#             [0.99],
#             # [0.99 ** 10],
#             3,
#         ),

#     # CQL params
#     "algo/high_level/critic/min_q_weight": 
#         (
#             "min_q", 
#             # [1.0],
#             [5.0],
#             3,
#         ),

#     "algo/high_level/actor/bc_start_steps": 
#         (
#             "bc",
#             # [40000], 
#             [0],
#             3,
#         ),

#     # gaussian decoding
#     "algo/low_level/vae/decoder/reconstruction_sum_across_elements": 
#         (
#             "gauss_dec", 
#             [True],
#             # [False],
#             6,
#         ),
#     "algo/low_level/vae/decoder/learn_variance": 
#         (
#             "", 
#             [True],
#             # [False],
#             6,
#         ),

#     # "algo/low_level/vae/encoder_layer_dims": 
#     #     (
#     #         "enc_ld", 
#     #         [[]],
#     #         # [(300, 400)],
#     #         6,
#     #     ),
#     # "algo/low_level/vae/decoder_layer_dims": 
#     #     (
#     #         "dec_ld", 
#     #         [[]],
#     #         # [(300, 400)],
#     #         6,
#     #     ),
#     # "algo/low_level/rnn/hidden_dim": 
#     #     (
#     #         "rnnd", 
#     #         [400],
#     #         6,
#     #     ),

#     "algo/low_level/vae/kl_weight": 
#         (
#             "kl", 
#             # [0.1, 0.1],
#             [0.1 / 8, 0.1 / 16],
#             10,
#         ),
#     "algo/low_level/vae/latent_dim": 
#         (
#             "ld", 
#             [8, 16],
#             10,
#         ),
# })

# # PLAS tuning
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 "~/Desktop/d4rl/converted/walker2d_medium_expert.hdf5",
#                 "~/Desktop/d4rl/converted/halfcheetah_random.hdf5",
#             ],
#             0,
#             [
#                 "walker2d_medium_expert",
#                 "halfcheetah_random",
#             ],
#         ),
#     "train/num_epochs": 
#         (
#             "nepoch", 
#             [51, 101],
#             # [101],
#             0,
#         ),
#     "algo/optim_params/latent_network/end_epoch": 
#         (
#             "pretrain", 
#             [25, 50],
#             # [50],
#             0,
#         ),
#     "algo/optim_params/critic/start_epoch": 
#         (
#             "", 
#             [25, 50],
#             # [50],
#             0,
#         ),
#     "algo/optim_params/actor/start_epoch": 
#         (
#             "", 
#             [25, 50],
#             # [50],
#             0,
#         ),
#     "experiment/rollout/warmstart": 
#         (
#             "", 
#             [25, 50],
#             # [50],
#             0,
#         ),

#     # lambda
#     "algo/critic/ensemble/weight": 
#         (
#             "lam", 
#             # [1.0, 0.75],
#             [0.75],
#             # [1.0],
#             1,
#         ),

#     # "algo/latent_network/vae/latent_dim": 
#     #     (
#     #         "ld", 
#     #         [12, 16],
#     #         3,
#     #     ),
#     # "algo/latent_network/vae/kl_weight": 
#     #     (
#     #         "kl", 
#     #         [0.5 / 12., 0.5 / 16.],
#     #         3,
#     #     ),

#     "algo/latent_network/flow/enabled": 
#         (
#             "flow", 
#             [True],
#             3,
#         ),
#     "algo/latent_network/vae/enabled": 
#         (
#             "", 
#             [False],
#             3,
#         ),
#     "algo/latent_network/flow/layer_dims": 
#         (
#             "ldims", 
#             [[256, 256]],
#             3,
#         ),
#     "algo/latent_network/flow/obs_layer_dims": 
#         (
#             "oldims", 
#             [[256, 256, 256]],
#             3,
#         ),
#     "algo/latent_network/flow/num_flows": 
#         (
#             "nf", 
#             [2, 3],
#             4,
#         ),
#     "algo/latent_network/flow/activation": 
#         (
#             "act", 
#             ["relu", "tanh"],
#             5,
#         ),
# })

# # GTI tuning
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 "~/Desktop/robosuite_v1_demos/panda_lift/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/sawyer_lift/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_done_2.hdf5",
#             ],
#             0,
#             [
#                 "v1_ds_panda_lift",
#                 "v1_ds_sawyer_lift",
#                 "v1_ds_panda_pick_place_can",
#                 "v1_ds_sawyer_pick_place_can",
#                 "v1_ds_panda_nut_assembly_square",
#                 "v1_ds_sawyer_nut_assembly_square",
#             ],
#         ),
#     "algo/actor/optim_params/policy/learning_rate/initial": 
#         (
#             "plr", 
#             [1e-3, 1e-3, 1e-4],
#             1,
#         ),
#     "algo/planner/optim_params/goal_network/learning_rate/initial": 
#         (
#             "glr", 
#             [1e-3, 1e-4, 1e-5],
#             1,
#         ),
#     "train/seq_length": 
#         (
#             "seq", 
#             [10],
#             2,
#         ),
#     "train/subgoal_horizons": 
#         (
#             "", 
#             [[10]],
#             2,
#         ),
#     "algo/actor/rnn/horizon": 
#         (
#             "", 
#             [10],
#             2,
#         ),
#     "algo/subgoal_update_interval": 
#         (
#             "", 
#             [10],
#             2,
#         ),
#     "train/num_epochs": 
#         (
#             "nepoch", 
#             [1001],
#             3,
#         ),

#     "algo/actor/actor_layer_dims":
#         (
#             "mlp", 
#             # [[], [300, 400]],
#             [[]],
#             4,
#         ),
#     "algo/actor/rnn/hidden_dim": 
#         (
#             "rnnd", 
#             [100, 400],
#             5,
#         ),

#     "algo/planner/vae/enabled": 
#         (
#             "vae", 
#             [True],
#             6,
#         ),
#     "algo/planner/vae/kl_weight": 
#         (
#             "kl", 
#             [0.05],
#             7,
#         ),
#     # "algo/planner/vae/latent_dim": 
#     #     (
#     #         "ld", 
#     #         # [1, 16],
#     #         [1],
#     #         7,
#     #     ),
#     # "algo/planner/vae/prior/use_categorical": 
#     #     (
#     #         "categ", 
#     #         [True],
#     #         8,
#     #     ),
#     # "algo/planner/vae/prior/categorical_dim": 
#     #     (
#     #         "cd", 
#     #         [10],
#     #         8,
#     #     ),
#     # "algo/planner/vae/prior/categorical_min_temp": 
#     #     (
#     #         "mtemp", 
#     #         [0.5],
#     #         8,
#     #     ),
#     # "algo/planner/vae/prior/categorical_gumbel_softmax_hard": 
#     #     (
#     #         "hard", 
#     #         # [True, False],
#     #         [False],
#     #         9,
#     #     ),
#     # "algo/planner/vae/prior/learn": 
#     #     (
#     #         "plearn", 
#     #         [True, False],
#     #         10,
#     #     ),
#     # "algo/planner/vae/prior/is_conditioned": 
#     #     (
#     #         "", 
#     #         [True, False],
#     #         10,
#     #     ),

# })

# # IRIS tuning
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count/states_combined.hdf5",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count/no_self_im/states_combined.hdf5",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count/ungrasp/states_combined.hdf5",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count/ungrasp/no_self_im/states_combined.hdf5",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count/ungrasp/no_self_im/states_combined_both.hdf5",

#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v2/ungrasp/states_combined.hdf5",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v2/ungrasp/no_self_im/states_combined.hdf5",
#                 "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v3/ungrasp/states_combined.hdf5",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v3/ungrasp/no_self_im/states_combined.hdf5",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v4/ungrasp/states_combined.hdf5",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v4/ungrasp/no_self_im/states_combined.hdf5",
#             ],
#             0,
#             [
#                 # "bad_100_random_30_count",
#                 # "bad_100_random_30_count_no_self_im",
#                 # "bad_100_random_30_count_ungrasp",
#                 # "bad_100_random_30_count_ungrasp_no_self_im",
#                 # "bad_100_random_30_count_both_no_self_im",

#                 # "bad_100_random_30_count_ungrasp_v2",
#                 # "bad_100_random_30_count_ungrasp_no_self_im_v2",
#                 "bad_100_random_30_count_ungrasp_v3",
#                 # "bad_100_random_30_count_ungrasp_no_self_im_v3",
#                 # "bad_100_random_30_count_ungrasp_v4",
#                 # "bad_100_random_30_count_ungrasp_no_self_im_v4",
#             ],
#         ),
#     "algo/rwp/weights": 
#         (
#             "", 
#             [
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v2/ungrasp/base_policy.pth",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v2/ungrasp/base_policy.pth",
#                 "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v3/ungrasp/base_policy.pth",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v3/ungrasp/base_policy.pth",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v4/ungrasp/base_policy.pth",
#                 # "~/Desktop/coffee/coffee_bad_100/intervention/random_30_1_count_v4/ungrasp/base_policy.pth",
#             ],
#             0,
#         ),
#     "algo/actor/optim_params/policy/learning_rate/initial": 
#         (
#             "plr", 
#             # [1e-3, 1e-3],
#             [1e-3],
#             1,
#         ),
#     "algo/value_planner/planner/optim_params/goal_network/learning_rate/initial": 
#         (
#             "glr", 
#             # [1e-3, 1e-4],
#             [1e-4],
#             1,
#         ),
#     "algo/value_planner/value/optim_params/critic/learning_rate/initial": 
#         (
#             "vlr1", 
#             [1e-3],
#             1,
#         ),
#     "algo/value_planner/value/optim_params/vae/learning_rate/initial": 
#         (
#             "vlr2", 
#             [1e-3],
#             1,
#         ),
#     "train/seq_length": 
#         (
#             "seq", 
#             [10],
#             2,
#         ),
#     "train/subgoal_horizons": 
#         (
#             "", 
#             [[10]],
#             2,
#         ),
#     "algo/actor/rnn/horizon": 
#         (
#             "", 
#             [10],
#             2,
#         ),
#     "algo/subgoal_update_interval": 
#         (
#             "", 
#             [10],
#             2,
#         ),
#     "train/num_epochs": 
#         (
#             "nepoch", 
#             [1501],
#             3,
#         ),

#     "algo/actor/actor_layer_dims":
#         (
#             "mlp", 
#             # [[], [300, 400]],
#             [[]],
#             4,
#         ),

#     "algo/value_planner/planner/vae/kl_weight": 
#         (
#             "kl", 
#             # [1e-4, 0.05, 0.5],
#             [0.5],
#             6,
#         ),
#     "algo/value_planner/planner/vae/latent_dim": 
#         (
#             "ld", 
#             # [1, 16],
#             [1],
#             7,
#         ),
#     "algo/value_planner/planner/vae/prior/use_categorical": 
#         (
#             "categ", 
#             [True],
#             8,
#         ),
#     "algo/value_planner/planner/vae/prior/categorical_dim": 
#         (
#             "cd", 
#             [10],
#             8,
#         ),
#     "algo/value_planner/planner/vae/prior/categorical_min_temp": 
#         (
#             "mtemp", 
#             [0.5],
#             8,
#         ),
#     "algo/value_planner/planner/vae/prior/categorical_gumbel_softmax_hard": 
#         (
#             "hard", 
#             # [True, False],
#             [False],
#             9,
#         ),

#     "algo/rwp/finetune/enabled": 
#         (
#             "finetune", 
#             [True],
#             20,
#         ),
#     "algo/rwp/use_balanced_sampling": 
#         (
#             "balanced_samp", 
#             # [True, False],
#             [False],
#             21,
#         ),
#     "algo/rwp/finetune/init_from_base_policy": 
#         (
#             "init", 
#             [False],
#             22,
#         ),

#     # waypoint config
#     "algo/rwp/use_waypoints": 
#         (
#             "wp", 
#             [True],
#             23,
#         ),
#     "algo/rwp/waypoints_on_all": 
#         (
#             "all", 
#             # [True, False],
#             [False],
#             24,
#         ),
#     "algo/rwp/closest_waypoint_ignore_radius": 
#         (
#             "r", 
#             [0],
#             # [10],
#             25,
#         ),


# })


# # HBCQ Latent
# PARAMETERS = OrderedDict({
#     "algo/latent_action/enabled": 
#         (
#             "latent", 
#             [True],
#             1,
#         ),
#     "algo/latent_action/prior_correction/enabled": 
#         (
#             "pc", 
#             [True, False],
#             2,
#         ),
#     "algo/latent_action/use_encoder_mean": 
#         (
#             "enc_mean", 
#             [True, False],
#             3,
#         ),
#     "algo/actor/enabled": 
#         (
#             "actor", 
#             [True, False],
#             4,
#         ),
# })


# # CQL Tuning
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 # "~/Desktop/d4rl/converted/walker2d_medium_expert.hdf5",
#                 # "~/Desktop/d4rl/converted/walker2d_medium.hdf5",
#                 "~/Desktop/d4rl/converted/halfcheetah_random.hdf5",
#                 # "~/Desktop/d4rl/converted/pen-v0_demos_clipped.hdf5",

#                 # "~/Desktop/d4rl/converted/antmaze_umaze_diverse.hdf5",
#                 # "~/Desktop/d4rl/converted/antmaze_medium_diverse.hdf5",

#                 # "~/Desktop/lift_rb/1/states_fixed_horz_done.hdf5",
#                 # "~/Desktop/lift_rb/1/states_fixed_horz_done_shaped.hdf5",
#                 # "~/Desktop/lift_rb/1/states_task_comp_done_shaped.hdf5",
#                 # "~/Desktop/lift_rb/1/states_task_comp_done_timeout_shaped.hdf5",

#                 # "~/Desktop/lift_suboptimal/states1.hdf5",
#                 # "~/Desktop/roboturk_v1/RoboTurkPilot/bins-Can/states.hdf5", # "fastest_225 filter key"
#                 # "~/Desktop/lift_suboptimal/states_dense.hdf5",
#             ],
#             0,
#             [
#                 # "walker2d_medium_expert",
#                 # "walker2d_medium",
#                 "halfcheetah_random",
#                 # "pen_human",

#                 # "antmaze_umaze_diverse",
#                 # "antmaze_medium_diverse",

#                 # "fixed_horz_sparse",
#                 # "fixed_horz_dense",
#                 # "task_comp_dense",
#                 # "task_comp_done_timeout_dense",

#                 # "lift_subopt",
#                 # "cans_top_225",
#                 # "lift_subopt_dense",
#             ],
#         ),
#     "train/hdf5_filter": 
#         (
#             "", 
#             [
#                 None,
#                 # "fastest_225",
#             ],
#             0,
#         ),
#     "algo/optim_params/critic/learning_rate/initial": 
#         (
#             "clr", 
#             [3e-4, 3e-4],
#             1,
#         ),
#     "algo/optim_params/actor/learning_rate/initial": 
#         (
#             "alr", 
#             [1e-4, 1e-4],
#             1,
#         ),
#     "train/batch_size": 
#         (
#             "bsize", 
#             [100, 256],
#             1,
#         ),
#     "train/num_epochs": 
#         (
#             "nepoch", 
#             # [55],
#             # [60],
#             [100],
#             # [2000],
#             # [1000],
#             # [801],
#             2,
#         ),
#     # "algo/dual/use_lagrange": 
#     #     (
#     #         "dual", 
#     #         [True],
#     #         # [False],
#     #         2,
#     #     ),
#     # "algo/dual/lagrange_threshold": 
#     #     (
#     #         "tau", 
#     #         # [10.0],
#     #         [5.0],
#     #         2,
#     #     ),
#     "algo/min_q_weight": 
#         (
#             "alpha", 
#             # [1.0],
#             # [10.0],
#             [5.0],
#             3,
#         ),
#     "algo/bc_updates": 
#         (
#             "bc_steps", 
#             [40000],
#             # [0, 40000],
#             # [0],
#             4,
#         ),

#     # "algo/critic/layer_dims": 
#     #     (
#     #         "ld", 
#     #         [[300, 400]],
#     #         5,
#     #     ),
#     # "algo/actor/layer_dims": 
#     #     (
#     #         "", 
#     #         [[300, 400]],
#     #         5,
#     #     ),

#     "algo/actor/gaussian/use_tanh": 
#         (
#             "tanh", 
#             [True],
#             5,
#         ),

# })



# # HBCQ Tuning
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 # "~/Desktop/lift_rb/1/states_fixed_horz_done.hdf5",
#                 # "~/Desktop/lift_rb/1/states_fixed_horz_done_shaped.hdf5",
#                 # "~/Desktop/lift_rb/1/states_task_comp_done_shaped.hdf5",
#                 # "~/Desktop/lift_rb/1/states_task_comp_done_timeout_shaped.hdf5",

#                 # "~/Desktop/lift_suboptimal/states1.hdf5",
#                 # "~/Desktop/roboturk_v1/RoboTurkPilot/bins-Can/states.hdf5", # "fastest_225 filter key"

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states_dense.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states_buffer_300.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states_dense_buffer_300.hdf5",

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_1.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_2.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_dense_done_2.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_2_succ.hdf5",

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_images_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_lift/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_lift/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/ajay/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/josiah/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/roberto/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/yuke/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/ajay/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/roberto/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/yuke/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can_paired2/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_lift/states_dense_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_lift/states_dense_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_dense_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_dense_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_dense_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_dense_done_2.hdf5",

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_careless/states.hdf5",

#                 # "~/Desktop/d4rl/converted/antmaze_umaze_diverse.hdf5",
#                 # "~/Desktop/d4rl/converted/antmaze_medium_diverse.hdf5",

#                 # "~/Desktop/d4rl/converted/pen-v0_demos_clipped.hdf5",

#                 # "~/Desktop/d4rl/converted/walker2d_medium_expert.hdf5",
#             ],
#             0,
#             [
#                 # "fixed_horz_sparse",
#                 # "fixed_horz_dense",
#                 # "task_comp_dense",
#                 # "task_comp_done_timeout_dense",

#                 # "lift_subopt",
#                 # "cans_top_225",

#                 # "lift_subopt_paired1",
#                 # "lift_subopt_paired1_dense",
#                 # "lift_subopt_paired1_buffer_300",
#                 # "lift_subopt_paired1_dense_buffer_300",

#                 # "lift_subopt_paired2_done_1",
#                 # "lift_subopt_paired2_done_2",
#                 # "lift_subopt_paired2_done_2_dense",
#                 # "lift_subopt_paired2_done_2_succ",

#                 # "lift_subopt_paired2_image",

#                 # "v1_ds_panda_lift",
#                 # "v1_ds_sawyer_lift",
#                 # "v1_ds_panda_pick_place_can",
#                 # "v1_ds_sawyer_pick_place_can",
#                 # "v1_ds_panda_nut_assembly_square",
#                 # "v1_ds_sawyer_nut_assembly_square",

#                 # "multi_panda_can_ajay",
#                 # "multi_panda_can_josiah",
#                 # "multi_panda_can_roberto",
#                 # "multi_panda_can_yuke",
#                 "multi_panda_can_all",

#                 # "multi_panda_square_ajay",
#                 # "multi_panda_square_josiah",
#                 # "multi_panda_square_roberto",
#                 # "multi_panda_square_yuke",
#                 "multi_panda_square_all",

#                 # "v1_ds_sawyer_pick_place_can_paired2",

#                 # "lift_subopt_careless",

#                 # "antmaze_umaze_diverse",
#                 # "antmaze_medium_diverse",

#                 # "pen_human",

#                 # "walker2d_medium_expert",
#             ],
#         ),
#      # "train/hdf5_filter": 
#      #    (
#      #        "", 
#      #        [
#      #            None,
#      #            # None, None,
#      #            # "fastest_225",
#      #        ],
#      #        0,
#      #    ),
#     "algo/optim_params/critic/learning_rate/initial": 
#         (
#             "clr", 
#             # [1e-3, 1e-4, 1e-4, 1e-4],
#             [1e-3],
#             1,
#         ),
#     "algo/optim_params/action_sampler/learning_rate/initial": 
#         (
#             "aslr", 
#             # [1e-3, 1e-4, 1e-3, 3e-4],
#             [1e-3],
#             1,
#         ),
#     # "algo/optim_params/actor/learning_rate/initial": 
#     #     (
#     #         "aclr", 
#     #         [1e-4],
#     #         1,
#     #     ),

#     # "algo/optim_params/critic/start_epoch": 
#     #     (
#     #         "c_start", 
#     #         [400, 400],
#     #         1,
#     #     ),
#     # "algo/optim_params/action_sampler/end_epoch": 
#     #     (
#     #         "as_end", 
#     #         [400, -1],
#     #         1,
#     #     ),

#     "algo/actor/enabled": 
#         (
#             "actor", 
#             # [True, False],
#             [False],
#             # [True],
#             2,
#         ),

#     # "train/num_epochs": 
#     #     (
#     #         "nepoch", 
#     #         [1601],
#     #         # [1001],
#     #         # [501],
#     #         6,
#     #     ),

#     # "train/hdf5_in_memory": 
#     #     (
#     #         "in_mem", 
#     #         [True],
#     #         6,
#     #     ),

#     # "algo/action_sampler/gmm/enabled": 
#     #     (
#     #         "gmm", 
#     #         # [True],
#     #         [False, True],
#     #         7,
#     #     ),
#     # "algo/action_sampler/vae/enabled": 
#     #     (
#     #         "vae", 
#     #         # [False],
#     #         [True, False],
#     #         7,
#     #     ),

#     "algo/target_tau": 
#         (
#             "tau", 
#             # [5e-3, 5e-3, 5e-4, 5e-4],
#             # [5e-3, 5e-3, 5e-4],
#             [5e-3, 5e-4],
#             # 8,
#             3,
#         ),
#     # "algo/critic/num_action_samples": 
#     #     (
#     #         "nact_samp", 
#     #         # [10, 100, 10, 100],
#     #         [10, 100, 10],
#     #         # 8,
#     #         3,
#     #     ),
#     # "algo/infinite_horizon": 
#     #     (
#     #         "inf", 
#     #         [True],
#     #         # 9,
#     #         4,
#     #     ),

#     # RNN-GMM
#     "train/seq_length": 
#         (
#             "seq", 
#             [10],
#             # [5, 10, 30, 50],
#             # [30],
#             5,
#         ),
#     "train/subgoal_horizons": 
#         (
#             "", 
#             [[10]],
#             # [[5], [10], [30], [50]],
#             # [[30]],
#             5,
#         ),
#     "algo/action_sampler/rnn/horizon": 
#         (
#             "", 
#             [10],
#             # [5, 10, 30, 50],
#             # [30],
#             5,
#         ),
#     "algo/critic/rnn/horizon": 
#         (
#             "", 
#             [10],
#             # [5, 10, 30, 50],
#             # [30],
#             5,
#         ),

#     "algo/action_sampler/rnn/hidden_dim": 
#         (
#             "as_rnnd", 
#             # [400, 1000],
#             # [1000],
#             # [400, 100],
#             [400],
#             6,
#         ),
#     "algo/action_sampler/rnn/enabled": 
#         (
#             "", 
#             # [True, True],
#             [True],
#             6,
#         ),
#     "algo/critic/rnn/hidden_dim": 
#         (
#             "c_rnnd", 
#             # [100, 100],
#             [100],
#             6,
#         ),
#     "algo/critic/rnn/enabled": 
#         (
#             "", 
#             # [True, True],
#             [True],
#             6,
#         ),

#     "algo/action_sampler/gmm/enabled": 
#         (
#             "gmm", 
#             # [True, True],
#             [True],
#             6,
#         ),
#     "algo/action_sampler/vae/enabled": 
#         (
#             "", 
#             # [False, False],
#             [False],
#             6,
#         ),

#     "algo/action_sampler/actor_layer_dims": 
#         (
#             "mlp", 
#             # [[], [300, 400]],
#             [[]],
#             # [[300, 400]],
#             7,
#         ),
#     "algo/critic/layer_dims": 
#         (
#             "", 
#             # [[], [300, 400]],
#             [[]],
#             # [[300, 400]],
#             7,
#         ),

#     "train/num_epochs": 
#         (
#             "nepoch", 
#             # [1601, 1601],
#             [1001],
#             # [2001, 2001],
#             8,
#         ),


# })


# # HBCQ Seq Tuning
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 # "~/Desktop/lift_rb/1/states_fixed_horz_done.hdf5",
#                 # "~/Desktop/lift_rb/1/states_fixed_horz_done_shaped.hdf5",
#                 # "~/Desktop/lift_rb/1/states_task_comp_done_shaped.hdf5",
#                 # "~/Desktop/lift_rb/1/states_task_comp_done_timeout_shaped.hdf5",

#                 # "~/Desktop/lift_suboptimal/states1.hdf5",
#                 # "~/Desktop/roboturk_v1/RoboTurkPilot/bins-Can/states.hdf5", # "fastest_225 filter key"
#                 # "~/Desktop/lift_suboptimal/states_dense.hdf5",

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states_dense.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states_buffer_300.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired1/states_dense_buffer_300.hdf5",

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_1.hdf5",
#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_paired2/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_lift/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_lift/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_done_2_combined.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_done_2_combined.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/ajay/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/josiah/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/roberto/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/yuke/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/panda_pick_place_can_multi/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/ajay/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/roberto/states_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/yuke/states_done_2.hdf5",
#                 "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/states_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can_paired2/states_done_2.hdf5",

#                 # "~/Desktop/d4rl_manip/lift_v1_subopt_careless/states.hdf5",

#                 # "~/Desktop/d4rl/converted/antmaze_umaze_diverse.hdf5",
#                 # "~/Desktop/d4rl/converted/antmaze_medium_diverse.hdf5",

#                 # "~/Desktop/d4rl/converted/pen-v0_demos_clipped.hdf5",
#             ],
#             0,
#             [
#                 # "fixed_horz_sparse",
#                 # "fixed_horz_dense",
#                 # "task_comp_dense",
#                 # "task_comp_done_timeout_dense",

#                 # "lift_subopt",
#                 # "cans_top_225",
#                 # "lift_subopt_dense",

#                 # "lift_subopt_paired1",
#                 # "lift_subopt_paired1_dense",
#                 # "lift_subopt_paired1_buffer_300",
#                 # "lift_subopt_paired1_dense_buffer_300",

#                 # "lift_subopt_paired2_done_1",
#                 # "lift_subopt_paired2_done_2",

#                 # "v1_ds_panda_lift",
#                 # "v1_ds_sawyer_lift",
#                 # "v1_ds_panda_pick_place_can",
#                 # "v1_ds_sawyer_pick_place_can",
#                 # "v1_ds_panda_nut_assembly_square",
#                 # "v1_ds_sawyer_nut_assembly_square",

#                 # "v1_ds_panda_nut_assembly_square_200",
#                 # "v1_ds_sawyer_nut_assembly_square_200",

#                 # "multi_panda_can_ajay",
#                 # "multi_panda_can_josiah",
#                 # "multi_panda_can_roberto",
#                 # "multi_panda_can_yuke",
#                 "multi_panda_can_all",

#                 # "multi_panda_square_ajay",
#                 # "multi_panda_square_josiah",
#                 # "multi_panda_square_roberto",
#                 # "multi_panda_square_yuke",
#                 "multi_panda_square_all",

#                 # "v1_ds_sawyer_pick_place_can_paired2",

#                 # "lift_subopt_careless",

#                 # "antmaze_umaze_diverse",
#                 # "antmaze_medium_diverse",

#                 # "pen_human",
#             ],
#         ),
#     # "train/hdf5_filter": 
#     #     (
#     #         "", 
#     #         [
#     #             None, None,
#     #             # "fastest_225",
#     #         ],
#     #         0,
#     #     ),
#     "algo/latent_action/enabled": 
#         (
#             "lt", 
#             [True],
#             1,
#             ["t"],
#         ),
#     "algo/action_sampler/rnn/enabled": 
#         (
#             "", 
#             # [True, True],
#             [True],
#             1,
#         ),
#     "algo/action_sampler/vae/enabled": 
#         (
#             "", 
#             [True],
#             1,
#         ),
#     # "algo/latent_action/prior_correction/enabled": 
#     #     (
#     #         "pc", 
#     #         # [True, False],
#     #         [False],
#     #         2,
#     #     ),
#     # "algo/latent_action/use_encoder_mean": 
#     #     (
#     #         "enc_mean", 
#     #         [True, False],
#     #         3,
#     #     ),
#     "algo/latent_action/value/use_single_step_return": 
#         (
#             "ss", 
#             [True],
#             # [False],
#             2,
#             ["t"],
#             # ["f"],
#         ),
#     "algo/latent_action/value/use_n_step_return": 
#         (
#             "", 
#             [False],
#             # [True],
#             2,
#         ),

#     "algo/optim_params/critic/learning_rate/initial": 
#         (
#             "clr", 
#             # [1e-3, 1e-3],
#             [1e-3],
#             4,
#         ),
#     "algo/optim_params/action_sampler/learning_rate/initial": 
#         (
#             "aslr", 
#             # [1e-3, 1e-4],
#             [1e-3],
#             4,
#         ),
#     # "algo/optim_params/actor/learning_rate/initial": 
#     #     (
#     #         "aclr", 
#     #         [1e-4],
#     #         4,
#     #     ),

#     "algo/actor/enabled": 
#         (
#             "actor", 
#             # [True, False],
#             # [False, False],
#             [False],
#             4,
#             ["f"],
#         ),


#     "train/seq_length": 
#         (
#             "seq", 
#             [10],
#             5,
#         ),
#     "train/subgoal_horizons": 
#         (
#             "", 
#             [[10]],
#             5,
#         ),
#     "algo/action_sampler/rnn/horizon": 
#         (
#             "", 
#             [10],
#             5,
#         ),

#     "algo/action_sampler/rnn/hidden_dim": 
#         (
#             "rnnd", 
#             # [400, 1000],
#             [400],
#             6,
#         ),
#     "train/num_epochs": 
#         (
#             "ne", 
#             # [1601],
#             [1001],
#             # [2001, 2001],
#             6,
#         ),

#     "algo/action_sampler/vae/kl_weight": 
#         (
#             "kl", 
#             # [5e-2, 5e-4, 5e-5, 5e-6],
#             # [5e-6],
#             # [0.05, 0.005],
#             # [0.05],
#             # [0.5, 5e-4],
#             # [5e-4, 5e-6],
#             [5e-5, 1e-4, 5e-3],
#             # [5e-4],
#             9,
#         ),
#     "algo/action_sampler/vae/decoder_layer_dims": 
#         (
#             "dld", 
#             # [[], [300, 400]],
#             [[]],
#             # [[300, 400]],
#             10,
#         ),
#     "algo/action_sampler/vae/encoder_layer_dims": 
#         (
#             "eld", 
#             # [[], [300, 400]],
#             # [[300, 400]],
#             [[]],
#             10,
#         ),

#     # gmm prior
#     "algo/action_sampler/vae/prior/learn": 
#         (
#             "p_gmm", 
#             # [False, True],
#             [True],
#             10,
#             ["t"],
#         ),
#     "algo/action_sampler/vae/prior/use_gmm": 
#         (
#             "", 
#             # [False, True],
#             [True],
#             10,
#         ),
#     "algo/action_sampler/vae/prior/gmm_learn_weights": 
#         (
#             "", 
#             # [False, True],
#             [True],
#             10,
#         ),
#     "algo/action_sampler/vae/prior/is_conditioned": 
#         (
#             "", 
#             # [False, True],
#             [True],
#             10,
#         ),
#     "algo/action_sampler/vae/prior/gmm_weights_are_conditioned": 
#         (
#             "", 
#             # [False, True],
#             [True],
#             10,
#         ),
#     "algo/action_sampler/vae/prior_layer_dims": 
#         (
#             "p_mlp", 
#             # [[], [300, 400]],
#             # [[300, 400]],
#             [[1024, 1024]],
#             11,
#         ),
#     # "algo/action_sampler/vae/prior/gmm_low_noise_eval": 
#     #     (
#     #         "lne", 
#     #         [True],
#     #         12,
#     #     ),

#     # # categorical
#     # "algo/action_sampler/vae/prior/use_categorical": 
#     #     (
#     #         "categ", 
#     #         [True],
#     #         10,
#     #     ),
#     # "algo/action_sampler/vae/latent_dim": 
#     #     (
#     #         "ld", 
#     #         [1],
#     #         10,
#     #     ),
#     # "algo/action_sampler/vae/prior/categorical_dim": 
#     #     (
#     #         "cd", 
#     #         [10],
#     #         10,
#     #     ),
#     # "algo/action_sampler/vae/prior/categorical_min_temp": 
#     #     (
#     #         # "mtemp", 
#     #         "",
#     #         [0.5],
#     #         10,
#     #     ),
#     # "algo/action_sampler/vae/prior/categorical_gumbel_softmax_hard": 
#     #     (
#     #         # "hard",
#     #         "", 
#     #         # [True, False],
#     #         [False],
#     #         11,
#     #     ),
#     # "algo/action_sampler/vae/prior/learn": 
#     #     (
#     #         "pl", 
#     #         # [False, True],
#     #         [False],
#     #         12,
#     #     ),
#     # "algo/action_sampler/vae/prior/is_conditioned": 
#     #     (
#     #         "", 
#     #         # [False, True],
#     #         [False],
#     #         12,
#     #     ),

#     # # normalizing flow prior
#     # "algo/action_sampler/vae/prior/learn": 
#     #     (
#     #         "p_flow", 
#     #         # [False, True],
#     #         [True],
#     #         10,
#     #         ["t"],
#     #     ),
#     # "algo/action_sampler/vae/prior/use_normalizing_flow": 
#     #     (
#     #         "", 
#     #         # [False, True],
#     #         [True],
#     #         10,
#     #     ),
#     # "algo/action_sampler/vae/prior/is_conditioned": 
#     #     (
#     #         "c", 
#     #         # [False, True],
#     #         # [True],
#     #         [False],
#     #         10,
#     #         # ["t"],
#     #         ["f"],
#     #     ),
#     # "algo/action_sampler/vae/prior/is_conditioned": 
#     #     (
#     #         "c", 
#     #         # [False, True],
#     #         [True],
#     #         # [False],
#     #         10,
#     #         ["t"],
#     #         # ["f"],
#     #     ),
#     # "algo/action_sampler/vae/prior/flow_layer_dims": 
#     #     (
#     #         "fld", 
#     #         # [[256, 256], [1024, 1024], [256, 256], [1024, 1024]],
#     #         [[256, 256], [256, 256], [256, 256], [256, 256]],
#     #         11,
#     #     ),
#     # "algo/action_sampler/vae/prior/flow_num": 
#     #     (
#     #         "nf", 
#     #         [2, 2, 3, 3],
#     #         11,
#     #     ),
#     # # "algo/action_sampler/vae/prior/flow_activation": 
#     # #     (
#     # #         "fact", 
#     # #         ["relu"],
#     # #         11,
#     # #     ),
#     # "algo/action_sampler/vae/prior_layer_dims": 
#     #     (
#     #         "p_mlp", 
#     #         # [[], [300, 400]],
#     #         # [[300, 400]],
#     #         [[256, 256, 256], [1024, 1024, 256], [256, 256, 256], [1024, 1024, 256]],
#     #         # [[1024, 1024]],
#     #         11,
#     #     ),

#     # "algo/action_sampler/rnn/kwargs/bidirectional": 
#     #     (
#     #         "bidir", 
#     #         [True],
#     #         12,
#     #     ),
#     "algo/action_sampler/vae/rnn/encoder_is_rnn": 
#         (
#             "ernn", 
#             # [False],
#             [True],
#             13,
#             ["t"],
#         ),
#     "algo/action_sampler/vae/rnn/decoder_is_rnn": 
#         (
#             "drnn", 
#             [True],
#             13,
#             ["t"],
#         ),
#     # "algo/action_sampler/vae/rnn/encoder_use_subgoals": 
#     #     (
#     #         "e_sg", 
#     #         # [True],
#     #         [False],
#     #         14,
#     #     ),
#     # "algo/action_sampler/vae/rnn/encoder_replace_actions": 
#     #     (
#     #         "", 
#     #         [True],
#     #         14,
#     #     ),
#     # "algo/action_sampler/vae/rnn/decoder_use_subgoals": 
#     #     (
#     #         "d_sg", 
#     #         # [True],
#     #         [False],
#     #         15,
#     #     ),

#     # "algo/optim_params/critic/start_epoch": 
#     #     (
#     #         "crit_st", 
#     #         [-1, 200],
#     #         16,
#     #     ),

#     # ## open-loop ##
#     # "algo/action_sampler/rnn/open_loop": 
#     #     (
#     #         "oloop", 
#     #         [True],
#     #         # [False],
#     #         17,
#     #     ),

#     # ## spirl ##
#     # "algo/action_sampler/spirl/enabled":
#     #     (
#     #         "spirl", 
#     #         [True],
#     #         18,
#     #         [""],
#     #     ),
#     # "algo/optim_params/skill_prior/learning_rate/initial": 
#     #     (
#     #         "slr", 
#     #         [1e-4],
#     #         19,
#     #     ),
#     # "algo/action_sampler/spirl/layer_dims":
#     #     (
#     #         "sld", 
#     #         # [(300, 400), (128, 128, 128, 128, 128), (1024, 1024)],
#     #         # [(128, 128, 128, 128, 128)],
#     #         [(1024, 1024)],
#     #         19,
#     #     ),
#     # "algo/action_sampler/gmm/enabled": 
#     #     (
#     #         "gmm", 
#     #         [True],
#     #         20,
#     #     ),
#     # "algo/action_sampler/gmm/min_std": 
#     #     (
#     #         "min", 
#     #         [1e-4],
#     #         20,
#     #     ),
#     # "algo/action_sampler/gmm/low_noise_eval": 
#     #     (
#     #         "lne", 
#     #         # [True, False],
#     #         # [True, True, False],
#     #         [True],
#     #         21,
#     #         # ["t", "t", "f"],
#     #         ["t"],
#     #     ),
#     # "algo/latent_action/spirl/skill_prior_eval_at_train": 
#     #     (
#     #         "tr", 
#     #         # [True, False, False],
#     #         [True],
#     #         21,
#     #         # ["t", "f", "f"],
#     #         ["t"],
#     #     ),
# })

# # BC image with random crops
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 # "~/Desktop/robosuite_v1_demos/panda_lift/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_lift/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_pick_place_can/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/states_images_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square/states_images_done_2.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/states_images_wrist_done_2.hdf5",

#                 "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2_120_120.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2_60_60.hdf5",
#             ],
#             0,
#             [
#                 # "v1_panda_lift_image",
#                 # "v1_sawyer_lift_image",
#                 # "v1_panda_can_image",
#                 # "v1_sawyer_can_image",
#                 # "v1_panda_square_image",
#                 # "v1_sawyer_square_image",

#                 # "v1_panda_can_image_wrist",

#                 "v1_panda_can_im_120_160",
#                 # "v1_panda_can_im_120_120",
#                 # "v1_panda_can_im_60_60",
#             ],
#         ),
#     "algo/optim_params/policy/learning_rate/initial": 
#         (
#             "plr", 
#             # [1e-3, 1e-4],
#             # [1e-3],
#             [1e-4],
#             1,
#         ),
#     "train/num_epochs": 
#         (
#             "nepoch", 
#             [601],
#             # [1001],
#             2,
#         ),
#     "algo/modalities/observation": 
#         (
#             "mod", 
#             # [["agentview_image"], ["agentview_image", "proprio"]],
#             # [["agentview_image", "proprio"]],
#             # [["agentview_image"]],
#             # [["agentview_image", "robot0_eye_in_hand_image"], ["agentview_image", "robot0_eye_in_hand_image", "proprio"]],
#             [["agentview_image", "robot0_eye_in_hand_image", "proprio"]],
#             # [["agentview_image", "robot0_eye_in_hand_image"]],
#             3,
#             # ["im", "im_prop"],
#             # ["im_prop"],
#             # ["im"],
#             # ["im_wrist", "im_wrist_prop"],
#             ["im_wrist_prop"],
#             # ["im_wrist"],
#         ),

#     "algo/obs_encoder/visual_core_kwargs/use_random_crops": 
#         (
#             "crop", 
#             [True],
#             4,
#             ["t"],
#         ),
#     "algo/obs_encoder/visual_core_kwargs/random_crop_h": 
#         (
#             "s", 
#             [60],
#             4,
#         ),
#     "algo/obs_encoder/visual_core_kwargs/random_crop_w": 
#         (
#             "", 
#             [60],
#             4,
#         ),
#     "algo/obs_encoder/visual_core_kwargs/random_crop_n": 
#         (
#             "n", 
#             [20],
#             4,
#         ),
#     "algo/obs_encoder/visual_core_kwargs/random_crop_pos_enc": 
#         (
#             "pos_enc", 
#             # [True],
#             [False],
#             4,
#         ),

#     "algo/obs_encoder/visual_core_kwargs/input_coord_conv": 
#         (
#             "coord", 
#             # [True, False],
#             # [True],
#             [False],
#             5,
#             # ["t", "f"],
#             # ["t"],
#             ["f"],
#         ),
#     "algo/obs_encoder/visual_core_kwargs/pretrained": 
#         (
#             "pt", 
#             # [True],
#             [False],
#             # [True, False],
#             6,
#             # ["t"],
#             ["f"],
#             # ["t", "f"],
#         ),
#     "algo/obs_encoder/visual_core_kwargs/concat_visual_obs": 
#         (
#             "cat", 
#             # [True],
#             [False],
#             # [True, False],
#             7,
#             # ["t"],
#             ["f"],
#             # ["t", "f"],
#         ),

#     # "algo/obs_encoder/use_spatial_softmax": 
#     #     (
#     #         "ssmax", 
#     #         # [True],
#     #         # [False],
#     #         [True, True, False],
#     #         8,
#     #         # ["t"],
#     #         # ["f"],
#     #         ["t_no_lin", "t", "f"],
#     #     ),
#     # "algo/obs_encoder/spatial_softmax_kwargs/temperature": 
#     #     (
#     #         "temp", 
#     #         [1.0, 10.0, 1.0],
#     #         8,
#     #     ),

#     # "algo/gmm/enabled": 
#     #     (
#     #         "gmm", 
#     #         [True],
#     #         9,
#     #     ),



#     # "train/seq_length": 
#     #     (
#     #         "seq", 
#     #         [10],
#     #         # [5, 10, 50, 100],
#     #         # [1],
#     #         2,
#     #     ),
#     # "train/subgoal_horizons": 
#     #     (
#     #         "", 
#     #         [[10]],
#     #         # [[5], [10], [50], [100]],
#     #         # [[1]],
#     #         2,
#     #     ),
#     # "algo/rnn/horizon": 
#     #     (
#     #         "", 
#     #         [10],
#     #         # [5, 10, 50, 100],
#     #         # [1],
#     #         2,
#     #     ),
#     # "algo/actor_layer_dims":
#     #     (
#     #         "mlp", 
#     #         # [[], [300, 400]],
#     #         [[]],
#     #         # [[300, 400], [256, 256, 256]],
#     #         # [[300, 400]],
#     #         # [[300, 400], [1024, 1024]],
#     #         # [[1024, 1024]],
#     #         3,
#     #     ),
#     # "algo/rnn/hidden_dim": 
#     #     (
#     #         "rnnd", 
#     #         # [128],
#     #         # [100, 400],
#     #         # [400, 1000],
#     #         [400],
#     #         # [1000],
#     #         # [100, 400, 1000],
#     #         4,
#     #     ),
#     # "algo/rnn/num_layers": 
#     #     (
#     #         "rnnnl", 
#     #         [8],
#     #         4,
#     #     ),

#     # "algo/rnn/enabled": 
#     #     (
#     #         "rnn", 
#     #         # [False, False],
#     #         [False],
#     #         5,
#     #     ),

#     # "algo/gmm/enabled": 
#     #     (
#     #         "gmm", 
#     #         [True, False],
#     #         # [True],
#     #         # [False],
#     #         5,
#     #     ),

#     # "algo/rnn/open_loop": 
#     #     (
#     #         "open_loop", 
#     #         [True],
#     #         # [True],
#     #         # [False],
#     #         6,
#     #     ),

#     # "algo/gmm/num_modes": 
#     #     (
#     #         "nm", 
#     #         [5, 5, 5, 100, 100],
#     #         6,
#     #     ),
#     # "algo/gmm/min_std": 
#     #     (
#     #         "min", 
#     #         [0.01, 1e-4, 1e-4, 0.01, 0.01],
#     #         6,
#     #     ),
#     # "algo/gmm/std_activation": 
#     #     (
#     #         "act", 
#     #         ["exp", "softplus", "exp", "softplus", "exp"],
#     #         6,
#     #     ),
# })


# # HAN ablations
# PARAMETERS = OrderedDict({
#     "train/data": 
#         (
#             "ds", 
#             [
#                 "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2_120_120.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2_60_60.hdf5",

#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_done_2_all.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_done_2_120_120_all.hdf5",
#                 # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_done_2_84_84_all.hdf5",
#             ],
#             0,
#             [                
#                 "v1_panda_can_im_120_160",
#                 # "v1_panda_can_im_120_120",
#                 # "v1_panda_can_im_60_60",

#                 # "v1_panda_square_all_im_120_160",
#                 # "v1_panda_square_all_im_120_120",
#                 # "v1_panda_square_all_im_84_84",
#             ],
#         ),

#     "algo/han/action_space/use_control_law": 
#         (
#             "ctrl_law", 
#             [False],
#             1,
#             ["f"],
#         ),

#     # "algo/han/action_space/use_eef_crop": 
#     #     (
#     #         "ee_crop", 
#     #         [False],
#     #         1,
#     #         ["f"],
#     #     ),

#     # "algo/han/action_space/use_confidence": 
#     #     (
#     #         "conf", 
#     #         [False],
#     #         1,
#     #         ["f"],
#     #     ),

#     # "algo/han/action_space/use_eef_pos": 
#     #     (
#     #         "ee_pos", 
#     #         [False],
#     #         1,
#     #         ["f"],
#     #     ),

#     # "algo/han/action_space/hard_attention_temp": 
#     #     (
#     #         "hard_att_temp", 
#     #         # [1.0, 10.0],
#     #         [1.0],
#     #         4,
#     #     ),


#     # "algo/obs_encoder/visual_core_kwargs/use_random_crops": 
#     #     (
#     #         "crop", 
#     #         [True],
#     #         # [False],
#     #         4,
#     #         ["t"],
#     #         # ["f"],
#     #     ),
#     # "algo/obs_encoder/visual_core_kwargs/random_crop_h": 
#     #     (
#     #         "s", 
#     #         # [60],
#     #         # [119],
#     #         [108],
#     #         4,
#     #     ),
#     # "algo/obs_encoder/visual_core_kwargs/random_crop_w": 
#     #     (
#     #         "", 
#     #         # [60],
#     #         # [159],
#     #         [146],
#     #         4,
#     #     ),
#     # "algo/obs_encoder/visual_core_kwargs/random_crop_n": 
#     #     (
#     #         "n", 
#     #         # [20],
#     #         [1],
#     #         4,
#     #     ),
# })

# BC image with random crops
PARAMETERS = OrderedDict({
    "train/data": 
        (
            "ds", 
            [
                # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_wrist_cam_done_2.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2.hdf5",
                # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2_120_120.hdf5",
                # "~/Desktop/robosuite_v1_demos/panda_pick_place_can/state_images_depth_done_2_60_60.hdf5",

                # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_wrist_cam_done_2_all.hdf5",
                "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_done_2_all.hdf5",
                # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_done_2_120_120_all.hdf5",
                # "~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_done_2_84_84_all.hdf5",
            ],
            0,
            [                
                "v1_panda_can_im_120_160",
                # "v1_panda_can_im_120_120",
                # "v1_panda_can_im_60_60",

                "v1_panda_square_all_im_120_160",
                # "v1_panda_square_all_im_120_120",
                # "v1_panda_square_all_im_84_84",
            ],
        ),
    "algo/obs_encoder/visual_core_kwargs/use_random_crops": 
        (
            "crop", 
            # [False, True, True],
            # [False, True],
            [True],
            1,
            # ["f", "t", "t"],
            # ["f", "t"],
            ["t"],
        ),
    "algo/obs_encoder/visual_core_kwargs/random_crop_h": 
        (
            "h", 
            # [120, 108, 60],
            # [120, 108, 60],
            # [60, 56, 30],
            # [84, 76, 42],
            # [120, 108],
            [108],
            1,
        ),
    "algo/obs_encoder/visual_core_kwargs/random_crop_w": 
        (
            "w", 
            # [160, 144, 60],
            # [120, 108, 60],
            # [60, 56, 30],
            # [84, 76, 42],
            # [160, 144],
            [144],
            1,
        ),
    "algo/obs_encoder/visual_core_kwargs/random_crop_n": 
        (
            "n", 
            # [1, 1, 10],
            # [1, 1],
            [1],
            1,
        ),
    # "algo/obs_encoder/visual_core_kwargs/random_crop_pos_enc": 
    #     (
    #         "pos_enc", 
    #         # [True],
    #         [False],
    #         0,
    #     ),

    "algo/optim_params/policy/learning_rate/initial": 
        (
            "plr", 
            # [1e-4, 1e-4, 1e-4],
            # [1e-4, 1e-4],
            [1e-4],
            1,
        ),
    "train/num_epochs": 
        (
            "nepoch", 
            [301],
            # [1001],
            2,
        ),
    "algo/modalities/observation": 
        (
            "mod", 
            # [["agentview_image"], ["agentview_image", "proprio"]],
            # [["agentview_image", "proprio"]],
            # [["agentview_image"]],
            # [["agentview_image", "robot0_eye_in_hand_image"], ["agentview_image", "robot0_eye_in_hand_image", "proprio"]],
            [["agentview_image", "robot0_eye_in_hand_image", "proprio"]],
            # [["agentview_image", "robot0_eye_in_hand_image", "proprio", "agentview_depth", "robot0_eye_in_hand_depth", "robot0_eye_in_hand_transform", "robot0_eye_in_hand_inverse_transform"]],
            # [["agentview_image", "robot0_eye_in_hand_image"]],
            2,
            # ["im", "im_prop"],
            # ["im_prop"],
            # ["im"],
            # ["im_wrist", "im_wrist_prop"],
            ["im_wrist_prop"],
            # ["im_wrist"],
        ),
    "algo/obs_encoder/visual_core_kwargs/concat_visual_obs": 
        (
            "cat", 
            # [True],
            [False],
            # [True, False],
            2,
            # ["t"],
            ["f"],
            # ["t", "f"],
        ),

    # for spirl-h
    "train/seq_length": 
        (
            "seq", 
            [50, 50],
            3,
        ),
    "algo/rnn/horizon": 
        (
            "l", 
            [10, 5],
            3,
        ),
    "algo/spirl/hierarchical/horizon": 
        (
            "h", 
            [5, 10],
            3,
        ),
    "algo/spirl/hierarchical/enabled": 
        (
            "", 
            [True, True],
            3,
        ),

    # "train/seq_length": 
    #     (
    #         "seq", 
    #         [10],
    #         3,
    #     ),
    # "train/subgoal_horizons": 
    #     (
    #         "", 
    #         [[10]],
    #         3,
    #     ),
    # "algo/rnn/horizon": 
    #     (
    #         "", 
    #         [10],
    #         3,
    #     ),
    "algo/actor_layer_dims":
        (
            "mlp", 
            [[]],
            # [[300, 400]],
            4,
        ),
    "algo/rnn/hidden_dim": 
        (
            "rnnd", 
            [400],
            # [1000],
            4,
        ),
    # for spirl-h
    "algo/spirl/hierarchical/hidden_dim": 
        (
            "", 
            [400],
            # [1000],
            4,
        ),


    # "algo/gmm/enabled": 
    #     (
    #         "gmm", 
    #         # [True, False],
    #         [True],
    #         5,
    #     ),

    "algo/vae/kl_weight": 
        (
            "kl", 
            # [5e-2, 5e-4, 5e-5, 5e-6],
            [5e-4, 5e-6],
            # [0.05],
            # [0.05, 5e-4],
            # [0.5, 0.05],
            5,
        ),
    # "algo/spirl/use_proprio": 
    #     (
    #         "prop", 
    #         [True],
    #         6,
    #         ["t"],
    #     ),
    
    # "algo/han/action_space/num_3d_kp": 
    #     (
    #         "nkp", 
    #         [3, 10],
    #         6,
    #     ),
    # "algo/han/action_space/use_eef_pos": 
    #     (
    #         "eef", 
    #         # [True, False],
    #         [False],
    #         7,
    #     ),
})

# an example of tuning for BC (no RNN)
PARAMETERS = OrderedDict({
    "train/data": 
        (
            "ds", 
            [
                # "~/Desktop/icra_public/coffee/ajay/base/states_images.hdf5",
                # "~/Desktop/icra_public/coffee/ajay/more/states_images_combined.hdf5",
                # "~/Desktop/icra_public/coffee/ajay/v3/no_gmm/bs_false_no_self_im/no_self_im/states_images_combined.hdf5",
                # "~/Desktop/icra_public/coffee/ajay/v3/no_gmm/bs_true/states_images_combined.hdf5",

                # for cross exp
                "~/Desktop/icra_public/coffee/ajay/v3/no_gmm/bs_false_no_self_im/states_images_combined.hdf5",
            ],
            0,
            [
                # "base",
                # "more",
                # "v3_hg_dagger",
                # "v3_iwr",
                "v3_hg_dagger_cross",
            ],
        ),
    "algo/optim_params/policy/learning_rate/initial": 
        (
            "plr", 
            # [1e-3, 1e-4],
            [1e-4],
            1,
        ),

    "algo/iwr/enabled": 
        (
            "iwr", 
            [True, False],
            2,
        ),

    # "algo/gmm/enabled": 
    #     (
    #         "gmm", 
    #         [True, False],
    #         6,
    #     ),
})


def load_json(json_file, verbose=True):
    with open(json_file, 'r') as f:
        config = json.load(f)
    if verbose:
        print('loading external config: =================')
        print(json.dumps(config, indent=4))
        print('==========================================')
    return config

def save_json(config, json_file):
    with open(json_file, 'w') as f:
        # preserve original key ordering
        json.dump(config, f, sort_keys=False, indent=4)

def name_for_experiment(base_name, parameter_setting, parameter_names):
    """
    Given the base name and the dictionary between parameter names
    and a specific parameter setting, this function generates
    the name for the experiment.
    """
    name = base_name
    for k in parameter_setting:
        # append parameter name and value to end of base name
        if len(PARAMETERS[k][0]) == 0:
            # empty string indicates that naming should be skipped
            continue
        if parameter_names[k] is not None:
            # take name from passed dictionary
            val_str = parameter_names[k]
        else:
            val_str = parameter_setting[k]
            if isinstance(parameter_setting[k], list) or isinstance(parameter_setting[k], tuple):
                # convert list to string to avoid weird spaces and naming problems
                val_str = "_".join([str(x) for x in parameter_setting[k]])
        name += '_{}_{}'.format(PARAMETERS[k][0], val_str)
    return name

def value_for_key(dic, k, v=None):
    """
    Get value for hierarchical dictionary with levels denoted by "/".
    If @v is not None, set that to be the new value.
    """
    val = dic
    subkeys = k.split('/')
    for s in subkeys[:-1]:
        val = val[s]
    if v is not None:
        val[subkeys[-1]] = v
    return val[subkeys[-1]]

def get_parameter_ranges(base_config, all_combinations=False):
    """
    Extract parameter ranges from base json file. If @all_combinations, take
    all possible combinations of the parameter ranges to generate an expanded
    set of values.
    """

    ### TODO: per parameter group, get range(n) as the list in ordered dict, then take all combs between groups, and then use indices to add params to each groups members ###

    # mapping from group id to list of indices to grab from each parameter's list 
    # of values in the parameter group
    parameter_group_indices = OrderedDict()
    for k in PARAMETERS:
        group_id = PARAMETERS[k][2]
        assert isinstance(PARAMETERS[k][1], list)
        num_param_values = len(PARAMETERS[k][1])
        if group_id not in parameter_group_indices:
            parameter_group_indices[group_id] = list(range(num_param_values))
        else:
            assert len(parameter_group_indices[group_id]) == num_param_values, \
                "error: inconsistent number of parameter values in group with id {}".format(group_id)

    if all_combinations:
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
    for k in PARAMETERS:
        parameter_values = PARAMETERS[k][1]
        group_id = PARAMETERS[k][2]
        inds = parameter_group_indices[group_id]
        parameter_ranges[k] = [parameter_values[ind] for ind in inds]

        # add in parameter names if supplied
        parameter_names[k] = None
        if len(PARAMETERS[k]) == 4:
            par_names = PARAMETERS[k][3]
            assert isinstance(par_names, list)
            assert len(par_names) == len(parameter_values)
            parameter_names[k] = [par_names[ind] for ind in inds]

    # ensure that the number of parameter settings is the same per parameter
    first_key = list(parameter_ranges.keys())[0]
    num_settings = len(parameter_ranges[first_key])
    for k in parameter_ranges:
        assert len(parameter_ranges[k]) == num_settings, "inconsistent number of values"

    return parameter_ranges, parameter_names

def generate_jsons(base_json_file, all_combinations=False):

    # base directory for saving jsons
    base_dir = os.path.dirname(base_json_file)

    # read base json
    base_config = load_json(base_json_file)

    # base exp name from this base config
    base_exp_name = base_config['experiment']['name']

    # use base json to determine the parameter ranges
    parameter_ranges, parameter_names = get_parameter_ranges(base_config, all_combinations=all_combinations)

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
        exp_name = name_for_experiment(base_exp_name, setting, maybe_parameter_names)

        # copy old json, but override name, and parameter values
        json_dict = deepcopy(base_config)
        json_dict['experiment']['name'] = exp_name
        for k in parameter_ranges:
            value_for_key(json_dict, k, v=parameter_ranges[k][i])

        # save file in same directory as old json
        json_path = os.path.join(base_dir, "{}.json".format(exp_name))
        save_json(json_dict, json_path)
        json_paths.append(json_path)

    return json_paths, base_config

def script_from_jsons(json_paths, batch, script_name):
    """
    Generates a bash script to run the experiments that correspond to
    the input jsons.
    """
    with open(script_name, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for path in json_paths:
            # output text file
            output_txt = os.path.basename(path)[:-5] # exclude .json

            # command to write to file
            cmd = "python train.py --config {} &> {}.txt\n".format(path, output_txt)
            f.write(cmd)


def main(args):
    # make experiment jsons
    json_paths, base_config = generate_jsons(
        base_json_file=args.config, 
        all_combinations=args.combinations,
    )

    # make a script to run them
    script_from_jsons(
        json_paths=json_paths, 
        batch=base_config['train']['data'], 
        script_name=args.script,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config to use for randomization
    parser.add_argument(
        "--config",
        type=str,
    )

    # Script name to generate
    parser.add_argument(
        "--script",
        type=str,
    )

    # Whether to generate all combinations of parameter ranges.
    parser.add_argument(
        "--combinations",
        action='store_true',
    )

    args = parser.parse_args()
    main(args)

