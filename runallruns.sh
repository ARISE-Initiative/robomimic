#!/bin/bash

# bash /mnt/fsx/surajnair/code/robomimic/runallruns.sh 
# List of commands to run

# # P4D1
# commands=(
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_eval_multi_cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_eval_foodinbowl_cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_eval_openmicro_cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_eval_closemicro_cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_balanced_broad_eval_multi_cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_balanced_broad_eval_foodinbowl_cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_balanced_broad_eval_openmicro_cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_balanced_broad_eval_closemicro_cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# )

# # P4D2
# commands=(
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-40-14/json/cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None_ds_eval_multi.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-40-14/json/cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None_ds_eval_foodinbowl.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-40-14/json/cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None_ds_eval_openmicro.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-40-14/json/cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None_ds_eval_closemicro.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-40-14/json/cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_multi.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-40-14/json/cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_foodinbowl.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-40-14/json/cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_openmicro.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-40-14/json/cams_wrist_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_closemicro.json"
# )

# # P4D3 (Ashwin P4D1)
commands=(
"/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_wrist_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_eval_multi.json"
"/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_wrist_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_eval_foodinbowl.json"
"/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_wrist_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_eval_openmicro.json"
"/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_wrist_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_eval_closemicro.json"
"/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_wrist_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_multi.json"
"/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_wrist_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_foodinbowl.json"
"/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_wrist_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_openmicro.json"
"/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_wrist_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_closemicro.json"
)

# # P4DE 1
# commands=(
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_eval_multi_cams_3cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_eval_foodinbowl_cams_3cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_eval_openmicro_cams_3cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_eval_closemicro_cams_3cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_balanced_broad_eval_multi_cams_3cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_balanced_broad_eval_foodinbowl_cams_3cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_balanced_broad_eval_openmicro_cams_3cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-16-01/json/ds_balanced_broad_eval_closemicro_cams_3cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-cam-lang_visenc_DeFiNeVisualCore_fuser_None.json"
# )

# # P4DE 2
# commands=(
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_3cams_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_eval_multi.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_3cams_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_eval_foodinbowl.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_3cams_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_eval_openmicro.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_3cams_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_eval_closemicro.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_3cams_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_multi.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_3cams_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_foodinbowl.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_3cams_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_openmicro.json"
# "/mnt/fsx/surajnair/mambaforge/envs/r2d2/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/train.py --config /mnt/fsx/surajnair/tmp/autogen_configs/ril/diffusion_policy/r2d2/im/12-29-None/12-29-23-01-43-10/json/cams_3cams_goal_mode_geom_truncated_geom_factor_0.3_ldkeys_proprio-cam_visenc_DeFiNeVisualCore_fuser_None_ds_balanced_broad_eval_closemicro.json"
# )



# Number of GPUs available (adjust as needed)
num_gpus=8

# Loop through the commands and launch them in separate screen sessions
for ((i=0; i<${#commands[@]}; i++)); do
    # Get the current command
    cmd="${commands[$i]}"

    # Calculate the GPU index for this command
    gpu_index=$((i % num_gpus))

    # Set the GPU index for this command
    export CUDA_VISIBLE_DEVICES=$gpu_index

    # Create a unique screen session for each command
    screen -dmS r2d2_$i $cmd

    # Sleep for a few seconds to allow screen to start
    sleep 5
done

echo "Launched ${#commands[@]} commands in separate screen sessions."
