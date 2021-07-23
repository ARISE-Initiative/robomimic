#!/bin/bash

### NOTE: script was updated from benchmark branch ###

#################################################################
# Preparation of demo.hdf5s                                     #
#################################################################

### All ###

# make sure we convert from internal teleop demo.hdf5 and remove unneeded keys
python teleop_to_env_meta.py --dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5
python remove_hdf5_attr.py --dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5

# can verify using this script
python print_hdf5_attr.py --dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5

### Machine-Generated ###

# datasets were truncated into subsets of trajectories to make it harder
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_dense_done_success.hdf5 --n 1500 --filter_key 1.5k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_done_success.hdf5 --n 1500 --filter_key 1.5k


### Multi-Human (and some Proficient-Human) ###


# first, each person's demos were postprocessed, converted to compatible demo.hdf5, and then split into train-val

# python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo.hdf5
# python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo.hdf5

### note: dataset filter keys were prepared beforehand ###

# first, each person's demos were postprocessed, converted to compatible demo.hdf5, and then split into train-val
python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo.hdf5

# then, the multi-human datasets were merged, keeping track of each peron's demos under filter keys
python merge_hdf5.py --batches \
~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/chen/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/danfei/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/josiah/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/roberto/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/yuke/demo.hdf5 \
--name demo_merged.hdf5 --valid \
--write_filter_keys ajay chen danfei josiah roberto yuke

mv ~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo_merged.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5

# then, additional filter keys were created by merging different people's keys together
python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys ajay josiah --name better --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys danfei yuke --name okay --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys chen roberto --name worse --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys danfei yuke ajay josiah --name okay_better --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys chen roberto ajay josiah --name worse_better --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys chen roberto danfei yuke --name worse_okay --valid

# verify dataset looks okay
python get_dataset_info.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 --verbose


## create dataset size ablation filter keys ##
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_single_expert/demo.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei


python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_single_expert/demo.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei


#################################################################
# Rename filter keys for anonymity.                             #
#################################################################

### rename MH filter keys ###
python rename_filter_keys.py --dataset ~/Desktop/final_benchmark_datasets/lift/mh/demo.hdf5 \
--filter_keys ajay josiah yuke danfei roberto chen \
--rename_filter_keys better_operator_1 better_operator_2 okay_operator_1 okay_operator_2 worse_operator_1 worse_operator_2 \
--valid

python rename_filter_keys.py --dataset ~/Desktop/final_benchmark_datasets/can/mh/demo.hdf5 \
--filter_keys ajay josiah yuke danfei roberto chen \
--rename_filter_keys better_operator_1 better_operator_2 okay_operator_1 okay_operator_2 worse_operator_1 worse_operator_2 \
--valid

python rename_filter_keys.py --dataset ~/Desktop/final_benchmark_datasets/square/mh/demo.hdf5 \
--filter_keys ajay josiah yuke danfei roberto chen \
--rename_filter_keys better_operator_1 better_operator_2 okay_operator_1 okay_operator_2 worse_operator_1 worse_operator_2 \
--valid

python rename_filter_keys.py --dataset ~/Desktop/final_benchmark_datasets/transport/mh/demo.hdf5 \
--filter_keys ajay_josiah yuke_danfei roberto_chen ajay_yuke josiah_chen roberto_danfei \
--rename_filter_keys better okay worse okay_better worse_better worse_okay \
--valid

#################################################################
# Observation Extraction (this part uses the new script)        #
#################################################################


### NOTE: we use done-mode 0 for MG (also include done-mode 2, just in case it works better, for low-dim) ###

### mg ###

python dataset_states_to_obs.py --done_mode 0 \
--dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5 \
--output_name low_dim_sparse.hdf5
python dataset_states_to_obs.py --done_mode 0 \
--dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5 \
--output_name image_sparse.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 0 \
--dataset ~/Desktop/final_benchmark_datasets/can/mg/demo.hdf5 \
--output_name low_dim_sparse.hdf5
python dataset_states_to_obs.py --done_mode 0 \
--dataset ~/Desktop/final_benchmark_datasets/can/mg/demo.hdf5 \
--output_name image_sparse.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# dense --shaped as well
python dataset_states_to_obs.py --done_mode 0 --shaped \
--dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5 \
--output_name low_dim_dense.hdf5
python dataset_states_to_obs.py --done_mode 0 --shaped \
--dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5 \
--output_name image_dense.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 0 --shaped \
--dataset ~/Desktop/final_benchmark_datasets/can/mg/demo.hdf5 \
--output_name low_dim_dense.hdf5
python dataset_states_to_obs.py --done_mode 0 --shaped \
--dataset ~/Desktop/final_benchmark_datasets/can/mg/demo.hdf5 \
--output_name image_dense.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# done mode 2 as well, for low-dim, to try it
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5 \
--output_name low_dim_sparse_done_2.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/can/mg/demo.hdf5 \
--output_name low_dim_sparse_done_2.hdf5
python dataset_states_to_obs.py --done_mode 2 --shaped \
--dataset ~/Desktop/final_benchmark_datasets/lift/mg/demo.hdf5 \
--output_name low_dim_dense_done_2.hdf5
python dataset_states_to_obs.py --done_mode 2 --shaped \
--dataset ~/Desktop/final_benchmark_datasets/can/mg/demo.hdf5 \
--output_name low_dim_dense_done_2.hdf5

### NOTE: we use done-mode 2 for PH / MH ###

### ph ###

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/lift/ph/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/lift/ph/demo.hdf5 \
--output_name image.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/can/ph/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/can/ph/demo.hdf5 \
--output_name image.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/square/ph/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/square/ph/demo.hdf5 \
--output_name image.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/transport/ph/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/transport/ph/demo.hdf5 \
--output_name image.hdf5 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/tool_hang/ph/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/tool_hang/ph/demo.hdf5 \
--output_name image.hdf5 --camera_names sideview robot0_eye_in_hand --camera_height 240 --camera_width 240

### mh ###

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/lift/mh/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/lift/mh/demo.hdf5 \
--output_name image.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/can/mh/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/can/mh/demo.hdf5 \
--output_name image.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/square/mh/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/square/mh/demo.hdf5 \
--output_name image.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/transport/mh/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/transport/mh/demo.hdf5 \
--output_name image.hdf5 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand --camera_height 84 --camera_width 84


### can-paired ###

python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/can/paired/demo.hdf5 \
--output_name low_dim.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset ~/Desktop/final_benchmark_datasets/can/paired/demo.hdf5 \
--output_name image.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

#################################################################
# Dataset Playback Videos                                       #
#################################################################


# dataset playback for RT benchmark datasets

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_success.hdf5 --filter_key 3.9k --video_path ~/Downloads/playback_can_rb_3.9k.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/subopt_can_paired/state_done_2.hdf5 --video_path ~/Downloads/playback_can_paired.mp4 --n 50

# worse (roberto, chen)
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key roberto --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_worse_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key chen --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_worse_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key roberto --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_worse_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key chen --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_worse_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key roberto --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_worse_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key chen --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_worse_2.mp4

# okay (yuke, danfei)
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key yuke --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_okay_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key danfei --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_okay_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key yuke --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_okay_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key danfei --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_okay_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key yuke --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_okay_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key danfei --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_okay_2.mp4

# better (ajay, josiah)
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key ajay --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_better_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key josiah --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_better_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key ajay --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_better_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key josiah --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_better_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key ajay --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_better_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key josiah --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_better_2.mp4


# 6 subsets for transport
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key ajay_josiah --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_better.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key yuke_danfei --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_okay.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key roberto_chen --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_worse.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key ajay_yuke --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_okay_better.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key roberto_danfei --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_worse_okay.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key josiah_chen --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_worse_better.mp4

# SE
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_lift_se.mp4 --n 50
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_can_se.mp4 --n 50
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_square_se.mp4 --n 50
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_transport_se.mp4 --n 50

# tool hang (sim)
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_tool_hanging_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_tool_hang_se.mp4 --n 100 --camera_names sideview


# observation space videos
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_obs/obs_can.mp4 --n 20 --camera_names agentview robot0_eye_in_hand
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_obs/obs_transport.mp4 --n 20 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_tool_hanging_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_obs/obs_tool_hang.mp4 --n 20 --camera_names sideview robot0_eye_in_hand


# real videos
python playback_dataset.py --dataset ~/Desktop/real_robot/lift/demo.hdf5 --use-obs --video_path ~/Downloads/rt_benchmark_playback/playback_lift_real.mp4 --video_skip 1 --n 50
python playback_dataset.py --dataset ~/Desktop/real_robot/can/demo.hdf5 --use-obs --video_path ~/Downloads/rt_benchmark_playback/playback_can_real.mp4 --video_skip 1 --n 50
python playback_dataset.py --dataset ~/Desktop/real_robot/tool_hanging_deadline/demo.hdf5 --use-obs --video_path ~/Downloads/rt_benchmark_playback/playback_tool_hang_real.mp4 --video_skip 1 --n 50


# get first frame videos for all tasks
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_lift_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_can_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_square_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_transport_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_tool_hanging_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_tool_hang_first.mp4 --camera_names sideview --first

### TODO: make sure to edit the file to only grab the relevant camera image first in playback_obs ###
python playback_dataset.py --dataset ~/Desktop/real_robot/lift/demo.hdf5 --use-obs --video_path ~/Downloads/playback_first/playback_lift_real_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/real_robot/can/demo.hdf5 --use-obs --video_path ~/Downloads/playback_first/playback_can_real_first.mp4 --first 
python playback_dataset.py --dataset ~/Desktop/real_robot/tool_hanging_deadline/demo.hdf5 --use-obs --video_path ~/Downloads/playback_first/playback_tool_hang_real_first.mp4 --first

# can paired BCQ rollouts
python test.py --agent /afs/cs.stanford.edu/u/amandlek/installed_libraries/benchmark/slurm/log/batchRL/batchrl_benchmark/hp_sweep/low_dim/hbcq/can_paired_bcq_low_dim/benchmark_can_paired_ld_bc_trained_models/hbcq_ds_can_paired_ld_seed_1/2021-06-12-15-12-19-179527/models/model_epoch_800_PickPlaceCan_success_0.46.pth \
--render_video --video_dir ~/Downloads/can_paired_rollouts --n_rollouts 50 --horizon 400 --seed 1

python test.py --agent /afs/cs.stanford.edu/u/amandlek/installed_libraries/benchmark/slurm/log/batchRL/batchrl_benchmark/hp_sweep/low_dim/bc/can_paired_bc_rnn_low_dim/benchmark_can_paired_ld_bc_trained_models/bc_rnn_ds_can_paired_ld_seed_1/2021-06-08-13-01-35-323829/models/model_epoch_300_PickPlaceCan_success_0.74.pth \
--render_video --video_dir ~/Downloads/can_paired_rollouts_bc_rnn --n_rollouts 50 --horizon 400 --seed 1


#################################################################
# Preparation of roboturk-pilot demo.hdf5s                      #
#################################################################

# NOTE: we have a new, better, mechanism in public release

# download raw data
wget http://cvgl.stanford.edu/projects/roboturk/RoboTurkPilot.zip
unzip RoboTurkPilot.zip

# in RobotTeleop
python process_demo_hdf5.py --folder ~/Desktop/roboturk_v1_test/RoboTurkPilot/bins-Can/

# convert to valid hdf5
python teleop_to_env_meta.py --dataset ~/Desktop/roboturk_v1_test/RoboTurkPilot/bins-Can/demo_new.hdf5 

# split into fastest 225
python split_fastest.py --dataset ~/Desktop/roboturk_v1_test/RoboTurkPilot/bins-Can/demo_new.hdf5 --n 225
python split_train_val.py --dataset ~/Desktop/roboturk_v1_test/RoboTurkPilot/bins-Can/demo_new.hdf5 --filter_key fastest_225

# playback dataset
python playback_dataset.py --dataset ~/Desktop/roboturk_v1_test/RoboTurkPilot/bins-Can/demo_new.hdf5 \
--n 10 --video_path ~/Downloads/playback_rt_cans_225.mp4 --render_image_names agentview --filter_key fastest_225

python playback_dataset.py --dataset ~/Desktop/roboturk_v1_test/RoboTurkPilot/bins-Can/demo_new.hdf5 \
--n 10 --video_path ~/Downloads/playback_rt_cans_act_225.mp4 --render_image_names agentview --filter_key fastest_225 --use-actions

# low dim
python dataset_states_to_obs.py --dataset ~/Desktop/roboturk_v1_test/RoboTurkPilot/bins-Can/demo_new.hdf5 \
--output_name low_dim.hdf5 --done_mode 2

# test image - first 10 traj
python dataset_states_to_obs.py --done_mode 2 --dataset ~/Desktop/roboturk_v1_test/RoboTurkPilot/bins-Can/demo_new.hdf5 \
--output_name image.hdf5 --camera_names agentview --camera_height 84 --camera_width 84 --n 5
