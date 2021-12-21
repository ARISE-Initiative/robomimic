#!/bin/bash

# This script holds the commands that were used to go from raw robosuite demo.hdf5 files
# to our processed low-dim and image hdf5 files.

BASE_DATASET_DIR="../../datasets"
POST_FIX="_offline_study_mj211"
echo "Using base dataset directory: $BASE_DATASET_DIR"


### ph ###

# square - ph
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/square/ph/demo.hdf5 \
--output_name low_dim_{$POST_FIX}.hdf5 &
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/square/ph/demo.hdf5 \
--output_name image_{$POST_FIX}.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 &


### mh ###

# can - mh
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/can/mh/demo.hdf5 \
--output_name low_dim_{$POST_FIX}.hdf5 &

# square - mh
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/square/mh/demo.hdf5 \
--output_name low_dim_{$POST_FIX}.hdf5 &
wait
