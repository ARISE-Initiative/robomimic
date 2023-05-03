#!/bin/bash

# This script holds the commands that were used to go from raw robosuite demo.hdf5 files
# to our processed low-dim and image hdf5 files.

BASE_DATASET_DIR="../../datasets"
echo "Using base dataset directory: $BASE_DATASET_DIR"


### NOTE: we use done-mode 0 for MG (dones on task success) ###


### mg ###


# lift - mg, sparse
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/lift/mg/demo_v141.hdf5 \
--output_name low_dim_sparse_v141.hdf5
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/lift/mg/demo_v141.hdf5 \
--output_name image_sparse_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# lift - mg, dense
python dataset_states_to_obs.py --done_mode 0 --shaped \
--dataset $BASE_DATASET_DIR/lift/mg/demo_v141.hdf5 \
--output_name low_dim_dense_v141.hdf5
python dataset_states_to_obs.py --done_mode 0 --shaped \
--dataset $BASE_DATASET_DIR/lift/mg/demo_v141.hdf5 \
--output_name image_dense_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# can - mg, sparse
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/can/mg/demo_v141.hdf5 \
--output_name low_dim_sparse_v141.hdf5
python dataset_states_to_obs.py --done_mode 0 \
--dataset $BASE_DATASET_DIR/can/mg/demo_v141.hdf5 \
--output_name image_sparse_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# can - mg, dense
python dataset_states_to_obs.py --done_mode 0 --shaped \
--dataset $BASE_DATASET_DIR/can/mg/demo_v141.hdf5 \
--output_name low_dim_dense_v141.hdf5
python dataset_states_to_obs.py --done_mode 0 --shaped \
--dataset $BASE_DATASET_DIR/can/mg/demo_v141.hdf5 \
--output_name image_dense_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84


### NOTE: we use done-mode 2 for PH / MH (dones on task success and end of trajectory) ###


### ph ###


# lift - ph
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/lift/ph/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/lift/ph/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# can - ph
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/can/ph/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/can/ph/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# square - ph
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/square/ph/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/square/ph/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# transport - ph
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/transport/ph/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/transport/ph/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand --camera_height 84 --camera_width 84

# tool hang - ph
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/tool_hang/ph/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/tool_hang/ph/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names sideview robot0_eye_in_hand --camera_height 240 --camera_width 240


### mh ###


# lift - mh
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/lift/mh/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/lift/mh/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# can - mh
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/can/mh/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/can/mh/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# square - mh
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/square/mh/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/square/mh/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

# transport - mh
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/transport/mh/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/transport/mh/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand --camera_height 84 --camera_width 84


### can-paired ###


python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/can/paired/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python dataset_states_to_obs.py --done_mode 2 \
--dataset $BASE_DATASET_DIR/can/paired/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
