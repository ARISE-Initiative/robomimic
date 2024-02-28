#!/bin/bash
datasets=(
    "austin_buds_dataset_converted_externally_to_rlds"    
    "jaco_play"
    "austin_sailor_dataset_converted_externally_to_rlds"     
    "kuka"
    "austin_sirius_dataset_converted_externally_to_rlds"     
    # "language_table"
    "bc_z"                                                   
    "nyu_door_opening_surprising_effectiveness"
    "berkeley_autolab_ur5"                                   
    "nyu_franka_play_dataset_converted_externally_to_rlds"
    "berkeley_cable_routing"                                 
    "ppgm"
    "berkeley_fanuc_manipulation"                            
    "roboturk"
    "bridge_dataset"                                         
    "stanford_hydra_dataset_converted_externally_to_rlds"
    # "bridge_orig"                                            
    "taco_play"
    "cmu_stretch"                                            
    "toto"
    "dlr_edan_shared_control_converted_externally_to_rlds"   
    "ucsd_kitchen_dataset_converted_externally_to_rlds"
    "fractal20220817_data"                                   
    "utaustin_mutex"
    "furniture_bench_dataset_converted_externally_to_rlds"   
    "viola"
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
)

for ((i=0; i<${#datasets[@]}; i++)); do
    # Get the current command
    d="${datasets[$i]}"

    /mnt/fsx/surajnair/mambaforge/envs/octo/bin/python /mnt/fsx/surajnair/code/robomimic/robomimic/scripts/conversion/oxe_to_h5.py --out_dir=/mnt/fsx/surajnair/datasets/oxe_hdf5/$d --dataset=$d; 
    done
done
