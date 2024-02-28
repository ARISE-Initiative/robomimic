import random
import json
import numpy as np
import os
import csv

EVAL_DATA_ROOT_DIR = "/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads-eval/"
RAW_ANNOTATIONS = os.path.join(EVAL_DATA_ROOT_DIR, "tri-multi-task-eval-lang-annotations.csv")
METADATA_DIR = "/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads-eval-json/"

def read_csv(file_path):
    data_list = []
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data_list.append(row)
    return data_list

# Update CSV to have dummy language annotations for single task data
single_task_evals = {
    "TRI_food_in_bowl": "Put the food item in the bowl", 
    "TRI_microwave_close": "Close the microwave", 
    "TRI_microwave_open": "Open the microwave"}

with open(RAW_ANNOTATIONS, 'a', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    for task in single_task_evals.keys():
        for traj_folder in os.listdir(os.path.join(EVAL_DATA_ROOT_DIR, task)):
            new_line = [os.path.join(task, traj_folder, "recordings"), single_task_evals[task]]
            csv_writer.writerow(new_line)

csv_data = read_csv(RAW_ANNOTATIONS)

aggregated_annotations_eval = {}
for _, annotation_info in enumerate(csv_data):
    path = annotation_info[0].split("/recordings")[0]
    lang = annotation_info[1]
    hdf5_path = os.path.join(path, "trajectory.h5")
    uuid = path.replace("/", "-")
    uuid_annotations = f"pilot/{uuid}.mp4"
    aggregated_annotations_eval[uuid_annotations] = {
        "language_instruction1": lang
    }
    metadata_path = os.path.join(METADATA_DIR, f"metadata_{uuid}.json")
    with open(metadata_path, 'w') as metadata_path:
        json.dump({"uuid": uuid, "hdf5_path": hdf5_path, "lab": ""}, metadata_path)

with open(os.path.join(METADATA_DIR, "aggregated_annotations_eval.json"), 'w') as annotations_path:
    json.dump(aggregated_annotations_eval, annotations_path, indent=4)