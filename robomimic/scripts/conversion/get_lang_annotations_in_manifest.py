import random
import json
import numpy as np
import os

DATA_ROOT_DIR = "/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads/"
ANNOTATIONS_DIR = "/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads-json/"
AGGREGATED_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, "aggregated-annotations.json")

EVAL_DATA_ROOT_DIR = "/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads-eval/"
EVAL_ANNOTATIONS_DIR = "/mnt/fsx/surajnair/datasets/r2d2-data/lab-uploads-eval-json/"
AGGREGATED_ANNOTATIONS_EVAL_PATH = os.path.join(EVAL_ANNOTATIONS_DIR, "aggregated_annotations_eval.json")

with open(AGGREGATED_ANNOTATIONS_PATH, "r") as a_file:
    ANNOTATIONS = json.load(a_file)

with open(AGGREGATED_ANNOTATIONS_EVAL_PATH, "r") as a_file:
    ANNOTATIONS_EVAL = json.load(a_file)

def fill_hdf5_to_lang_map(broaddataset_lang, annotations, data_root, annotations_dir):
    print("ANNOTATIONS DIR: ", annotations_dir)
    for i, uuid in enumerate(list(annotations.keys())):
        if i % 1000 == 0:
            print("I: ", i)
        ann = annotations[uuid]
        lang_keys = [l_key for l_key in ann.keys() if "language_instruction" in l_key]
        lang_annotations = [ann[l_key] for l_key in lang_keys]
        # Select random annotation
        lang_annotation = np.random.choice(lang_annotations)
        # Get metadata file
        uuid = uuid.split("/")[-1].split(".")[0]
        metadata_path = os.path.join(annotations_dir, f"metadata_{uuid}.json")
        if not os.path.exists(metadata_path) or lang_annotation in [""]:
            continue
        with open(metadata_path, "r") as m_file:
            metadata = json.load(m_file)
        hdf5_path = os.path.join(data_root, metadata["lab"], metadata["hdf5_path"]).replace('trajectory', 'trajectory_im128')
        broaddataset_lang.append({"path": hdf5_path, "lang": lang_annotation})

broaddataset_lang = []

# Get hdf5 to lang for broad data
fill_hdf5_to_lang_map(broaddataset_lang, ANNOTATIONS, DATA_ROOT_DIR, ANNOTATIONS_DIR)
# Get hdf5 to lang for eval data
fill_hdf5_to_lang_map(broaddataset_lang, ANNOTATIONS_EVAL, EVAL_DATA_ROOT_DIR, EVAL_ANNOTATIONS_DIR)

# Save broaddataset_lang to a new manifest file called manifest_lang.json
json.dump(broaddataset_lang, open(os.path.join(DATA_ROOT_DIR.split("lab-uploads")[0], "manifest_lang.json"), "w"), indent=4)