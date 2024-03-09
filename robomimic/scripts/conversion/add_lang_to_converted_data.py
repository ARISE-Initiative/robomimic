"""
Add image information to existing droid hdf5 file
"""
import h5py
import os
import numpy as np
import glob
from tqdm import tqdm
import argparse
import shutil
import torch
import random
import json
import cv2

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16)
model.to('cuda')

"""
Follow instructions here to setup zed:
https://www.stereolabs.com/docs/installation/linux/
"""
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

def add_lang(path, raw_lang, args):
    output_path = os.path.join(os.path.dirname(path), "trajectory_im{}.h5".format(args.imsize))
    f = h5py.File(output_path, "a")
    print(f.keys())

    # Extract language data
    if "lang_fixed" not in f["observation"]:
        f["observation"].create_group("lang_fixed")
    lang_grp = f["observation/lang_fixed"]

    H = f["observation/robot_state/cartesian_position"].shape[0]
    encoded_input = tokenizer(raw_lang, return_tensors='pt').to('cuda')
    outputs = model(**encoded_input)
    encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(0).repeat(H, 1)
    if "language_raw" not in f["observation/lang_fixed"]:
        lang_grp.create_dataset("language_raw", data=[raw_lang]*H)
        lang_grp.create_dataset("language_distilbert", data=encoded_lang.cpu().detach().numpy())
    else:
        f["observation/lang_fixed/language_raw"][...] = [raw_lang]*H
        f["observation/lang_fixed/language_distilbert"][...] = encoded_lang.cpu().detach().numpy()
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest_file",
        type=str,
        help="manifest file path",
    )

    parser.add_argument(
        "--imsize",
        type=int,
        default=128,
        help="image size (w and h)",
    )
    
    args = parser.parse_args()

    with open(args.manifest_file, 'r') as file:
        datasets = json.load(file)

    print("adding lang to datasets...")
    random.shuffle(datasets)
    for item in tqdm(datasets):
        d, l = item['path'], item['lang']
        d = os.path.expanduser(d)
        add_lang(d, l, args)
