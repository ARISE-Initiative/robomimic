# DROID Policy Learning and Evaluation

This repository contains code for training and evaluating policies on the DROID(TODO(Karl/Sasha): add link to DROID website) dataset. DROID is a large-scale, in-the-wild robot manipulation dataset. This codebase is built as a fork of [`robomimic`](https://robomimic.github.io/), a popular repository for imitation learning algorithm development. For more information about DROID, please see the following links: [**[Homepage]**](XXX) &ensp; [**[Documentation]**](XXX) &ensp; [**[Paper]**](XXX) &ensp; [**[Dataset Visualizer]**](XXX).

-------
## Installation
Create a python3 conda environment (tested with Python 3.10) and run the following:

1. Create python 3.10 conda environment: `conda create --name droid_policy_learning python=3.10`
2. Activate the conda environment: `conda activate droid_policy_learning`
3. Install [octo](https://github.com/octo-models/octo) (used for data loading)
4. Run `pip install -e .` in `robomimic`

With this you are all set up for training policies on DROID. If you want to evaluate your policies on a real robot DROID setup, 
please install the DROID robot controller in the same conda environment (follow the instructions [here](https://github.com/AlexanderKhazatsky/DROID)).

-------
## Preparing Datasets
We provide all DROID datasets in RLDS format, which makes it easy to co-train with various other robot-learning datasets (such as those in the [Open X-Embodiment](https://robotics-transformer-x.github.io/)).

To download the DROID dataset from the Google cloud bucket, install the [gsutil package](https://cloud.google.com/storage/docs/gsutil_install) and run the following command (Note: the full dataset is XXX TB in size):
```
gsutil -m cp -r XXX <path_to_your_target_dir>
```

We also provide a small (2GB) example dataset with 100 DROID trajectories that uses the same format as the full RLDS dataset and can be used for code prototyping and debugging:
```
gsutil -m cp -r XXX <path_to_your_target_dir>
```

For good performance of DROID policies in your target setting, it is helpful to include a small number of demonstrations in your target domain into the training mix ("co-training"). 
Please follow the instructions [here](XXX) for collecting a small teleoperated dataset in your target domain and converting it to the RLDS training format.
Make sure that all datasets you want to train on are under the same root directory `DATA_PATH`.

-------
## Training
To train policies, update `DATA_PATH`, `EXP_LOG_PATH`, and `EXP_NAMES` in `robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py` and then run:

`python robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py --wandb_proj_name <WANDB_PROJ_NAME>`

This will generate a python command that can be run to launch training. You can also update other training parameters within `robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py`. Please see the `robomimic` documentation for more information on how `robomimic` configs are defined. The three
most important parameters in this file are:

- `DATA_PATH`: This is the directory in which all RLDS datasets were prepared.
- `EXP_LOG_PATH`: This is the path at which experimental data (eg. policy checkpoints) will be stored.
- `EXP_NAMES`: This defines the name of each experiment (as will be logged in `wandb`), the RLDS datasets corresponding to that experiment, and the desired sample weights between those datasets. See `robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py` for a template on how this should be formatted.

During training, we use a [_shuffle buffer_](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) to ensure that training samples are properly randomized. It is important to use a large enough shuffle buffer size.
The default `shuffle_buffer_size` is set to `500000`, but you may need to reduce this based on your RAM availability. For best results, we recommend using `shuffle_buffer_size >= 100000` if possible. All polices were trained on a single NVIDIA A100 GPU.

To specify your information for Weights and Biases logging, make sure to update the `WANDB_ENTITY` and `WANDB_API_KEY` values in `robomimic/macros.py`.

-------
## Evaluation on a DROID robot station
Make sure [DROID](https://github.com/AlexanderKhazatsky/DROID) is installed and follow the policy evaluation instructions at the bottom of the README. 

-------

## Training Policies with HDF5 Format
While we recommend using RLDS, HDF5 can be useful for debugging on smaller datasets, and is supported by robomimic natively. 

### Convert from raw DROID data format to HDF5

First you need to make sure to install the ZED SDK, follow the instructions [here](https://www.stereolabs.com/docs/installation/linux/) for your CUDA version and the accompanying `pyzed` package. Then run
`python robomimic/scripts/conversion/convert_droid.py  --folder <PATH_TO_DROID_DATA_FOLDER> --imsize 128`
which will populate each demo folder with an HDF5 file `trajectory_im128.h5` which contains the full observations and actions for that demo. 

### Composing a manifest file
You may want to subselect certain demos to train on. As a result, we assume that you define a manifest json file which contains a list of demos, including the path to each H5 file and the associated language instruction. For example:
```
[
    {
        "path": "/fullpathA/trajectory_im128.h5",
        "lang": "Put the apple on the plate"
    },
    {
        "path": "/fullpathB/trajectory_im128.h5",
        "lang": "Move the fork to the sink"
    },
    ...
]
```

### Adding language embeddings to HDF5
For the files and language specified in the above manifest JSON, run:
`python robomimic/scripts/conversion/add_lang_to_converted_data.py --manifest_file <PATH_TO_MANIFEST_FILE> --imsize 128`
to compute DistilBERT embeddings of each language instruction and add it as an observation key to the HDF5. 

### Run training
To train policies, update `MANIFEST_PATH`, `EXP_LOG_PATH`, in `robomimic/scripts/config_gen/droid_runs_language_conditioned.py` and then run:

`python robomimic/scripts/config_gen/droid_runs_language_conditioned.py --wandb_proj_name <WANDB_PROJ_NAME>`

This will generate a python command that can be run to launch training. You can also update other training parameters within `robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py`. Please see the `robomimic` documentation for more information on how `robomimic` configs are defined. The three
most important parameters in this file are:

- `MANIFEST_PATH`: This is the manifest JSON for the training data you want to use.
- `MANIFEST_2_PATH`: You can optionally set a second manfiest for another dataset to do 50-50 co-training with. 
- `EXP_LOG_PATH`: This is the path at which experimental data (eg. policy checkpoints) will be stored.

------------
## Citation

```
@misc{droid_2024,
    title={DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
    author = {XXX},
    howpublished  = {\url{XXX}},
    year = {2024},
}
```
