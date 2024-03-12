# DROID Policy Learning and Evaluation

This repository contains code for training and evaluating policies on the DROID(TODO(Karl/Sasha): add link to DROID website) dataset. DROID is a large-scale, in-the-wild robot manipulation dataset. This codebase is built as a fork of [`robomimic`](https://robomimic.github.io/), a popular repository for imitation learning algorithm development. For more information about DROID, please see the following links: [**[Homepage]**](XXX) &ensp; [**[Documentation]**](XXX) &ensp; [**[Paper]**](XXX) &ensp; [**[Dataset Visualizer]**](XXX).

-------
## Installation
Create a python3 conda environment (tested with Python 3.10) and run the following:

1. Create python 3.10 conda environment: `conda create --name droid_policy_learning python=3.10`
2. Activate the conda environment: `conda activate droid_policy_learning`
3. Install [octo](https://github.com/octo-models/octo) (used for data loading)
4. Run `pip install -e .` in `robomimic`. Make sure you are on the `r2d2` branch.

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

We also provide a stand-alone example to load data from DROID [here](examples/droid_dataloader.py).

-------
## Code Structure

|                           | File                                                    | Description                                                                   |
|---------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------|
| Hyperparameters           | [droid_runs_language_conditioned_rlds.py](robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py)     | Generates a config based on defined hyperparameters  |
| Training Loop             | [train.py](robomimic/scripts/train.py)                  | Main training script.                                                         |
| Datasets                  | [dataset.py](octo/data/dataset.py)                      | Functions for creating datasets and computing dataset statistics,             |
| RLDS Data Processing      | [rlds_utils.py](robomimic/utils/rlds_utils.py)    | Processing to convert RLDS dataset into dataset compatible for DROID training                      |
| General Algorithm Class   | [algo.py](robomimic/algo/algo.py)             | Defines a high level template for all algorithms (eg. diffusion policy) to extend           |
| Diffusion Policy          | [diffusion_policy.py](robomimic/algo/diffusion_policy.py)    | Implementation of diffusion policy |
| Observation Processing    | [obs_nets.py](robomimic/models/obs_nets.py)    | General observation pre-processing/encoding |
| Visualization             | [vis_utils.py](robomimic/utils/vis_utils.py) | Utilities for generating trajectory visualizations                      |

-------

## Evaluating Trained Policies
To evaluate policies, make sure that you additionally install [DROID](https://github.com/AlexanderKhazatsky/DROID) in your conda environment and then run:
```python
python scripts/evaluation/evaluate_policy.py
```
from the DROID root directory. Make sure to use the appropriate command line arguments for the model checkpoint path and whether to do goal or language conditioning, and then follow
all resulting prompts in the terminal. To replicate experiments from the paper, use the language conditioning mode.

-------

## Training Policies with HDF5 Format
Natively, robomimic uses HDF5 files to store and load data. While we mainly support RLDS as the data format for training with DROID, [here](https://github.com/ashwin-balakrishna96/robomimic/tree/r2d2/README_hdf5.md) are instructions for how to run training with the HDF5 data format.

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
