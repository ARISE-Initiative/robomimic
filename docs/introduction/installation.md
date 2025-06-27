# Installation

## Requirements

- Mac OS X or Linux machine
- Python >= 3.6 (recommended 3.8.0)
- [conda](https://www.anaconda.com/products/individual) 
  - [virtualenv](https://virtualenv.pypa.io/en/latest/) is also an acceptable alternative, but we assume you have conda installed in our examples below

## Install robomimic

<div class="admonition note">
<p class="admonition-title">1. Create and activate conda environment</p>

```sh
$ conda create -n robomimic_venv python=3.8.0
$ conda activate robomimic_venv
```

</div>

<div class="admonition note">
<p class="admonition-title">2. Install PyTorch</p>

[PyTorch](https://pytorch.org/) reference

<details>
  <summary><b>Option 1: Mac</b></summary>
<p>

```sh
# Can change pytorch, torchvision versions
# We don't install cudatoolkit since Mac does not have NVIDIA GPU
$ conda install pytorch==2.0.0 torchvision==0.15.1 -c pytorch
```

</p>
</details>

<details>
  <summary><b>Option 2: Linux</b></summary>
<p>

```sh
# Can change pytorch, torchvision versions
$ conda install pytorch==2.0.0 torchvision==0.15.1 -c pytorch
```

</p>
</details>

</div>


<div class="admonition note">
<p class="admonition-title">3. Install robomimic</p>

<details>
  <summary><b>Option 1: Install from source <i>(recommended)</i></b></summary>
<p>

```sh
$ cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
$ git clone https://github.com/ARISE-Initiative/robomimic.git
$ cd robomimic
$ pip install -e .
```

</p>
</details>

<details>
  <summary><b>Option 2: Install via pip</b></summary>
<p>

```sh
$ pip install robomimic
```

</p>
</details>

</div>

<div class="admonition warning">
<p class="admonition-title">Warning! Additional dependencies might be required</p>

This is all you need for using the suite of algorithms and utilities packaged with robomimic. However, to use our demonstration datasets, you may need additional dependencies. Please see the [datasets page](../datasets/overview.html) for more information on downloading datasets and reproducing experiments, and see [the simulators section below](installation.html#install-simulators).
</div>


# Optional Installations

## Downloading datasets and reproducing experiments

See the [datasets page](../datasets/overview.html) for more information on downloading datasets and reproducing experiments.

## Install simulators

If you would like to run robomimic examples and work with released datasets, please install the following simulators:

<details>
  <summary><b>robosuite</b></summary>
<p>
 Required for running most robomimic examples and released datasets. Compatible with robosuite v1.2+. Install via:

```sh
# From source (recommended)
$ cd <PATH_TO_INSTALL_DIR>
$ git clone https://github.com/ARISE-Initiative/robosuite.git
$ cd robosuite
$ pip install -r requirements.txt
OR
# Via pip
$ pip install robosuite
```

**(Optional)** to use our released datasets and reproduce our experiments, switch to the `v1.5.1` branch (requires installing robosuite from source):

```sh
git checkout v1.5.1
```

<!-- <div class="admonition warning">
<p class="admonition-title">mujoco-py dependency!</p>

Robosuite requires [mujoco-py](https://github.com/openai/mujoco-py). If you are on an Ubuntu machine with a GPU, you should make sure that the `GPU` version of `mujoco-py` gets built, so that image rendering is fast (crucial for working with image datasets!).

An easy way to ensure this is to clone the repository, change [this line](https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/builder.py#L74) to `Builder = LinuxGPUExtensionBuilder`, and install from source by running `pip install -e .` in the `mujoco-py` root directory.

</div> -->

</p>
</details>


<details>
  <summary><b>D4RL</b></summary>
<p>

Useful for running some of our algorithms on the [D4RL](https://arxiv.org/abs/2004.07219) datasets.

Install via the instructions [here](https://github.com/rail-berkeley/d4rl).

</p>
</details>


## Test your installation
This assumes you have installed robomimic from source.

Run a quick debugging (dummy) training loop to make sure robomimic is installed correctly:
```sh
$ cd <PATH_TO_ROBOMIMIC_INSTALL_DIR>
$ python examples/train_bc_rnn.py --debug
```

Run a much more thorough test of several algorithms and scripts (**Warning: this script may take several minutes to finish!**):
```sh
$ cd <PATH_TO_ROBOMIMIC_INSTALL_DIR>/tests
$ bash test.sh
```

To run some easy examples, see the [Getting Started](./getting_started.html) section.

## Install documentation dependencies

If you plan to contribute to the repository and add new features, you must install the additional requirements required to build the documentation locally:

```sh
$ pip install -r requirements-docs.txt
```

You can test generating the documentation and viewing it locally in a web browser:
```sh
$ cd <PATH_TO_ROBOMIMIC_INSTALL_DIR>/docs
$ make clean
$ make apidoc
$ make html
$ make prep
$ cp -r images _build/html/
```

There should be a generated `_build` folder - navigate to `_build/html/` and open `index.html` in a web browser to view the documentation.