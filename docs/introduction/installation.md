# Installation

## Requirements

- Mac OS X or Linux machine
- Python >= 3.6 (recommended 3.7.9)
- [conda](https://www.anaconda.com/products/individual) 
  - [virtualenv](https://virtualenv.pypa.io/en/latest/) is also an acceptable alternative, but we assume you have conda installed in our examples below

## Installation Steps
1. Create and activate conda environment
```sh
$ conda create -n robomimic_venv python=3.7.9
$ conda activate robomimic_venv
```

2. Install [PyTorch](https://pytorch.org/)
<details>
  <summary><b>Mac</b></summary>
<p>

```sh
# Can change pytorch, torchvision versions
# We don't install cudatoolkit since Mac does not have NVIDIA GPU
$ conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
```

</p>
</details>

<details>
  <summary><b>Linux</b></summary>
<p>

```sh
# Can change pytorch, torchvision versions
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

</p>
</details>


3. Install robomimic

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


## Testing


<div class="admonition note">
<p class="admonition-title">Careful, this may change your GPU drivers!</p>

There are several system dependencies to correctly run iGibson on Linux, mostly related to Nvidia drivers and Cuda.
In case your system is a clean Ubuntu 20.04, you can run the following commands as root/superuser to install all required dependencies:

<details>
  <summary>Click to expand the code to install the dependencies including Nvidia drivers in headless mode to use for example in a cluster:</summary>
<p>

```bash
# Add the nvidia ubuntu repositories
apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# The following cuda libraries are required to compile igibson
apt-get update && apt-get update && apt-get install -y --no-install-recommends \
    nvidia-headless-470 \
    cuda-cudart-11-1=11.1.74-1 \
    cuda-compat-11-1 \
    cuda-command-line-tools-11-1=11.1.1-1 \
    cuda-libraries-dev-11-1=11.1.1-1 \

# For building and running igibson
apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    g++ \
    libegl-dev
```
</p>
</details>
<details>
  <summary>Click to expand the code to install the dependencies including Nvidia drivers to render on screen for example on a desktop computer:</summary>
<p>

```bash
# Add the nvidia ubuntu repositories
apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# The following cuda libraries are required to compile igibson
apt-get update && apt-get update && apt-get install -y --no-install-recommends \
    xserver-xorg-video-nvidia-470 \
    cuda-cudart-11-1=11.1.74-1 \
    cuda-compat-11-1 \
    cuda-command-line-tools-11-1=11.1.1-1 \
    cuda-libraries-dev-11-1=11.1.1-1 \

# For building and running igibson
apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    g++ \
    libegl-dev
```

</p>
</details>

</div>

## Install from source (preferred)

First, clone the repository from github.

```sh
# clone the repository
$ git clone https://github.com/ARISE-Initiative/robomimic.git
$ cd robomimic
```

Next, install the repository in editable mode with pip.

```sh
# install such that changes to source code will be reflected directly in the installation
$ pip install -e .
```

To run a quick test, without any dependence on simulators, run the following example

```sh
$ python examples/simple_train_loop.py
```

For maximum functionality though, we also recommend installing [robosuite](https://robosuite.ai/) -- see the section on simulators below.

## Install from pip

While not preferred, the repository can also be installed directly via pip.

```sh
$ pip install robomimic
```

## Install simulators

While the **robomimic** repository does not depend on particular simulators, installing the following simulators is strongly encouraged, in order to run the examples provided with the repository and work with released datasets.

### Robosuite

Most of our examples and released datasets use [robosuite](https://robosuite.ai/), so we strongly recommend installing it. Install it using [the instructions here](https://robosuite.ai/docs/installation.html), and once again, we recommend installing from source. While the repository is compatible with robosuite `v1.2+`, switch to the `offline_study` branch (by running `git checkout offline_study` in the `robosuite` root directory) in order to easily work with our released datasets and reproduce our experiments.

**Note:** robosuite has a dependency on [mujoco-py](https://github.com/openai/mujoco-py). If you are on an Ubuntu machine with a GPU, you should make sure that the `GPU` version of `mujoco-py` gets built, so that image rendering is fast (this is extremely important for working with image datasets). An easy way to ensure this is to clone the repository, change [this line](https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/builder.py#L74) to `Builder = LinuxGPUExtensionBuilder`, and install from source by running `pip install -e .` in the `mujoco-py` root directory.

### D4RL

We also have examples to run some of our algorithms on the [D4RL](https://arxiv.org/abs/2004.07219) datasets. Follow the instructions [here](https://github.com/rail-berkeley/d4rl) to install them, in order to reproduce our results or run further evaluations on these datasets.

## Test your installation

To run a quick test, run the following script (see the [Getting Started](./quickstart.html#run-a-quick-example) section for more information).

```sh
$ python examples/train_bc_rnn.py --debug
```

To run a much more thorough test of several algorithms and scripts, navigate to the `tests` directory and run the following command. **Warning: this script may take several minutes to finish.**

```sh
$ bash test.sh
```

## Downloading released datasets

To download and get started with the suite of released datasets, please see [this section](./results.html#downloading-released-datasets).

## Installation for generating docs

If you plan to contribute to the repository and add new features, you may want to install additional requirements required to build the documentation locally (in case the docs need to be updated).

```sh
$ pip install -r requirements-docs.txt
```

Then, you can test generating the documentation and viewing it locally in a web browser. Run the following commands to generate documentation.

```sh
$ cd docs/
$ make clean
$ make apidoc
$ make html
```

There should be a generated `_build` folder - navigate to `_build/html/` and open `index.html` in a web browser to view the documentation.