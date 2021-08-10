# Installation

**robomimic** officially supports Mac OS X and Linux on Python 3. We strongly recommend using a virtual environment with [conda](https://www.anaconda.com/products/individual) ([virtualenv](https://virtualenv.pypa.io/en/latest/) is also an acceptable alternative). To get started, create a virtual env (we use conda in our examples below).

```sh
# create a python 3.7 virtual environment
$ conda create -n robomimic_venv python=3.7.9
# activate virtual env
$ conda activate robomimic_venv
```

Next, install [PyTorch](https://pytorch.org/) (in our example below, we chose to use version `1.6.0` with CUDA `10.2`). You can omit the `cudatoolkit=10.2` if you're on a machine without a CUDA-capable GPU (such as a Macbook).

```sh
# install pytorch with specific version of cuda
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

Next, we'll install the repository and its requirements. We provide two options - installing from source, and installing from pip. **We strongly recommend installing from source**, as it allows greater flexibility and easier access to scripts and examples.

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

## Installing released datasets

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