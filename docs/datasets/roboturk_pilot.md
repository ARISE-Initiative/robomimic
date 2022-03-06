# RoboTurk Pilot Datasets

The first [RoboTurk paper](https://arxiv.org/abs/1811.02790) released [large-scale pilot datasets](https://roboturk.stanford.edu/dataset_sim.html) collected with robosuite `v0.3`. These datasets consist of over 1000 task demonstrations each on several Sawyer `PickPlace` and `NutAssembly` task variants, collected by several human operators. This repository is fully compatible with these datasets. 

![roboturk_pilot](../images/roboturk_pilot.png)

To get started, first download the dataset [here](http://cvgl.stanford.edu/projects/roboturk/RoboTurkPilot.zip) (~9 GB download), and unzip the file, resulting in a `RoboTurkPilot` folder. This folder has subdirectories corresponding to each task, each with a raw hdf5 file. You can convert the demonstrations using a command like the one below.

```sh
# convert the Can demonstrations, and also create a "fastest_225" filter_key (prior work such as IRIS has trained on this subset)
$ python conversion/convert_roboturk_pilot.py --folder /path/to/RoboTurkPilot/bins-Can --n 225
```

Next, make sure that you're on the [roboturk_v1](https://github.com/ARISE-Initiative/robosuite/tree/roboturk_v1) branch of robosuite, which is a modified version of v0.3. **You should always be on the roboturk_v1 branch when using these datasets.** Finally, follow the instructions in the "Extracting Observations from MuJoCo states" section of the docs (see [here](./robosuite.html#extracting-observations-from-mujoco-states))to extract observations from the raw converted `demo.hdf5` file, in order to produce an hdf5 ready for training.