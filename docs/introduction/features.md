# Features Overview

## Summary

In this section, we briefly summarize some key features and where you should look to learn more about them.

1. **Datasets supported by robomimic**
   - See a list of supported datasets [here](./features.html#supported-datasets).<br><br>
2. **Visualizing datasets**
   - Learn how to visualize dataset trajectories [here](./datasets.html#view-dataset-structure-and-videos).<br><br>
3. **Reproducing paper experiments**
   - Easily reproduce experiments from the following papers
     - robomimic: [here](./results.html)
     - MOMART: [here](https://sites.google.com/view/il-for-mm/datasets)<br><br>
4. **Making your own dataset**
   - Learn how to make your own collected dataset compatible with this repository [here](./datasets.html#dataset-structure). 
   - Note that **all datasets collected through robosuite are also readily compatible** (see [here](./datasets.html#converting-robosuite-hdf5-datasets)).<br><br>
5. **Using filter keys to easily train on subsets of a dataset**
   - See [this section](./datasets.html#filter-keys-and-train-valid-splits) on how to use filter keys.<br><br>
6. **Running hyperparameter scans easily**
   - See [this guide](./advanced.html#using-the-hyperparameter-helper-to-launch-runs) on running hyperparameter scans.<br><br>
7. **Using pretrained models in the model zoo**
   - See [this link](./model_zoo.html) to download and use pretrained models.<br><br>
8. **Getting familiar with configs**
   - Learn about how configs work [here](../modules/configs.html).<br><br>
9. **Getting familiar with operations over tensor collections**
   - Learn about using useful tensor utilities [here](../modules/utils.html#tensorutils).<br><br>
10. **Creating your own observation modalities**
    - Learn how to make your own observation modalities and process them with custom network architectures [here](../modules/observations.html).<br><br>
11. **Creating your own algorithm**
    - Learn how to implement your own learning algorithm [here](../modules/algorithms.html#building-your-own-algorithm).<br><br>

## Supported Datasets

This is a list of datasets that we currently support, along with links on how to work with them. This list will be expanded as more datasets are made compatible with robomimic.

- [robomimic](./results.html#downloading-released-datasets)
- [robosuite](./datasets.html#converting-robosuite-hdf5-datasets)
- [MOMART](./datasets.html#momart-datasets)
- [D4RL](./results.html#d4rl)
- [RoboTurk Pilot](./datasets.html#roboturk-pilot-datasets)