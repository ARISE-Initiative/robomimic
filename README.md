# robomimic

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Pre-processing datasets
1. Convert the raw robosuite dataset to robomimic format
```
python robomimic/scripts/conversion/convert_robosuite.py --dataset <ds-path> --filter_num_demos <list-of-numbers>
```
`--filter_num_demos` corresponds to the number of demos to filter by. It's a list, eg. `10 30 50 100 200 500 1000`

This script will extract absolute actions, extract the action dict, and add filter keys.

2. Extract image observations from robomimic dataset
```
python robomimic/scripts/dataset_states_to_obs.py --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand --compress --exclude-next-obs --dataset <ds-path>
```
This script will generate a new dataset with the suffix `_im84.hdf5` in the same directory as `--dataset`
