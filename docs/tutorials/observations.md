# Multimodal Observations

**robomimic** natively supports multiple different observation modalities, and provides integrated support for modifying observations and adding your own custom ones.

First, we highlight semantic distinctions when referring to different aspects of observations:

- **Keys** are individual observations that are received from an environment / dataset. For example, `rgb_wrist`, `eef_pos`, and `joint_vel` could be keys, depending on the dataset / environment.
- **Modalities** are different observation modes. For example, low dimensional states are considered a single mode, whereas RGB observations might be another mode. **robomimic** natively supports four modalities: `low_dim`, `rgb`, `depth`, and `scan`. Each modality owns it own set of observation keys.
- **Groups** consist of potentially multiple modalities and multiple keys per modality, which are together passed to a learning model. For example, **robomimic** commonly uses three different groups: `obs`, which contains the normal observations passed to any model using these as inputs, and `goal` / `subgoal`, which means any specified modalities / keys correspond to a goal / subgoal to be learned.

Observations are handled in the following way:
1. Each observation key is according to their modality via their `Modality` class,
2. All observations for a given modality are concatenated and passed through an `ObservationEncoder` for that modality,
3. All processed observations over all modalities are concatenated together and passed to a learning network

## Modifying and Adding Your Own Observation Modalities

**robomimic** natively supports the following modalities:
- `low_dim`: low-dimensional states
- `rgb`: RGB images
- `depth`: depth images
- `scan`: scan arrays

The way each of these modalities are processed and encoded can be easily specified by modifying their respective `encoder` parameters in your `Config` class.

You may want to specify your own custom modalities that get processed and encoded in a certain way (e.g.: semantic segmentation, optical flow, etc...). This can also easily be done, and we refer you to our [example script](https://github.com/ARISE-Initiative/robomimic/blob/master/examples/simple_obs_nets.py) which walks through the process.
