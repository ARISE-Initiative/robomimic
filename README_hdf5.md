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