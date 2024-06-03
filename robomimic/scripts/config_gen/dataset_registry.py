from collections import OrderedDict
from copy import deepcopy
import os

SINGLE_STAGE_TASK_DATASETS = OrderedDict(
    PnPCounterToCab=dict(
        env_meta_update_dict=dict(env_kwargs=dict()),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    PnPCabToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict()),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-05-04-22-10-37_and_2024-05-07-07-40-14",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    PnPCounterToSink=dict(
        env_meta_update_dict=dict(env_kwargs=dict()),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17",
        mg_filter_key="3000_demos",
        horizon=700,
    ),
    PnPSinkToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict()),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    PnPCounterToMicrowave=dict(
        env_meta_update_dict=dict(env_kwargs=dict()),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17",
        mg_filter_key="3000_demos",
        horizon=600,
    ),
    PnPMicrowaveToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict()),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-05-04-22-14-26_and_2024-05-07-07-41-42",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    PnPCounterToStove=dict(
        env_meta_update_dict=dict(env_kwargs=dict()),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-05-04-22-14-20",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    PnPStoveToCounter=dict(
        env_meta_update_dict=dict(env_kwargs=dict()),
        human_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_pnp/PnPStoveToCounter/mg/2024-05-04-22-14-40",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    OpenSingleDoor=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    CloseSingleDoor=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    OpenDoubleDoor=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_doors/OpenDoubleDoor/mg/2024-05-04-22-35-53",
        mg_filter_key="3000_demos",
        horizon=1000,
    ),
    CloseDoubleDoor=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_doors/CloseDoubleDoor/mg/2024-05-04-22-22-42_and_2024-05-08-06-02-36",
        mg_filter_key="3000_demos",
        horizon=700,
    ),
    OpenDrawer=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_drawer/OpenDrawer/2024-05-03",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_drawer/OpenDrawer/mg/2024-05-04-22-38-42",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    CloseDrawer=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_drawer/CloseDrawer/2024-04-30",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_drawer/CloseDrawer/mg/2024-05-09-09-32-19",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    TurnOnSinkFaucet=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-05-04-22-17-46",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    TurnOffSinkFaucet=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-05-04-22-17-26",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    TurnSinkSpout=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_sink/TurnSinkSpout/mg/2024-05-09-09-31-12",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    TurnOnStove=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOnStove/2024-05-02",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    TurnOffStove=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOffStove/2024-05-02",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_stove/TurnOffStove/mg/2024-05-08-09-20-45",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    CoffeeSetupMug=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-05-04-22-22-13_and_2024-05-08-05-52-13",
        mg_filter_key="3000_demos",
        horizon=600,
    ),
    CoffeeServeMug=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-05-04-22-21-50",
        mg_filter_key="3000_demos",
        horizon=600,
    ),
    CoffeePressButton=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-05-04-22-21-32",
        mg_filter_key="3000_demos",
        horizon=300,
    ),
    TurnOnMicrowave=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    TurnOffMicrowave=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25",
        human_filter_key="50_demos",
        mg_path="~/robocasa/datasets/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-05-04-22-39-23",
        mg_filter_key="3000_demos",
        horizon=500,
    ),
    NavigateKitchen=dict(
        human_path="~/robocasa/datasets/single_stage/kitchen_navigate/NavigateKitchen/2024-05-09",
        human_filter_key="50_demos",
        horizon=500,
    ),
)


MULTI_STAGE_TASK_DATASETS = OrderedDict(
    ArrangeVegetables=dict(
        human_path="~/robocasa/datasets/multi_stage/chopping_food/ArrangeVegetables/2024-05-11",
        human_filter_key="50_demos",
        horizon=1200,
        activity="chopping_food",
    ),
    MicrowaveThawing=dict(
        human_path="~/robocasa/datasets/multi_stage/defrosting_food/MicrowaveThawing/2024-05-11",
        human_filter_key="50_demos",
        horizon=1000,
        activity="defrosting_food",
    ),
    RestockPantry=dict(
        human_path="~/robocasa/datasets/multi_stage/restocking_supplies/RestockPantry/2024-05-10",
        human_filter_key="50_demos",
        horizon=1000,
        activity="restocking_supplies",
    ),
    PreSoakPan=dict(
        human_path="~/robocasa/datasets/multi_stage/washing_dishes/PreSoakPan/2024-05-10",
        human_filter_key="50_demos",
        horizon=1500,
        activity="washing_dishes",
    ),
    PrepareCoffee=dict(
        human_path="~/robocasa/datasets/multi_stage/brewing/PrepareCoffee/2024-05-07",
        human_filter_key="50_demos",
        horizon=1000,
        activity="brewing",
    ),
)


def get_ds_cfg(ds_names, exclude_ds_names=None, overwrite_ds_lang=False, src="human", filter_key=None, eval=None, gen_tex=True, rand_cams=True):
    assert src in ["human", "mg"]
    all_datasets = {}
    all_datasets.update(SINGLE_STAGE_TASK_DATASETS)
    all_datasets.update(MULTI_STAGE_TASK_DATASETS)

    if ds_names == "all":
        ds_names = list(all_datasets.keys())
    elif ds_names == "single_stage":
        ds_names = list(SINGLE_STAGE_TASK_DATASETS.keys())
    elif ds_names == "multi_stage":
        ds_names = list(MULTI_STAGE_TASK_DATASETS.keys())
    elif ds_names == "pnp":
        ds_names = [name for name in all_datasets.keys() if "PnP" in name]
    elif isinstance(ds_names, str):
        ds_names = [ds_names]

    if exclude_ds_names is not None:
        ds_names = [name for name in ds_names if name not in exclude_ds_names]

    ret = []
    for name in ds_names:
        ds_meta = all_datasets[name]

        cfg = dict(
            horizon=ds_meta["horizon"]
        )
        
        # determine whether we are performing eval on dataset
        if eval is None or name in eval:
            cfg["do_eval"] = True
        else:
            cfg["do_eval"] = False

        # if applicable overwrite the language stored in the dataset
        if overwrite_ds_lang is True:
            cfg["lang"] = ds_meta["lang"]
        
        # determine dataset path
        path_list = ds_meta.get(f"{src}_path", None)
        # skip if entry does not exist for this dataset src
        if path_list is None:
            continue
        
        if isinstance(path_list, str):
            path_list = [path_list]

        for path_i, path in enumerate(path_list):
            cfg_for_path = deepcopy(cfg)

            # determine dataset filter key
            if filter_key is not None:
                cfg_for_path["filter_key"] = filter_key
            else:
                cfg_for_path["filter_key"] = ds_meta[f"{src}_filter_key"]

            if "env_meta_update_dict" in ds_meta:
                cfg_for_path["env_meta_update_dict"] = ds_meta["env_meta_update_dict"]
            
            if not path.endswith(".hdf5"):
                # determine path
                if gen_tex is True and rand_cams is True:
                    path = os.path.join(path, "demo_gentex_im128_randcams.hdf5")
                elif gen_tex is True and rand_cams is False:
                    path = os.path.join(path, "demo_gentex_im128.hdf5")
                elif gen_tex is False and rand_cams is False:
                    path = os.path.join(path, "demo_im128.hdf5")
                else:
                    raise ValueError
            cfg_for_path["path"] = path

            if path_i > 0:
                cfg_for_path["do_eval"] = False

            ret.append(cfg_for_path)

    return ret


def scan_datasets(folder, postfix=".h5"):
    dataset_paths = []
    for root, dirs, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            if f.endswith(postfix):
                dataset_paths.append(os.path.join(root, f))
    return dataset_paths