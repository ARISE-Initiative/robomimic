from robomimic.scripts.config_gen.dataset_registry import SINGLE_STAGE_TASK_DATASETS
import os

ds_paths = []

# task_nems = SINGLE_STAGE_TASK_DATASETS.keys()
task_names = [
    "PnPCounterToCab",
    "PnPCabToCounter",
    # "PnPCounterToSink",
    # "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    # "PnPMicrowaveToCounter",
    # "PnPCounterToStove",
    "OpenDoorDoubleHinge",
    "CloseDoorDoubleHinge",
    "OpenDoorSingleHinge",
    "CloseDoorSingleHinge",
    # "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
]

for task in task_names:
    task_spec = SINGLE_STAGE_TASK_DATASETS[task]
    mg_path = os.path.join(task_spec["mg_5scenes_path"], "demo_gentex_im128.hdf5")
    ds_paths.append(mg_path)

for path in ds_paths:
    print("python ~/scripts/internal/add_train_filter_key.py --dataset {path}; python ~/scripts/filter_dataset_size.py --dataset {path} --input_filter_key train".format(path=path))
    print()