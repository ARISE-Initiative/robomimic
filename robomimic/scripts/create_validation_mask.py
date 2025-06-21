import h5py
import argparse
import random
import numpy as np

def list_demos(h5file):
    # Try to find demos under /data
    if "data" in h5file:
        return list(h5file["data"].keys())
    # Fallback: search for demo_*
    demos = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.startswith("demo_"):
            demos.append(name)
    h5file.visititems(visitor)
    return demos

def main():
    parser = argparse.ArgumentParser(description="Create a validation and training mask for a robomimic dataset.")
    parser.add_argument("dataset", help="Path to the HDF5 dataset file")
    parser.add_argument("--val_pct", type=float, required=True, help="Validation percentage (e.g., 0.02 for 2%)")
    parser.add_argument("--mask_name", type=str, required=True, help="Base name for the new validation and training masks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    with h5py.File(args.dataset, "a") as f:
        demos = list_demos(f)
        n_val = max(1, int(len(demos) * args.val_pct))
        random.seed(args.seed)
        np.random.seed(args.seed)
        val_demos = random.sample(demos, n_val)
        train_demos = [d for d in demos if d not in val_demos]
        print(f"Selected {n_val} demos for validation mask '{args.mask_name}_valid':")
        for demo in val_demos:
            print(demo)
        print(f"Selected {len(train_demos)} demos for training mask '{args.mask_name}_train':")
        for demo in train_demos:
            print(demo)
        # Save the masks
        mask_group = f.require_group("mask")
        # Validation mask
        valid_mask_name = f"{args.mask_name}_valid"
        if valid_mask_name in mask_group:
            print(f"Mask '{valid_mask_name}' already exists. Overwriting.")
            del mask_group[valid_mask_name]
        mask_group.create_dataset(valid_mask_name, data=[s.encode("utf-8") for s in val_demos])
        # Training mask
        train_mask_name = f"{args.mask_name}_train"
        if train_mask_name in mask_group:
            print(f"Mask '{train_mask_name}' already exists. Overwriting.")
            del mask_group[train_mask_name]
        mask_group.create_dataset(train_mask_name, data=[s.encode("utf-8") for s in train_demos])
        print(f"Masks '{valid_mask_name}' and '{train_mask_name}' created.")

if __name__ == "__main__":
    main()
