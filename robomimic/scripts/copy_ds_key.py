import argparse
import h5py
import numpy as np

def copy_ds_group(src, target, keys):
    f_src = h5py.File(src, "r")
    f_target = h5py.File(target, "a")

    for ep in f_src["data"].keys():
        for key in keys:
            src_ep = f_src["data"][ep]
            targ_ep = f_target["data"][ep]

            if isinstance(src_ep[key], h5py._hl.dataset.Dataset):
                v = np.array(src_ep[key][:])
                if key in targ_ep.keys():
                    targ_ep[key][:] = v
                else:
                    targ_ep.create_dataset(key, data=v)
            else:
                if not key in f_target["data"][ep]:
                    targ_ep.create_group(key)
                for k in src_ep[key].keys():
                    v = np.array(src_ep[key][k][:])
                    if k in targ_ep[key].keys():
                        targ_ep[key][k][:] = v
                    else:
                        targ_ep[key].create_dataset(k, data=v)

    f_src.close()
    f_target.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="path to output hdf5 dataset",
    )

    parser.add_argument(
        "--keys",
        type=str,
        nargs='+',
        default=[],
        help="key names to copy",
    )

    args = parser.parse_args()

    copy_ds_group(args.src, args.target, args.keys)
