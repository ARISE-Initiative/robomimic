"""
Internal conversion tool to remove extra hdf5 attributes in demo.hdf5 files.
"""
import h5py
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
    )
    args = parser.parse_args()

    # open file for editing
    f = h5py.File(args.dataset, "a")

    # check important global attributes and remove all others
    assert "env_args" in f["data"].attrs
    assert "total" in f["data"].attrs
    for attr_k in f["data"].attrs:
        if attr_k not in ["env_args", "total"]:
            del f["data"].attrs[attr_k]

    for ep in f["data"]:
        # check important per-demo attributes and remove all others
        assert "num_samples" in f["data/{}".format(ep)].attrs
        for attr_k in f["data/{}".format(ep)].attrs:
            if attr_k not in ["model_file", "num_samples"]:
                del f["data/{}".format(ep)].attrs[attr_k]

    f.close()
