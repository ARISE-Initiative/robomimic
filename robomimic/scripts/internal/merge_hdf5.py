"""
A convenience script to merge two or more hdf5s together into one file.
"""

import os
import json
import h5py
import argparse
import numpy as np


def print_warning(str_to_print):
    print("*" * 50)
    print("WARNING: {}".format(str_to_print))
    print("*" * 50)
    print("")


def maybe_copy_attribute_from_source_files(new_file_group, source_files, attr_name, json_load=False):
    """
    Helper function for attributes that should be copied over from source files. 
    It prints warnings for attributes that do not match across all the source files.

    Args:
        new_file_group: base group for new merged file (usually just new_f["data"])
        source_files: list of source file handles
        attr_name (str): attribute name to maybe copy
        json_load (bool): if True, assumes that value to copy is a json string, so it will
            be loaded as a dictionary before checking for equality across files. This is
            so order will not matter when making the comparison.
    """
    n_files = len(source_files)
    vals = []
    val_fids = []
    for f_id, source_f in enumerate(source_files):
        if attr_name in source_f["data"].attrs:
            vals.append(source_f["data"].attrs[attr_name])
            val_fids.append(f_id)

    if len(vals) == 0:
        print_warning("not writing attribute {} since not found in any file".format(attr_name))
        return

    # only bother checking equality if all files have the attribute
    all_equal = False
    if len(vals) == n_files:
        vals_to_check = vals
        if json_load:
            vals_to_check = [json.loads(val) for val in vals]
        all_equal = (vals_to_check[:-1] == vals_to_check[1:]) # check if all elements are equal

    if not all_equal:
        print_warning("attribute {} is not equal in all files - copying from file {}".format(attr_name, val_fids[0]))

    # write value
    new_file_group.attrs[attr_name] = vals[0]


def main(args):
    # source hdf5s
    assert (len(args.datasets) > 1), "must provide more than one file!"
    assert (len(set(args.datasets)) == len(args.datasets)), "duplicate file!"
    read_filter_keys = len(args.read_filter_keys) > 0
    if read_filter_keys:
        assert len(args.read_filter_keys) == len(args.datasets), "must provide filter key per dataset"
    write_filter_keys = len(args.write_filter_keys) > 0
    if write_filter_keys:
        assert len(args.write_filter_keys) == len(args.datasets), "must provide filter key per dataset"
    source_files = []
    for bf in args.datasets:
        source_files.append(h5py.File(bf, "r"))

    # new hdf5
    demo_path = os.path.dirname(args.datasets[0])
    new_path = os.path.join(demo_path, args.name)
    new_f = h5py.File(new_path, "w")
    new_f_grp = new_f.create_group("data")

    num_demos = 0
    lengths = []
    train_demos = []
    valid_demos = []
    for f_id, f in enumerate(source_files):
        # maybe reduce demos to merge using filter key
        demos = list(f["data"].keys())
        has_read_filter_key = False
        if read_filter_keys:
            if "mask/{}".format(args.read_filter_keys[f_id]) in f:
                print("using read filter key {} for file {}".format(args.read_filter_keys[f_id], f_id))
                has_read_filter_key = True
                demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.read_filter_keys[f_id])][:])])
            else:
                print_warning("read filter key {} does not exist in file {}".format(args.read_filter_keys[f_id], f_id))
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        # maybe copy train-val split as well
        if args.valid:
            filter_key_str = "{}_".format(args.read_filter_keys[f_id]) if has_read_filter_key else ""
            f_train = [x.decode("utf-8") for x in f["mask/{}train".format(filter_key_str)][:]]
            f_valid = [x.decode("utf-8") for x in f["mask/{}valid".format(filter_key_str)][:]]
            assert set(f_train + f_valid) == set(demos), "train-val split has issues"

        new_demo_groups = []
        new_train_demos = []
        new_valid_demos = []
        for i in range(len(demos)):
            ep = demos[i]
            print('%s to demo_%i' % (ep, num_demos))
            grp_name = "demo_{}".format(num_demos)
            ep_data_grp = new_f_grp.create_group(grp_name)

            new_demo_groups.append(grp_name)
            if args.valid:
                # preserve train / valid split across the file merge
                if ep in f_valid:
                    new_valid_demos.append(grp_name)
                    valid_demos.append(grp_name)
                else:
                    new_train_demos.append(grp_name)
                    train_demos.append(grp_name)

            # write the datasets
            for k in f["data/{}".format(ep)]:
                if k in ["obs", "next_obs"]:
                    # dataset per modality
                    for m in f["data/{}/{}".format(ep, k)]:
                        ep_data_grp.create_dataset("{}/{}".format(k, m), 
                            data=np.array(f["data/{}/{}/{}".format(ep, k, m)]),
                        )
                elif isinstance(f["data/{}/{}".format(ep, k)], h5py.Dataset):
                    ep_data_grp.create_dataset(k, data=np.array(f["data/{}/{}".format(ep, k)]))

            # write the metadata present in attributes as well
            for k in f["data/{}".format(ep)].attrs:
                ep_data_grp.attrs[k] = f["data/{}".format(ep)].attrs[k]

            lengths.append(ep_data_grp.attrs["num_samples"])
            num_demos += 1

        if write_filter_keys:
            # write the list of demos to filter key in new merged file
            print_warning("write file {} to filter key {}".format(f_id, args.write_filter_keys[f_id]))
            new_f["mask/{}".format(args.write_filter_keys[f_id])] = np.array(new_demo_groups, dtype='S')
            # maybe copy train-val split to this filter key as well
            if args.valid:
                new_f["mask/{}_train".format(args.write_filter_keys[f_id])] = np.array(new_train_demos, dtype='S')
                new_f["mask/{}_valid".format(args.write_filter_keys[f_id])] = np.array(new_valid_demos, dtype='S')


    if args.valid:
        # write the train-valid split
        new_f["mask/train"] = np.array(train_demos, dtype='S')
        new_f["mask/valid"] = np.array(valid_demos, dtype='S')

    # write dataset attributes (metadata)
    new_f_grp.attrs["total"] = np.sum(lengths)
    maybe_copy_attribute_from_source_files(new_file_group=new_f_grp, source_files=source_files, attr_name="env", json_load=False)
    maybe_copy_attribute_from_source_files(new_file_group=new_f_grp, source_files=source_files, attr_name="teleop_config", json_load=True)
    maybe_copy_attribute_from_source_files(new_file_group=new_f_grp, source_files=source_files, attr_name="teleop_env_metadata", json_load=True)
    maybe_copy_attribute_from_source_files(new_file_group=new_f_grp, source_files=source_files, attr_name="env_args", json_load=True)

    new_f.close()
    for f in source_files:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # datasets to merge
    parser.add_argument(
        "--datasets",
        nargs='+',
    )

    # puts merged batch in same folder as batch 1, with the passed name
    parser.add_argument(
        "--name",
        type=str,
        default="states.hdf5"
    )

    # flag for merging train-valid splits as well
    parser.add_argument(
        "--valid",
        action='store_true',
    )

    # if provided, use the filter key per dataset for merging (will print a warning if it doesn't exist)
    parser.add_argument(
        "--read_filter_keys",
        type=str,
        nargs='+',
        default=[],
    )

    # if provided, store the list of demonstrations for each dataset in the corresponding filter key in the merged file
    parser.add_argument(
        "--write_filter_keys",
        type=str,
        nargs='+',
        default=[],
    )

    args = parser.parse_args()
    main(args)
