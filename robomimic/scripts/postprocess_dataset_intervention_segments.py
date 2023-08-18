"""
Script to postprocess a dataset by splitting each trajectory up into new trajectories 
that only consists of continuous intervention segments.
"""
import os
import json
import h5py
import argparse
import numpy as np

import robomimic.utils.file_utils as FileUtils


def write_intervention_segments_as_trajectories(
    src_ep_grp,
    dst_grp,
    start_ep_write_ind,
    same=False,
):
    """
    Helper function to extract intervention segments from a source demonstration and use their indices to
    write the corresponding subset of each trajectory to a new trajectory.

    Returns:
        end_ep_write_ind (int): updated episode index after writing trajectories to destination file
        num_traj (int): number of trajectories written to destination file
        total_samples (int): total number of samples written to destination file
        same (bool): if True, write all intevrention segments to the same trajectory
    """

    # get segments
    interventions = src_ep_grp["interventions"][()].reshape(-1).astype(int)
    segments = FileUtils.get_intervention_segments(interventions)

    ep_write_ind = start_ep_write_ind
    total_samples = 0
    num_traj = len(segments)
    keys_to_try_and_copy = ["states", "obs", "next_obs", "rewards", "dones", "actions_abs", "datagen_info"]

    if same:
        # concatenate information across intervention segments and write to single episode
        num_traj = 1
        dst_grp_name = "demo_{}".format(ep_write_ind)
        dst_ep_grp = dst_grp.create_group(dst_grp_name)
        for k in keys_to_try_and_copy:
            should_compress = (k in ["obs", "next_obs"])
            if k in src_ep_grp:
                if isinstance(src_ep_grp[k], h5py.Group):
                    for k2 in src_ep_grp[k]:
                        assert isinstance(src_ep_grp[k][k2], h5py.Dataset)
                        data = np.concatenate(
                            [src_ep_grp[k][k2][seg_start_ind : seg_end_ind] for seg_start_ind, seg_end_ind in segments],
                            axis=0,
                        )
                        if should_compress:
                            dst_ep_grp.create_dataset("{}/{}".format(k, k2), data=data, compression="gzip")
                        else:
                            dst_ep_grp.create_dataset("{}/{}".format(k, k2), data=data)
                else:
                    assert isinstance(src_ep_grp[k], h5py.Dataset)
                    data = np.concatenate(
                        [src_ep_grp[k][seg_start_ind : seg_end_ind] for seg_start_ind, seg_end_ind in segments],
                        axis=0,
                    )
                    if should_compress:
                        dst_ep_grp.create_dataset("{}".format(k), data=data, compression="gzip")
                    else:
                        dst_ep_grp.create_dataset("{}".format(k), data=data)

        # manually copy actions since they might need truncation
        actions = np.concatenate([src_ep_grp["actions"][seg_start_ind : seg_end_ind] for seg_start_ind, seg_end_ind in segments], axis=0)
        if actions.shape[-1] != 7:
            actions = actions[..., :7]
        dst_ep_grp.create_dataset("actions", data=actions)

        # mimicgen metadata
        if "src_demo_inds" in src_ep_grp:
            dst_ep_grp.create_dataset("src_demo_inds", data=np.array(src_ep_grp["src_demo_inds"][:]))
        if "src_demo_labels" in src_ep_grp:
            dst_ep_grp.create_dataset("src_demo_labels", data=np.array(src_ep_grp["src_demo_labels"][:]))

        # copy attributes too
        for k in src_ep_grp.attrs:
            dst_ep_grp.attrs[k] = src_ep_grp.attrs[k]
        dst_ep_grp.attrs["num_samples"] = np.sum([(seg_end_ind - seg_start_ind) for seg_start_ind, seg_end_ind in segments])

        # update variables for next iter
        ep_write_ind += 1
        total_samples += dst_ep_grp.attrs["num_samples"]
        print("  wrote trajectory to dst grp {} with num samples {}".format(dst_grp_name, dst_ep_grp.attrs["num_samples"]))
    else:
        # write each segment to new episode
        for seg_start_ind, seg_end_ind in segments:
            dst_grp_name = "demo_{}".format(ep_write_ind)
            dst_ep_grp = dst_grp.create_group(dst_grp_name)

            # copy over subsequence from source trajectory into destination trajectory
            for k in keys_to_try_and_copy:
                should_compress = (k in ["obs", "next_obs"])
                if k in src_ep_grp:
                    if isinstance(src_ep_grp[k], h5py.Group):
                        for k2 in src_ep_grp[k]:
                            assert isinstance(src_ep_grp[k][k2], h5py.Dataset)
                            if should_compress:
                                dst_ep_grp.create_dataset("{}/{}".format(k, k2), data=np.array(src_ep_grp[k][k2][seg_start_ind : seg_end_ind]), compression="gzip")
                            else:
                                dst_ep_grp.create_dataset("{}/{}".format(k, k2), data=np.array(src_ep_grp[k][k2][seg_start_ind : seg_end_ind]))
                    else:
                        assert isinstance(src_ep_grp[k], h5py.Dataset)
                        if should_compress:
                            dst_ep_grp.create_dataset("{}".format(k), data=np.array(src_ep_grp[k][seg_start_ind : seg_end_ind]), compression="gzip")
                        else:
                            dst_ep_grp.create_dataset("{}".format(k), data=np.array(src_ep_grp[k][seg_start_ind : seg_end_ind]))

            # manually copy actions since they might need truncation
            actions = np.array(src_ep_grp["actions"][seg_start_ind : seg_end_ind])
            if actions.shape[-1] != 7:
                actions = actions[..., :7]
            dst_ep_grp.create_dataset("actions", data=actions)

            # mimicgen metadata
            if "src_demo_inds" in src_ep_grp:
                dst_ep_grp.create_dataset("src_demo_inds", data=np.array(src_ep_grp["src_demo_inds"][:]))
            if "src_demo_labels" in src_ep_grp:
                dst_ep_grp.create_dataset("src_demo_labels", data=np.array(src_ep_grp["src_demo_labels"][:]))

            # copy attributes too
            for k in src_ep_grp.attrs:
                dst_ep_grp.attrs[k] = src_ep_grp.attrs[k]
            dst_ep_grp.attrs["num_samples"] = (seg_end_ind - seg_start_ind)

            # update variables for next iter
            ep_write_ind += 1
            total_samples += dst_ep_grp.attrs["num_samples"]
            print("  wrote trajectory to dst grp {} with num samples {}".format(dst_grp_name, dst_ep_grp.attrs["num_samples"]))

    return ep_write_ind, num_traj, total_samples


def postprocess_dataset_intervention_segments(args):
    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("\ninput file: {}".format(args.dataset))
    print("output file: {}\n".format(output_path))

    ep_write_ind = 0
    num_traj = 0
    total_samples = 0
    for ind in range(len(demos)):
        ep = demos[ind]
        src_ep_grp = f["data/{}".format(ep)]
        print("src grp: {} with {} samples".format(ep, src_ep_grp.attrs["num_samples"]))
        ep_write_ind, ep_num_traj, ep_total_samples = write_intervention_segments_as_trajectories(
            src_ep_grp=src_ep_grp,
            dst_grp=data_grp,
            start_ep_write_ind=ep_write_ind,
            same=args.same,
        )
        num_traj += ep_num_traj
        total_samples += ep_total_samples

    # TODO: update filter keys based on which source demos created which target demos
    # if "mask" in f:
    #     f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = f["data"].attrs["env_args"] # environment info
    print("\nWrote {} trajectories from src with {} trajectories to {}".format(num_traj, len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # write all intervention segments to the same demo key (so they will be treated as a contiguous trajectory in time)
    parser.add_argument(
        "--same",
        action='store_true',
        help="write all intervention segments to the same demo key (so they will be treated as a contiguous trajectory in time",
    )

    args = parser.parse_args()
    postprocess_dataset_intervention_segments(args)
