"""
Script to remove idle segments from a real robot hdf5.
"""
import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
from robomimic.scripts.postprocess_dataset_intervention_segments import postprocess_dataset_intervention_segments


def get_idle_segments_in_trajectory(
    ep_grp,
    obs_pos_key="ee_pose",
    min_segment_length=1,
    threshold=1e-4,
    verbose=False,
):
    """
    Returns a mask that corresponds to idle segments in the trajectory.

    Args:
        ep_grp (h5py.Group): hdf5 group that corresponds to a demo key (such as "demo_0")
        obs_pos_key (str): key for eef pos observations
        min_segment_length (int): minimum length of idle segment
        threshold (float): threshold for delta eef pos differences - everything below this threshold
            value is considered idle
        verbose (bool): if True, print some helpful info

    Returns:
        idle_segment_mask (np.array): array with value of 1 during an idle segment
    """
    if verbose:
        print(ep_grp)
    eef_pos = ep_grp["obs/{}".format(obs_pos_key)][:, :3]
    delta_eef_pos_norms = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)

    # note: pad with 0 at start to make sure indices correspond to indices in @eef_pos (otherwise we're off by one due to the difference calculation)
    idle_segment_mask = np.array([0] + (delta_eef_pos_norms < threshold).astype(int).tolist())
    idle_segments = FileUtils.get_intervention_segments(idle_segment_mask)

    # filter out segments that are too short
    ret_mask = np.zeros(eef_pos.shape[0]).astype(int)
    for seg in idle_segments:
        if seg[1] - seg[0] >= min_segment_length:
            ret_mask[seg[0] : seg[1]] = 1
        
            if verbose:
                print("segment: {}".format(seg))
                # print norms N timesteps before and after window to get a sense of nearby values
                prev_norms = delta_eef_pos_norms[max(seg[0] - 6, 0) : seg[0] - 1]
                print("prev_norms")
                print(prev_norms)
                post_norms = delta_eef_pos_norms[seg[1] - 1 : min(seg[1] + 4, eef_pos.shape[0] - 1)]
                print("post_norms")
                print(post_norms)

    return ret_mask


def write_non_idle_segments_as_interventions(hdf5_path, n=None, min_segment_length=1, threshold=1e-4):
    """
    Modifies the hdf5 in-place by splitting each trajectory into idle and non-idle segments, and
    writing the result as an "interventions" key in each trajectory, where the interventions correspond
    to non-idle segments.
    """
    
    # get demo keys
    f = h5py.File(args.dataset, "a")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    if args.n is not None:
        demos = demos[:args.n]

    # for each demo key, get idle segment, and write to interventions
    for demo_key in tqdm(demos):
        ep_grp = f["data/{}".format(demo_key)]
        idle_seg_mask = get_idle_segments_in_trajectory(
            ep_grp=ep_grp,
            obs_pos_key="ee_pose",
            min_segment_length=min_segment_length,
            threshold=threshold,
        )

        # write non-idle segment mask as interventions
        non_idle_seg_mask = 1 - idle_seg_mask
        if "interventions" in ep_grp:
            del ep_grp["interventions"]
        ep_grp.create_dataset("interventions", data=non_idle_seg_mask)

    f.close()


def combine_intervention_segments(hdf5_path, output_name, n=None):
    """
    Helper function to combine intervention segments in each demo trajectory together, and discard
    non-intervention segments. This repurposes the postprocess_dataset_intervention_segments.py to
    essentially remove the idle segments (which are non-intervention segments).
    """
    args = argparse.Namespace()
    args.dataset = os.path.expandvars(os.path.expanduser(hdf5_path))
    args.output_name = output_name
    args.n = n
    args.same = True
    postprocess_dataset_intervention_segments(args)


def remove_idle_segments(args):
    if args.debug:
        # print idle segments for the demos

        # get demo keys
        f = h5py.File(args.dataset, "r")
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        if args.n is not None:
            demos = demos[:args.n]

        for demo_key in demos:
            idle_seg_mask = get_idle_segments_in_trajectory(
                ep_grp=f["data/{}".format(demo_key)],
                obs_pos_key="ee_pose",
                # min_segment_length=1,
                min_segment_length=7,
                threshold=1e-4,
                # threshold=3e-4,
                # verbose=True,
                verbose=False,
            )
            idle_segs = FileUtils.get_intervention_segments(idle_seg_mask)
            print(demo_key)
            # print(len(idle_segs))
            print("idle segments")
            print(idle_segs)
            print("segment lengths")
            print([seg[1] - seg[0] for seg in idle_segs])

        f.close()
        exit()

    assert args.output_name is not None

    # split each trajectory into idle and non-idle segments and write to "interventions" key
    print("writing non-idle segments as interventions...")
    write_non_idle_segments_as_interventions(
        hdf5_path=args.dataset,
        n=args.n,
        # some good candidates below
        min_segment_length=7,
        threshold=1e-4,
        # min_segment_length=7,
        # threshold=3e-4,
    )

    # write new dataset, keeping only interventions
    print("combining interventions into new dataset...")
    combine_intervention_segments(
        hdf5_path=args.dataset,
        output_name=args.output_name,
        n=args.n,
    )

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
        default=None,
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

    parser.add_argument(
        "--debug",
        action='store_true',
        help="just print the idle and non-idle segment splits instead of actually doing any dataset processing",
    )

    args = parser.parse_args()
    remove_idle_segments(args)