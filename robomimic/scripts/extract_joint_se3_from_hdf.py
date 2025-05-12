import os
import h5py
import numpy as np
import torch
import pytorch_kinematics as pk

# Path to MJCF robot model (Franka Panda)
MJCF_PATH = os.path.join(os.path.dirname(__file__), '../algo/panda/robot.xml')

# Name of the new dataset to add
NEW_KEY = "robot0_joint_se3"

# Helper: Convert rotation matrix to 6D representation (Zhou et al. 2019)
def matrix_to_6d(R):
    # R: (..., 3, 3)
    return R[..., :3, 0:2].reshape(*R.shape[:-2], 6)

def main(h5_path):
    # Load kinematic chain
    with open(MJCF_PATH, 'r') as f:
        mjcf = f.read()
    chain = pk.build_serial_chain_from_mjcf(mjcf, "right_hand")
    num_joints = len(chain.get_joint_parameter_names())

    with h5py.File(h5_path, 'a') as f:
        demos = list(f["data"].keys())
        for demo in demos:
            grp = f["data/"][demo]
            # Temporary fix: swap first 7 and next 7 columns in object observations
            # obj_dset = grp["obs"]["object"]
            # obj_data = obj_dset[...]
            # if obj_data.shape[1] >= 14:
            #     swapped = obj_data.copy()
            #     swapped[:, :7] = obj_data[:, 7:14]
            #     swapped[:, 7:14] = obj_data[:, :7]
            #     obj_dset[...] = swapped
            #     print(f"{demo}: swapped object obs columns")

            if NEW_KEY in grp["obs"]:
                # Delete from the group if it exists
                del grp["obs"][NEW_KEY]
                print(f"{demo}: deleted {NEW_KEY} from obs group")
                continue
            qpos = grp["obs/robot0_joint_pos"][:]
            n_steps = qpos.shape[0]
            se3s = np.zeros((n_steps, num_joints * 9), dtype=np.float32)
            qpos_torch = torch.tensor(qpos, dtype=torch.float32)
            fk = chain.forward_kinematics(qpos_torch, end_only=False)
            for j in range(num_joints):
                link_name = f"link{j}"
                if link_name not in fk:
                    raise RuntimeError(f"Link {link_name} not found in FK result.")
                mat = fk[link_name].get_matrix().cpu().numpy()  # (n_steps, 4, 4)
                pos = mat[:, :3, 3]  # (n_steps, 3)
                rot = mat[:, :3, :3]  # (n_steps, 3, 3)
                rot6d = matrix_to_6d(torch.tensor(rot)).numpy()  # (n_steps, 6)
                se3 = np.concatenate([pos, rot6d], axis=1)  # (n_steps, 9)
                se3s[:, j*9:(j+1)*9] = se3
            grp["obs"].create_dataset(NEW_KEY, data=se3s, compression="gzip")
            print(f"{demo}: wrote {NEW_KEY} with shape {se3s.shape}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <path_to_h5>")
        exit(1)
    main(sys.argv[1])
