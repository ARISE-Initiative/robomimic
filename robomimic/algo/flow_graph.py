"""
FlowMatching Graph Attention Network (FlowGAT) Algorithm for Robomimic.

"""

import math
import os
from typing import Callable, Union

from collections import OrderedDict, deque
from typing import Dict, Optional, Tuple
from packaging.version import parse as parse_version

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx, unbatch

# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.models.flow_policy import FlowPolicy
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_nets as ObsNets


@register_algo_factory_func("flow_gat")
def algo_config_to_class(algo_config):
    """Factory function for FLOWGAT algorithm."""
    return FLOW_GAT, {}


class NodeFeatureProcessor(nn.Module):
    """
    Processes raw observation dictionaries into structured node features and
    constructs graph representations (spatial and temporal) for the GNN.
    """

    def __init__(self, num_joints: int = 7, chain: Optional[pk.SerialChain] = None):
        super().__init__()
        """
        Initializes the processor.

        Args:
            num_joints (int): Number of robot joints to include as nodes.
            chain (Optional[pk.SerialChain]): Pre-loaded PyTorch Kinematics chain
                                               for forward kinematics calculations.
        """
        self.chain = chain
        self.num_joints = num_joints
        # Offset from world origin to robot base (adjust if necessary)
        # self.robot_base_offset = torch.tensor([-0.5, -0.1, 0.912]) # Bin task
        table_length = 0.8
        self.robot_base_offset = torch.tensor(
            [-0.16 - table_length / 2, 0, 1]
        )  # table task


        # Will be set after processing first batch
        self.node_feature_dim: Optional[int] = None
        self.node_keys: Optional[list] = None
        self.batch_size: Optional[int] = None
        self.frame_stack: Optional[int] = None

    def process_features(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts and formats node features from the observation dictionary.

        Args:
            obs_dict (Dict[str, torch.Tensor]): Dictionary of observations, where each
                tensor has shape (batch_size, frame_stack, feature_dim).

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping node names (e.g., "joint_0",
                "eef", "object") to their feature tensors of shape
                (batch_size, frame_stack, node_feature_dim). Features are padded
                to ensure consistent dimensionality across node types.
        """
        # Infer batch_size and frame_stack from an arbitrary observation tensor
        any_obs_tensor = next(iter(obs_dict.values()))
        self.batch_size = any_obs_tensor.shape[0]
        self.frame_stack = any_obs_tensor.shape[1]
        device = any_obs_tensor.device

        # Ensure robot base offset is on the correct device
        self.robot_base_offset = self.robot_base_offset.to(device=device)

        # Extract relevant observations
        joint_pos = obs_dict["robot0_joint_pos"].float()
        eef_pos = obs_dict["robot0_eef_pos"].float()
        eef_quat = obs_dict["robot0_eef_quat"].float()
        gripper_qpos = obs_dict["robot0_gripper_qpos"].float()
        object_features = obs_dict["object"].float()

        # --- Get or Compute SE(3) Pose for Joints ---
        def matrix_to_6d(R):
            # R: (..., 3, 3)
            return R[..., :3, 0:2].reshape(*R.shape[:-2], 6)

        if "robot0_joint_se3" in obs_dict:
            se3_pose = (
                obs_dict["robot0_joint_se3"]
                .float()
                .view(self.batch_size, self.frame_stack, self.num_joints, 9)
            )
        elif self.chain is not None:
            # During inference, we need to compute the FK for the joints
            B, T, J = joint_pos.shape
            joint_pos_flat = joint_pos.reshape(B * T, J)
            fk_result = self.chain.forward_kinematics(joint_pos_flat, end_only=False)
            se3s = []  # Will be (B * T, num_joints, 9)
            for j in range(self.num_joints):
                link_name = f"link{j}"
                if link_name not in fk_result:
                    raise ValueError(
                        f"Link '{link_name}' not found in kinematic chain results. Available links: {list(fk_result.keys())}"
                    )
                mat = fk_result[link_name].get_matrix()  # (B * T, 4, 4)
                pos = mat[:, :3, 3]  # (B * T, 3)
                rot = mat[:, :3, :3]  # (B * T, 3, 3)
                rot6d = matrix_to_6d(rot)  # (B * T, 6)
                se3 = torch.cat([pos, rot6d], dim=1)  # (B * T, 9)
                se3s.append(se3)
            se3_pose = torch.stack(se3s, dim=1)  # (B * T, num_joints, 9)
            se3_pose = se3_pose.view(B, T, self.num_joints, 9)
        else:
            raise ValueError(
                "Missing 'robot0_joint_se3' in observations and no kinematic chain (self.chain) provided to compute it."
            )

        node_dict = {}

        # --- Joint Features ---
        for i in range(self.num_joints):
            node_name = f"joint_{i}"
            # Use the computed or extracted se3_pose here
            base_features = torch.cat(
                [se3_pose[:, :, i, :], joint_pos[:, :, i].unsqueeze(-1)], dim=-1
            )
            # Concatenate base features and one-hot encoding
            node_dict[node_name] = torch.cat([base_features], dim=-1)

        # --- End Effector Features ---
        node_name = "eef"
        # Align eef_pose with the robot base frame
        eef_pos = eef_pos - self.robot_base_offset
        base_features = torch.cat([eef_pos, eef_quat, gripper_qpos], dim=-1)
        node_dict[node_name] = torch.cat([base_features], dim=-1)

        # --- Object Features ---
        node_name = "object"

        # Switch object features due to bug in dataset
        temp = object_features[:, :, 0:7].clone()
        object_features[:, :, 0:7] = object_features[:, :, 7:14]
        object_features[:, :, 7:14] = temp
        object_pos = object_features[:, :, :3] - self.robot_base_offset
        base_features = torch.cat([object_pos, object_features[:, :, 3:]], dim=-1)
        node_dict[node_name] = torch.cat([base_features], dim=-1)

        # --- Base Frame Features ---
        # node_name = "base_frame"
        # base_features = object_features[:, :, :14]
        # # Align base frame with the robot base frame. Assume first 3 features are position.
        # base_features[:, :, :3] = base_features[:, :, :3] - self.robot_base_offset

        # node_dict[node_name] = torch.cat([base_features], dim=-1)

        # --- Insertion Hook Features ---
        # node_name = "insertion_hook"
        # base_features = object_features[:, :, 14:28]
        # # Align insertion hook with the robot base frame. Assume first 3 features are position.
        # base_features[:, :, :3] = base_features[:, :, :3] - self.robot_base_offset

        # node_dict[node_name] = torch.cat([base_features], dim=-1)

        # --- Wrench Features ---
        # node_name = "wrench"
        # base_features = object_features[:, :, 28:42]
        # # Align wrench with the robot base frame. Assume first 3 features are position.
        # base_features[:, :, :3] = base_features[:, :, :3] - self.robot_base_offset

        # node_dict[node_name] = torch.cat([base_features], dim=-1)

        return node_dict


# --- Start of FLOW_GAT Class ---


class FLOW_GAT(PolicyAlgo):
    def __init__(
        self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
    ):
        # Call parent __init__ first - it sets up device, config, etc. and calls _create_networks
        super().__init__(
            algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
        )

        # Setup CUDNN, matmul precision
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        self.batch_size = self.global_config.train.batch_size
        self.t_o = global_config.train.frame_stack  # Observation window size
        self.t_p = self.global_config.train.seq_length  # Prediction horizon
        self.t_a = self.algo_config.t_a  # Action execution horizon

        self.tg = self.algo_config.graph_frame_stack  # Observation window for GNN
        assert self.tg <= self.t_o, "graph_frame_stack cannot exceed frame_stack"
        assert (
            self.t_a <= self.t_p
        ), f"Action execution horizon (t_a={self.t_a}) cannot be greater than prediction horizon (t_p={self.t_p})"
        self.num_feedback_actions = self.t_p - self.t_a
        print(
            f"Receding Horizon: Prediction Horizon t_p={self.t_p}, Execution Horizon t_a={self.t_a}, Feedback Actions={self.num_feedback_actions}"
        )

        # --- Kinematics Loading ---
        self.chain = None
        try:
            # Use a relative path or ensure the absolute path is correct
            script_dir = os.path.dirname(
                __file__
            )  # Gets directory where this script is located
            mjcf_path = os.path.join(
                script_dir, "panda", "robot.xml"
            )  # Example relative path

            if os.path.exists(mjcf_path):
                self.chain = pk.build_serial_chain_from_mjcf(
                    open(mjcf_path).read(), "right_hand"
                ).to(dtype=torch.float32, device=self.device)
                print("Successfully loaded kinematic chain from robot.xml.")
            else:
                print(f"Warning: Kinematic definition file not found at {mjcf_path}")
        except ImportError:
            print("Warning: pytorch_kinematics not found. Kinematics will not be used.")
        except FileNotFoundError:
            print(f"Warning: Kinematic definition file not found at {mjcf_path}")
        except Exception as e:
            print(f"Warning: Failed to load robot.xml for kinematics: {e}")

        num_joints = self.algo_config.num_joints

        node_processor_instance = NodeFeatureProcessor(
            num_joints=num_joints,
            chain=self.chain,
        )
        self.node_feature_processor = node_processor_instance
        print("NodeFeatureProcessor initialized.")

        # --- Action Feedback Buffer (for Inference) ---
        # Stores the t_p - t_a actions from the *previous* prediction that were *not* executed.
        # Initialize with zeros. Shape (t_p - t_a, action_dim)
        self.previous_unexecuted_actions_inf = torch.zeros(
            self.num_feedback_actions, self.ac_dim, device=self.device
        )

        # --- Buffer for actions from the *current* prediction sequence (for Inference) ---
        # Holds the t_a actions to execute *this* cycle.
        # Shape (t_a, action_dim) when populated.
        self._actions_to_execute = deque()

        # --- Counter for steps within the t_a execution window (for Inference) ---
        self._steps_since_last_prediction = 0

    def _create_networks(self):
        """
        Creates the Flow Policy model and related components.
        Called by the parent class's __init__.
        """

        self.nets = nn.ModuleDict()

        flow_model = FlowPolicy(
            algo_config=self.algo_config,
            global_config=self.global_config,
            device=self.device,
        )
        flow_model_target = FlowPolicy(
            algo_config=self.algo_config,
            global_config=self.global_config,
            device=self.device,
        )
        nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {
                        "flow_model": flow_model,
                        "flow_model_target": flow_model_target,
                    }
                )
            }
        )
        nets = nets.float().to(self.device)

        print("Policy Network initialized.")

        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(
                parameters=nets["policy"]["flow_model"].parameters(), decay=self.algo_config.ema.power
            )

        self.nets = nets
        self.ema = ema

    def process_batch_for_training(self, batch: Dict) -> Dict:
        """
        Prepares a raw batch from the dataset for training.
        Includes slicing for observations, actions, and *deriving action feedback*.
        """
        t_o = self.t_o  # Observation window size (e.g., frame_stack)
        t_p = self.t_p  # Prediction horizon (sequence length for actions)

        num_feedback = self.num_feedback_actions

        # Total sequence length available in the batch item
        total_seq_len = batch["actions"].shape[1]

        # [-t_o-1],[-2],[-1],[0],[1],[2],[3],...,[t_p-1]
        # |<------- t_o ----|<->|---------- t_p ------->|
        # --- Observation Window ---
        obs_data = {k: batch["obs"][k][:, :t_o, ...] for k in batch["obs"]}
        next_obs_data = {
            k: batch["next_obs"][k][:, :t_o, ...] for k in batch["next_obs"]
        }

        # --- Action Window ---
        action_start_idx = (
            t_o - 1
        )  # Action corresponding to the *last* observation state
        action_data = batch["actions"][:, action_start_idx:, :]  # Shape [B, t_p, A]

        # --- Action Feedback (Derived from Batch) ---
        # We need the num_feedback actions *preceding* the action_data sequence.
        # These are the actions that would have been "unexecuted" if this was inference.
        feedback_start_idx = action_start_idx - num_feedback
        feedback_end_idx = action_start_idx
        assert (
            feedback_start_idx >= 0 and feedback_end_idx <= total_seq_len
        ), f"Invalid feedback indices: start_idx={feedback_start_idx}, end_idx={feedback_end_idx}, total_seq_len={total_seq_len}"
        feedback_actions = batch["actions"][
            :, feedback_start_idx:feedback_end_idx, :
        ]  # Shape [B, num_feedback, A]

        processed_batch = {
            "obs": obs_data,  # Observations [B, t_obs, O]
            "next_obs": next_obs_data,  # Next observations [B, t_obs, O]
            "actions": action_data,  # Target actions [B, t_p, A]
            "feedback_actions": feedback_actions,  # Derived feedback [B, num_feedback, A]
            # "goal_obs": batch.get("goal_obs", None), # Pass along goals if present
        }

        # Move tensors to device and ensure float32
        processed_batch = TensorUtils.to_float(
            TensorUtils.to_device(processed_batch, self.device)
        )

        # Build graph node features from observations
        # Only take the t_g last steps for graph_data. T_o >= t_g
        obs_for_graph = {
            k: v[:, -self.tg :, ...]  # slice off the last tg steps
            for k, v in processed_batch["obs"].items()
        }
        next_obs_for_graph = {
            k: v[:, -self.tg :, ...]  # slice off the last tg steps
            for k, v in processed_batch["next_obs"].items()
        }
        processed_batch["graph_data"] = self.node_feature_processor.process_features(
            obs_for_graph
        )
        processed_batch["next_graph_data"] = (
            self.node_feature_processor.process_features(next_obs_for_graph)
        )

        return processed_batch

    def process_batch_for_inference(self, obs_dict: Dict) -> Dict:
        """
        Prepares observations for inference/rollout.
        Ensures correct shape [1, t_obs, O] and processes features.
        """
        # Ensure tensors have a batch dimension [1, t_obs, O]
        obs_data = obs_dict  # Assume input obs_dict already has [B, t_obs, O] shape from env/wrapper
        first_key = next(iter(obs_data))
        if obs_data[first_key].ndim == 2:  # Shape [t_obs, O] -> Add batch dim
            obs_data = {k: v.unsqueeze(0) for k, v in obs_data.items()}

        # Ensure we only process keys expected by the model/processor
        obs_data_filtered = {
            k: obs_data[k] for k in self.obs_key_shapes if k in obs_data
        }

        # Move tensors to device and ensure float32
        processed_batch = {
            "obs": TensorUtils.to_float(
                TensorUtils.to_device(obs_data_filtered, self.device)
            ),
        }

        obs_for_graph = {
            k: v[:, -self.tg :, ...] for k, v in processed_batch["obs"].items()
        }
        processed_batch["graph_data"] = self.node_feature_processor.process_features(
            obs_for_graph
        )

        return processed_batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Performs a single training or validation step on a processed batch.
        Uses Corrected Conditional Flow Matching (CFM) formulation.
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Extract data from the processed batch
            obs = batch["obs"]  # Observations [B, t_o, O]
            actions = batch["actions"]  # Target actions [B, t_p, A]
            graph_data = batch["graph_data"]  # Dict of processed observation features
            next_graph_data = batch["next_graph_data"]  # Dict of processed next observation features
            feedback_actions = batch[
                "feedback_actions"
            ]  # Derived feedback [B, num_feedback, A]
            B, T, A = actions.shape  # T is t_p (prediction horizon)

            # --- CFM Implementation ---
            t = torch.rand(B, T, 1, device=self.device)  # Sample time [B, T, 1]
            eps = torch.randn_like(actions)  # Sample noise [B, T, A]
            x_t = t * actions + (1.0 - t) * eps  # Interpolated state [B, T, A]
            u_t = actions - eps  # Target vector field [B, T, A]
            # --- End CFM ---

            # Predict the vector field using the Policy Network
            # Pass the derived feedback_actions from the batch here
            predicted_flow,pred_q, _, pred_next_g_emb = self.nets["policy"]["flow_model"](
                action=x_t,  # Interpolated state x_t
                timestep=t,  # Sampled times t
                graph_data=graph_data,  # Observation condition
                previous_unexecuted_actions=feedback_actions,  # Feedback actions from batch
            )  # Shape [B, T, A]

            with torch.no_grad():
                # Predict the next graph embedding using the next_graph_data as target
                _,_, next_g_emb, _ = self.nets["policy"]["flow_model_target"](
                    action=None,  
                    timestep=None,  
                    graph_data=next_graph_data,  
                    previous_unexecuted_actions=None,  
                )

            # Compute loss
            flow_loss = F.huber_loss(predicted_flow, u_t, delta=1.0)

            # Q loss
            # q = obs["robot0_joint_pos"][:, -1, :].float()
            # q_loss = F.huber_loss(pred_q, q, delta=1.0)
            # Dynamic loss
            dyn_loss = F.huber_loss(pred_next_g_emb, next_g_emb, delta=1.0)

            losses = OrderedDict()
            losses["flow_loss"] = flow_loss
            losses["dyn_loss"] = dyn_loss
            # losses["q_loss"] = q_loss
            # Total loss
            total_loss = flow_loss + dyn_loss #+ q_loss
            losses["total_loss"] = total_loss 
            info = {"losses": TensorUtils.detach(losses)}

            # Backpropagation and Optimization Step (if training)
            if not validate:
                step_info = {}
                policy_grad_norm = TorchUtils.backprop_for_loss(
                    net=self.nets["policy"],
                    optim=self.optimizers["policy"],
                    loss=total_loss,
                    max_grad_norm=self.algo_config.grad_clip,  # Use configured grad clipping
                )
                step_info["policy_grad_norms"] = policy_grad_norm

                # Update EMA
                if self.ema is not None:
                    self.ema.step(self.nets["policy"]["flow_model"].parameters())

                # --- TARGET NETWORK UPDATE ---
                tau_target = self.algo_config.get("target_network_update_tau", 0.005)
                target_model_params = self.nets["policy"]["flow_model_target"].parameters()

                with torch.no_grad():
                    if self.ema is not None and self.algo_config.get("update_target_from_ema", True): # Add a config for this
                        # Update target from the EMA shadow parameters of the online model
                        # self.ema.shadow_params should be a list of tensors in the same order as model parameters
                        ema_online_shadow_params = self.ema.shadow_params
                        for target_param, ema_online_param_data in zip(target_model_params, ema_online_shadow_params):
                            target_param.data.copy_(
                                tau_target * ema_online_param_data.data + (1.0 - tau_target) * target_param.data
                            )
                    else:
                        # Update target from the regular (non-EMA) online model parameters
                        online_model_params = self.nets["policy"]["flow_model"].parameters()
                        for target_param, online_param in zip(target_model_params, online_model_params):
                            target_param.data.copy_(
                                tau_target * online_param.data + (1.0 - tau_target) * target_param.data
                            )

                        
                info.update(step_info)

        return info

    def log_info(self, info: Dict) -> Dict:
        """Logs training/validation information."""
        log = super().log_info(info)
        log["Loss"] = info["losses"]["total_loss"].item()
        log["Flow_Loss"] = info["losses"]["flow_loss"].item()
        log["Dyn_Loss"] = info["losses"]["dyn_loss"].item()
        # log["Q_Loss"] = info["losses"]["q_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norm"] = info["policy_grad_norms"]  # Changed key slightly

        # Log parameter count once
        if not hasattr(self, "_num_params_logged") or not self._num_params_logged:
            num_params = sum(
                p.numel() for p in self.nets.parameters() if p.requires_grad
            )
            log["NumParams"] = num_params
            self._num_params_logged = True

        return log

    def get_action(
        self, obs_dict: Dict, goal_dict: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Gets the next action for the environment step using receding horizon.
        Predicts t_p actions every t_a steps, executes t_a, feeds back t_p - t_a.
        Assumes obs_dict already contains t_obs stacked frames with shape [1, t_obs, O].
        """
        assert (
            next(iter(obs_dict.values())).shape[0] == 1
        ), "get_action expects batch size 1"
        assert (
            next(iter(obs_dict.values())).shape[1] == self.t_o
        ), f"get_action expects {self.t_o} observation frames"

        # --- Core Receding Horizon Logic ---

        # Trigger a NEW prediction cycle if it's the first step OR t_a steps executed
        if self._steps_since_last_prediction == 0:

            # 1. Process the current observation to get graph_data
            # process_batch_for_inference returns a dict containing 'graph_data'
            processed_batch = self.process_batch_for_inference(obs_dict)
            current_graph_data = processed_batch["graph_data"]  # Extract the dictionary

            # 2. Get Previous Unexecuted Actions (from the buffer)
            # self.previous_unexecuted_actions_inf has shape [num_feedback, A]
            # Add batch dim for the sample function: [1, num_feedback, A]
            feedback_actions_batched = self.previous_unexecuted_actions_inf.unsqueeze(0)

            # 3. Sample a NEW action sequence using the model
            sampled_action_sequence = self.sample(
                graph_data=current_graph_data,  # Observation condition (dict)
                previous_unexecuted_actions=feedback_actions_batched,  # Action feedback [1, num_feedback, A]
                K=self.algo_config.get(
                    "inference_euler_steps", 10
                ), 
            )  # Output shape [1, t_p, A]

            sampled_action_sequence = sampled_action_sequence.to(self.device)

            # Remove batch dimension for storage: shape [t_p, A]
            action_sequence_no_batch = sampled_action_sequence.squeeze(0)

            # 4. Store the NEXT t_a actions to be executed
            self._actions_to_execute = deque(
                action_sequence_no_batch[: self.t_a, :].tolist()
            )  # Shape [t_a, A]

            # 5. Store the REMAINING t_p - t_a actions for the *next* prediction's feedback
            # Update the inference feedback buffer
            self.previous_unexecuted_actions_inf = action_sequence_no_batch[
                self.t_a :, :
            ].detach()  # Shape [t_p - t_a, A]

            # Reset the step counter (already 0, but explicit)
            self._steps_since_last_prediction = 0

        # Get the action corresponding to the current step within the t_a window
        current_step_action = self._actions_to_execute.popleft()  # Shape [A,]

        # If the deque is empty, reset it to trigger prediction next time
        if not self._actions_to_execute:
            self._steps_since_last_prediction = 0

        # Return the action for the current step (add batch dim back for robomimic env)
        return torch.tensor(current_step_action, device=self.device).unsqueeze(0)  # Shape [1, A]

    @torch.no_grad()
    def sample(
        self,
        graph_data: Dict,  # Observation condition (dict of features)
        previous_unexecuted_actions: torch.Tensor,  # Action feedback [1, num_feedback, A]
        K: int = 10,  # Number of Euler steps
    ) -> torch.Tensor:
        """
        Generates an action sequence using Euler integration of the learned ODE.
        Starts from noise (t=0) and integrates to data (t=1).

        Args:
            graph_data (Dict): Dictionary of processed observation features.
            previous_unexecuted_actions (torch.Tensor): Feedback actions [1, num_feedback, A].
            K (int): Number of integration steps.

        Returns:
            torch.Tensor: The generated clean action sequence, shape [1, t_p, A].
        """
        B = 1  # Inference batch size is 1
        T = self.t_p  # Prediction horizon
        A = self.ac_dim
        num_feedback = self.num_feedback_actions

        # Validate feedback shape
        expected_feedback_shape = (B, num_feedback, A)
        if previous_unexecuted_actions.shape != expected_feedback_shape:
            raise ValueError(
                f"Sample received previous_unexecuted_actions with shape {previous_unexecuted_actions.shape}, expected {expected_feedback_shape}"
            )

        # Use EMA weights if available
        if self.ema is not None:
            self.ema.store(self.nets["policy"]["flow_model"].parameters())
            self.ema.copy_to(self.nets["policy"]["flow_model"].parameters())

        # 1. Initialize starting point from noise N(0, I) at t=0
        x_t = torch.randn(B, T, A, device=self.device)  # Shape [1, t_p, A]

        # 2. Define time steps for Euler integration
        dt = 1.0 / K

        # 3. Perform Euler integration from t=0 to t=1
        for i in range(K):
            t_current = torch.full(
                (B, T, 1), (i * dt), device=self.device, dtype=torch.float32
            )  # Time for current step [1, t_p, 1]

            # Predict flow v(x_t, t, cond, feedback)
            predicted_flow,_, _, _ = self.nets["policy"]["flow_model"](
                action=x_t,
                timestep=t_current,
                graph_data=graph_data,  # Pass the dictionary
                previous_unexecuted_actions=previous_unexecuted_actions,
            )  # Output shape [1, t_p, A]

            # Euler step: x_{t+dt} = x_t + dt * v(x_t, t, ...)
            x_t = x_t + dt * predicted_flow

        # Restore parameters from EMA if used
        if self.ema is not None:
            self.ema.restore(self.nets["policy"]["flow_model"].parameters())

        return x_t.contiguous()  # Shape: [1, t_p, A]
