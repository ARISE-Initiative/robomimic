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
# import networkx as nx
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
from robomimic.algo.flow_gat_files.graph_converter import JsonTemporalGraphConverter


@register_algo_factory_func("flow_gat")
def algo_config_to_class(algo_config):
    """Factory function for FLOWGAT algorithm."""
    return FLOW_GAT, {}


class FLOW_GAT(PolicyAlgo):
    def __init__(
        self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
    ):
        # Call parent __init__ first - it sets up device, config, etc. and calls _create_networks
        super().__init__(
            algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
        )
        self.converter = JsonTemporalGraphConverter(
            json_path=global_config.train.graph_config, device=self.device
        )
        print(self.global_config.train.graph_config)
        print(device)

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

        print(
            f"Receding Horizon: Prediction Horizon t_p={self.t_p}, Execution Horizon t_a={self.t_a}"
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

        nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {
                        "flow_model": flow_model,
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
                parameters=nets["policy"]["flow_model"].parameters(),
                decay=self.algo_config.ema.power
            ).to(self.device)

        self.nets = nets
        self.ema = ema

    def process_batch_for_training(self, batch: Dict) -> Dict:
        """
        Prepares a raw batch from the dataset for training.
        Includes slicing for observations, actions, and *deriving action feedback*.
        """
        t_o = self.t_o  # Observation window size (e.g., frame_stack)
        t_p = self.t_p  # Prediction horizon (sequence length for actions)

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

        processed_batch = {
            "obs": obs_data,  # Observations [B, t_obs, O]
            "next_obs": next_obs_data,  # Next observations [B, t_obs, O]
            "actions": action_data,  # Target actions [B, t_p, A]
        }

        # Move tensors to device and ensure float32
        processed_batch = TensorUtils.to_float(
            TensorUtils.to_device(processed_batch, self.device)
        )

        obs_tensor = torch.cat(
            [v for k, v in obs_data.items()], dim=-1
        )
        # Convert to float32
        obs_tensor = TensorUtils.to_float(obs_tensor)
        processed_batch["graph"] = self.converter.convert(obs_tensor, temporal_edges=True)

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

        obs_tensor = torch.cat(
            [v for k, v in obs_data_filtered.items()], dim=-1
        )
        # Convert to float32
        obs_tensor = TensorUtils.to_float(obs_tensor)
        processed_batch["graph"] = self.converter.convert(obs_tensor, temporal_edges=True)

        return processed_batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Performs a single training or validation step on a processed batch.
        Uses Corrected Conditional Flow Matching (CFM) formulation.
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Extract data from the processed batch
            actions = batch["actions"]  # Target actions [B, t_p, A]
            graph = batch["graph"]

            B, T, A = actions.shape  # T is t_p (prediction horizon)

            # --- CFM Implementation ---
            t = torch.rand(B, 1, 1, device=self.device)  # Sample time [B, T, 1]
            eps = torch.randn_like(actions)  # Sample noise [B, T, A]
            x_t = t * actions + (1.0 - t) * eps  # Interpolated state [B, T, A]
            u_t = actions - eps  # Target vector field [B, T, A]
            # --- End CFM ---

            

            predicted_flow, pred_q, _, pred_next_g_emb = self.nets["policy"][
                "flow_model"
            ](
                action=x_t,  # Interpolated state x_t
                timestep=t,  # Sampled times t
                obs=None,
                graph=graph,  # Optional graph data
            )  # Shape [B, T, A]

            # Compute loss
            flow_loss = F.huber_loss(predicted_flow, u_t, delta=1.0)



            losses = OrderedDict()
            losses["flow_loss"] = flow_loss

            # Total loss
            total_loss = flow_loss
            losses["total_loss"] = total_loss
            info = {"losses": TensorUtils.detach(losses)}

            # fusion_gates = self.nets["policy"].flow_model.graph_encoder.fusion.last_gates
            # info.update({"fusion_gates": fusion_gates})

            # # --- dump gates every N steps (existing) ----------------------------
            # if not hasattr(self, "_fusion_dump_counter"):
            #     self._fusion_dump_counter = 0
            #     self._fusion_dump_every   = 10
            # self._fusion_dump_counter += 1
            # if self._fusion_dump_counter % self._fusion_dump_every == 0:
            #     import json
            #     # --- dump fusion gates ---
            #     with open("fusion_gates_log.jsonl", "a") as f:
            #         rec = {"epoch": epoch, "gates": fusion_gates.tolist()}
            #         f.write(json.dumps(rec) + "\n")

            #     # --- dump attention pooling weights ---
            #     pool = self.nets["policy"].flow_model.graph_encoder.pooling
            #     attn_w = pool.last_attn_weights        # [num_nodes,1]
            #     batch_i = pool.last_attn_batch_idx     # [num_nodes]
            #     with open("attention_pool_log.jsonl", "a") as f2:
            #         rec2 = {
            #             "epoch": epoch,
            #             "attn_weights": attn_w.flatten().tolist(),
            #             "batch_idx":    batch_i.tolist()
            #         }
            #         f2.write(json.dumps(rec2) + "\n")
            # --------------------------------------------------------------------

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

                info.update(step_info)

        return info

    def log_info(self, info: Dict) -> Dict:
        """Logs training/validation information."""
        log = super().log_info(info)
        log["Loss"] = info["losses"]["total_loss"].item()
        log["Flow_Loss"] = info["losses"]["flow_loss"].item()
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
        Receding‐horizon action: predict t_p steps every t_a calls
        and execute one action at a time.
        """
        assert not self.nets.training  # Ensure we're in eval mode
        # If no actions queued, sample a new sequence
        if not self._actions_to_execute:
            graph = self.process_batch_for_inference(obs_dict)["graph"]

            # [1, t_p, A] → [t_p, A]
            seq = self.sample(
                K=self.algo_config.get("inference_euler_steps", 5), graph=graph
            ).squeeze(0)

            # Queue t_a steps for execution
            self._actions_to_execute = deque(seq[: self.t_a].cpu().tolist())

        # Pop one action and return
        action = self._actions_to_execute.popleft()
        return torch.tensor(action, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def sample(
        self,
        K: int = 10,  # Number of Euler steps
        graph: Optional[Data] = None,  # Optional graph data for the model
    ) -> torch.Tensor:
        """
        Generates an action sequence using Euler integration of the learned ODE.
        Starts from noise (t=0) and integrates to data (t=1).

        Args:
            graph_data (Dict): Dictionary of processed observation features.
            K (int): Number of integration steps.

        Returns:
            torch.Tensor: The generated clean action sequence, shape [1, t_p, A].
        """
        B = 1  # Inference batch size is 1
        T = self.t_p  # Prediction horizon
        A = self.ac_dim

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
                (B, 1, 1), (i * dt), device=self.device, dtype=torch.float32
            )  # Time for current step [1, t_p, 1]
            # Predict flow v(x_t, t, cond, feedback)
            predicted_flow, _, _, _ = self.nets["policy"]["flow_model"](
                action=x_t,
                timestep=t_current,
                graph=graph,
            )  # Output shape [1, t_p, A]


            # Euler step: x_{t+dt} = x_t + dt * v(x_t, t, ...)
            x_t = x_t + dt * predicted_flow

        # Restore parameters from EMA if used
        if self.ema is not None:
            self.ema.restore(self.nets["policy"]["flow_model"].parameters())

        return x_t.contiguous()  # Shape: [1, t_p, A]

    def reset(self):
        """
        Reset the internal state of the algorithm.
        Clears the action execution queue and resets the step counter.
        """
        self._actions_to_execute.clear()
        self._steps_since_last_prediction = 0

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {"nets": self.nets.state_dict(), 
                "ema": self.ema.state_dict() if self.ema is not None else None}

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if self.ema is not None and "ema" in model_dict:
            self.ema.load_state_dict(model_dict["ema"])

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.nets.eval()
        

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.nets.train()