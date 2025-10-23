"""
FlowMatching Graph Attention Network (FlowGAT) Algorithm for Robomimic.
"""

import os
from collections import OrderedDict, deque
from typing import Dict, Optional, Tuple, Callable
from packaging.version import parse as parse_version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from diffusers.training_utils import EMAModel

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.models.flow_policy import FlowPolicy
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
            json_path=global_config.train.graph_config2, device=self.device
        )



        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        self.batch_size = global_config.train.batch_size
        self.obs_window = global_config.train.frame_stack
        self.pred_horizon = algo_config.t_p
        self.exec_horizon = algo_config.t_a
        self.temp_edges = algo_config.temp_edges
        self.has_edge_attr = algo_config.has_edge_attr
        assert self.exec_horizon <= self.pred_horizon, "t_a cannot be greater than t_p"

        self.chain = None
        try:
            script_dir = os.path.dirname(__file__)
            mjcf_path = os.path.join(script_dir, "panda", "robot.xml")
            if os.path.exists(mjcf_path):
                import pytorch_kinematics as pk
                self.chain = pk.build_serial_chain_from_mjcf(
                    open(mjcf_path).read(), "right_hand"
                ).to(dtype=torch.float32, device=self.device)
        except Exception:
            pass

        # --- Buffer for actions from the *current* prediction sequence (for Inference) ---
        # Holds the t_a actions to execute *this* cycle.
        # Shape (t_a, action_dim) when populated.
        self._actions_to_execute = deque()

        # --- Counter for steps within the t_a execution window (for Inference) ---
        self._steps_since_last_prediction = 0

        

    def _create_networks(self):
        """Create Flow Policy model and related components."""
        flow_model = FlowPolicy(
            algo_config=self.algo_config,
            global_config=self.global_config,
            device=self.device,
        )
        self.nets = nn.ModuleDict({
            "policy": nn.ModuleDict({"flow_model": flow_model})
        }).float().to(self.device)
        self.ema = (
            EMAModel(
                parameters=self.nets["policy"]["flow_model"].parameters(),
                decay=self.algo_config.ema.power
            ).to(self.device)
            if self.algo_config.ema.enabled else None
        )

        print("Policy Network initialized.")

    def process_batch_for_training(self, batch: Dict) -> Dict:
        """
        Prepare a raw batch for training: slice obs/actions, normalize, and build graph.
        """
        obs_data = {k: batch["obs"][k][:, :self.obs_window] for k in batch["obs"]}
        action_data = batch["actions"][:, self.obs_window - 1:].to(self.device, dtype=torch.float32)
        obs_tensor = torch.cat(list(obs_data.values()), dim=-1)
        obs_tensor = TensorUtils.to_float(TensorUtils.to_device(obs_tensor, self.device))
        action_data = TensorUtils.to_float(TensorUtils.to_device(action_data, self.device))
        processed_batch = {
            "obs_tensor": obs_tensor,
            "actions": action_data,
            "graph": self.converter.convert(
               obs_tensor, 
               temporal_edges=self.temp_edges, 
               has_edge_attr=self.has_edge_attr,
               edge_features=self.algo_config.edge_features
            )
        }
        return processed_batch

    def process_batch_for_inference(self, obs_dict: Dict) -> Dict:
        """
        Prepare observations for inference/rollout. Ensures shape [1, t_obs, O] and processes features.
        """
        obs_data = {k: v.unsqueeze(0) if v.ndim == 2 else v for k, v in obs_dict.items()}
        obs_data = {k: obs_data[k] for k in self.obs_key_shapes if k in obs_data}
        obs_tensor = torch.cat(list(obs_data.values()), dim=-1)
        obs_tensor = TensorUtils.to_float(TensorUtils.to_device(obs_tensor, self.device))
        return {
            "obs_tensor": obs_tensor,
            "graph": self.converter.convert(
               obs_tensor, 
               temporal_edges=self.temp_edges, 
               has_edge_attr=self.has_edge_attr,
               edge_features=self.algo_config.edge_features
            )
        }

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Performs a single training or validation step on a processed batch using Corrected Conditional Flow Matching (CFM).
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            actions = batch["actions"]
            graph = batch["graph"]
            batch_size, seq_len, action_dim = actions.shape

            # CFM: sample time and noise, interpolate state, compute target vector field
            t = torch.rand(batch_size, 1, 1, device=self.device)
            eps = torch.randn_like(actions)
            x_t = t * actions + (1.0 - t) * eps
            u_t = actions - eps

            predicted_flow = self.nets["policy"]["flow_model"](
                action=x_t,
                timestep=t,
                graph=graph,
                # obs = batch["obs_tensor"]
            )

            flow_loss = F.huber_loss(predicted_flow, u_t, delta=1.0)
            losses = OrderedDict(flow_loss=flow_loss)
            losses["total_loss"] = flow_loss
            info = {"losses": TensorUtils.detach(losses)}

            if not validate:
                step_info = {}
                policy_grad_norm = TorchUtils.backprop_for_loss(
                    net=self.nets["policy"],
                    optim=self.optimizers["policy"],
                    loss=flow_loss,
                    max_grad_norm=self.algo_config.grad_clip,
                )
                step_info["policy_grad_norms"] = policy_grad_norm
                if self.ema is not None:
                    self.ema.step(self.nets["policy"]["flow_model"].parameters())
                info.update(step_info)

        return info

    def log_info(self, info: Dict) -> Dict:
        """Log training/validation info, including losses and parameter count."""
        log = super().log_info(info)
        log["Loss"] = info["losses"]["total_loss"].item()
        log["Flow_Loss"] = info["losses"]["flow_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norm"] = info["policy_grad_norms"]
        if not getattr(self, "_num_params_logged", False):
            log["NumParams"] = sum(p.numel() for p in self.nets.parameters() if p.requires_grad)
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
            batch = self.process_batch_for_inference(obs_dict)
            graph = batch['graph']
            # graph = None

            # [1, t_p, A] → [t_p, A]
            seq = self.sample(
                K=self.algo_config["inference_euler_steps"], graph=graph, obs=batch['obs_tensor']
            ).squeeze(0)

            # Queue t_a steps for execution
            self._actions_to_execute = deque(seq[: self.exec_horizon].cpu().tolist())

        # Pop one action and return
        action = self._actions_to_execute.popleft()

        action = torch.tensor(action, device=self.device, dtype=torch.float32)

        return action.unsqueeze(0)  # Return as [1, A] shape

    @torch.no_grad()
    def sample(
        self,
        K: int = 10,  # Number of Euler steps
        graph: Optional[Data] = None,  # Optional graph data for the model
        obs: Optional[torch.Tensor] = None,  # Optional observation features
    ) -> torch.Tensor:
        """
        Generate an action sequence by integrating the learned ODE from noise to data using Euler's method.

        Args:
            K (int): Number of integration steps.
            graph (Optional[Data]): Optional graph data for the model.
            obs (Optional[torch.Tensor]): Optional observation features.

        Returns:
            torch.Tensor: Generated action sequence of shape [1, t_p, A].
        """
        batch_size = 1
        seq_len = self.pred_horizon
        action_dim = self.ac_dim

        # Use EMA weights if available
        if self.ema is not None:
            self.ema.store(self.nets["policy"]["flow_model"].parameters())
            self.ema.copy_to(self.nets["policy"]["flow_model"].parameters())

        # Start from noise at t=0
        x_t = torch.randn(batch_size, seq_len, action_dim, device=self.device)
        dt = 1.0 / K

        for i in range(K):
            t_current = torch.full((batch_size, 1, 1), i * dt, device=self.device)
            predicted_flow = self.nets["policy"]["flow_model"](
                action=x_t,
                timestep=t_current,
                graph=graph,
                # obs = obs
            )
            x_t = x_t + dt * predicted_flow

        if self.ema is not None:
            self.ema.restore(self.nets["policy"]["flow_model"].parameters())

        return x_t.contiguous()

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

