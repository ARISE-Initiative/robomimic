import torch
import torch.utils
from robomimic.algo import register_algo_factory_func, PolicyAlgo
from torch import nn
from collections import OrderedDict

from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, GATv2Conv
import pytorch_kinematics as pk
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.loss_utils as LossUtils
from typing import Dict, Any
import torch.nn.functional as F


@register_algo_factory_func("gat_bc")
def algo_config_to_class(algo_config):
    return GAT_BC, {}


class GAT_BC(PolicyAlgo):
    def __init__(self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device):
        super().__init__(algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device)
        self.chain = pk.build_serial_chain_from_mjcf(
            open("robomimic/algo/panda/robot.xml").read(), "link7"
        ).to(dtype=torch.float32)  # Use float32 for consistency
        # Initialize policy optimizer; ensures self.optimizers["policy"] exists
        self.optimizers = {}
        self.optimizers["policy"] = torch.optim.Adam(
            self.nets.parameters(),
            lr=self.algo_config.optim_params.policy.learning_rate.initial
        )

    def _create_networks(self):
        """
        Create and initialize the graph-based policy network.
        """
        # 1. Node Feature Dimensions (Simplified)
        node_dims = {
            **{f"joint_{i}": 4 for i in range(7)},  # pos(3) + angle(1)
            "eef": 9,    # pos(3) + quat(4) + gripper(2)
            "object": 14,  # pos(3) + rot_mat(9) OR pos(3) + quat(4) + gripper(2)
            "toolhang": 16,  # pos(3) + rot_mat(9) OR pos(3) + quat(4) + gripper(2) + state(1) + toolhang_assembled(1)
            "base": 14,  # pos(3) + rot_mat(9) OR pos(3) + quat(4) + gripper(2)
        }

        # 2. Node Encoder (Simplified MLP)
        hidden_dim = self.algo_config.node_encoder_dims[-1]  # Use final dimension
        self.node_encoder = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(dim, hidden_dim)
            ) for key, dim in node_dims.items()
        })

        # 3. Graph Encoder
        self.graph_encoder = nn.ModuleList()  # Use ModuleList for sequential layers
        in_channels = hidden_dim  # Now using single hidden dimension
        out_channels = self.algo_config.hidden_dim
        for i in range(self.algo_config.num_layers):
            heads = (
                self.algo_config.heads if i < self.algo_config.num_layers - 1 else 1
            )  # Last layer has one head
            self.graph_encoder.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    edge_dim=3,  # Ensure edge_dim is consistent
                    dropout=self.algo_config.dropout,
                    add_self_loops=True, # Prevent self-loops,
                )
            )
            in_channels = out_channels * heads  # Update in_channels for next layer

        # 4. Output Layer (After Graph Encoder)
        self.output_layer = nn.Linear(
            self.algo_config.hidden_dim, self.algo_config.action_dim
        )

        # 5. Register Networks (More Organized)
        self.nets = nn.ModuleDict(
            {"node_encoder": self.node_encoder, "graph_encoder": self.graph_encoder, "output": self.output_layer}
        )

        self.nets = self.nets.to(self.device)


    def process_batch_for_training(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation batch for training."""
        input_batch = {
            "obs":  {k: batch["obs"][k][:, 0, :] for k in batch["obs"]},
            "goal_obs": batch.get("goal_obs", None),
            "actions": batch["actions"][:, :, :].view(batch["actions"].shape[0], -1) if batch["actions"] is not None else None,
        }
        processor = NodeFeatureProcessor(chain=self.chain)
        input_batch["obs"] = processor.process_features(input_batch["obs"])
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def process_single_obs_for_eval(self, obs_dict):
        """Process single observation for evaluation."""
        obs_dict = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
        batch = {"obs": obs_dict, "actions": None}  # Add dummy actions for consistency
        return self.process_batch_for_training(batch)["obs"]

    def train_on_batch(self, batch, epoch, validate=False):
        """Train or validate on a batch of data."""
        with TorchUtils.maybe_no_grad(no_grad=validate):
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info = {
                "predictions": TensorUtils.detach(predictions),
                "losses": TensorUtils.detach(losses),
            }

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)
        return info

    def _forward_training(self, batch):
        """Forward pass for training."""
        obs = batch["obs"]

        # 1. Encode Nodes
        encoded_nodes = {
            key: self.node_encoder[key](value) for key, value in obs.items()
        }

        # 2. Build Graph
        graph = self._build_graph(encoded_nodes, obs)

        # 3. Graph Encoding (Simplified)
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        for layer in self.graph_encoder:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)  # Apply ReLU activation
            x = F.dropout(x, p=self.algo_config.dropout)
            
        # 4. Global Pooling
        x = global_mean_pool(x, graph.batch)

        # 5. Output Layer
        x = self.output_layer(x)
        actions = torch.tanh(x)

        return {"actions": actions}

    def _build_graph(self, encoded_nodes, obs):
        """Build the graph from encoded node features."""

        adjacencies = torch.tensor(
            [
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # joint_0
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # joint_1
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # joint_2
            [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],  # joint_3
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],  # joint_4
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],  # joint_5
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # joint_6
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],  # eef
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # object
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # toolhang
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # base
            ],
            dtype=torch.bool,
            device=self.device,
        )
        edge_index = adjacencies.nonzero(as_tuple=False).t().contiguous()

        # Stack node features (correct order and handling of batch size)
        node_names = list(encoded_nodes.keys()) # Get keys in a consistent order
        node_features = torch.stack([encoded_nodes[key] for key in node_names], dim=1)
        if len(node_features.shape) == 2:  # Add batch dimension if needed
            node_features = node_features.unsqueeze(0)

        # Create edge attributes based on 3D positions
        positions = torch.stack([obs[key][:, :3] for key in node_names], dim=1)
        edge_features_list = []
        for i in range(positions.shape[0]): # Iterate over batch
                edge_features = torch.abs(
                    positions[i, edge_index[0]] - positions[i, edge_index[1]]
                )
                edge_features_list.append(edge_features)

        # Batch from Data list
        data_list = [
            Data(x=node_features[i], edge_index=edge_index, edge_attr=edge_features_list[i])
            for i in range(node_features.shape[0])
        ]
        batch_graph = Batch.from_data_list(data_list).to(self.device)
        return batch_graph

    def _compute_losses(self, predictions, batch):
        """Compute losses."""
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]

        # Use combined loss (more robust)
        losses["mse_loss"] = nn.MSELoss()(actions, a_target)
        losses["smooth_l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        losses["cos_loss"] = LossUtils.cosine_loss(
            actions[..., :3], a_target[..., :3]
        )  # Cosine loss on direction

        # Weighted sum of losses
        losses["action_loss"] = (
            self.algo_config.loss.l2_weight * losses["mse_loss"]
            + self.algo_config.loss.l1_weight * losses["smooth_l1_loss"]
            + self.algo_config.loss.cos_weight * losses["cos_loss"]
        )
        return losses

    def _train_step(self, losses):
        """Perform a single training step."""
        info = OrderedDict()
        self.optimizers["policy"].zero_grad()  # Use the correct optimizer key
        losses["action_loss"].backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.nets.parameters(), self.algo_config.grad_clip
        )  # Gradient clipping
        self.optimizers["policy"].step()
        info["policy_grad_norms"] = policy_grad_norm
        return info

    def log_info(self, info):
        """Log training information."""
        log = super().log_info(info)
        log["Number of Parameters"] = sum(
            p.numel() for p in self.nets.parameters() if p.requires_grad
        )
        log["Loss"] = info["losses"]["action_loss"].item()
        log["MSE_Loss"] = info["losses"]["mse_loss"].item()
        log["Smooth_L1_Loss"] = info["losses"]["smooth_l1_loss"].item()
        log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"].item()
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """Get action from the policy."""
        self.nets.eval()  # Set to evaluation mode
        with torch.no_grad():
            obs = self.process_single_obs_for_eval(obs_dict)
            action = self._forward_training({"obs": obs})["actions"][:,:7]
        return action


class NodeFeatureProcessor:
    """Processes raw observations into node features."""

    def __init__(self, num_joints: int = 7, chain: pk.SerialChain = None):
        self.chain = chain
        self.num_joints = num_joints

    def process_features(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract node features from observation dictionary."""
        joint_pos = obs["robot0_joint_pos"].float() # Ensure float32
        eef_pos = obs["robot0_eef_pos"].float()
        eef_quat = obs["robot0_eef_quat"].float()
        gripper_qpos = obs["robot0_gripper_qpos"].float()
        object = obs["object"].float()


        self.chain.to(device=joint_pos.device)
        fk = self.chain.forward_kinematics(joint_pos, end_only=False)

        node_dict = {}
        # Joint features: position (3) + joint angle (1)
        for i in range(self.num_joints):
            t_ = fk[f"link{i + 1}"].get_matrix()
            node_dict[f"joint_{i}"] = torch.cat(
                [t_[:, :3, 3], joint_pos[:, i].unsqueeze(-1)], dim=-1
            )

        # End effector features: position (3) + quaternion (4) + gripper (2)
        node_dict["eef"] = torch.cat([eef_pos, eef_quat, gripper_qpos, ], dim=-1)

        # Object features: 14-dimensional features (absolute position + relative position to eef)
        node_dict["object"] = object[:,:14]


        # Toolhang features: 16-dimensional features (absolute position + relative position to eef + state if tool is hanging and tool hang assembled)
        node_dict["toolhang"] = torch.cat([object[:,28:42],object[:,42:44]], dim=-1)
        # Base frame features: 14-dimensional features (absolute position + relative position to eef)
        node_dict["base"] = object[:,14:28]

        return node_dict