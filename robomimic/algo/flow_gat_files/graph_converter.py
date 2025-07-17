import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import (
    ToUndirected,
    AddSelfLoops,
)  # Not used directly in visualize
from typing import List, Dict, Union, Optional, Any, Tuple  # Add Tuple
import json
# import networkx as nx
import matplotlib.pyplot as plt  # Ensure this is at the top of your file
import matplotlib.colors as mcolors  # For more color options
from collections import defaultdict
import pytorch_kinematics as pk
from typing import Callable, Optional
import numpy as np

class JsonTemporalGraphConverter:
    def __init__(self, json_path: str, device: str = "cpu"):
        self.json_path = json_path
        self.config = json.loads(open(json_path, "r").read())
        self.device = device
        self.chain = self._load_pk_chain()
        self.nodes = self.config["nodes"]
        self.edges = self.config["edges"]

        self.node_types = [v["type"] for v in self.nodes]
        self.node_ids = [v["id"] for v in self.nodes]
        self.node_id_map = {v: i for i, v in enumerate(self.node_ids)}
        self.node_id_to_idx = {v["id"]: i for i, v in enumerate(self.nodes)}
        self.unique_node_types = {t: i for i, t in enumerate(sorted(list(set(self.node_types))))}

        self.global_features_indices = self.config.get("global_features", [])

        # self.jax_kin = JaxKinematics(config=self.config)

    def convert(
        self,
        feat_vec: torch.Tensor,
        extra_fn: Optional[
            Callable[
                [
                    "JsonTemporalGraphConverter",
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                torch.Tensor,
            ]
        ] = None,
        temporal_edges: bool = True,
        has_edge_attr: bool = True,
        edge_features: Optional[List[str]] = None
    ) -> Data:
        """
        Convert the feature vector to a PyTorch Geometric Data object.
        """
        B, T, _ = feat_vec.shape
        feat_vec = feat_vec.to(self.device)
        new_feat_vec = self.preprocess(feat_vec, extra_fn=extra_fn) # (B * T, N, F)
        _, _, F = new_feat_vec.shape  # Get the number of features after preprocessing

        global_features = None
        if self.global_features_indices:
            global_features = feat_vec[:, :, self.global_features_indices]
            global_features = global_features.view(B * T, -1)

        node_type_map = {v["id"]: v["type"] for v in self.nodes}
        graph_edge_index = [
            [self.node_id_map[e["source_id"]], self.node_id_map[e["target_id"]]]
            for e in self.edges
        ]
        graph_edge_index = (
            torch.tensor(graph_edge_index, dtype=torch.long, device=self.device)
            .t()
            .contiguous()
        )

        node_pos = new_feat_vec[:, :, :3]  # shape (B, num_nodes, 3)

        # ----------------------------------------------------------------------------
        # Create edge attributes (vectorized)
        src_idx = graph_edge_index[0]  # (E,)
        dst_idx = graph_edge_index[1]  # (E,)
        # Δ position per edge
        dp = node_pos[:, dst_idx, :] - node_pos[:, src_idx, :]  # (B * T, E, 3)
        # distance per edge
        dist = dp.norm(dim=2, keepdim=True)  # (B * T, E, 1)

        # build a length-E mask for kinematic edges
        edge_src = [node_type_map[self.node_ids[i]] for i in src_idx.tolist()]
        edge_dst = [node_type_map[self.node_ids[i]] for i in dst_idx.tolist()]
        kin_mask = torch.tensor(
            [
                (
                    1
                    if s.startswith("joint")
                    and (d.startswith("joint") or d.startswith("eef"))
                    else 0
                )
                for s, d in zip(edge_src, edge_dst)
            ],
            device=self.device,
            dtype=torch.float32,
        )  # (E,)

        # one-hot encode it and expand to batch
        etype = torch.stack([kin_mask, 1 - kin_mask], dim=1)  # (E, 2)
        etype = etype.unsqueeze(0).repeat(B * T, 1, 1)  # (B * T, E, 2)

        # final edge_attr
        if has_edge_attr:
            # Default edge features if not specified
            if edge_features is None:
                edge_features = ['relative_position', 'distance', 'edge_type']
            
            edge_attr_components = []
            
            # Add features based on the list
            if 'relative_position' in edge_features:
                edge_attr_components.append(dp)
            if 'distance' in edge_features:
                edge_attr_components.append(dist)
            if 'edge_type' in edge_features:
                edge_attr_components.append(etype)
            
            # Concatenate enabled features
            if edge_attr_components:
                edge_attr = torch.cat(edge_attr_components, dim=2)
            else:
                edge_attr = None
        else:
            edge_attr = None

        # Create Batched PyTorch Geometric Data object
        num_nodes_per_graph = len(self.node_ids)  # Number of nodes in a single graph example

        # Adjust edge_index for batching and temporal stacking:
        batch_offsets = torch.arange(B * T, device=self.device) * num_nodes_per_graph  # (B*T,)
        batch_offsets = batch_offsets.view(B * T, 1, 1)  # (B*T,1,1)
        batched_edge_index = graph_edge_index.unsqueeze(0) + batch_offsets  # (B*T,2,E)
        batched_edge_index = batched_edge_index.permute(1,0,2).reshape(2, -1)  # (2, B*T*E)


        if temporal_edges:
            # --- Vectorized temporal edges ---
            # For each batch, for each node, connect v_{t-1} <-> v_t for t=1..T-1
            node_indices = torch.arange(num_nodes_per_graph, device=self.device)  # (N,)
            batch_indices = torch.arange(B, device=self.device)  # (B,)
            time_indices = torch.arange(1, T, device=self.device)  # (T-1,)

            # Compute source and target indices for temporal edges
            # For each batch b, time t, node n:
            #   src = b * T * N + (t-1) * N + n
            #   tgt = b * T * N + t * N + n
            src_temporal = (
                batch_indices[:, None, None] * T * num_nodes_per_graph
                + (time_indices[None, :, None] - 1) * num_nodes_per_graph
                + node_indices[None, None, :]
            )  # (B, T-1, N)
            tgt_temporal = (
                batch_indices[:, None, None] * T * num_nodes_per_graph
                + time_indices[None, :, None] * num_nodes_per_graph
                + node_indices[None, None, :]
            )  # (B, T-1, N)

            src_temporal = src_temporal.reshape(-1)
            tgt_temporal = tgt_temporal.reshape(-1)

            # Stack both directions for undirected temporal edges
            temporal_edge_index = torch.stack([
                torch.cat([src_temporal, tgt_temporal], dim=0),
                torch.cat([tgt_temporal, src_temporal], dim=0)
            ], dim=0)  # (2, 2*B*(T-1)*N)

            # Temporal edge attributes: zeros, same feature dim as edge_attr
            num_temporal_edges = temporal_edge_index.shape[1]
            edge_attr_dim = edge_attr.shape[-1]
            temporal_edge_attr = torch.zeros((num_temporal_edges, edge_attr_dim), device=self.device)

            # --- Combine spatial and temporal edges ---
            final_edge_index = torch.cat([batched_edge_index, temporal_edge_index], dim=1)
            final_edge_attr = torch.cat([edge_attr.contiguous().view(-1, edge_attr_dim), temporal_edge_attr], dim=0)
        else:
            # No temporal edges, just use the spatial edges
            final_edge_index = batched_edge_index
            if edge_attr is not None:
                final_edge_attr = edge_attr.contiguous().view(-1, edge_attr.shape[-1])
            else:
                final_edge_attr = None

        node_type_mask = torch.tensor(
            [self.unique_node_types[v["type"]] for v in self.nodes], device=self.device
        )
        # Maps each node to its temporal idx in the batch
        node_temporal_mask = torch.arange(T, device=self.device).view(1, T, 1).repeat(B, 1, num_nodes_per_graph)
        node_temporal_mask = node_temporal_mask.view(B * T, num_nodes_per_graph)  # Shape: (B * T, num_nodes_per_graph)
        node_temporal_mask = node_temporal_mask.view(-1) # Flatten to (B * T * num_nodes_per_graph,)


        data = Data(
            x=new_feat_vec.view(-1, F),  # Shape: (B * num_nodes_per_graph, F)
            edge_index=final_edge_index,  # Shape: (2, B * num_edges_single_graph)
            edge_attr=final_edge_attr,  # Shape: (B * num_edges_single_graph, num_edge_features)
            num_nodes=num_nodes_per_graph * B * T,  # Total number of nodes in the batch
            node_type=node_type_mask.unsqueeze(0).repeat(
                B, 1
            ), 
            t = T, # Shape: (B, num_nodes_per_graph)
            node_temporal_mask=node_temporal_mask
        )
        if global_features is not None:
            data.global_features = global_features
        # Make undirected and add self-loops
        # data = ToUndirected()(data)

        data.batch = (
            torch.arange(B, device=self.device)
            .view(-1, 1)
            .repeat(1, num_nodes_per_graph * T) 
            .flatten()
        )
        data = data.to(self.device)
        return data

    def preprocess(
        self,
        feat_vec: torch.Tensor,
        extra_fn: Optional[
            Callable[
                [
                    "JsonTemporalGraphConverter",
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                torch.Tensor,
            ]
        ] = None,
    ) -> torch.Tensor:
        """
        Preprocess the feature vector and convert it to a PyTorch Geometric Data object.
        """
        B, T, F = feat_vec.shape  # Reshape to (B*T, F)
        base_offset = self.config.get("robot_base_offset", [0, 0, 0])
        feat_vec = feat_vec.view(-1, F)

        se3_pose = torch.zeros(
            (B * T, len(self.nodes), 9), device=self.device, dtype=feat_vec.dtype
        )

        # ----------------------------------------------------------------------------
        # Compute SE(3) transformations from joint positions (pos + 6D rotation)
        joint_pos_node_idx = [
            (self.node_id_to_idx[k], v["joint_pos"][0])
            for k, v in self.config["node_feature_map"].items()
            if v.get("type") == "joint"
        ]
        joint_node_idx, joint_pos_idx = zip(*joint_pos_node_idx)
        joint_node_idx = torch.tensor(
            joint_node_idx, device=self.device, dtype=torch.long
        )
        joint_pos_idx = torch.tensor(
            joint_pos_idx, device=self.device, dtype=torch.long
        )
        joint_pos = feat_vec[:, joint_pos_idx]
        joint_se3_pose = self._compute_se3_from_qpos(
            joint_pos
        )  # Shape: (B, num_joints, 9)

        # 3) Fill the SE(3) pose for joints
        se3_pose[:, joint_node_idx, :] = joint_se3_pose[:,3:10,:]

        # ----------------------------------------------------------------------------
        # Gather remaining SE(3) transformations (EEF, object, etc.)
        # Get positions
        positions_idx = [
            (self.node_id_to_idx[k], v["position"])
            for k, v in self.config["node_feature_map"].items()
            if v.get("type") != "joint" and v.get("position") is not None
        ]
        # Get rotations
        rotations_idx = [
            (self.node_id_to_idx[k], v["rotation"])
            for k, v in self.config["node_feature_map"].items()
            if v.get("type") != "joint" and v.get("rotation") is not None
        ]
        positions_node_idx, positions_idx = zip(*positions_idx)
        rotations_node_idx, rotations_idx = zip(*rotations_idx)
        positions_node_idx = torch.tensor(
            positions_node_idx, device=self.device, dtype=torch.long
        )
        rotations_node_idx = torch.tensor(
            rotations_node_idx, device=self.device, dtype=torch.long
        )

        pos_feat = feat_vec[:, positions_idx]  # Shape: (B, num_nodes, 3)
        # Add base offset to positions
        pos_feat = pos_feat - torch.tensor(
            base_offset, device=self.device, dtype=pos_feat.dtype
        )
        quats = feat_vec[:, rotations_idx]
        # Replace all zero-norm quaternions with [1, 0, 0, 0]
        norms = torch.norm(quats, dim=-1, keepdim=True)
        zero_norm_mask = norms == 0
        quats = torch.where(
            zero_norm_mask.expand_as(quats),
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=quats.device, dtype=quats.dtype),
            quats,
        )
        rot_feat = pk.matrix_to_rotation_6d(
            pk.quaternion_to_matrix(quats)
        )  # Shape: (B, num_nodes, 6)
        # Ensure rotation is orthonormalized
        rot_feat = self._get_orthonormalized_rotation(rot_feat)
        # Concatenate joint SE(3) transforms with other poses
        node_se3_pose = torch.cat([pos_feat, rot_feat], dim=-1)

        # Fill the SE(3) pose for non-joint nodes
        se3_pose[:, positions_node_idx, :3] = node_se3_pose[:, :, :3]
        se3_pose[:, rotations_node_idx, 3:] = node_se3_pose[:, :, 3:]


        # ----------------------------------------------------------------------------
        # Get remaining node features from config
        node_feature_map = self.config["node_feature_map"]
        # Collect all unique extra keys and their max lengths
        extra_keys = {}
        for v in node_feature_map.values():
            for key in v:
                if key not in ("position", "rotation", "type"):
                    val = v[key]
                    n = len(val) if isinstance(val, list) else 1
                    if key not in extra_keys or extra_keys[key] < n:
                        extra_keys[key] = n

        # Build ordered list of (key, subindex) for columns
        ordered_keys = []
        for key, n in extra_keys.items():
            for i in range(n):
                ordered_keys.append((key, i))

        # Create tensor for remaining features: (B, num_nodes, num_extra_columns)
        num_extra_columns = len(ordered_keys)
        remaining_feat = torch.zeros(
            (B * T, len(self.nodes), num_extra_columns),
            device=self.device,
            dtype=torch.float32,
        )

        # Fill the tensor
        for node_idx, node in enumerate(self.nodes):
            node_id = node["id"]
            if node_id not in node_feature_map:
                continue
            node_map = node_feature_map[node_id]
            for col_idx, (key, subidx) in enumerate(ordered_keys):
                if key in node_map:
                    val = node_map[key]
                    if isinstance(val, list):
                        if subidx < len(val):
                            idx = val[subidx]
                            remaining_feat[:, node_idx, col_idx] = feat_vec[:, idx]
                    elif subidx == 0:
                        remaining_feat[:, node_idx, col_idx] = feat_vec[:, val]

        # ----------------------------------------------------------------------------
        # Create one-hot encoding for node types
        one_hot_node_types = torch.functional.F.one_hot(
            torch.tensor(
                [self.unique_node_types[t] for t in self.node_types], device=self.device
            ),
            num_classes=len(self.unique_node_types),
        )
        one_hot_node_types = one_hot_node_types.unsqueeze(0).repeat(B * T, 1, 1)
        # ----------------------------------------------------------------------------
        # Combine all features into a single tensor
        base_feats = torch.cat([se3_pose, one_hot_node_types, remaining_feat], dim=-1)

        if extra_fn is not None:
            extra = extra_fn(
                self, feat_vec, se3_pose, one_hot_node_types, remaining_feat
            )
            # sanity check
            assert extra.shape[0] == B and extra.shape[1] == len(
                self.nodes
            ), f"extra_fn must return (B, N, E), got {extra.shape}"
        else:
            # no extra features
            extra = torch.zeros((B * T, len(self.nodes), 0), device=self.device)

        new_feat_vec = torch.cat([base_feats, extra], dim=-1)

        return new_feat_vec

    def _load_pk_chain(self) -> pk.SerialChain:
        """
        Load the kinematic chain from the configuration.
        """
        urdf_path = self.config.get("urdf_path", None)
        if not urdf_path:
            raise ValueError("URDF path is not specified in the configuration.")
        try:
            end_link = self.config.get("urdf_end_link_name", "panda_hand_tcp")
            chain = chain = pk.build_serial_chain_from_mjcf(
                open(urdf_path).read(), end_link_name=end_link
            ).to(dtype=torch.float32, device=self.device)
            return chain
        except Exception as e:
            raise ValueError(f"Failed to load kinematic chain from URDF: {e}")
        
    def _load_pk_chain2(self) -> pk.SerialChain:
        """
        Load the kinematic chain from the configuration.
        """
        urdf_path = "robomimic/algo/panda_urdf/panda_v2.urdf"
        if not urdf_path:
            raise ValueError("URDF path is not specified in the configuration.")
        try:
            end_link = "panda_hand_tcp"
            chain = pk.build_serial_chain_from_urdf(
                open(urdf_path).read(), end_link_name=end_link
            ).to(dtype=torch.float32, device=self.device)
            return chain
        except Exception as e:
            raise ValueError(f"Failed to load kinematic chain from URDF: {e}")
        
    def _compute_se3_from_qpos2(self, q_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute the SE(3) transformations from joint for all joints in the kinematic chain.
        Rotation is converted to 6D representation.
        """
        transformation: Dict = self.chain2.forward_kinematics(q_pos, end_only=False)
        se3_transforms = []
        for key in transformation:
            m = transformation[key].get_matrix()
            pos = m[:, :3, 3]
            rot = pk.matrix_to_rotation_6d(m[:, :3, :3])
            orthonormalized_rot = self._get_orthonormalized_rotation(rot)
            se3_transforms.append(torch.cat([pos, orthonormalized_rot], dim=1))
        se3_transforms = torch.stack(
            se3_transforms, dim=1
        )  # Shape: (B * T, num_joints, 9)
        # Only return the transforms for link 1 to 7 as they correspond to joint 0 to 6
        se3_transforms = se3_transforms
        return se3_transforms


    def _compute_se3_from_qpos(self, q_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute the SE(3) transformations from joint for all joints in the kinematic chain.
        Rotation is converted to 6D representation.
        """
        transformation: Dict = self.chain.forward_kinematics(q_pos, end_only=False)
        se3_transforms = []
        for key in transformation:
            m = transformation[key].get_matrix()
            pos = m[:, :3, 3]
            rot = pk.matrix_to_rotation_6d(m[:, :3, :3])
            orthonormalized_rot = self._get_orthonormalized_rotation(rot)
            se3_transforms.append(torch.cat([pos, orthonormalized_rot], dim=1))
        se3_transforms = torch.stack(
            se3_transforms, dim=1
        )  # Shape: (B * T, num_joints, 9)
        # Only return the transforms for link 1 to 7 as they correspond to joint 0 to 6
        se3_transforms = se3_transforms
        return se3_transforms

    def _get_orthonormalized_rotation(
        self, rot6d: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert 6D rotation to orthonormalized 3D rotation.
        """
        r1 = rot6d[..., :3]
        r2 = rot6d[..., 3:]
        epsilon = 1e-7

        # Normalize the first vector (u1)
        u1 = r1 / (torch.norm(r1, p=2, dim=-1, keepdim=True) + epsilon)

        # Project r2 onto u1: proj_r2_onto_u1 = dot(u1, r2) * u1
        dot_u1_r2 = torch.sum(u1 * r2, dim=-1, keepdim=True)

        # Subtract the projection from r2 to get a vector v2 orthogonal to u1
        v2 = r2 - dot_u1_r2 * u1

        # Normalize the second vector (u2)
        u2 = v2 / (torch.norm(v2, p=2, dim=-1, keepdim=True) + epsilon)

        # Concatenate the orthonormalized 3D vectors to get the new 6D rotation
        return torch.cat([u1, u2], dim=-1)

    def _compute_relative_se3(self, se3_pose):
        # se3_pose: (batch, num_links, 9)
        batch, num_links, _ = se3_pose.shape
        pos = se3_pose[..., :3]  # (batch, num_links, 3)
        rot6d = se3_pose[..., 3:]  # (batch, num_links, 6)
        rotmat = pk.rotation_6d_to_matrix(rot6d)
        T = torch.eye(4, device=se3_pose.device).repeat(batch, num_links, 1, 1)
        T[..., :3, :3] = rotmat
        T[..., :3, 3] = pos

        T_inv = torch.linalg.inv(T)
        T_rel = T_inv[:, :-1] @ T[:, 1:]  # (batch, num_links-1, 4, 4)
        rel_pos = T_rel[..., :3, 3]
        rel_rot6d = pk.matrix_to_rotation_6d(T_rel[..., :3, :3].reshape(-1, 3, 3)).view(
            batch, num_links - 1, 6
        )
        return torch.cat([rel_pos, rel_rot6d], dim=-1)  # (batch, num_links-1, 9)

    def visualize_graph(
        self,
        data: Data,
        batch_idx: int = 0,
        with_labels: bool = True,
        node_color: str = "skyblue",
        edge_color: str = "gray",
        figsize: tuple = (10, 8),
        use_3d: bool = True,
        use_node_positions: bool = True,
    ):
        """
        Visualize the graph structure for a single batch using networkx and matplotlib.

        Args:
            data: PyTorch Geometric Data object
            batch_idx: Index of the batch to visualize
            with_labels: Whether to show node labels
            node_color: Color for nodes
            edge_color: Color for edges
            figsize: Figure size
            use_3d: Whether to use 3D visualization (requires use_node_positions=True)
            use_node_positions: Whether to use the first 3 features as node positions
        """
        import numpy as np
        import torch
        import networkx as nx
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Get mask for nodes in the selected batch
        if hasattr(data, "batch"):
            node_mask = data.batch == batch_idx
            node_indices = node_mask.nonzero(as_tuple=True)[0]
        else:
            node_indices = torch.arange(data.num_nodes)

        # Get node features for this batch
        node_features = data.x[node_indices].cpu().numpy()

        # Map global node indices to local indices for this batch
        idx_map = {int(idx): i for i, idx in enumerate(node_indices.tolist())}

        # Filter edges: both source and target must be in this batch
        edge_index = data.edge_index.cpu().numpy()
        mask = np.isin(edge_index[0], node_indices.cpu().numpy()) & np.isin(
            edge_index[1], node_indices.cpu().numpy()
        )
        filtered_edges = edge_index[:, mask]

        # Remap node indices for visualization
        remapped_edges = np.array(
            [
                [idx_map[int(s)], idx_map[int(t)]]
                for s, t in zip(filtered_edges[0], filtered_edges[1])
            ]
        ).T

        # Build graph
        G = nx.DiGraph()
        for i, idx in enumerate(node_indices.tolist()):
            G.add_node(i, label=str(idx))
        for s, t in remapped_edges.T:
            G.add_edge(s, t)

        # Set up positions
        if use_node_positions and node_features.shape[1] >= 3:
            # Use first 3 features as positions (SE(3) poses are now at the beginning)
            positions_3d = node_features[:, :3]

            if use_3d:
                # 3D visualization
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection="3d")

                # Plot nodes
                xs, ys, zs = positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2]
                ax.scatter(xs, ys, zs, c=node_color, s=100, alpha=0.8)

                # Plot edges
                for s, t in remapped_edges.T:
                    if s < len(positions_3d) and t < len(positions_3d):
                        ax.plot(
                            [positions_3d[s, 0], positions_3d[t, 0]],
                            [positions_3d[s, 1], positions_3d[t, 1]],
                            [positions_3d[s, 2], positions_3d[t, 2]],
                            color=edge_color,
                            alpha=0.6,
                        )

                # Add labels
                if with_labels:
                    for i, (x, y, z) in enumerate(positions_3d):
                        ax.text(x, y, z, G.nodes[i].get("label", str(i)), fontsize=8)

                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position")
                ax.set_zlabel("Z Position")
                ax.set_title(f"3D Graph Structure (Batch {batch_idx})")

            else:
                # 2D visualization using first 2 position coordinates
                plt.figure(figsize=figsize)
                pos = {
                    i: (positions_3d[i, 0], positions_3d[i, 1])
                    for i in range(len(positions_3d))
                }

                nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=100)
                nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.6)

                if with_labels:
                    node_labels = {i: G.nodes[i].get("label", str(i)) for i in G.nodes}
                    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

                plt.xlabel("X Position")
                plt.ylabel("Y Position")
                plt.title(f"2D Graph Structure (Batch {batch_idx})")
                plt.axis("equal")
        else:
            # Fallback to spring layout if positions are not available or not requested
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(G)
            node_labels = {i: G.nodes[i].get("label", str(i)) for i in G.nodes}
            nx.draw_networkx_nodes(G, pos, node_color=node_color)
            nx.draw_networkx_edges(G, pos, edge_color=edge_color)
            if with_labels:
                nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
            plt.title(f"Graph Structure (Batch {batch_idx}) - Spring Layout")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def stack_cube_gripper_extra_features(
    converter,  # JsonTemporalGraphConverter instance
    feat_vec,  # raw (B*N, F) after view
    se3_pose,  # (B, N, 9)
    one_hot_node_types,  # (B, N, C)
    remaining_feat,  # (B, N, R)
    ):
    B, N, _ = se3_pose.shape
    device = feat_vec.device

    # 1) Which dims in feat_vec are the two TCP joint angles?
    eef_map = converter.config["node_feature_map"]["eef"]
    idxs = eef_map["joint_pos"]  # e.g. [7,8]
    gpos = feat_vec[:, idxs]  # shape (B, 2)

    # 2) Compute width + binary grasp flag
    grip_width = (gpos[:, 0] - gpos[:, 1]).abs().unsqueeze(-1)  # (B,1)
    is_grasp = (grip_width < 0.01).float()  # (B,1)

    # 3) Scatter into an (B, N, 2) tensor
    extra = torch.zeros((B, N, 2), device=device)
    node_ids = [n["id"] for n in converter.nodes]
    eef_idx = node_ids.index("eef")
    extra[:, eef_idx, 0] = grip_width.squeeze(-1)
    extra[:, eef_idx, 1] = is_grasp.squeeze(-1)

    return extra



