"""
Task-Specific Graph Configuration for FlowGAT.
Defines robot/environment-specific parameters for graph construction.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from typing import Optional

class BaseTaskGraphConfig:
    """Base class for task-specific graph configurations."""
    def __init__(self):
        self.robot_base_offset: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
        self.adjacency_matrix: np.ndarray = np.array([[]])
        self.node_order: List[str] = []
        self.node_type_map: Dict[str, str] = {} # Maps node_name -> general_type (e.g. 'joint', 'eef', 'object')
        self.node_type_to_id: Dict[str, int] = {} # Maps general_type -> integer_id
        self.num_joints: int = 0

        # Defines how to slice features from the raw 'object' observation tensor
        self.object_node_feature_slicing: Dict[str, Tuple[int, Optional[int]]] = {}
        self.end_effector_link_name: str = "right_hand" # Default, can be overridden by specific task

    def _validate_and_derive(self):
        if not self.node_order:
            raise ValueError("node_order must be defined in the task configuration.")
        if not self.node_type_map:
            raise ValueError("node_type_map must be defined.")
        if len(self.node_order) != len(self.adjacency_matrix):
            raise ValueError(
                f"Length of node_order ({len(self.node_order)}) must match "
                f"adjacency_matrix dimension ({len(self.adjacency_matrix)})."
            )
        if not all(node_name in self.node_type_map for node_name in self.node_order):
            raise ValueError("All nodes in node_order must have a type defined in node_type_map.")

        # Validate object node slicing
        for node_name, node_type in self.node_type_map.items():
            if node_type == 'object':
                if node_name not in self.object_node_feature_slicing:
                    raise ValueError(
                        f"Object node '{node_name}' is defined but missing "
                        f"from object_node_feature_slicing in task config."
                    )
                # Further validation (e.g. non-overlapping slices) could be added if needed

        self.num_joints = sum(1 for node_name in self.node_order if self.node_type_map.get(node_name) == 'joint')
        unique_types = sorted(list(set(self.node_type_map.values())))
        self.node_type_to_id = {name: i for i, name in enumerate(unique_types)}


class PandaSquareNutTaskConfig(BaseTaskGraphConfig):
    def __init__(self):
        super().__init__()
        # --- Robot Base Offset ---
        # Adjust this based on your specific Square Nut task setup in robosuite/robomimic
        # Example: if robot base is at world origin and table is centered.
        # self.robot_base_offset = torch.tensor([0.0, 0.0, 0.913]) # Common for Panda on table
        # For robomimic's default Square task (Panda)
        # table_offset = [0.1, 0, 0.8] (approx center of table relative to world)
        # panda_base_offset_from_table_center = [-0.26, -0.3, 0.0] (approx)
        # So, robot_base_offset approx = [-0.16, -0.3, 0.8]
        # This needs to be precise for your specific environment setup.
        # For now, using an example offset from one of the earlier configs.
        table_length = 0.8 # This was from a "table task" comment. Square nut might be different.
        self.robot_base_offset = torch.tensor(
             [-0.16 - table_length / 2, 0, 1.0] # Placeholder - VERIFY FOR SQUARE NUT
        )
        self.end_effector_link_name = "panda_hand" # Common for robosuite Panda, or "panda_link7" for others

        # --- Node Definition ---
        self.node_order = [
            'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', # 7 Panda joints
            'eef',
            'square_nut_obj', # The square nut object
            # Add other objects if present, e.g. 'peg_obj' if it's a separate observed entity
        ]

        self.node_type_map = {name: 'joint' for name in self.node_order if 'joint' in name}
        self.node_type_map['eef'] = 'eef'
        self.node_type_map['square_nut_obj'] = 'object'

        # --- Object Feature Slicing ---
        # Robomimic's Square task typically has one object (the nut) in 'object' obs.
        # It has 14 features:
        #   - object_pos (3)
        #   - object_quat (4)
        #   - object_to_eef_pos (3) (relative position)
        #   - object_to_eef_quat (4) (relative orientation - not always used/meaningful)
        # For 'square_nut_obj', we take all 14 features from the 'object' observation.
        self.object_node_feature_slicing = {
            'square_nut_obj': (0, 14), # Takes features from index 0 up to (but not including) 14
            # If you had another object, e.g., 'peg_obj', and it was the next 14 features:
            # 'peg_obj': (14, 28),
        }

        # --- Adjacency Matrix ---
        # Order: J0-J6, EEF, square_nut_obj (9x9)
        # Adjust if more objects are added to node_order
        _adj = [
            # J0 J1 J2 J3 J4 J5 J6 EEF NUT
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # J0
            [1, 0, 1, 0, 0, 0, 0, 0, 0],  # J1
            [0, 1, 0, 1, 0, 0, 0, 0, 0],  # J2
            [0, 0, 1, 0, 1, 0, 0, 0, 0],  # J3
            [0, 0, 0, 1, 0, 1, 0, 0, 0],  # J4
            [0, 0, 0, 0, 1, 0, 1, 0, 0],  # J5
            [0, 0, 0, 0, 0, 1, 0, 1, 0],  # J6
            [0, 0, 0, 0, 0, 0, 1, 0, 1],  # EEF (connects to J6 and square_nut_obj)
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # square_nut_obj (connects to EEF)
        ]
        self.adjacency_matrix = np.array(_adj, dtype=bool)
        np.fill_diagonal(self.adjacency_matrix, True) # Add self-loops

        self._validate_and_derive()


TASK_CONFIGS = {
    "PandaSquareNutTask": PandaSquareNutTaskConfig,
}

def get_task_graph_config(name: str) -> BaseTaskGraphConfig:
    if name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task graph configuration: {name}. Available: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[name]()