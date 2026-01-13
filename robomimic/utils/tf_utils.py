import torch
import numpy as np
import warnings

# Suppress warnings from scipy/numpy matrix logarithm operations
warnings.filterwarnings('ignore', message='logm result may be inaccurate')
warnings.filterwarnings('ignore', category=np.ComplexWarning)

try:
    from geomstats.geometry.special_euclidean import SpecialEuclidean
    from geomstats.learning.frechet_mean import FrechetMean
except ImportError:
    raise ImportError(
        "geomstats is required for compute_transform_stats. "
        "Install it with: pip install geomstats"
    )

###########################
# --- PRIVATE METHODS --- #
###########################
# rotation conversions
def _quat_to_rot_mat(quaternions, w_first=True):
    """
    Highly optimized batch conversion of quaternions to rotation matrices.
    Uses vectorized operations for maximum performance on large batches.
    
    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4) or (..., 4) representing quaternions (qw, qx, qy, qz).
        
    Returns:
        torch.Tensor: A tensor of shape (N, 3, 3) or (..., 3, 3) representing rotation matrices.
    """
    # Ensure input is at least 2D for batch processing
    original_shape = quaternions.shape
    if quaternions.dim() == 1:
        quaternions = quaternions.unsqueeze(0)
        was_1d = True
    else:
        was_1d = False
    
    # Reshape to (batch_size, 4) for efficient processing
    batch_size = quaternions.shape[0]
    quaternions = quaternions.view(-1, 4)
    
    # Normalize quaternions for numerical stability
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    if not w_first:
        # If w is not first, rearrange from (qx, qy, qz, qw) to (qw, qx, qy, qz)
        quaternions = torch.cat([quaternions[:, 3:4], quaternions[:, :3]], dim=1).contiguous()

    # Extract components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Precompute all required terms using broadcasting
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    # Build rotation matrices using vectorized operations
    batch_rot_matrices = torch.stack([
        torch.stack([w2 + x2 - y2 - z2, 2*(xy - wz), 2*(xz + wy)], dim=1),
        torch.stack([2*(xy + wz), w2 - x2 + y2 - z2, 2*(yz - wx)], dim=1),
        torch.stack([2*(xz - wy), 2*(yz + wx), w2 - x2 - y2 + z2], dim=1)
    ], dim=1)
    
    # Reshape back to original batch dimensions
    if was_1d:
        return batch_rot_matrices.squeeze(0)
    else:
        target_shape = original_shape[:-1] + (3, 3)
        return batch_rot_matrices.view(target_shape)

def _rot_mat_to_quat(rotation_matrices, w_first=True):
    """
    Convert a batch of rotation matrices to quaternions using Shepperd's method for numerical stability.
    
    Args:
        rotation_matrices (torch.Tensor): A tensor of shape (..., 3, 3) representing rotation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing quaternions (qw, qx, qy, qz).
    """
    batch_shape = rotation_matrices.shape[:-2]
    device = rotation_matrices.device
    dtype = rotation_matrices.dtype
    
    # Flatten batch dimensions for easier processing
    R = rotation_matrices.view(-1, 3, 3)
    batch_size = R.shape[0]
    
    # Initialize output quaternions
    q = torch.zeros((batch_size, 4), device=device, dtype=dtype)
    
    # Extract diagonal elements
    R00, R11, R22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    
    # Compute trace
    trace = R00 + R11 + R22
    
    # Use Shepperd's method for numerical stability
    # Case 1: trace > 0 (most common case)
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        q[mask1, 0] = 0.25 * s  # qw
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # qx
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # qy
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # qz
    
    # Case 2: R00 > R11 and R00 > R22
    mask2 = (~mask1) & (R00 > R11) & (R00 > R22)
    if mask2.any():
        s = torch.sqrt(1.0 + R00[mask2] - R11[mask2] - R22[mask2]) * 2  # s = 4 * qx
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # qw
        q[mask2, 1] = 0.25 * s  # qx
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # qy
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # qz
    
    # Case 3: R11 > R22
    mask3 = (~mask1) & (~mask2) & (R11 > R22)
    if mask3.any():
        s = torch.sqrt(1.0 + R11[mask3] - R00[mask3] - R22[mask3]) * 2  # s = 4 * qy
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # qw
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # qx
        q[mask3, 2] = 0.25 * s  # qy
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # qz
    
    # Case 4: else (R22 is largest)
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R22[mask4] - R00[mask4] - R11[mask4]) * 2  # s = 4 * qz
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # qw
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # qx
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # qy
        q[mask4, 3] = 0.25 * s  # qz
    
    # Ensure quaternion is normalized and positive w (canonical form)
    q = q / torch.norm(q, dim=-1, keepdim=True)
    
    # Make quaternion canonical (positive w)
    negative_w = q[:, 0] < 0
    q[negative_w] = -q[negative_w]
    
    # Reshape back to original batch shape
    quat = q.view(*batch_shape, 4)

    if not w_first:
        # If w was not first, rearrange to (qx, qy, qz, qw)
        quat = quat[..., 1:4].contiguous()
    
    return quat

def _ortho6d_to_quat(ortho6d, w_first=True):
    """
    Convert a batch of 6-DOF rotation representations (x and y vectors) back to quaternions.
    
    Args:
        ortho6d (torch.Tensor): A tensor of shape (..., 6) representing 6-DOF rotations (x1, x2, x3, y1, y2, y3).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing quaternions (qw, qx, qy, qz).
    """
    # Extract x and y vectors
    x_vectors = ortho6d[..., :3]  # First three elements
    y_vectors = ortho6d[..., 3:]   # Last three elements

    # Compute the z vector using the cross product
    z_vectors = torch.cross(x_vectors, y_vectors, dim=-1)

    # Recompute the y vector using the cross product to ensure orthogonality
    y_vectors = torch.cross(z_vectors, x_vectors, dim=-1)
    
    # Normalize the vectors
    x_norm = torch.norm(x_vectors, dim=-1, keepdim=True)
    y_norm = torch.norm(y_vectors, dim=-1, keepdim=True)
    z_norm = torch.norm(z_vectors, dim=-1, keepdim=True)

    # Handle potential division by zero
    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)
    z_norm = torch.clamp(z_norm, min=1e-8)

    x_unit = x_vectors / x_norm
    y_unit = y_vectors / y_norm
    z_unit = z_vectors / z_norm

    # Construct rotation matrix from orthonormal vectors
    batch_shape = ortho6d.shape[:-1]
    rotation_matrices = torch.zeros((*batch_shape, 3, 3), dtype=ortho6d.dtype, device=ortho6d.device)
    
    rotation_matrices[..., 0, :] = x_unit  # First column
    rotation_matrices[..., 1, :] = y_unit  # Second column
    rotation_matrices[..., 2, :] = z_unit  # Third column

    # Convert rotation matrices to quaternions using the batch function
    return _rot_mat_to_quat(rotation_matrices, w_first=w_first)

def _quat_to_ortho6d(quaternions, w_first=True):
    """
    Convert a batch of quaternions to 6-DOF rotation representations (x and y vectors of the rotation).
    
    Args:
        quaternions (torch.Tensor): A tensor of shape (..., 4) representing quaternions (qw, qx, qy, qz).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 6) representing the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
    """
    # Get the rotation matrices for the batch of quaternions
    rotation_matrices = _quat_to_rot_mat(quaternions, w_first=w_first)

    # Extract the x and y vectors from the rotation matrices
    x_vector = rotation_matrices[..., 0, :]  # First column
    y_vector = rotation_matrices[..., 1, :]  # Second column

    # Concatenate the x and y vectors to form the 6-DOF representation
    ortho6d = torch.cat([x_vector, y_vector], dim=-1)   

    return ortho6d

def _axis_angle_to_quat(axis_angle, w_first=True):
    """
    Convert axis-angle representation to quaternion.
    
    Args:
        axis_angle (torch.Tensor): A tensor of shape (..., 3) representing axis-angle rotation (axis * angle).
        w_first (bool): Whether to return quaternion in (qw, qx, qy, qz) format.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing quaternions.
    """
    # Compute rotation angle from axis-angle vector
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    
    # Compute rotation axis (handling zero angle case)
    # When angle is near zero, use a default axis and the result will have sin(angle/2) ≈ 0 anyway
    safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
    axis = axis_angle / safe_angle
    
    # Convert axis-angle to quaternion
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)
    
    quaternion = torch.cat([
        cos_half,           # qw
        axis * sin_half     # qx, qy, qz
    ], dim=-1)
    
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    
    if not w_first:
        # Rearrange to (qx, qy, qz, qw)
        quaternion = torch.cat([quaternion[..., 1:], quaternion[..., 0:1]], dim=-1)
    
    return quaternion

def _rot_mat_to_ortho6d(rotation_matrices, w_first=True):
    """
    Convert a batch of rotation matrices to 6-DOF rotation representations (x and y vectors of the rotation).
    
    Args:
        rotation_matrices (torch.Tensor): A tensor of shape (..., 3, 3) representing rotation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 6) representing the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
    """
    # Extract the x and y vectors from the rotation matrices
    x_vector = rotation_matrices[..., 0, :]  # First column
    y_vector = rotation_matrices[..., 1, :]  # Second column

    # Concatenate the x and y vectors to form the 6-DOF representation
    ortho6d = torch.cat([x_vector, y_vector], dim=-1)   

    return ortho6d

def _ortho6d_to_rot_mat(ortho6d, w_first=True):
    """
    Convert a batch of 6-DOF rotation representations (x and y vectors) to rotation matrices.
    
    Args:
        ortho6d (torch.Tensor): A tensor of shape (..., 6) representing 6-DOF rotations (x1, x2, x3, y1, y2, y3).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing rotation matrices.
    """
    # Extract x and y vectors
    x_vectors = ortho6d[..., :3]  # First three elements
    y_vectors = ortho6d[..., 3:]   # Last three elements

    # Compute the z vector using the cross product
    z_vectors = torch.cross(x_vectors, y_vectors, dim=-1)

    # Recompute the y vector using the cross product to ensure orthogonality
    y_vectors = torch.cross(z_vectors, x_vectors, dim=-1)
    
    # Normalize the vectors
    x_norm = torch.norm(x_vectors, dim=-1, keepdim=True)
    y_norm = torch.norm(y_vectors, dim=-1, keepdim=True)
    z_norm = torch.norm(z_vectors, dim=-1, keepdim=True)

    # Handle potential division by zero
    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)
    z_norm = torch.clamp(z_norm, min=1e-8)

    x_unit = x_vectors / x_norm
    y_unit = y_vectors / y_norm
    z_unit = z_vectors / z_norm

    # Construct rotation matrices from orthonormal vectors
    batch_shape = ortho6d.shape[:-1]
    rotation_matrices = torch.zeros((*batch_shape, 3, 3), dtype=ortho6d.dtype, device=ortho6d.device)
    
    rotation_matrices[..., 0, :] = x_unit  # First column
    rotation_matrices[..., 1, :] = y_unit  # Second column
    rotation_matrices[..., 2, :] = z_unit  # Third column

    return rotation_matrices

def _axis_angle_to_quat(axis_angle, w_first=True):
    """
    Convert axis-angle representation to quaternion.
    
    Args:
        axis_angle (torch.Tensor): A tensor of shape (..., 3) representing axis-angle rotation (axis * angle).
        w_first (bool): Whether to return quaternion in (qw, qx, qy, qz) format.
    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing quaternions.
    """
    # Compute rotation angle from axis-angle vector
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    
    # Compute rotation axis (handling zero angle case)
    # When angle is near zero, use a default axis and the result will have sin(angle/2) ≈ 0 anyway
    safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
    axis = axis_angle / safe_angle
    
    # Convert axis-angle to quaternion
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)
    
    quaternion = torch.cat([
        cos_half,           # qw
        axis * sin_half     # qx, qy, qz
    ], dim=-1)
    
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    
    if not w_first:
        # Rearrange to (qx, qy, qz, qw)
        quaternion = torch.cat([quaternion[..., 1:], quaternion[..., 0:1]], dim=-1)
    
    return quaternion

# transformation conversions
def _pose_quat_to_T(pose_quat, w_first=True):
    """
    Convert a batch of position and quaternion tensors to transformation matrices.
    Args:
        pose_quat (torch.Tensor): A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                                     and the last 4 elements are the quaternion (qw, qx, qy, qz).
    Returns:
        torch.Tensor: A tensor of shape (..., 4, 4) representing transformation matrices.
    """
    position = pose_quat[..., :3]  # First three elements are position
    quaternion = pose_quat[..., 3:]  # Last four elements are quaternion

    # Convert quaternion to rotation matrix
    rotation_matrix = _quat_to_rot_mat(quaternion, w_first=w_first)

    # Create transformation matrices
    batch_shape = position.shape[:-1]
    T = torch.zeros((*batch_shape, 4, 4), dtype=position.dtype, device=position.device)
    
    T[..., :3, :3] = rotation_matrix
    T[..., :3, 3] = position
    T[..., 3, 3] = 1.0  # Homogeneous coordinate
    
    return T

def _T_to_pose_quat(T, w_first=True):
    """
    Convert a batch of transformation matrices to position and quaternion tensors.
    
    Args:
        T (torch.Tensor): A tensor of shape (..., 4, 4) representing transformation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                      and the last 4 elements are the quaternion (qw, qx, qy, qz).
    """
    position = T[..., :3, 3]  # Extract position (x, y, z)
    
    # Extract rotation matrix from transformation matrix
    rotation_matrix = T[..., :3, :3]
    
    # Convert rotation matrix to quaternion
    quaternion = _rot_mat_to_quat(rotation_matrix, w_first=w_first)
    
    # Concatenate position and quaternion
    pose_quat = torch.cat([position, quaternion], dim=-1)
    
    return pose_quat

def _T_to_pose_ortho6d(T, w_first=True):
    """
    Convert a batch of transformation matrices to position and 6-DOF rotation representation.
    
    Args:
        T (torch.Tensor): A tensor of shape (..., 4, 4) representing transformation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 9) where the first 3 elements are position (x, y, z)
                      and the last 6 elements are the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
    """
    position = T[..., :3, 3]  # Extract position (x, y, z)
    
    # Extract rotation matrix from transformation matrix
    rotation_matrix = T[..., :3, :3]
    
    # Convert rotation matrix to 6-DOF rotation representation
    ortho6d = _rot_mat_to_ortho6d(rotation_matrix, w_first=w_first)
    
    # Concatenate position and 6-DOF rotation
    pose_ortho6d = torch.cat([position, ortho6d], dim=-1)
    
    return pose_ortho6d

def _pose_ortho6d_to_T(pose_ortho6d, w_first=True):
    """
    Convert a batch of position and 6-DOF rotation tensors to transformation matrices.
    
    Args:
        pose_ortho6d (torch.Tensor): A tensor of shape (..., 9) where the first 3 elements are position (x, y, z)
                                          and the last 6 elements are the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 4, 4) representing transformation matrices.
    """
    position = pose_ortho6d[..., :3]  # First three elements are position
    ortho6d = pose_ortho6d[..., 3:]  # Last six elements are 6-DOF rotation

    # Convert 6-DOF rotation representation to rotation matrix
    rotation_matrix = _ortho6d_to_rot_mat(ortho6d, w_first=w_first)

    # Create transformation matrices
    batch_shape = position.shape[:-1]
    T = torch.zeros((*batch_shape, 4, 4), dtype=position.dtype, device=position.device)
    
    T[..., :3, :3] = rotation_matrix
    T[..., :3, 3] = position
    T[..., 3, 3] = 1.0  # Homogeneous coordinate
    
    return T

def _pose_quat_to_pose_ortho6d(pose_quat, w_first=True):
    """
    Convert a batch of position and quaternion tensors to a 6-DOF rotation representation.
    
    Args:
        pose_quat (torch.Tensor): A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                                     and the last 4 elements are the quaternion (qw, qx, qy, qz).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 9) where the first 3 elements are position (x, y, z)
                      and the last 6 elements are the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
    """
    position = pose_quat[..., :3]  # First three elements are position
    quaternion = pose_quat[..., 3:]  # Last four elements are quaternion

    # Convert quaternion to 6-DOF rotation representation
    ortho6d = _quat_to_ortho6d(quaternion, w_first=w_first)

    # Concatenate position and 6-DOF rotation
    pose_ortho6d = torch.cat([position, ortho6d], dim=-1)

    return pose_ortho6d

def _pose_ortho6d_to_pose_quat(pose_ortho6d, w_first=True):
    """
    Convert a batch of position and 6-DOF rotation tensors back to position and quaternion representation.
    
    Args:
        pose_ortho6d (torch.Tensor): A tensor of shape (..., 9) where the first 3 elements are position (x, y, z)
                                          and the last 6 elements are the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                      and the last 4 elements are the quaternion (qw, qx, qy, qz).
    """
    position = pose_ortho6d[..., :3]  # First three elements are position
    ortho6d = pose_ortho6d[..., 3:]  # Last six elements are 6-DOF rotation

    # Convert 6-DOF rotation representation to quaternion
    quaternion = _ortho6d_to_quat(ortho6d, w_first=w_first)

    # Concatenate position and quaternion
    pose_quat = torch.cat([position, quaternion], dim=-1)

    return pose_quat

def _pose_axis_angle_to_pose_quat(pose_axis_angle, w_first=True):
    """
    Convert a batch of position and axis-angle tensors to position and quaternion representation.
    
    Args:
        pose_axis_angle (torch.Tensor): A tensor of shape (..., 6) where the first 3 elements are position (xyz)
                                         and the last 3 elements are the axis-angle rotation (axis * angle).
    Returns:
        torch.Tensor: A tensor of shape (..., 7) representing the pose in position and quaternion representation.
    """
    position = pose_axis_angle[..., :3]  # First three elements are position
    axis_angle = pose_axis_angle[..., 3:]  # Last three elements are axis-angle

    # Convert axis-angle to quaternion
    quaternion = _axis_angle_to_quat(axis_angle, w_first=w_first)

    # Concatenate position and quaternion
    pose_quat = torch.cat([position, quaternion], dim=-1)

    return pose_quat

def _pose_quat_to_pose_axis_angle(pose_quat, w_first=True):
    """
    Convert a batch of position and quaternion tensors to position and axis-angle representation.
    
    Args:
        pose_quat (torch.Tensor): A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                                     and the last 4 elements are the quaternion (qw, qx, qy, qz).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 6) where the first 3 elements are position (x, y, z)
                      and the last 3 elements are the axis-angle rotation (axis * angle).
    """
    position = pose_quat[..., :3]  # First three elements are position
    quaternion = pose_quat[..., 3:]  # Last four elements are quaternion

    # Convert quaternion to rotation matrix
    rotation_matrix = _quat_to_rot_mat(quaternion, w_first=w_first)

    # Convert rotation matrix to axis-angle
    batch_shape = rotation_matrix.shape[:-2]
    R = rotation_matrix.view(-1, 3, 3)
    batch_size = R.shape[0]

    angle = torch.acos(torch.clamp((torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2, -1.0, 1.0))
    sin_angle = torch.sin(angle)

    axis = torch.zeros((batch_size, 3), dtype=rotation_matrix.dtype, device=rotation_matrix.device)

    mask = sin_angle > 1e-6
    if mask.any():
        axis[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / (2 * sin_angle[mask])
        axis[mask, 1] = (R[mask, 0, 2] - R[mask, 2, 0]) / (2 * sin_angle[mask])
        axis[mask, 2] = (R[mask, 1, 0] - R[mask, 0, 1]) / (2 * sin_angle[mask])

    axis_angle = axis * angle.unsqueeze(-1)

    axis_angle = axis_angle.view(*batch_shape, 3)

    # Concatenate position and axis-angle
    pose_axis_angle = torch.cat([position, axis_angle], dim=-1)
    return pose_axis_angle

def _convert_pose_any_to_pose_quat(pose, w_first=True):
    """
    Convert a batch of poses from any supported representation to position and quaternion representation.
    
    Args:
        pose (torch.Tensor): A tensor of shape (..., D) where D is 7 for 'quat' and 9 for 'ortho6d'.
        w_first (bool): Whether the quaternion representation uses (qw, qx, qy, qz) format.

    Returns:
        torch.Tensor: A tensor of shape (..., 9) representing the pose in position and 6-DOF rotation representation.
    """
    representation_from = 'quat' if pose.shape[-1] == 7 \
        else 'ortho6d' if pose.shape[-1] == 9 \
            else 'T_mat' if (pose.shape[-1] == 4) and (pose.shape[-2] == 4) \
                else 'axis_angle' if pose.shape[-1] == 6 \
                    else None

    if representation_from == 'ortho6d':
        return _pose_ortho6d_to_pose_quat(pose, w_first=w_first), representation_from
    elif representation_from == 'T_mat':
        return _T_to_pose_quat(pose, w_first=w_first), representation_from
    elif representation_from == 'axis_angle':
        return _pose_axis_angle_to_pose_quat(pose, w_first=w_first), representation_from
    elif representation_from == 'quat':
        if not w_first:
            # Rearrange to (qw, qx, qy, qz)
            position = pose[..., :3]
            quaternion = pose[..., 3:]
            quaternion = torch.cat([quaternion[..., 3:4], quaternion[..., :3]], dim=-1)
            pose = torch.cat([position, quaternion], dim=-1)
        return pose, representation_from
    else:
        raise ValueError(f"Unsupported representation_from. Supported: 'quat', 'T_matrix', 'ortho6d', 'axis_angle'.")

def _convert_pose_quat_to_pose_any(pose_quat, representation_to, w_first=True):
    """
    Convert a batch of poses from position and quaternion to any supported representation.
    
    Args:
        pose_quat (torch.Tensor): A tensor of shape (..., 9) representing the pose in position and 6-DOF rotation representation.
        representation_to (str): The target representation: 'quat', 'ortho6d', 'axis_angle', or 'T_mat'.
        w_first (bool): Whether the quaternion representation uses (qw, qx, qy, qz) format.

    Returns:
        torch.Tensor: A tensor of shape (..., D) where D is 7 for 'quat', 9 for 'ortho6d', 6 for 'axis_angle', or (4,4) for 'T_mat'.
    """
    if representation_to == 'ortho6d':
        return _pose_quat_to_pose_ortho6d(pose_quat, w_first=w_first)
    elif representation_to == 'T_mat':
        return _pose_quat_to_T(pose_quat, w_first=w_first)
    elif representation_to == 'axis_angle':
        return _pose_quat_to_pose_axis_angle(pose_quat, w_first=w_first)
    elif representation_to == 'quat':
        if not w_first:
            # Rearrange to (qx, qy, qz, qw)
            position = pose_quat[..., :3]
            quaternion = pose_quat[..., 3:]
            quaternion = torch.cat([quaternion[..., 1:], quaternion[..., 0:1]], dim=-1)
            pose_quat = torch.cat([position, quaternion], dim=-1)
        return pose_quat
    else:
        raise ValueError(f"Unsupported representation_to. Supported: 'quat', 'T_matrix', 'ortho6d'.")

def _pose_axis_angle_to_T(pose_axis_angle, w_first=True):
    """
    Convert a batch of position and axis-angle tensors to transformation matrices.
    Handles NaN entries by replacing them with zeros and then replacing those zeros with NaNs again after the transformation
    Args:
        pose_axis_angle (torch.Tensor): A tensor of shape (..., 6) where the first 3 elements are position (xyz)
                                         and the last 3 elements are the axis-angle rotation (axis * angle).
    Returns:
        torch.Tensor: A tensor of shape (..., 4, 4) representing transformation matrices.
    """
    # Extract position and axis-angle components
    position = pose_axis_angle[..., :3]  # First three elements are position
    axis_angle = pose_axis_angle[..., 3:]  # Last three elements are axis-angle
    
    # Track NaN positions for later restoration
    nan_mask = torch.isnan(pose_axis_angle)
    has_nans = nan_mask.any()
    
    # Replace NaNs with zeros for computation
    if has_nans:
        position = torch.where(torch.isnan(position), torch.zeros_like(position), position)
        axis_angle = torch.where(torch.isnan(axis_angle), torch.zeros_like(axis_angle), axis_angle)
    
    # Convert axis-angle to quaternion
    quaternion = _axis_angle_to_quat(axis_angle, w_first=True)
    
    # Convert quaternion to rotation matrix
    rotation_matrix = _quat_to_rot_mat(quaternion, w_first=True)
    
    # Create transformation matrices
    batch_shape = position.shape[:-1]
    T = torch.zeros((*batch_shape, 4, 4), dtype=position.dtype, device=position.device)
    
    T[..., :3, :3] = rotation_matrix
    T[..., :3, 3] = position
    T[..., 3, 3] = 1.0  # Homogeneous coordinate
    
    # Restore NaNs in the output
    if has_nans:
        # If any component of position had NaN, make entire position column NaN
        pos_nan_mask = nan_mask[..., :3].any(dim=-1, keepdim=True)
        T[..., :3, 3] = torch.where(
            pos_nan_mask.expand_as(T[..., :3, 3]),
            torch.full_like(T[..., :3, 3], float('nan')),
            T[..., :3, 3]
        )
        
        # If any component of axis-angle had NaN, make entire rotation matrix NaN
        rot_nan_mask = nan_mask[..., 3:].any(dim=-1, keepdim=True).unsqueeze(-1)
        T[..., :3, :3] = torch.where(
            rot_nan_mask.expand_as(T[..., :3, :3]),
            torch.full_like(T[..., :3, :3], float('nan')),
            T[..., :3, :3]
        )
    
    return T

# quaternion math ops
def _quaternion_multiply(q1, q2):
        """
        Multiply two quaternions together (q1 * q2)
        
        Args:
            q1: First quaternion of shape [..., 4] in (qw, qx, qy, qz) format
            q2: Second quaternion of shape [..., 4] in (qw, qx, qy, qz) format
            
        Returns:
            Quaternion product of shape [..., 4]
        """
        w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
        w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        
        return torch.cat([w, x, y, z], dim=-1)

def _quaternion_difference(q1, q2):
    """
    Calculate the quaternion difference q_diff = q2 * q1^(-1)
    
    Args:
        q1: First quaternion of shape [..., 4] in (qw, qx, qy, qz) format
        q2: Second quaternion of shape [..., 4] in (qw, qx, qy, qz) format
        
    Returns:
        Quaternion difference of shape [..., 4]
    """
    # For unit quaternions, inverse is just the conjugate
    q1_inv = torch.cat([q1[..., 0:1], -q1[..., 1:4]], dim=-1)
    
    # Quaternion multiplication q2 * q1_inv
    w1, x1, y1, z1 = q1_inv[..., 0:1], q1_inv[..., 1:2], q1_inv[..., 2:3], q1_inv[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    q_diff = torch.cat([w, x, y, z], dim=-1)
    
    # Normalize to fix numerical errors
    q_diff = q_diff / torch.norm(q_diff, dim=-1, keepdim=True)
    
    return q_diff

def _quaternion_mean(q, dim=-1):
    """
    Computes the means of a batch of quaternions using Markley's method
    Args:
        q: Torch.tensor with shape [...,N,...,4], quaternions (w,x,y,z)
        dim: int, the dimension along which to take the mean. If -1 (default), takes the average across all dimensions except the last (quaternion components)
    Returns:
        q_mean: Torch.tensor with shape [...,1,...,4] or [4], mean quaternions (w,x,y,z) accross dim 
        V: Torch.tensor with shape [...,1] or scalar, dispersion metric V = 1 - lambda1 (largest eigenvalue)
    """
    original_shape = q.shape
    
    if dim == -1:
        # Average across all quaternions (flatten all dims except last)
        q = q.reshape(-1, 4)
        N = q.shape[0]
        batch_size = 1
        batch_shape = ()
    else:
        # Move the averaging dimension to the end if it's not already there
        if dim != q.dim() - 2:
            q = q.transpose(dim, -2)
        
        # Get the shape for processing
        batch_shape = original_shape[:dim] + original_shape[dim+1:-1]
        N = original_shape[dim]  # Number of quaternions to average
        
        # Reshape to [..., N, 4] for easier processing
        q = q.reshape(-1, N, 4)
        batch_size = q.shape[0]
    
    # Normalize quaternions
    q = q / torch.norm(q, dim=-1, keepdim=True)
    
    # Markley's method: construct the 4x4 matrix M and find its dominant eigenvector
    q_mean = torch.zeros((batch_size, 1, 4), dtype=q.dtype, device=q.device)
    V = torch.zeros((batch_size, 1), dtype=q.dtype, device=q.device)
    
    for i in range(batch_size):
        # Build the M matrix as sum of outer products
        M = torch.zeros((4, 4), dtype=q.dtype, device=q.device)
        for j in range(N):
            qj = q[i, j, :]  # Shape [4]
            M += torch.outer(qj, qj)
        
        M = M / N  # Average
        
        # Find the eigenvector corresponding to the largest eigenvalue
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        
        # The eigenvector with the largest eigenvalue is the mean quaternion
        q_mean[i, 0, :] = eigenvectors[:, -1]
        
        # Calculate V = 1 - lambda1 (dispersion metric)
        lambda1 = eigenvalues[-1]
        V[i, 0] = 1.0 - lambda1
    
    # Ensure positive w component (canonical form)
    negative_w = q_mean[:, 0, 0] < 0
    q_mean[negative_w, 0, :] = -q_mean[negative_w, 0, :]
    
    if dim == -1:
        # Return scalar results for averaging across all dims
        return q_mean.squeeze(0).squeeze(0), V.squeeze(0).squeeze(0)
    else:
        # Reshape back to original batch dimensions
        target_shape = batch_shape + (1, 4)
        q_mean = q_mean.reshape(target_shape)
        V = V.reshape(batch_shape + (1,))
        
        # If we transposed earlier, transpose back
        if dim != original_shape.index(original_shape[dim]):
            q_mean = q_mean.transpose(dim, -2)
        
        return q_mean, V

def _quaternion_tan_space_covar(q, q_mean, dim=-1):
    """
    Computes the 3x3 tangent space covariance of a set of quaternions
    Args:
        q: Torch.tensor [...,N,...,4], quaternions (w,x,y,z)
        q_mean: Torch.tensor [...,1,...,4] or [4], the mean quaternion (w,x,y,z)
        dim: int, the dimension of q along which to measure the covariance, if dim is -1 and q_mean is [4] then measures
                the covariance accross all dims, otherwise measures the covariance accross the desired dim
    Returns:
        tan_cov: Torch.tensor [...,1,...3,3], the tangent space covariance matrix
    """
    original_shape = q.shape
    is_scalar_mean = q_mean.dim() == 1  # Check if q_mean is shape [4]
    
    if dim == -1:
        # Average across all quaternions (flatten all dims except last)
        q = q.reshape(-1, 4)
        N = q.shape[0]
        batch_size = 1
        batch_shape = ()
        
        # Ensure q_mean has the right shape for broadcasting
        if is_scalar_mean:
            q_mean = q_mean.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        else:
            q_mean = q_mean.reshape(1, 1, 4)
    else:
        # Move the averaging dimension to the end if it's not already there
        if dim != q.dim() - 2:
            q = q.transpose(dim, -2)
        
        # Get the shape for processing
        batch_shape = original_shape[:dim] + original_shape[dim+1:-1]
        N = original_shape[dim]  # Number of quaternions to average
        
        # Reshape to [..., N, 4] for easier processing
        q = q.reshape(-1, N, 4)
        batch_size = q.shape[0]
        
        # Reshape q_mean to match batch processing
        if not is_scalar_mean:
            q_mean = q_mean.reshape(batch_size, 1, 4)
    
    # Normalize quaternions
    q = q / torch.norm(q, dim=-1, keepdim=True)
    q_mean = q_mean / torch.norm(q_mean, dim=-1, keepdim=True)
    
    # Compute tangent space vectors for each quaternion
    # For each q_i, compute the quaternion difference q_diff = q_mean^(-1) * q_i
    q_mean_inv = torch.cat([q_mean[..., 0:1], -q_mean[..., 1:4]], dim=-1)
    
    # Initialize covariance matrix
    tan_cov = torch.zeros((batch_size, 1, 3, 3), dtype=q.dtype, device=q.device)
    
    for i in range(batch_size):
        # Get mean for this batch element
        q_m = q_mean[i, 0, :] if batch_size > 1 else q_mean[0, 0, :]
        q_m_inv = torch.cat([q_m[0:1], -q_m[1:4]], dim=-1)
        
        # Compute tangent vectors for all quaternions in this batch
        tangent_vectors = torch.zeros((N, 3), dtype=q.dtype, device=q.device)
        
        for j in range(N):
            # Compute q_diff = q_mean_inv * q[i, j]
            q_j = q[i, j, :]
            q_diff = _quaternion_multiply(q_m_inv.unsqueeze(0), q_j.unsqueeze(0)).squeeze(0)
            
            # Convert quaternion difference to tangent space (axis-angle representation)
            # For small rotations, the tangent vector is approximately 2 * [qx, qy, qz]
            qw = q_diff[0]
            qxyz = q_diff[1:4]
            
            # Compute rotation angle
            angle = 2 * torch.acos(torch.clamp(qw, min=-1.0, max=1.0))
            
            # Compute rotation axis
            norm_qxyz = torch.norm(qxyz)
            if norm_qxyz > 1e-6:
                axis = qxyz / norm_qxyz
                tangent_vectors[j] = axis * angle
            else:
                tangent_vectors[j] = torch.zeros(3, dtype=q.dtype, device=q.device)
        
        # Compute covariance matrix: (1/N) * sum(v_i * v_i^T)
        for j in range(N):
            v = tangent_vectors[j].unsqueeze(-1)  # [3, 1]
            tan_cov[i, 0] += torch.matmul(v, v.transpose(-2, -1))  # [3, 3]
        
        tan_cov[i, 0] /= N
    
    if dim == -1:
        # Return scalar results for averaging across all dims
        return tan_cov.squeeze(0).squeeze(0)
    else:
        # Reshape back to original batch dimensions
        target_shape = batch_shape + (1, 3, 3)
        tan_cov = tan_cov.reshape(target_shape)
        
        # If we transposed earlier, transpose back
        if dim != original_shape.index(original_shape[dim]):
            # Need to transpose the result appropriately
            tan_cov = tan_cov.transpose(dim, -3)
        
        return tan_cov

# transformation math ops
def _compute_transform_stats(T, dim=-1):
    """
    Measures the Frechet mean (riemannian center of mass) and the covariance in the tangent space
    Args:
        T: torch.tensor [...,N,...,4,4], transformation matrices (can contain NaN values)
        dim: int, the dimension along which to take the mean. If -1 (default), takes the average across 
                all dimensions except the last two. NaN transformations along this dimension are ignored.
    Returns:
        T_mean: torch.tensor [...,1,...,4,4] or [4,4], the mean transformations across dim (or mean transformation if across all dims)
        T_cov: torch.tensor [...,1,...,6,6] or [6,6], the variation in translation and rotation
    """

    
    original_shape = T.shape
    device = T.device
    dtype = T.dtype
    
    # Initialize SE(3) group with PyTorch backend
    SE3 = SpecialEuclidean(n=3, point_type='matrix')
    
    if dim == -1:
        # Average across all transformations (flatten all dims except last two)
        T_flat = T.reshape(-1, 4, 4)
        N = T_flat.shape[0]
        batch_size = 1
        batch_shape = ()
    else:
        # Move the averaging dimension to position -3 (before the 4x4 matrix dims)
        if dim != T.dim() - 3:
            T = T.transpose(dim, -3)
        
        # Get the shape for processing
        batch_shape = original_shape[:dim] + original_shape[dim+1:-2]
        N = original_shape[dim]  # Number of transformations to average
        
        # Reshape to [..., N, 4, 4] for easier processing
        T_flat = T.reshape(-1, N, 4, 4)
        batch_size = T_flat.shape[0]
    
    # Convert to numpy for geomstats (it will handle PyTorch backend internally)
    T_np = T_flat.cpu().numpy()
    
    # Compute Frechet mean and tangent space covariance for each batch
    T_mean_list = []
    T_cov_list = []
    
    for i in range(batch_size):
        if dim == -1:
            # Single batch - all transformations
            data_points = T_np  # [N, 4, 4]
        else:
            data_points = T_np[i]  # [N, 4, 4]
        
        # Filter out NaN transformations by checking if any element is NaN
        # A transformation is valid if no elements in the 4x4 matrix are NaN
        valid_mask = ~np.isnan(data_points).any(axis=(1, 2))  # [N]
        valid_data = data_points[valid_mask]  # [N_valid, 4, 4]
        
        # If no valid transformations, set mean and covariance to NaN
        if valid_data.shape[0] == 0:
            T_mean_list.append(np.full((4, 4), np.nan))
            T_cov_list.append(np.full((6, 6), np.nan))
            continue
        
        # If only one valid transformation, use it as mean and set covariance to zero
        if valid_data.shape[0] == 1:
            T_mean_list.append(valid_data[0])
            T_cov_list.append(np.zeros((6, 6)))
            continue
        
        # Compute Frechet mean using geomstats
        frechet_mean = FrechetMean(space=SE3)
        T_mean_np = frechet_mean.fit(valid_data).estimate_
        
        # Compute tangent vectors at the mean for covariance
        # Log map: T_i -> log_{T_mean}(T_i) gives tangent vectors
        tangent_vecs = SE3.metric.log(valid_data, base_point=T_mean_np)  # Should be [N_valid, 6]
        
        # Handle various possible shapes from geomstats
        if tangent_vecs.ndim == 1:
            # Single vector case: reshape to [1, 6]
            tangent_vecs = tangent_vecs.reshape(1, -1)
        elif tangent_vecs.ndim == 3:
            # If it returns [N_valid, 4, 4] or similar, we need to extract the 6D representation
            # Flatten the last dimensions and take first 6 elements
            tangent_vecs = tangent_vecs.reshape(tangent_vecs.shape[0], -1)[:, :6]
        
        # Ensure we have the right shape [N_valid, 6]
        N_valid = valid_data.shape[0]
        if tangent_vecs.shape[0] != N_valid or tangent_vecs.shape[1] != 6:
            # Fallback: use zeros for covariance if shape is wrong
            print(f"Warning: tangent_vecs has unexpected shape {tangent_vecs.shape}, expected ({N_valid}, 6)")
            T_cov_np = np.zeros((6, 6))
        else:
            # Compute covariance matrix in tangent space
            # Center the tangent vectors (they should already be centered at 0)
            tangent_vecs_centered = tangent_vecs - tangent_vecs.mean(axis=0, keepdims=True)  # [N_valid, 6]
            T_cov_np = (tangent_vecs_centered.T @ tangent_vecs_centered) / N_valid  # [6, 6]
        
        T_mean_list.append(T_mean_np)
        T_cov_list.append(T_cov_np)
    
    # Convert back to PyTorch tensors
    if dim == -1:
        # Return scalar results
        T_mean = torch.from_numpy(T_mean_list[0]).to(device=device, dtype=dtype)
        T_cov = torch.from_numpy(T_cov_list[0]).to(device=device, dtype=dtype)
    else:
        # Stack and reshape back to original batch dimensions
        T_mean_stacked = torch.from_numpy(np.stack(T_mean_list, axis=0)).to(device=device, dtype=dtype)
        T_cov_stacked = torch.from_numpy(np.stack(T_cov_list, axis=0)).to(device=device, dtype=dtype)
        
        # Reshape to target shape
        target_shape_mean = batch_shape + (1, 4, 4)
        target_shape_cov = batch_shape + (1, 6, 6)
        
        T_mean = T_mean_stacked.reshape(target_shape_mean)
        T_cov = T_cov_stacked.reshape(target_shape_cov)
        
        # If we transposed earlier, transpose back
        if dim != original_shape.index(original_shape[dim]):
            T_mean = T_mean.transpose(dim, -3)
            T_cov = T_cov.transpose(dim, -3)
    
    return T_mean, T_cov


##########################
# --- PUBLIC METHODS --- #
##########################
# twist math ops
def compute_twist_between_poses(pose1, pose2=None, dt=1.0, relative_pose=None, w_first=True):
    """
    Compute the twist (spatial velocity) between two sets of poses. Handles NaN entries by replacing NaNs with zeros during computation
    and then restoring NaNs in the output twist where either input pose had NaNs.
    
    Args:
        pose1: First pose as tensor of shape [..., 7], [..., 9], or [..., 4, 4]
        pose2: Second pose as tensor of shape [..., 7], [..., 9], or [..., 4, 4] or None. If None, 
                convert pose1 into a twist relative to the world frame (identity).
        dt: Time difference between poses (default=1.0)
        relative_pose: pose that the twist is relative to (default: None) of 
                shape [..., 7], [..., 9], or [..., 4, 4]. If None, twist is relative
                to world frame
        w_first: Whether the input quaternion representation uses (qw, qx, qy, qz) format (default: True)
        
    Returns:
        Twist vector of shape [..., 6] containing [vx, vy, vz, ωx, ωy, ωz]
    """

    # Handle case where pose2 is None (convert pose1 to twist from identity)
    if pose2 is None:
        # Create identity pose with same shape as pose1
        batch_shape = pose1.shape[:-1]
        identity_pose = torch.zeros((*batch_shape, 7), device=pose1.device, dtype=pose1.dtype)
        identity_pose[..., 3] = 1.0  # Set qw = 1 for identity quaternion
        pose2 = pose1
        pose1 = identity_pose

    # Set up relative_pose (default to identity/world frame)
    if relative_pose is None:
        # Create identity pose with same shape as pose1
        batch_shape = pose1.shape[:-1]
        relative_pose = torch.zeros((*batch_shape, 7), device=pose1.device, dtype=pose1.dtype)
        relative_pose[..., 3] = 1.0  # Set qw = 1 for identity quaternion
    else:
        # put the relative pose on the correct device and dtype
        relative_pose = relative_pose.to(pose1.device).to(pose1.dtype)

    # convert poses to position and quaternion (x, y, z, qw, qx, qy, qz)
    pose1, _ = _convert_pose_any_to_pose_quat(pose1, w_first=w_first)
    pose2, _ = _convert_pose_any_to_pose_quat(pose2, w_first=w_first)
    relative_pose, _ = _convert_pose_any_to_pose_quat(relative_pose, w_first=w_first)

    # check that pose1 and pose2 batch dimensions are compatible
    if pose1.shape[:-1] != pose2.shape[:-1]:
        raise ValueError(f"Pose1 batch dim {pose1.shape[:-1]} and Pose2 batch dim {pose2.shape[:-1]} are not compatible.")

    # Track NaN entries in input poses for later restoration
    nan_mask_pose1 = torch.isnan(pose1).any(dim=-1)  # [...] True where any component is NaN
    nan_mask_pose2 = torch.isnan(pose2).any(dim=-1)  # [...] True where any component is NaN
    nan_mask = nan_mask_pose1 | nan_mask_pose2  # Combined mask
    
    # Replace NaNs with safe default values for computation
    # For positions: use 0, for quaternions: use identity (1, 0, 0, 0) for w,x,y,z
    pose1_safe = torch.where(torch.isnan(pose1), torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=pose1.device, dtype=pose1.dtype), pose1)
    pose2_safe = torch.where(torch.isnan(pose2), torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=pose2.device, dtype=pose2.dtype), pose2)

    # Extract positions and quaternions
    pos1, quat1 = pose1_safe[..., :3], pose1_safe[..., 3:7]
    pos2, quat2 = pose2_safe[..., :3], pose2_safe[..., 3:7]
    rel_pos, rel_quat = relative_pose[..., :3], relative_pose[..., 3:7]
    
    # Compute linear velocity in world frame (position difference)
    linear_vel_world = (pos2 - pos1) / dt
    
    # Compute quaternion difference (relative rotation)
    quat_diff = _quaternion_difference(quat1, quat2)
    
    # Convert quaternion difference to angular velocity in world frame
    qw = quat_diff[..., 0]
    qxyz = quat_diff[..., 1:4]
    
    # Compute rotation angle from quaternion
    angle = 2 * torch.acos(torch.clamp(qw, min=-1.0, max=1.0))
    
    # Compute rotation axis (handling edge cases)
    norm_qxyz = torch.norm(qxyz, dim=-1, keepdim=True)
    safe_norm_qxyz = torch.where(norm_qxyz > 1e-6, norm_qxyz, torch.ones_like(norm_qxyz))
    axis = qxyz / safe_norm_qxyz
    
    # Angular velocity = axis * angle / dt (in world frame)
    angular_vel_world = axis * angle.unsqueeze(-1) / dt
    
    # Transform twist to relative frame using inverse rotation of relative_pose
    # Get conjugate (inverse) of relative quaternion
    rel_quat_inv = torch.cat([rel_quat[..., 0:1], -rel_quat[..., 1:4]], dim=-1)
    rel_rot_mat = _quat_to_rot_mat(rel_quat_inv, w_first=True)
    
    # Rotate velocities into relative frame
    linear_vel = torch.matmul(rel_rot_mat, linear_vel_world.unsqueeze(-1)).squeeze(-1)
    angular_vel = torch.matmul(rel_rot_mat, angular_vel_world.unsqueeze(-1)).squeeze(-1)
    
    # Combine linear and angular velocity into twist
    twist = torch.cat([linear_vel, angular_vel], dim=-1)
    
    # Restore NaNs in output where either input pose had NaNs
    twist = torch.where(nan_mask.unsqueeze(-1), torch.tensor(float('nan'), device=twist.device, dtype=twist.dtype), twist)
    
    return twist

def add_twist_to_pose(pose, twist, dt, w_first=True):
    """
    Add a twist over time dt to a pose
    
    Args:
        pose: Tensor of shape [..., 7], [..., 9], or [..., 4, 4] depending on representation
        twist: Tensor of shape [..., 6] (vx, vy, vz, ωx, ωy, ωz)
        dt: Tensor of shape [..., 1] Time period to apply twist for
        w_first: Whether the input quaternion representation uses (qw, qx, qy, qz) format (default: True)
        
    Returns:
        Updated pose of shape [..., 7]
    """
   
    # convert pose to position and quaternion (x, y, z, qw, qx, qy, qz)
    pose, pose_rep = _convert_pose_any_to_pose_quat(pose, w_first=w_first)

    # Ensure dt has the correct shape [..., 1]
    if dt.dim() == 0:  # scalar tensor
        dt = dt.unsqueeze(-1)
    elif dt.shape[-1] != 1:
        dt = dt.unsqueeze(-1)

    # check that twist and pose batch dimensions are compatible
    if pose.shape[:-1] != twist.shape[:-1] or pose.shape[:-1] != dt.shape[:-1] or twist.shape[:-1] != dt.shape[:-1]:
        raise ValueError(f"Pose batch dim {pose.shape[:-1]}, twist batch dim {twist.shape[:-1]}, and dt batch dim {dt.shape[:-1]} are not compatible.")

    # Extract components
    position = pose[..., :3]
    quaternion = pose[..., 3:7]
    linear_vel = twist[..., :3]
    angular_vel = twist[..., 3:6]
    
    # 1. Update position (simple integration)
    new_position = position + linear_vel * dt
    
    # 2. Update orientation using exponential map
    # Calculate the rotation angle from angular velocity
    angle = torch.norm(angular_vel, dim=-1, keepdim=True) * dt
    
    # Create a unit rotation axis (handling zero angular velocity case)
    axis_norm = torch.norm(angular_vel, dim=-1, keepdim=True)
    axis = torch.where(
        axis_norm > 1e-6,
        angular_vel / axis_norm,
        torch.tensor([1.0, 0.0, 0.0], device=pose.device).expand_as(angular_vel)
    )
    
    # Create the rotation quaternion (from axis-angle)
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    
    rot_quat = torch.cat([
        torch.cos(half_angle),
        axis * sin_half
    ], dim=-1)
    
    # Apply the rotation using quaternion multiplication
    new_quaternion = _quaternion_multiply(quaternion, rot_quat)
    
    # Normalize the resulting quaternion
    new_quaternion = new_quaternion / torch.norm(new_quaternion, dim=-1, keepdim=True)
    
    # Combine into new pose
    new_pose = torch.cat([new_position, new_quaternion], dim=-1)

    # convert pose back to original representation (and match the quat order to the input if needed)
    new_pose = _convert_pose_quat_to_pose_any(new_pose, representation_to=pose_rep, w_first=w_first)
    
    return new_pose

def compute_quaternion_stats(q, dim=-1):
    """
    Measures the mean quaternion, the variation of that mean quaternion, and the tangent space covariance
    Args:
        q: Torch.tensor [...,4], quaternions
        dim: int, dimension along which to compute (defualt -1 is compute along all dimensions)
    Returns:
        q_mean: Torch.tensor with shape [...,1,...,4] or [4], mean quaternions (w,x,y,z) accross dim 
        V: Torch.tensor with shape [...,1] or scalar, dispersion metric V = 1 - lambda1 (largest eigenvalue)
        tan_cov: Torch.tensor [...,1,...3,3], the tangent space covariance matrix
    """

    q_mean, V = _quaternion_mean(q,dim)

    tan_cov = _quaternion_tan_space_covar(q,q_mean,dim)

    return q_mean, V, tan_cov

def compute_action_stats(actions, dim=-1):
    """
    Compute statistics for actions in pose + axis-angle format
    Args:
        actions: torch.tensor [...,N,...,7], actions in (x, y, z, axis*angle, gripper_cmd) format (can contain NaN values)
        dim: int, the dimension along which to take the mean. If -1 (default), takes the average across 
                all dimensions except the last dimension. NaN actions along this dimension are ignored.
    Returns:
        T_mean: torch.tensor [...,1,...,4,4] or [4,4], the mean transformation across dim
        T_cov: torch.tensor [...,1,...,6,6] or [6,6], the tangent space covariance
        gripper_mean: torch.tensor [...,1,...] or scalar, the mean gripper command across dim
        gripper_sd: torch.tensor [...,1,...] or scalar, the standard deviation of gripper command across dim
    """

    # seperate out gripper command
    gripper_cmd = actions[...,-1]
    pose = actions[...,:-1]

    # Convert pos and axis-angle actions to transformations
    T = _pose_axis_angle_to_T(pose, w_first=True)
    
    # Compute the mean and covariance of the transformation matrices
    T_mean, T_cov = _compute_transform_stats(T, dim=dim)

    # compute mean and sd of gripper command (ignoring NaN values)
    if dim == -1:
        # Average across all dimensions
        gripper_mean = torch.nanmean(gripper_cmd)
        gripper_sd = torch.sqrt(torch.nanmean((gripper_cmd - gripper_mean) ** 2))
    else:
        # Average along specified dimension, keeping dimensions
        gripper_mean = torch.nanmean(gripper_cmd, dim=dim, keepdim=True)
        gripper_sd = torch.sqrt(torch.nanmean((gripper_cmd - gripper_mean) ** 2, dim=dim, keepdim=True))
    
    return T_mean, T_cov, gripper_mean, gripper_sd
    

