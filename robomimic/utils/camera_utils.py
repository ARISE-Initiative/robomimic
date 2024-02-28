"""
Almost all the code in this file has been copied from TRI's VIDAR
https://github.com/TRI-ML/vidar
"""


from abc import ABC
import abc
import torch
import torch.nn as nn
import torch.nn.functional as tfn
import os
import numpy as np
from collections import OrderedDict
import torchvision
from einops import rearrange

def is_int(data):
    """Checks if data is an integer."""
    return isinstance(data, int)

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor

def to_global_pose_broken(pose, zero_origin=False):
    """Get global pose coordinates from current and context poses"""
    tgt = 0 if 0 in pose else (0, 0)
    base = None if zero_origin else pose[tgt].T.clone()
    pose[tgt].T = torch.eye(
        4, device=pose[tgt].device, dtype=pose[tgt].dtype).repeat(pose[tgt].T.shape[0], 1, 1)

    keys = pose.keys()
    # steps = sorted(set([key[0] for key in keys]))
    steps = {key[1]: [key2[0] for key2 in keys if key2[1] == key[1]] for key in keys}
    cams = sorted(set([key[1] for key in keys]))
    
    for cam in cams:
        if cam != tgt[1]:
            pose[(tgt[0], cam)].T = (pose[(tgt[0], cam)] * pose[tgt]).T.float()
    for cam in cams:
        for step in steps[cam]:
            if step != tgt[0]:
                pose[(step, cam)] = (pose[(step, cam)] * pose[(tgt[0], cam)])
    # for step in steps:
    #     if step != tgt[0]:
    #         for cam in cams:
    #             pose[(step, cam)] = (pose[(step, cam)] * pose[(tgt[0], cam)])
    if not zero_origin:
        for key in keys:
            pose[key].T = pose[key].T @ base
    return pose

def to_global_pose(pose, zero_origin=False):
    """Get global pose coordinates from current and context poses"""
    tgt = 0 if 0 in pose else (0, 0)
    base = None if zero_origin else pose[tgt].T[[0]].clone()
    pose[tgt].T[[0]] = torch.eye(4, device=pose[tgt].device, dtype=pose[tgt].dtype)
    for b in range(1, len(pose[tgt])):
        pose[tgt].T[[b]] = (pose[tgt][b] * pose[tgt][0]).T.float()
    for key in pose.keys():
        if key != tgt:
            pose[key] = pose[key] * pose[tgt]
    if not zero_origin:
        for key in pose.keys():
            for b in range(len(pose[key])):
                pose[key].T[[b]] = pose[key].T[[b]] @ base
    return pose

def from_dict_sample(T, to_global=False, zero_origin=False, to_matrix=False, broken=False):
    """Helper function to convert sample poses to Pose objects"""
    pose = {key: Pose(val) for key, val in T.items()}
    if to_global:
        to_global_pose_fn = to_global_pose_broken if broken else to_global_pose
        pose = to_global_pose_fn(pose, zero_origin=zero_origin)
    if to_matrix:
        pose = {key: val.T for key, val in pose.items()}
    return pose


def pose_vec2mat(vec, mode='euler', dtype=None):
    """Convert Euler parameters to transformation matrix."""
    if mode is None:
        return vec
    trans, rot = vec[:, :3].unsqueeze(-1), vec[:, 3:]
    if mode == 'euler':
        rot_mat = euler2mat(rot, dtype)
    else:
        raise ValueError('Rotation mode not supported {}'.format(mode))
    mat = torch.cat([rot_mat, trans], dim=2)  # [B,3,4]
    return mat


def euler2mat(angle, dtype=None):
    """Convert euler angles to rotation matrix"""
    B = angle.size(0)
    if dtype is not None:
        angle = angle.to(dtype)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([ cosz, -sinz, zeros,
                         sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([ cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat

def is_tuple(data):
    """Checks if data is a tuple."""
    return isinstance(data, tuple)

def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list) or isinstance(data, torch.nn.ModuleList)

def is_seq(data):
    """Checks if data is a list or tuple."""
    return is_tuple(data) or is_list(data)

def is_dict(data):
    """Checks if data is a dictionary."""
    return isinstance(data, dict) or isinstance(data, torch.nn.ModuleDict)

def iterate1(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var, *args, **kwargs):
        if is_seq(var):
            return [func(v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(var, *args, **kwargs)
    return inner

@iterate1
def invert_pose(T):
    """Inverts a [B,4,4] torch.tensor pose"""
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv
    # return torch.linalg.inv(T)

def from_dict_batch(T, **kwargs):
    """Helper function to convert a dicionary of tensor poses to Pose objects"""
    pose_batch = [from_dict_sample({key: val[b] for key, val in T.items()}, **kwargs)
                  for b in range(T[0].shape[0])]
    return {key: torch.stack([v[key] for v in pose_batch], 0) for key in pose_batch[0]}


def align_corners(value=None):
    """Get the align_corners value from the environment variable"""
    if value is not None:
        return value
    if os.getenv('ALIGN_CORNERS') is not None:
        return True if os.getenv('ALIGN_CORNERS') == 'True' else \
               False if os.getenv('ALIGN_CORNERS') == 'False' else None
    return True


@iterate1
def interpolate(tensor, size, scale_factor, mode):
    """Helper function to interpolate tensors"""
    if size is None and scale_factor is None:
        return tensor
    if is_tensor(size):
        size = size.shape[-2:]
    return tfn.interpolate(
        tensor, size=size, scale_factor=scale_factor,
        recompute_scale_factor=False, mode=mode,
        align_corners=None if mode == 'nearest' else align_corners(),
    )


def interpolate_nearest(tensor, size=None, scale_factor=None):
    """Helper function to interpolate tensors using nearest neighbor interpolation"""
    if size is not None and is_tensor(size):
        size = size.shape[-2:]
    return interpolate(tensor.float(), size, scale_factor, mode='nearest')


def masked_average(loss, mask, eps=1e-7):
    """Mask average for loss given mask"""
    return (loss * mask).sum() / (mask.sum() + eps)


def multiply_mask(data, mask):
    """Multiply data with masks"""
    return data if (data is None or mask is None) else data * mask


def multiply_args(*args):
    """Multiply all arguments"""
    valids = [v for v in args if v is not None]
    return None if not valids else reduce((lambda x, y: x * y), valids)


def grid_sample(tensor, grid, padding_mode, mode):
    """Helper function for grid sampling"""
    return tfn.grid_sample(
        tensor, grid,
        padding_mode=padding_mode, mode=mode,
        align_corners=align_corners(),
    )


def grid_sample_volume(tensor, grid, padding_mode, mode):
    """Helper function for multi-layer grid sampling"""
    b, d, h, w, _ = grid.shape
    return grid_sample(
        tensor, grid.reshape(b, d, h * w, 2),
        padding_mode=padding_mode, mode=mode
    ).reshape(b, 3, d, h, w)


def pixel_grid(hw, b=None, with_ones=False, device=None, normalize=False, shake=False):
    """Helper function to generate a pixel grid given [H,W] or [B,H,W]"""
    if is_tensor(hw):
        b, hw, device = hw.shape[0], hw.shape[-2:], hw.device
    if is_tensor(device):
        device = device.device
    if align_corners():
        hi, hf = 0, hw[0] - 1
        wi, wf = 0, hw[1] - 1
    else:
        hi, hf = 0.5, hw[0] - 0.5
        wi, wf = 0.5, hw[1] - 0.5
    yy, xx = torch.meshgrid([torch.linspace(hi, hf, hw[0], device=device),
                             torch.linspace(wi, wf, hw[1], device=device)], indexing='ij')
    if with_ones:
        grid = torch.stack([xx, yy, torch.ones(hw, device=device)], 0)
    else:
        grid = torch.stack([xx, yy], 0)
    if b is not None:
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    if shake:
        if align_corners():
            rand = torch.rand((b, 2, *hw), device=device)
        else:
            rand = torch.rand((b, 2, *hw), device=device) - 0.5
        grid[:, :2, :, :] += rand
    if normalize:
        grid = norm_pixel_grid(grid)
    return grid


def norm_pixel_grid(grid, hw=None, in_place=False):
    """Normalize a pixel grid from [W,H] to [-1,+1]."""
    if hw is None:
        hw = grid.shape[-2:]
    if not in_place:
        grid = grid.clone()
    if align_corners():
        grid[:, 0] = 2.0 * grid[:, 0] / (hw[1] - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / (hw[0] - 1) - 1.0
    else:
        grid[:, 0] = 2.0 * grid[:, 0] / hw[1] - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / hw[0] - 1.0
    return grid


def unnorm_pixel_grid(grid, hw=None, in_place=False):
    """Unnormalize a pixel grid from [-1,+1] to [W,H]."""
    if hw is None:
        hw = grid.shape[-2:]
    if not in_place:
        grid = grid.clone()
    if align_corners():
        grid[:, 0] = 0.5 * (hw[1] - 1) * (grid[:, 0] + 1)
        grid[:, 1] = 0.5 * (hw[0] - 1) * (grid[:, 1] + 1)
    else:
        grid[:, 0] = 0.5 * hw[1] * (grid[:, 0] + 1)
        grid[:, 1] = 0.5 * hw[0] * (grid[:, 1] + 1)
    return grid


def match_scales(image, targets, num_scales, mode='bilinear'):
    """Match scales of image given target scales and resolutions

    Parameters
    ----------
    image : torch.Tensor
        Input image to be scaled
    targets : list of torch.Tensor
        Targes to match scale
    num_scales : int
        Number of scales to match
    mode : str, optional
        Interpolation mode, by default 'bilinear'

    Returns
    -------
    _type_
        _description_
    """
    # For all scales
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        # If image shape is equal to target shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            # Otherwise, interpolate
            images.append(interpolate_image(image, target_shape, mode=mode))
    # Return scaled images
    return images


def cat_channel_ones(tensor, n=1):
    """
    Concatenate tensor with an extra channel of ones

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be concatenated
    n : int
        Which channel will be concatenated

    Returns
    -------
    cat_tensor : torch.Tensor
        Concatenated tensor
    """
    # Get tensor shape with 1 channel
    shape = list(tensor.shape)
    shape[n] = 1
    # Return concatenation of tensor with ones
    return torch.cat([tensor, torch.ones(shape,
                      device=tensor.device, dtype=tensor.dtype)], n)


def same_shape(shape1, shape2):
    """Checks if two shapes are the same"""
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True


def interpolate_image(image, shape=None, scale_factor=None,
                      mode='bilinear', recompute_scale_factor=False):
    """
    Interpolate an image to a different resolution

    Parameters
    ----------
    image : torch.Tensor
        Image to be interpolated [B,?,h,w]
    shape : torch.Tensor or tuple
        Output shape [H,W]
    scale_factor : float
        Scale factor for output shape
    mode : str
        Interpolation mode
    recompute_scale_factor : bool
        True if scale factor is recomputed

    Returns
    -------
    image : torch.Tensor
        Interpolated image [B,?,H,W]
    """
    assert shape is not None or scale_factor is not None, 'Invalid option for interpolate_image'
    # Take last two dimensions as shape
    if shape is not None:
        if is_tensor(shape):
            shape = shape.shape
        if len(shape) > 2:
            shape = shape[-2:]
        # If the shapes are the same, do nothing
        if same_shape(image.shape[-2:], shape):
            return image
    # Interpolate image to match the shape
    return interpolate(image, size=shape, scale_factor=scale_factor, mode=mode)


def check_assert(pred, gt, atol=1e-5, rtol=1e-5):
    """Check if two dictionaries are equal"""
    for key in gt.keys():
        if key in pred.keys():
            # assert key in pred and key in gt
            if is_dict(pred[key]):
                check_assert(pred[key], gt[key])
            elif is_seq(pred[key]):
                for val1, val2 in zip(pred[key], gt[key]):
                    if is_tensor(val1):
                        assert torch.allclose(val1, val2, atol=atol, rtol=rtol), \
                            f'Assert error in {key} : {val1.mean().item()} x {val2.mean().item()}'
                    else:
                        assert val1 == val2, \
                            f'Assert error in {key} : {val1} x {val2}'
            else:
                if is_tensor(pred[key]):
                    assert torch.allclose(pred[key], gt[key], atol=atol, rtol=rtol), \
                        f'Assert error in {key} : {pred[key].mean().item()} x {gt[key].mean().item()}'
                else:
                    assert pred[key] == gt[key], \
                        f'Assert error in {key} : {pred[key]} x {gt[key]}'


def interleave(data, b):
    """Interleave data across multiple batches"""
    data_interleave = data.unsqueeze(1).expand(-1, b, *data.shape[1:])
    return data_interleave.reshape(-1, *data.shape[1:])

def invert_intrinsics(K):
    """Invert camera intrinsics"""

    Kinv = K.clone()
    Kinv[:, 0, 0] = 1. / K[:, 0, 0]
    Kinv[:, 1, 1] = 1. / K[:, 1, 1]
    Kinv[:, 0, 2] = -1. * K[:, 0, 2] / K[:, 0, 0]
    Kinv[:, 1, 2] = -1. * K[:, 1, 2] / K[:, 1, 1]
    return Kinv


def scale_intrinsics(K, ratio):
    """Scale intrinsics given x_scale and y_scale factors"""

    if is_seq(ratio):
        ratio_h, ratio_w = ratio
    else:
        ratio_h = ratio_w = ratio

    K = K.clone()

    K[..., 0, 0] *= ratio_w
    K[..., 1, 1] *= ratio_h

    K[..., 0, 2] = K[..., 0, 2] * ratio_w
    K[..., 1, 2] = K[..., 1, 2] * ratio_h

    return K

class Pose:
    """
    Pose class, that encapsulates a [4,4] transformation matrix
    for a specific reference frame
    """
    def __init__(self, T=1):
        """
        Initializes a Pose object.

        Parameters
        ----------
        T : int or torch.Tensor
            Transformation matrix [B,4,4]
        """
        if is_int(T):
            T = torch.eye(4).repeat(T, 1, 1)
        self.T = T if T.dim() == 3 else T.unsqueeze(0)

    def __len__(self):
        """Batch size of the transformation matrix"""
        return len(self.T)

    def __getitem__(self, idx):
        """Return batch-wise pose"""
        if not is_tensor(idx):
            idx = [idx]
        return Pose(self.T[idx])

    def __mul__(self, data):
        """Transforms the input (Pose or 3D points) using this object"""
        if isinstance(data, Pose):
            return Pose(self.T.bmm(data.T))
        elif isinstance(data, torch.Tensor):
            return self.T[:, :3, :3].bmm(data) + self.T[:, :3, -1].unsqueeze(-1)
        else:
            raise NotImplementedError()

    def detach(self):
        """Detach pose from graph"""
        return Pose(self.T.detach())

    def clone(self):
        """Clone pose"""
        return type(self)(
            T=self.T.clone()
        )

    @property
    def shape(self):
        """Return pose shape"""
        return self.T.shape

    @property
    def device(self):
        """Return pose device"""
        return self.T.device

    @property
    def dtype(self):
        """Return pose type"""
        return self.T.dtype

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        """Initializes as a [4,4] identity matrix"""
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))

    @staticmethod
    def from_dict(T, to_global=False, zero_origin=False, to_matrix=False, broken=False):
        if T is None:
            return None
        tgt = (0, 0) if broken else 0
        if T[tgt].dim() == 3:
            return from_dict_sample(
                T, to_global=to_global, zero_origin=zero_origin, to_matrix=to_matrix, broken=broken)
        elif T[tgt].dim() == 4:
            return from_dict_batch(
                T, to_global=to_global, zero_origin=zero_origin, to_matrix=True, broken=broken)

    @classmethod
    def from_vec(cls, vec, mode):
        """Initializes from a [B,6] batch vector"""
        mat = pose_vec2mat(vec, mode)
        pose = torch.eye(4, device=vec.device, dtype=vec.dtype).repeat([len(vec), 1, 1])
        pose[:, :3, :3] = mat[:, :3, :3]
        pose[:, :3, -1] = mat[:, :3, -1]
        return cls(pose)

    def repeat(self, *args, **kwargs):
        """Repeats the transformation matrix multiple times"""
        self.T = self.T.repeat(*args, **kwargs)
        return self

    def inverse(self):
        """Returns a new Pose that is the inverse of this one"""
        return Pose(invert_pose(self.T))

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.T = self.T.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Moves pose to GPU"""
        self.to('cuda')
        return self

    def translate(self, xyz):
        """Translate pose by a vector"""
        self.T[:, :3, -1] = self.T[:, :3, -1] + xyz.to(self.device)
        return self

    def rotate(self, rpw):
        """Rotate pose by a vector"""
        rot = euler2mat(rpw)
        T = invert_pose(self.T).clone()
        T[:, :3, :3] = T[:, :3, :3] @ rot.to(self.device)
        self.T = invert_pose(T)
        return self

    def rotateRoll(self, r):
        """Rotate pose by a roll angle"""
        return self.rotate(torch.tensor([[0, 0, r]]))

    def rotatePitch(self, p):
        """Rotate pose by a pitch angle"""
        return self.rotate(torch.tensor([[p, 0, 0]]))

    def rotateYaw(self, w):
        """Rotate pose by a yaw angle"""
        return self.rotate(torch.tensor([[0, w, 0]]))

    def translateForward(self, t):
        """Translate pose on its z axis (forward)"""
        return self.translate(torch.tensor([[0, 0, -t]]))

    def translateBackward(self, t):
        """Translate pose on its z axis (backward)"""
        return self.translate(torch.tensor([[0, 0, +t]]))

    def translateLeft(self, t):
        """Translate pose on its x axis (left)"""
        return self.translate(torch.tensor([[+t, 0, 0]]))

    def translateRight(self, t):
        """Translate pose on its x axis (right)"""
        return self.translate(torch.tensor([[-t, 0, 0]]))

    def translateUp(self, t):
        """Translate pose on its y axis (up)"""
        return self.translate(torch.tensor([[0, +t, 0]]))

    def translateDown(self, t):
        """Translate pose on its y axis (down)"""
        return self.translate(torch.tensor([[0, -t, 0]]))


class CameraBase(nn.Module, ABC):
    """Base camera class

    Parameters
    ----------
    hw : tuple
        Camera resolution
    Twc : torch.Tensor, optional
        Camera pose (world to camera), by default None
    Tcw : torch.Tensor, optional
        Camera pose (camera to world), by default None
    """
    def __init__(self, hw, Twc=None, Tcw=None):
        super().__init__()
        assert Twc is None or Tcw is None

        # Pose

        if Twc is None and Tcw is None:
            self._Twc = torch.eye(
                4, dtype=self._K.dtype, device=self._K.device).unsqueeze(0).repeat(self._K.shape[0], 1, 1)
        else:
            self._Twc = invert_pose(Tcw) if Tcw is not None else Twc
        if is_tensor(self._Twc):
            self._Twc = Pose(self._Twc)

        # Resolution

        self._hw = hw
        if is_tensor(self._hw):
            self._hw = self._hw.shape[-2:]

    def __getitem__(self, idx):
        """Return batch-wise pose"""
        if is_seq(idx):
            return type(self).from_list([self.__getitem__(i) for i in idx])
        else:
            if not is_tensor(idx):
                idx = [idx]
            return type(self)(
                K=self._K[idx],
                Twc=self._Twc[idx] if self._Twc is not None else None,
                hw=self._hw,
            )

    def __len__(self):
        """Return length as intrinsics batch"""
        return self._K.shape[0]

    def __eq__(self, cam):
        """Check camera equality"""
        if not isinstance(cam, type(self)):
            return False
        if self._hw[0] != cam.hw[0] or self._hw[1] != cam.hw[1]:
            return False
        if not torch.allclose(self._K, cam.K):
            return False
        if not torch.allclose(self._Twc.T, cam.Twc.T):
            return False
        return True

    def clone(self):
        """Clone camera"""
        return type(self)(
            K=self.K.clone(),
            Twc=self.Twc.clone(),
            hw=[v for v in self._hw],
        )

    @property
    def pose(self):
        """Return camera pose (world to camera)"""
        return self._Twc.T

    @property
    def K(self):
        """Return camera intrinsics"""
        return self._K

    @K.setter
    def K(self, K):
        """Set camera intrinsics"""
        self._K = K

    @property
    def batch_size(self):
        """Return batch size"""
        return self._Twc.T.shape[0]

    @property
    def b(self):
        """Return batch size"""
        return self._Twc.T.shape[0]

    @property
    def bhw(self):
        """Return batch size and resolution"""
        return self.b, self.hw

    @property
    def bdhw(self):
        """Return batch size, device, and resolution"""
        return self.b, self.device, self.hw

    @property
    def hw(self):
        """Return camera resolution"""
        return self._hw

    @hw.setter
    def hw(self, hw):
        """Set camera resolution"""
        self._hw = hw

    @property
    def wh(self):
        """Return camera resolution"""
        return self._hw[::-1]

    @property
    def n_pixels(self):
        """Return number of pixels"""
        return self._hw[0] * self._hw[1]

    @property
    def Tcw(self):
        """Return camera pose (camera to world)"""
        return None if self._Twc is None else self._Twc.inverse()

    @Tcw.setter
    def Tcw(self, Tcw):
        """Set camera pose (camera to world)"""
        self._Twc = Tcw.inverse()

    @property
    def Twc(self):
        """Return camera pose (world to camera)"""
        return self._Twc

    @Twc.setter
    def Twc(self, Twc):
        """Set camera pose (world to camera)"""
        self._Twc = Twc

    @property
    def dtype(self):
        """Get camera data type"""
        return self._K.dtype

    @property
    def device(self):
        """Get camera device"""
        return self._K.device

    def detach_pose(self):
        """Detach camera pose"""
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def detach_K(self):
        """Detach camera intrinsics"""
        return type(self)(K=self._K.detach(), hw=self._hw, Twc=self._Twc)

    def detach(self):
        """Detach camera"""
        return type(self)(K=self._K.detach(), hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def inverted_pose(self):
        """Return camera with inverted pose"""
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.inverse() if self._Twc is not None else None)

    def no_translation(self):
        """Return camera with no translation"""
        Twc = self.pose.clone()
        Twc[:, :-1, -1] = 0
        return type(self)(K=self._K, hw=self._hw, Twc=Twc)

    def no_pose(self):
        """Return camera with no pose"""
        return type(self)(K=self._K, hw=self._hw)

    def interpolate(self, rgb):
        """Interpolate RGB image"""
        if rgb.dim() == 5:
            rgb = rearrange(rgb, 'b n c h w -> (b n) c h w')
        return interpolate(rgb, scale_factor=None, size=self.hw, mode='bilinear')

    def interleave_K(self, b):
        """Interleave intrinsics for multiple batches"""
        return type(self)(
            K=interleave(self._K, b),
            Twc=self._Twc,
            hw=self._hw,
        )

    def interleave_Twc(self, b):
        """Interleave pose for multiple batches"""
        return type(self)(
            K=self._K,
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def interleave(self, b):
        """Interleave camera for multiple batches"""
        return type(self)(
            K=interleave(self._K, b),
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def repeat_bidir(self):
        """Repeat camera for bidirectional training"""
        return type(self)(
            K=self._K.repeat(2, 1, 1),
            Twc=torch.cat([self._Twc.T, self.Tcw.T], 0),
            hw=self.hw,
        )

    def Pwc(self, from_world=True):
        """Return camera projection matrix (world to camera)"""
        return self._K[:, :3] if not from_world or self._Twc is None else \
            torch.matmul(self._K, self._Twc.T)[:, :3]

    def to_world(self, points):
        """Moves pointcloud to world coordinates"""
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self.Tcw is None else self.Tcw * points

    def from_world(self, points):
        """Moves pointcloud to camera coordinates"""
        if points.dim() > 3:
            shape = points.shape
            points = points.reshape(points.shape[0], 3, -1)
        else:
            shape = None
        local_points = points if self._Twc is None else \
            torch.matmul(self._Twc.T, cat_channel_ones(points, 1))[:, :3]
        return local_points if shape is None else local_points.view(shape)
    
    def from_world2(self, points):
        """Moves pointcloud to camera coordinates"""
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self._Twc is None else \
            torch.matmul(self._Twc.T[:, :3, :3], points[:, :3]) + self._Twc.T[:, :3, 3:]

    def to(self, *args, **kwargs):
        """Moves camera to device"""
        self._K = self._K.to(*args, **kwargs)
        if self._Twc is not None:
            self._Twc = self._Twc.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Moves camera to GPU"""
        return self.to('cuda')

    def relative_to(self, cam):
        """Returns camera relative to another camera"""
        return type(self)(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc.inverse())

    def global_from(self, cam):
        """Returns global camera from one relative to another camera"""
        return type(self)(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc)

    def pixel_grid(self, shake=False):
        """Returns pixel grid"""
        return pixel_grid(
            b=self.batch_size, hw=self.hw, with_ones=True,
            shake=shake, device=self.device).view(self.batch_size, 3, -1)

    def reconstruct_depth_map(self, depth, to_world=False, grid=None, scene_flow=None, world_scene_flow=None):
        """Reconstruct 3D pointcloud from z-buffer depth map"""
        if depth is None:
            return None
        b, _, h, w = depth.shape
        if grid is None:
            grid = pixel_grid(depth, with_ones=True, device=depth.device).view(b, 3, -1)
        points = self.lift(grid) * depth.view(depth.shape[0], 1, -1)
        if scene_flow is not None:
            points = points + scene_flow.view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
            if world_scene_flow is not None:
                points = points + world_scene_flow.view(b, 3, -1)
        return points.view(b, 3, h, w)

    def reconstruct_euclidean(self, depth, to_world=False, grid=None, scene_flow=None, world_scene_flow=None):
        """Reconstruct 3D pointcloud from euclidean depth map"""
        if depth is None:
            return None
        b, _, h, w = depth.shape
        rays = self.get_viewdirs(normalize=True, to_world=False, grid=grid).view(b, 3, -1)
        points = rays * depth.view(depth.shape[0], 1, -1)
        if scene_flow is not None:
            points = points + scene_flow.view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
            if world_scene_flow is not None:
                points = points + world_scene_flow.view(b, 3, -1)
        return points.view(b, 3, h, w)

    def reconstruct_volume(self, depth, euclidean=False, **kwargs):
        """Reconstruct 3D pointcloud from depth volume"""
        if euclidean:
            return torch.stack([self.reconstruct_euclidean(depth[:, :, i], **kwargs)
                                for i in range(depth.shape[2])], 2)
        else:
            return torch.stack([self.reconstruct_depth_map(depth[:, :, i], **kwargs)
                                for i in range(depth.shape[2])], 2)

    def reconstruct_cost_volume(self, volume, to_world=True, flatten=True):
        """Reconstruct 3D pointcloud from cost volume"""
        c, d, h, w = volume.shape
        grid = pixel_grid((h, w), with_ones=True, device=volume.device).view(3, -1).repeat(1, d)
        points = torch.stack([
            (volume.view(c, -1) * torch.matmul(invK[:3, :3].unsqueeze(0), grid)).view(3, d * h * w)
            for invK in self.invK], 0)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        if flatten:
            return points.view(-1, 3, d, h * w).permute(0, 2, 1, 3)
        else:
            return points.view(-1, 3, d, h, w)

    def project_points(self, points, from_world=True, normalize=True,
                       return_z=False, return_e=False, flag_invalid=True):
        """Projects 3D points to 2D image plane"""

        is_depth_map = points.dim() == 4
        hw = self._hw if not is_depth_map else points.shape[-2:]
        return_depth = return_z or return_e

        if is_depth_map:
            points = points.reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        coords, depth = self.unlift(points, from_world=from_world, euclidean=return_e)

        if not is_depth_map:
            if normalize:
                coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
                if flag_invalid:
                    invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                              (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth < 0)
                    coords[invalid.unsqueeze(1).repeat(1, 2, 1)] = -2
            if return_depth:
                return coords.permute(0, 2, 1), depth
            else:
                return coords.permute(0, 2, 1)

        coords = coords.view(b, 2, *hw)
        depth = depth.view(b, 1, *hw)

        if normalize:
            coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
            if flag_invalid:
                invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                          (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth[:, 0] < 0)
                coords[invalid.unsqueeze(1).repeat(1, 2, 1, 1)] = -2

        if return_depth:
            return coords.permute(0, 2, 3, 1), depth
        else:
            return coords.permute(0, 2, 3, 1)

    def project_cost_volume(self, points, from_world=True, normalize=True):
        """Projects 3D points from a cost volume to 2D image plane"""
    
        if points.dim() == 4:
            points = points.permute(0, 2, 1, 3).reshape(points.shape[0], 3, -1)
        b, _, n = points.shape
    
        points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))
    
        coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-7)
        coords = coords.view(b, 2, -1, *self._hw).permute(0, 2, 3, 4, 1)
    
        if normalize:
            coords[..., 0] /= self._hw[1] - 1
            coords[..., 1] /= self._hw[0] - 1
            return 2 * coords - 1
        else:
            return coords

    def create_radial_volume(self, bins, to_world=True):
        """Create a volume of radial depth bins"""
        ones = torch.ones((1, *self.hw), device=self.device)
        volume = torch.stack([depth * ones for depth in bins], 1).unsqueeze(0)
        return self.reconstruct_volume(volume, to_world=to_world)

    def project_volume(self, volume, from_world=True):
        """Project a volume to 2D image plane"""
        b, c, d, h, w = volume.shape
        return self.project_points(volume.view(b, c, -1), from_world=from_world).view(b, d, h, w, 2)

    def coords_from_cost_volume(self, volume, ref_cam=None):
        """Project a cost volume to 2D image plane"""
        if ref_cam is None:
            return self.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=False), from_world=True)
        else:
            return ref_cam.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=True), from_world=True)

    def z2e(self, z_depth):
        """Convert z-buffer depth to euclidean depth"""
        points = self.reconstruct_depth_map(z_depth, to_world=False)
        return self.project_points(points, from_world=False, return_e=True)[1]

    def e2z(self, e_depth):
        """Convert euclidean depth to z-buffer depth"""
        points = self.reconstruct_euclidean(e_depth, to_world=False)
        return self.project_points(points, from_world=False, return_z=True)[1]

    def control(self, draw, tvel=0.2, rvel=0.1):
        """Control camera with keyboard (requires camviz)"""
        change = False
        if draw.UP:
            self.Twc.translateForward(tvel)
            change = True
        if draw.DOWN:
            self.Twc.translateBackward(tvel)
            change = True
        if draw.LEFT:
            self.Twc.translateLeft(tvel)
            change = True
        if draw.RIGHT:
            self.Twc.translateRight(tvel)
            change = True
        if draw.KEY_Z:
            self.Twc.translateUp(tvel)
            change = True
        if draw.KEY_X:
            self.Twc.translateDown(tvel)
            change = True
        if draw.KEY_A:
            self.Twc.rotateYaw(-rvel)
            change = True
        if draw.KEY_D:
            self.Twc.rotateYaw(+rvel)
            change = True
        if draw.KEY_W:
            self.Twc.rotatePitch(+rvel)
            change = True
        if draw.KEY_S:
            self.Twc.rotatePitch(-rvel)
            change = True
        if draw.KEY_Q:
            self.Twc.rotateRoll(-rvel)
            change = True
        if draw.KEY_E:
            self.Twc.rotateRoll(+rvel)
            change = True
        return change


class CameraNerf(CameraBase):
    """Camera class with NeRF functionalities"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Matrix to convert from vidar to NeRF coordinate system
        self.convert_matrix = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0)

    @staticmethod
    def from_list(cams):
        """Get a CameraNerf object from a list of cameras"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return CameraNerf(K=K, Twc=Twc, hw=cams[0].hw)

    @staticmethod
    def from_dict(K, hw, Twc=None):
        """Get a CameraNerf object from a dictionary of cameras"""
        return {key: CameraNerf(K=K[0], hw=hw[0], Twc=val) for key, val in Twc.items()}

    def switch(self):
        """Convert from vidar to NeRF coordinate system"""
        T = self.convert_matrix.to(self.device)
        Twc = T @ self.Twc.T @ T
        return type(self)(K=self.K, Twc=Twc, hw=self.hw)

    def bwd(self):
        """Go from NeRF to vidar coordinate system"""
        T = self.convert_matrix.to(self.device)
        Tcw = T @ self.Twc.T @ T
        return type(self)(K=self.K, Tcw=Tcw, hw=self.hw)

    def fwd(self):
        """Go to vidar from NeRF coordinate system"""
        T = self.convert_matrix.to(self.device)
        Twc = T @ self.Tcw.T @ T
        return type(self)(K=self.K, Twc=Twc, hw=self.hw)

    def look_at(self, at, up=torch.Tensor([0, 1, 0])):
        """Point a camera to a particular 3D point"""

        eps = 1e-5
        eye = self.Tcw.T[:, :3, -1]

        at = at.unsqueeze(0)
        up = up.unsqueeze(0).to(at.device)
        up /= up.norm(dim=-1, keepdim=True) + eps

        z_axis = at - eye
        z_axis /= z_axis.norm(dim=-1, keepdim=True) + eps

        up = up.expand(z_axis.shape)
        x_axis = torch.cross(up, z_axis)
        x_axis /= x_axis.norm(dim=-1, keepdim=True) + eps

        y_axis = torch.cross(z_axis, x_axis)
        y_axis /= y_axis.norm(dim=-1, keepdim=True) + eps

        R = torch.stack((x_axis, y_axis, z_axis), dim=-1)

        Tcw = self.Tcw
        Tcw.T[:, :3, :3] = R
        self.Twc = Tcw.inverse()

    def get_origin(self, flatten=False):
        """Get camera origin"""
        orig = self.Tcw.T[:, :3, -1].view(len(self), 3, 1, 1).repeat(1, 1, *self.hw)
        if flatten:
            orig = orig.reshape(len(self), 3, -1).permute(0, 2, 1)
        return orig

    def get_viewdirs(self, normalize=False, flatten=False, to_world=False):
        """Get camera viewing rays"""
        ones = torch.ones((len(self), 1, *self.hw), dtype=self.dtype, device=self.device)
        rays = self.reconstruct_depth_map(ones, to_world=False)
        if normalize:
            rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        if to_world:
            rays = self.to_world(rays).reshape(len(self), 3, *self.hw)
        if flatten:
            rays = rays.reshape(len(self), 3, -1).permute(0, 2, 1)
        return rays

    def get_render_rays(self, near=None, far=None, n_rays=None, gt=None):
        """Get camera render rays"""
        b = len(self)

        ones = torch.ones((b, 1, *self.hw), dtype=self.dtype, device=self.device)

        rays = self.reconstruct_depth_map(ones, to_world=False)
        rays = rays / torch.norm(rays, dim=1).unsqueeze(1)

        rays[:, 1] = - rays[:, 1]
        rays[:, 2] = - rays[:, 2]

        orig = self.pose[:, :3, -1].view(b, 3, 1, 1).repeat(1, 1, *self.hw)
        rays = self.no_translation().inverted_pose().to_world(rays).reshape(b, 3, *self.hw)

        info = [orig, rays]
        if near is not None:
            info = info + [near * ones]
        if far is not None:
            info = info + [far * ones]
        if gt is not None:
            info = info + [gt]

        rays = torch.cat(info, 1)
        rays = rays.permute(0, 2, 3, 1).reshape(b, -1, rays.shape[1])

        if n_rays is not None:
            idx = torch.randint(0, self.n_pixels, (n_rays,))
            rays = rays[:, idx, :]

        return rays

    def get_plucker(self):
        """Get camera plucker rays"""
        b = len(self)
        ones = torch.ones((b, 1, *self.hw), dtype=self.dtype, device=self.device)
        rays = self.reconstruct_depth_map(ones, to_world=False)
        rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        orig = self.Tcw.T[:, :3, -1].view(b, 3, 1, 1).repeat(1, 1, *self.hw)

        orig = orig.view(1, 3, -1).permute(0, 2, 1)
        rays = rays.view(1, 3, -1).permute(0, 2, 1)

        cross = torch.cross(orig, rays, dim=-1)
        plucker = torch.cat((rays, cross), dim=-1)

        return plucker

    def project_pointcloud(self, pcl_src, rgb_src, thr=1):
        """Project pointcloud to the image plane"""

        if rgb_src.dim() == 4:
            rgb_src = rgb_src.view(*rgb_src.shape[:2], -1)

        # Get projected coordinates and depth values
        uv_all, z_all = self.project_points(pcl_src, return_z=True, from_world=True)

        rgbs_tgt, depths_tgt = [], []

        b = pcl_src.shape[0]
        for i in range(b):
            uv, z = uv_all[i].reshape(-1, 2), z_all[i].reshape(-1, 1)

            # Remove out-of-bounds coordinates and points behind the camera
            idx = (uv[:, 0] >= -1) & (uv[:, 0] <= 1) & \
                  (uv[:, 1] >= -1) & (uv[:, 1] <= 1) & (z[:, 0] > 0.0)

            # Unormalize and stack coordinates for scatter operation
            uv = (unnorm_pixel_grid(uv[idx], self.hw)).round().long()
            uv = uv[:, 0] + uv[:, 1] * self.hw[1]

            # Min scatter operation (only keep the closest depth)
            depth_tgt = 1e10 * torch.ones((self.hw[0] * self.hw[1], 1), device=pcl_src.device)
            depth_tgt, argmin = scatter_min(src=z[idx], index=uv.unsqueeze(1), dim=0, out=depth_tgt)
            depth_tgt[depth_tgt == 1e10] = 0.

            num_valid = (depth_tgt > 0).sum()
            if num_valid > thr:

                # Substitute invalid values with zero
                invalid = argmin == argmin.max()
                argmin[invalid] = 0
                rgb_tgt = rgb_src[i].permute(1, 0)[idx][argmin]
                rgb_tgt[invalid] = -1

            else:

                rgb_tgt = -1 * torch.ones(1, self.n_pixels, 3, device=self.device, dtype=self.dtype)

            # Reshape outputs
            rgb_tgt = rgb_tgt.reshape(1, self.hw[0], self.hw[1], 3).permute(0, 3, 1, 2)
            depth_tgt = depth_tgt.reshape(1, 1, self.hw[0], self.hw[1])

            rgbs_tgt.append(rgb_tgt)
            depths_tgt.append(depth_tgt)

        rgb_tgt = torch.cat(rgbs_tgt, 0)
        depth_tgt = torch.cat(depths_tgt, 0)

        return rgb_tgt, depth_tgt

    def reconstruct_depth_map_rays(self, depth, to_world=False):
        """Reconstruct 3D points from depth map"""
        if depth is None:
            return None
        b, _, h, w = depth.shape
        rays = self.get_viewdirs(normalize=True, to_world=False)
        points = (rays * depth).view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        return points.view(b, 3, h, w)


class CameraPinhole(CameraBase):
    """Pinhole camera model"""
    def __init__(self, K, *args, **kwargs):
        # Intrinsics
        if same_shape(K.shape[-2:], (3, 3)):
            self._K = torch.eye(4, dtype=K.dtype, device=K.device).repeat(K.shape[0], 1, 1)
            self._K[:, :3, :3] = K
        else:
            self._K = K
        super().__init__(*args, **kwargs)

        self.convert_matrix = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0)

    @staticmethod
    def from_list(cams):
        """Create a single camera from a list of cameras"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return CameraPinhole(K=K, Twc=Twc, hw=cams[0].hw)

    @staticmethod
    def from_dict(K, hw, Twc=None, broken=False):
        """Create a single camera from a dictionary of cameras"""
        if broken:
            return {key: CameraPinhole(
                K=K[key] if key in K else K[(0, key[1])],
                hw=hw[key], Twc=val
            ) for key, val in Twc.items()}
        else:
            return {key: CameraPinhole(
                K=K[key] if key in K else K[0],
                hw=hw[key], Twc=val
            ) for key, val in Twc.items()}

    @property
    def fx(self):
        """Focal length in x direction"""
        return self._K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y direction"""
        return self._K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x direction"""
        return self._K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y direction"""
        return self._K[:, 1, 2]

    @property
    def fxy(self):
        """Focal length in x and y direction"""
        return torch.tensor([self.fx, self.fy], dtype=self.dtype, device=self.device)

    @property
    def cxy(self):
        """Principal point in x and y direction"""
        return torch.tensor([self.cx, self.cy], dtype=self.dtype, device=self.device)

    @property
    def invK(self):
        """Inverse of camera intrinsics"""
        return invert_intrinsics(self._K)

    def offset_start(self, start):
        """Offset the principal point"""
        new_cam = self.clone()
        if is_seq(start):
            new_cam.K[:, 0, 2] -= start[1]
            new_cam.K[:, 1, 2] -= start[0]
        else:
            start = start.to(self.device)
            new_cam.K[:, 0, 2] -= start[:, 1]
            new_cam.K[:, 1, 2] -= start[:, 0]
        return new_cam

    def scaled(self, scale_factor):
        """Scale the camera intrinsics"""
        if scale_factor is None or scale_factor == 1:
            return self
        if is_seq(scale_factor):
            if len(scale_factor) == 4:
                scale_factor = scale_factor[-2:]
            scale_factor = [float(scale_factor[i]) / float(self._hw[i]) for i in range(2)]
        else:
            scale_factor = [scale_factor] * 2
        return type(self)(
            K=scale_intrinsics(self._K, scale_factor),
            hw=[int(self._hw[i] * scale_factor[i]) for i in range(len(self._hw))],
            Twc=self._Twc
        )

    def lift(self, grid):
        """Lift a grid of points to 3D"""
        return torch.matmul(self.invK[:, :3, :3], grid)

    def unlift(self, points, from_world=True, euclidean=False):
        """Unlift a grid of points to 2D"""
        projected = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))
        coords = projected[:, :2] / (projected[:, 2].unsqueeze(1) + 1e-7)
        if not euclidean:
            depth = projected[:, 2]
        else:
            points = self.from_world(points) if from_world else points
            depth = torch.linalg.vector_norm(points, dim=1, keepdim=True)
        return coords, depth

    def switch(self):
        """Switch the camera from world to camera coordinates"""
        T = self.convert_matrix.to(self.device)
        Twc = T @ self.Twc.T @ T
        return type(self)(K=self.K, Twc=Twc, hw=self.hw)

    def bwd(self):
        """Switch the camera from world to camera coordinates"""
        T = self.convert_matrix.to(self.device)
        Tcw = T @ self.Twc.T @ T
        return type(self)(K=self.K, Tcw=Tcw, hw=self.hw)

    def fwd(self):
        """Switch the camera from world to camera coordinates"""
        T = self.convert_matrix.to(self.device)
        Twc = T @ self.Tcw.T @ T
        return type(self)(K=self.K, Twc=Twc, hw=self.hw)

    def up(self):
        """Get the up vector of the camera"""
        up = self.clone()
        up.Twc.translateUp(1)
        return up.get_center() - self.get_center()

    def forward(self):
        """Get the forward vector of the camera"""
        forward = self.clone()
        forward.Twc.translateForward(1)
        return forward.get_center() - self.get_center()

    def look_at(self, at, up=None):
        """Look at a point"""

        if up is None:
            up = self.up()

        eps = 1e-5
        eye = self.get_center()

        at = at.unsqueeze(0)
        up = up.unsqueeze(0).to(at.device)
        up /= up.norm(dim=-1, keepdim=True) + eps

        z_axis = at - eye
        z_axis /= z_axis.norm(dim=-1, keepdim=True) + eps

        up = up.expand(z_axis.shape)
        x_axis = torch.cross(up, z_axis)
        x_axis /= x_axis.norm(dim=-1, keepdim=True) + eps

        y_axis = torch.cross(z_axis, x_axis)
        y_axis /= y_axis.norm(dim=-1, keepdim=True) + eps

        R = torch.stack((x_axis, y_axis, z_axis), dim=-1)

        Tcw = self.Tcw
        Tcw.T[:, :3, :3] = R
        self.Twc = Tcw.inverse()

    def get_center(self):
        """Get the center of the camera"""
        return self.Tcw.T[:, :3, -1]

    def get_origin(self, flatten=False):
        """Get the origin of the camera"""
        orig = self.get_center().view(len(self), 3, 1, 1).repeat(1, 1, *self.hw)
        if flatten:
            orig = orig.reshape(len(self), 3, -1).permute(0, 2, 1)
        return orig

    def get_viewdirs(self, normalize=None, to_world=None, flatten=False, reflect=False, grid=None):
        """Get the view directions of the camera"""

        ones = torch.ones((len(self), 1, *self.hw), dtype=self.dtype, device=self.device)
        rays = self.reconstruct_depth_map(ones, to_world=False, grid=grid)

        if reflect:
            rays[:, 1] = - rays[:, 1]
            rays[:, 2] = - rays[:, 2]

        if normalize is True or normalize == 'unit':
            rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        if normalize == 'plane':
            rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
            rays = rays / rays[:, [2]]

        if to_world:
            # rays = self.to_world(rays).reshape(len(self), 3, *self.hw)
            rays = self.no_translation().to_world(rays).reshape(len(self), 3, *self.hw)

        if flatten:
            rays = rays.reshape(len(self), 3, -1).permute(0, 2, 1)

        return rays

    def get_render_rays(self, near=None, far=None, n_rays=None, gt=None):
        """Get the rays for rendering"""

        b = len(self)

        ones = torch.ones((b, 1, *self.hw), dtype=self.dtype, device=self.device)

        rays = self.reconstruct_depth_map(ones, to_world=False)
        rays = rays / torch.norm(rays, dim=1).unsqueeze(1)

        rays[:, 1] = - rays[:, 1]
        rays[:, 2] = - rays[:, 2]

        orig = self.pose[:, :3, -1].view(b, 3, 1, 1).repeat(1, 1, *self.hw)
        rays = self.no_translation().inverted_pose().to_world(rays).reshape(b, 3, *self.hw)

        info = [orig, rays]
        if near is not None:
            info = info + [near * ones]
        if far is not None:
            info = info + [far * ones]
        if gt is not None:
            info = info + [gt]

        rays = torch.cat(info, 1)
        rays = rays.permute(0, 2, 3, 1).reshape(b, -1, rays.shape[1])

        if n_rays is not None:
            idx = torch.randint(0, self.n_pixels, (n_rays,))
            rays = rays[:, idx, :]

        return rays

    def get_plucker(self):
        """Get the plucker coordinates of the camera"""

        b = len(self)
        ones = torch.ones((b, 1, *self.hw), dtype=self.dtype, device=self.device)
        rays = self.reconstruct_depth_map(ones, to_world=False)
        rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        orig = self.get_center().view(b, 3, 1, 1).repeat(1, 1, *self.hw)

        orig = orig.view(1, 3, -1).permute(0, 2, 1)
        rays = rays.view(1, 3, -1).permute(0, 2, 1)

        cross = torch.cross(orig, rays, dim=-1)
        plucker = torch.cat((rays, cross), dim=-1)

        return plucker

    def project_pointcloud(self, pcl_src, rgb_src=None, thr=1):
        """Project a pointcloud to the image plane"""

        if rgb_src is not None and rgb_src.dim() == 4:
            rgb_src = rgb_src.view(*rgb_src.shape[:2], -1)

        # Get projected coordinates and depth values
        uv_all, z_all = self.project_points(pcl_src, return_z=True, from_world=True)

        rgbs_tgt, depths_tgt = [], []

        b = pcl_src.shape[0]
        for i in range(b):
            uv, z = uv_all[i].reshape(-1, 2), z_all[i].reshape(-1, 1)

            # Remove out-of-bounds coordinates and points behind the camera
            idx = (uv[:, 0] >= -1) & (uv[:, 0] <= 1) & \
                  (uv[:, 1] >= -1) & (uv[:, 1] <= 1) & (z[:, 0] > 0.0)

            # Unormalize and stack coordinates for scatter operation
            uv = (unnorm_pixel_grid(uv[idx], self.hw)).round().long()
            uv = uv[:, 0] + uv[:, 1] * self.hw[1]

            # Min scatter operation (only keep the closest depth)
            depth_tgt = 1e10 * torch.ones((self.hw[0] * self.hw[1], 1), device=pcl_src.device)
            depth_tgt, argmin = scatter_min(src=z[idx], index=uv.unsqueeze(1), dim=0, out=depth_tgt)
            depth_tgt[depth_tgt == 1e10] = 0.

            num_valid = (depth_tgt > 0).sum()
            if num_valid > thr:

                # Substitute invalid values with zero
                invalid = argmin == argmin.max()
                argmin[invalid] = 0
                if rgb_src is not None:
                    rgb_tgt = rgb_src[i].permute(1, 0)[idx][argmin]
                    rgb_tgt[invalid] = 0

            else:

                if rgb_src is not None:
                    rgb_tgt = -1 * torch.ones(1, self.n_pixels, 3, device=self.device, dtype=self.dtype)

            # Reshape outputs
            depth_tgt = depth_tgt.reshape(1, 1, self.hw[0], self.hw[1])
            depths_tgt.append(depth_tgt)

            if rgb_src is not None:
                rgb_tgt = rgb_tgt.reshape(1, self.hw[0], self.hw[1], 3).permute(0, 3, 1, 2)
                rgbs_tgt.append(rgb_tgt)

        if rgb_src is not None:
            rgb_tgt = torch.cat(rgbs_tgt, 0)
        else:
            rgb_tgt = None

        depth_tgt = torch.cat(depths_tgt, 0)

        return rgb_tgt, depth_tgt

    def reconstruct_depth_map_rays(self, depth, to_world=False):
        """Reconstruct a depth map from a depth image"""
        if depth is None:
            return None
        b, _, h, w = depth.shape
        rays = self.get_viewdirs(normalize=True, to_world=False)
        points = (rays * depth).view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        return points.view(b, 3, h, w)

    def to_ndc_rays(self, rays_o, rays_d, near=1.0):
        """Transform rays from camera coordinates to NDC coordinates"""
        H, W = self.hw
        focal = self.fy[0].item()

        t = - (near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d

    def from_ndc(self, xyz_ndc):
        """Transform points from NDC coordinates to camera coordinates"""
        wh = self.wh
        fx = fy = self.fy[0].item()

        z_e = 2. / (xyz_ndc[..., 2:3] - 1. + 1e-6)
        x_e = - xyz_ndc[..., 0:1] * z_e * wh[0] / (2. * fx)
        y_e = - xyz_ndc[..., 1:2] * z_e * wh[1] / (2. * fy)

        return torch.cat([x_e, y_e, z_e], -1)


class PerceiverAbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
    """Perceiver abstract position encoding."""

    @property
    @abc.abstractmethod
    def num_dimensions(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def output_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch_size, pos):
        raise NotImplementedError


class PerceiverTrainablePositionEncoding(PerceiverAbstractPositionEncoding):
    """Trainable position encoding."""

    def __init__(self, index_dims, num_channels=128):
        super().__init__()
        self._num_channels = num_channels
        self._index_dims = index_dims
        index_dim = np.prod(index_dims)
        self.position_embeddings = nn.Parameter(
            torch.randn(index_dim, num_channels))

    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kwargs) -> int:
        return self._num_channels

    def forward(self, batch_size):
        position_embeddings = self.position_embeddings

        if batch_size is not None:
            position_embeddings = position_embeddings.expand(
                batch_size, -1, -1)
        return position_embeddings

def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """
    Generate an array of position indices for an N-D input array.

    Args:
      index_dims (`List[int]`):
        The shape of the index dimensions of the input array.
      output_range (`Tuple[float]`, *optional*, defaults to `(-1.0, 1.0)`):
        The min and max values taken by each input index dimension.

    Returns:
      `torch.FloatTensor` of shape `(index_dims[0], index_dims[1], .., index_dims[-1], N)`.
    """

    def _linspace(n_xels_per_dim):
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges)

    return torch.stack(array_index_grid, dim=-1)

def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """
    Checks or builds spatial position features (x, y, ...).

    Args:
      pos (`torch.FloatTensor`):
        None, or an array of position features. If None, position features are built. Otherwise, their size is checked.
      index_dims (`List[int]`):
        An iterable giving the spatial/index size of the data to be featurized.
      batch_size (`int`):
        The batch size of the data to be featurized.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, prod(index_dims))` an array of position features.
    """
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
    # else:
    #     # Just a warning label: you probably don't want your spatial features to
    #     # have a different spatial layout than your pos coordinate system.
    #     # But feel free to override if you think it'll work!
    #     if pos.shape[-1] != len(index_dims):
    #         raise ValueError("Spatial features have the wrong number of dimensions.")
    return pos

def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
    """
    Generate a Fourier frequency position encoding with linear spacing.

    Args:
      pos (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`):
        The Tensor containing the position of n points in d dimensional space.
      num_bands (`int`):
        The number of frequency bands (K) to use.
      max_resolution (`Tuple[int]`, *optional*, defaults to (224, 224)):
        The maximum resolution (i.e. the number of pixels per dim). A tuple representing resolution for each dimension.
      concat_pos (`bool`, *optional*, defaults to `True`):
        Whether to concatenate the input position encoding to the Fourier features.
      sine_only (`bool`, *optional*, defaults to `False`):
        Whether to use a single phase (sin) or two (sin/cos) for each frequency band.

    Returns:
      `torch.FloatTensor` of shape `(batch_size, sequence_length, n_channels)`: The Fourier position embeddings. If
      `concat_pos` is `True` and `sine_only` is `False`, output dimensions are ordered as: [dim_1, dim_2, ..., dim_d,
      sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ..., sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d), cos(pi*f_1*dim_1),
      ..., cos(pi*f_K*dim_1), ..., cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)], where dim_i is pos[:, i] and f_k is the
      kth frequency band.
    """

    batch_size = pos.shape[0]
    device = pos.device

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device) for res in max_resolution], dim=0
    )

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(
        per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat(
            [pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features

class PerceiverFourierPositionEncoding(PerceiverAbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def forward(self, index_dims, batch_size, device, pos=None):
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device)
        return fourier_pos_enc


def build_position_encoding(
    position_encoding_type,
    out_channels=None,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None,
):
    """
    Builds the position encoding.

    Args:

    - out_channels: refers to the number of channels of the position encodings.
    - project_pos_dim: if specified, will project the position encodings to this dimension.

    """

    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError(
                "Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = PerceiverTrainablePositionEncoding(
            **trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        # We don't use the index_dims argument, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError(
                "Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = PerceiverFourierPositionEncoding(
            **fourier_position_encoding_kwargs)
    else:
        raise ValueError(
            f"Unknown position encoding type: {position_encoding_type}.")

    # Optionally, project the position encoding to a target dimension:
    positions_projection = nn.Linear(
        out_channels, project_pos_dim) if project_pos_dim > 0 else nn.Identity()

    return output_pos_enc, positions_projection

import torchvision.models as models
RESNET_VERSIONS = { # general resblock
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}

RESNET_WEIGHTS = 'IMAGENET1K_V1'

class ResNetMultiInput(models.ResNet, ABC):
    def __init__(self, block_type, block_channels, num_input_rgb):
        """Multi-input ResNet model

        Parameters
        ----------
        block_type : nn.Module
            Residual block type
        block_channels : int
            Number of channels in each block
        num_input_rgb : int
            Number of input images
        """
        super().__init__(block_type, block_channels)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_rgb * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_type,  64, block_channels[0])
        self.layer2 = self._make_layer(block_type, 128, block_channels[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, block_channels[2], stride=2)
        self.layer4 = self._make_layer(block_type, 512, block_channels[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multi_input(num_layers, num_input_rgb, pretrained=True):
    """Generates a multi-input ResNet model

    Parameters
    ----------
    num_layers : int
        Number of layers in the ResNet
    num_input_rgb : int
        Number of input images
    pretrained : bool, optional
        True is pre-trained ImageNet weights are used, by default True

    Returns
    -------
    nn.Module
        Multi-input ResNet model
    """
    assert num_layers in [18, 50], 'Can only run with 18 or 50 layer resnet'

    block_channels = {
        18: [2, 2, 2, 2],
        50: [3, 4, 6, 3]
    }[num_layers]

    block_type = {
        18: models.resnet.BasicBlock,
        50: models.resnet.Bottleneck
    }[num_layers]

    model = ResNetMultiInput(block_type, block_channels, num_input_rgb)

    if pretrained:
        loaded = RESNET_VERSIONS[num_layers](weights=RESNET_WEIGHTS).state_dict()
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_rgb, 1) / num_input_rgb
        model.load_state_dict(loaded)

    return model

'''
appends spatial co-ordinates to the feature map
the range of the spatial co-ordinates can be defined as [min_val, max_val]

Arguments:
    feature_maps : input feature maps (N,C,H,W)
    min_val : minimum value of spatial co-ordinate
    max_val : maximum value of spatial co-ordinate

Returns:
    feature_maps : input feature maps concatenated with spatial co-ordinates (N,C+2,H,W) 
'''
def append_spatial_location(feature_maps, min_val=-1, max_val=1):
    batch_size, channels, height, width = feature_maps.data.shape

    # arange spatial co-ordiantes for height
    h_array = Variable(torch.stack([torch.linspace(min_val, max_val, height)]*width, dim=1).cuda())
    # arange spatial co-ordinates for width
    w_array = Variable(torch.stack([torch.linspace(min_val, max_val, width)]*height, dim=0).cuda())
    # stack the h_array and w_array to get the spatial co-ordinates at each location
    spatial_array = torch.stack([h_array, w_array], dim=0)
    # expand the spatial co-ordinates across the batch size
    spatial_array = torch.stack([spatial_array]*batch_size, dim=0)
    # concatenate feature maps with spatial co-ordinates
    feature_maps = torch.cat([feature_maps, spatial_array], dim=1)

    return feature_maps


class DynamicFC(nn.Module):

    def __init__(self):
        super(DynamicFC, self).__init__()

        self.in_planes = None
        self.out_planes = None
        self.activation = None
        self.use_bias = None

        self.activation = None
        self.linear = None

    '''
    Arguments:
        embedding : input to the MLP (N,*,C)
        out_planes : total ch      dddannels in the output
        activation : 'relu' or 'tanh'
        use_bias : True / False

    Returns:
        out : output of the MLP (N,*,out_planes)
    '''
    def forward(self, embedding, out_planes=1, activation=None, use_bias=True):

        self.in_planes = embedding.data.shape[-1]
        self.out_planes = out_planes
        self.use_bias = use_bias

        self.linear = nn.Linear(self.in_planes, self.out_planes, bias=use_bias).cuda()
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True).cuda()
        elif activation == 'tanh':
            self.activation = nn.Tanh().cuda()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if self.use_bias:
                    nn.init.constant(m.bias, 0.1)

        out = self.linear(embedding)
        if self.activation is not None:
            out = self.activation(out)

        return out


'''
A very basic FiLM layer with a linear transformation from context to FiLM parameters
'''
class FilmLayer(nn.Module):
    
    def __init__(self, feature_size):
        super(FilmLayer, self).__init__()

        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None
        self.feature_size = feature_size

        # self.fc = DynamicFC().cuda()
        self.fc = nn.Linear(768, self.feature_size * 2).cuda() # Hardcode language embedding shape for now

    '''
    Arguments:
        feature_maps : input feature maps (N,C,H,W)
        context : context embedding (N,L)

    Return:
        output : feature maps modulated with betas and gammas (FiLM parameters)
    '''
    def forward(self, feature_maps, context):
        self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape
        # FiLM parameters needed for each channel in the feature map
        # hence, feature_size defined to be same as no. of channels
        # self.feature_size = feature_maps.data.shape[1]

        # import pdb; pdb.set_trace()

        # linear transformation of context to FiLM parameters
        # film_params = self.fc(context, out_planes=2*self.feature_size, activation=None)
        film_params = self.fc(context)

        # stack the FiLM parameters across the spatial dimension
        film_params = torch.stack([film_params]*self.height, dim=2)
        film_params = torch.stack([film_params]*self.width, dim=3)

        # slice the film_params to get betas and gammas
        gammas = film_params[:, :self.feature_size, :, :]
        betas = film_params[:, self.feature_size:, :, :]

        # modulate the feature map with FiLM parameters
        output = (1 + gammas) * feature_maps + betas

        return output

'''
Modualted ResNet block with FiLM layer 
'''
class FilmResBlock(nn.Module):

    '''
    Arguments:
        in_channels : no. of channels in the input
        feature_size : feature size required
        spatial_location : whether to append spatial co-ordinates
    '''
    def __init__(self, in_channels, feature_size, spatial_location=True):
        super(FilmResBlock, self).__init__()

        self.spatial_location = spatial_location
        self.feature_size = feature_size
        self.in_channels = in_channels
        # add 2 channels for spatial location
        if spatial_location:
            self.in_channels += 2

        # modulated resnet block with FiLM layer
        self.conv1 = nn.Conv2d(self.in_channels, self.feature_size, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.feature_size)
        self.film_layer = FilmLayer(self.feature_size).cuda()
        self.relu2 = nn.ReLU()

    '''
    Arguments:
        feature_maps : input feature maps (N,C,H,W)
        context : context embedding (N,L)

    Returns:
        out : input feature maps modulated with FiLM parameters computed using context embedding
    '''
    def forward(self, feature_maps, context):

        if self.spatial_location:
            feature_maps = append_spatial_location(feature_maps)

        conv1 = self.conv1(feature_maps)
        out1 = self.relu1(conv1)

        conv2 = self.conv2(out1)
        bn = self.bn2(conv2)
        film_out = self.film_layer(bn, context)
        out2 = self.relu2(film_out)
        
        # residual connection
        out = out1 + out2

        return out


class ResNetFiLMFeaturizer(nn.Module, ABC):
    """
    Single-frame depth encoder

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, version=18, num_rgb_in=1, pretrained=False):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        # features = []

        assert version in RESNET_VERSIONS, f'Invalid ResNet version: {version}'

        if num_rgb_in > 1:
            self.encoder = resnet_multi_input(
                version, num_rgb_in, pretrained)
        else:
            self.encoder = RESNET_VERSIONS[version](
                weights=None if not pretrained else RESNET_WEIGHTS)

        if version > 34:
            self.num_ch_enc[1:] *= 4
            
        self.film1 = FilmLayer(self.num_ch_enc[1]).cuda()
        self.film2 = FilmLayer(self.num_ch_enc[2]).cuda()
        self.film3 = FilmLayer(self.num_ch_enc[3]).cuda()
        self.film4 = FilmLayer(self.num_ch_enc[4]).cuda()

    def forward(self, input_image, context):
        """Network forward pass"""

        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        features.clear()
        features.append(self.encoder.relu(x))
        if context is None:
            features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
            features.append(self.encoder.layer2(features[-1]))
            features.append(self.encoder.layer3(features[-1]))
            features.append(self.encoder.layer4(features[-1]))
        else:
            features.append(self.film1(self.encoder.layer1(self.encoder.maxpool(features[-1])), context))
            features.append(self.film2(self.encoder.layer2(features[-1]), context))
            features.append(self.film3(self.encoder.layer3(features[-1]), context))
            features.append(self.film4(self.encoder.layer4(features[-1]), context))

        return features
    

class ResNetFiLMEncoder(nn.Module, ABC):
    """
    Single-frame depth encoder

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, pretrained=False, input_channel=3):
        super().__init__()
        self.resnet_film_featurizer = ResNetFiLMFeaturizer(version=50, pretrained=pretrained, num_rgb_in=int(input_channel / 3))

    def forward(self, image, lang_emb):
        """Network forward pass"""
        film_features =  self.resnet_film_featurizer(image, lang_emb)[-1]
        film_features_flat = film_features.view(*film_features.shape[:-2], -1).permute(0, 2, 1)
        return film_features_flat

    def output_shape(self, input_shape):
        return ( (input_shape[1] // 29) * (input_shape[2] // 29), 2048)
    


class DeFiNeImageCamEncoder(nn.Module, ABC):
    def __init__(self, pretrained=False, input_coord_conv=False, use_cam=True, downsample=False, input_channel=3):
        super().__init__()
        
        """
        pulled these from here:
        https://github.com/TRI-ML/vidar/blob/6fdbee65bff2cc94a42aaedd121a2179ac48d441/configs/papers/define/hub_define_temporal.yaml#L21
        """
        num_bands_orig = 20
        num_bands_dirs = 10
        max_resolution_orig = 60
        max_resolution_dirs = 60

        # whether to use camera information (intrinsics + extrinsics) in the encoder
        self.use_cam = use_cam
        self.downsample = downsample

        self.image_feature_encoder = ResNetFiLMFeaturizer(version=50, pretrained=pretrained, num_rgb_in=int(input_channel / 3))

        # Fourier encoding for camera origin
        self._fourier_encoding_orig, _ = build_position_encoding(
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs={
                'num_bands': num_bands_orig,
                'max_resolution': [max_resolution_orig] * 3,
                'concat_pos': True,
                'sine_only': False,
            }
        )

        # Fourier encoding for camera viewing rays
        self._fourier_encoding_dirs, _ = build_position_encoding(
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs={
                'num_bands': num_bands_dirs,
                'max_resolution': [num_bands_dirs] * 3,
                'concat_pos': True,
                'sine_only': False,
            }
        )

        if self.downsample:
            # Adding in another Conv2 layer that is applied [H/4, W/4, features] to
            # reduce size
            self.num_cam_channels = 186
            self.num_image_channels = 3840 # ResNet50
            if self.use_cam:
                num_channels = self.num_image_channels + self.num_cam_channels
            else:
                num_channels = self.num_image_channels
            self.finalconv = torch.nn.Conv2d(num_channels, 1024, 2, stride=2)


    def get_rgb_feat(self, rgb, lang=None, rgb_feat_type="resnet_all"):
        """Exract image features"""
        if rgb_feat_type == 'convnet':
            return {
                'feat': self.image_feature_encoder(rgb, lang)
            }
        elif rgb_feat_type == 'resnet':
            return {
                'feat': self.image_feature_encoder(rgb, lang)[1]
            }
        elif rgb_feat_type.startswith('resnet_all'):
            all_feats = self.image_feature_encoder(rgb, lang)
            feats = all_feats[1:]
            for i in range(1, len(feats)):
                feats[i] = interpolate(
                    feats[i], size=feats[0], scale_factor=None, mode='bilinear')
            if rgb_feat_type.endswith('rgb'):
                feats = feats + [interpolate(
                    rgb, size=feats[0], scale_factor=None, mode='bilinear')]
            feat = torch.cat(feats, 1)
            return {
                'all_feats': all_feats,
                'feat': feat
            }


    def embeddings(self, data, sources, downsample):
        """Compute embeddings for encoder and decoder data"""
        if 'rgb' in sources:
            assert 'rgb' in data[0].keys()
            b = [datum['rgb'].shape[0] for datum in data]

            rgb = torch.cat([datum['rgb'] for datum in data], 0)

            if 'lang' in sources:
                lang = torch.cat([datum['lang'] for datum in data], 0)
                output_feats = self.get_rgb_feat(rgb, lang)
            else:
                output_feats = self.get_rgb_feat(rgb)

            feats = torch.split(output_feats['feat'], b)
            for i in range(len(data)):
                data[i]['feat'] = feats[i]

        encodings = []
        for datum in data:

            encoding = OrderedDict()

            # Camera embeddings
            if 'cam' in sources:
                assert 'cam' in data[0].keys()  

                cam = datum['cam'].scaled(1. / downsample)
                orig = cam.get_origin(flatten=True)

                to_world = True # should it be True or False? idk
                
                if to_world:
                    dirs = cam.get_viewdirs(normalize=True, flatten=True, to_world=True)
                else:
                    dirs = cam.no_translation().get_viewdirs(normalize=True, flatten=True, to_world=True)

                orig_encodings = self._fourier_encoding_orig(
                    index_dims=None, pos=orig, batch_size=orig.shape[0], device=orig.device)
                dirs_encodings = self._fourier_encoding_dirs(
                    index_dims=None, pos=dirs, batch_size=dirs.shape[0], device=dirs.device)

                encoding['cam'] = torch.cat([orig_encodings, dirs_encodings], -1)

            # Image embeddings
            if 'rgb' in sources:
                rgb = datum['feat']
                rgb_flat = rgb.view(*rgb.shape[:-2], -1).permute(0, 2, 1)
                encoding['rgb'] = rgb_flat

            # Adding in another Conv2 layer that is applied [H/4, W/4, features] to
            # reduce size
            if self.downsample:
                stacked = torch.cat([val for val in encoding.values()], -1)
                st_sh = stacked.shape
                stacked = stacked.reshape(st_sh[0], int(np.sqrt(st_sh[1])), int(np.sqrt(st_sh[1])), st_sh[-1])
                stacked = self.finalconv(stacked.permute(0, 3, 1, 2))
                encoding['all'] = stacked.reshape(stacked.shape[0], stacked.shape[1], -1).permute(0, 2, 1)
            else:
                encoding['all'] = torch.cat([val for val in encoding.values()], -1)
            encodings.append(encoding)

        return encodings
    
    def forward(self, image, extrinsics, intrinsics, lang_emb):
        if extrinsics is None or intrinsics is None:
            sources = ["rgb"] # camera info not present
            data = [{
                "rgb": image,
            }]
        else:
            sources = ["rgb", "cam"]
            cam = CameraPinhole(
                K=intrinsics,
                Twc=Pose(extrinsics).to(image.device),
                hw=image.shape[-2:],
            )
            data = [{
                "cam": cam,
                "rgb": image,
            }]
        if lang_emb is not None:
            sources += ["lang"]
            data[0]["lang"] = lang_emb 

        emb = self.embeddings(
            data,
            sources=sources,
            downsample=4,
        )
        result = torch.cat([emb[i]["all"] for i in range(len(emb))])
        return result
    
    def output_shape(self, input_shape):
        if self.downsample:
            # num_cam_channels = 186
            # num_image_channels = 3840 # ResNet18: 512 + 256 + 128 + 64
            num_sptial_feats = int(np.floor((input_shape[1] / 8))) * int(np.floor(input_shape[1] / 8))
            # Updated to have the shapes outputted by the final conv2.
            return (num_sptial_feats, 1024)
        else:
            num_cam_channels = 186
            num_image_channels = 3840 # ResNet18: 512 + 256 + 128 + 64
            num_sptial_feats = int(input_shape[1] / 4 * input_shape[2] / 4)
            if self.use_cam:
                num_channels = num_image_channels + num_cam_channels
            else:
                num_channels = num_image_channels
            return (num_sptial_feats, num_channels)


class VoltronImageCamEncoder(nn.Module, ABC):
    def __init__(self, pretrained=False, input_coord_conv=False, use_cam=True, downsample=False, input_channel=3):
        super().__init__()

        from vtx import load
        from torchvision import transforms as T
        
        from torchvision.transforms.functional import InterpolationMode

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.img_transform, self.ctcn = load("ctcn-base+x7", "/mnt/fsx/x-voltron/ctcn-base+x7/ctcn-base+x7.pt", device=device, freeze_backbone=False)
    
    def forward(self, image, lang_emb):
        all_langs = []
        for b in range(len(lang_emb[0])):
            for h in range(len(lang_emb)):
                all_langs.append(lang_emb[h][b])
        processed_images = torch.stack([self.img_transform(i) for i in image]).to(self.device)
        embedding: torch.Tensor = self.ctcn.embed(processed_images, all_langs)
        return embedding
    
    def output_shape(self, input_shape):
        return (768)


class R3MImageCamEncoder(nn.Module, ABC):
    def __init__(self, pretrained=False, input_coord_conv=False, use_cam=True, downsample=False, input_channel=3):
        super().__init__()

        from r3m import load_r3m
        from torchvision import transforms as T

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.r3m = load_r3m("resnet50") # resnet18, resnet34
        self.r3m.train()
        self.r3m.to(self.device)

        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ]) 
    
    def forward(self, image, lang_emb):
        processed_images = torch.stack([self.transforms(i) for i in image]).to(self.device)
        embedding: torch.Tensor = self.r3m(processed_images * 255.0)
        return embedding
    
    def output_shape(self, input_shape):
        return (2048,)

class VC1ImageCamEncoder(nn.Module, ABC):
    def __init__(self, pretrained=False, input_coord_conv=False, use_cam=True, downsample=False, input_channel=3):
        super().__init__()

        from vc_models.models.vit import model_utils
        self.model, self.embd_size, self.model_transforms, self.model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
        self.model.train()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
    
    def forward(self, image, lang_emb):
        processed_images = self.model_transforms(image)
        embedding: torch.Tensor = self.model(processed_images)
        return embedding
    
    def output_shape(self, input_shape):
        return (768,)