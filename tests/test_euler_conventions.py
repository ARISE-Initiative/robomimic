"""Regression coverage for the public Euler-to-6D conversion API."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from robomimic.utils import torch_utils as utils

CONVENTIONS = (
    "XYZ",
    "XZY",
    "YXZ",
    "YZX",
    "ZXY",
    "ZYX",
    "XYX",
    "XZX",
    "YXY",
    "YZY",
    "ZXZ",
    "ZYZ",
)


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_euler_to_6d_matches_independent_rotation_reference(convention):
    angles = torch.tensor(
        [
            [[0.21, 0.83, -0.43], [-0.51, 1.17, 0.29]],
            [[0.71, 0.63, -0.31], [0.14, 0.92, 0.55]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = utils.euler_angles_to_rot_6d(angles, convention=convention)
    matrices = utils.rotation_6d_to_matrix(result)
    # SciPy uppercase conventions are intrinsic, matching R_a R_b R_c.
    expected = (
        Rotation.from_euler(convention, angles.detach().numpy().reshape(-1, 3))
        .as_matrix()
        .reshape(2, 2, 3, 3)
    )
    np.testing.assert_allclose(
        matrices.detach().numpy(), expected, atol=1e-12, rtol=1e-12
    )
    recovered = utils.rot_6d_to_euler_angles(result, convention=convention)
    recovered_matrices = utils.euler_angles_to_matrix(recovered, convention)
    torch.testing.assert_close(recovered_matrices, matrices, atol=1e-12, rtol=1e-12)
    # Preserve differentiability through the public conversion pair.
    weighted = matrices * torch.arange(9, dtype=torch.float64).reshape(3, 3)
    weighted.sum().backward()
    assert torch.isfinite(angles.grad).all()
    assert (angles.grad.abs().sum(-1) > 0).all()


def test_default_convention_remains_xyz():
    angles = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float64)
    torch.testing.assert_close(
        utils.euler_angles_to_rot_6d(angles),
        utils.euler_angles_to_rot_6d(angles, "XYZ"),
    )


@pytest.mark.parametrize("convention", ["BAD", "XXZ", "XY"])
def test_invalid_convention_is_rejected(convention):
    with pytest.raises(ValueError):
        utils.euler_angles_to_rot_6d(
            torch.tensor([[0.2, -0.4, 0.7]]), convention=convention
        )
