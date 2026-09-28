"""Projection of a categorical value distribution onto its atom grid."""
import torch

from robomimic.utils.loss_utils import project_values_onto_atoms


def test_projection_keeps_a_value_on_the_last_atom():
    projected = project_values_onto_atoms(
        torch.tensor([[2.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([0.0, 1.0, 2.0]),
    )
    assert torch.allclose(projected, torch.tensor([[0.0, 0.0, 1.0]]))


def test_projection_splits_a_value_between_its_neighbors():
    projected = project_values_onto_atoms(
        torch.tensor([[0.2]]),
        torch.tensor([[1.0]]),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.allclose(projected, torch.tensor([[0.8, 0.2]]))


if __name__ == "__main__":
    test_projection_keeps_a_value_on_the_last_atom()
    test_projection_splits_a_value_between_its_neighbors()
