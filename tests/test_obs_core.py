"""Tests for observation encoder cores and randomizers."""

import torch

from robomimic.models.obs_core import CropRandomizer


def test_crop_randomizer_eval_preserves_positional_encoding():
    inputs = torch.arange(2 * 3 * 8 * 10, dtype=torch.float32).reshape(2, 3, 8, 10)
    randomizer = CropRandomizer(
        input_shape=(3, 8, 10),
        crop_height=4,
        crop_width=6,
        num_crops=1,
        pos_enc=True,
    )
    randomizer.eval()

    outputs = randomizer.forward_in(inputs)

    assert outputs.shape == (2, 5, 4, 6)
    assert torch.equal(outputs[:, :3], inputs[:, :, 2:6, 2:8])

    expected_y = torch.arange(2, 6, dtype=torch.float32).view(4, 1).expand(4, 6) / 8
    expected_x = torch.arange(2, 8, dtype=torch.float32).view(1, 6).expand(4, 6) / 10
    assert torch.equal(outputs[0, 3], expected_y)
    assert torch.equal(outputs[0, 4], expected_x)
