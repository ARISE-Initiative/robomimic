"""
CPU tests for image crop utilities. Run with python tests/test_obs_utils.py.
"""
import unittest
from unittest.mock import patch

import torch

import robomimic.utils.obs_utils as ObsUtils


class TestImageCrops(unittest.TestCase):
    def test_crop_at_image_edges(self):
        for leading_shape in [(), (2,), (2, 3)]:
            with self.subTest(leading_shape=leading_shape):
                images = torch.arange(3 * 5 * 7).reshape(3, 5, 7)
                images = images.expand(*leading_shape, 3, 5, 7)
                indices = torch.tensor([[0, 0], [2, 0], [0, 3], [2, 3]])
                indices = indices.expand(*leading_shape, 4, 2)
                crops = ObsUtils.crop_image_from_indices(images, indices, 3, 4)
                expected = torch.stack([
                    images[..., :3, :4],
                    images[..., 2:, :4],
                    images[..., :3, 3:],
                    images[..., 2:, 3:],
                ], dim=-4)
                self.assertTrue(torch.equal(crops, expected))

    def test_single_crop_at_bottom_right(self):
        images = torch.arange(3 * 5 * 7).reshape(3, 5, 7)
        crop = ObsUtils.crop_image_from_indices(images, torch.tensor([2, 3]), 3, 4)
        self.assertTrue(torch.equal(crop, images[:, 2:, 3:]))

    def test_reject_out_of_bounds_indices(self):
        images = torch.zeros(3, 5, 7)
        for index in [[-1, 0], [0, -1], [3, 0], [0, 4]]:
            with self.subTest(index=index), self.assertRaises(AssertionError):
                ObsUtils.crop_image_from_indices(images, torch.tensor(index), 3, 4)

    def test_random_crops_can_reach_bottom_right(self):
        images = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
        # A draw near one should select the last valid origin in each dimension.
        with patch.object(torch, "rand", side_effect=lambda *shape: torch.full(shape, 0.999)):
            crops, indices = ObsUtils.sample_random_image_crops(images, 3, 4, 2)
        expected_indices = torch.tensor([2, 3]).expand(2, 2, 2)
        self.assertTrue(torch.equal(indices, expected_indices))
        expected = images[:, None, :, 2:, 3:].expand(2, 2, 3, 3, 4)
        self.assertTrue(torch.equal(crops, expected))

    def test_random_crops_matching_image_dimensions(self):
        images = torch.arange(3 * 5 * 7).reshape(3, 5, 7)
        for height, width in [(5, 4), (3, 7), (5, 7)]:
            with self.subTest(height=height, width=width):
                crops, indices = ObsUtils.sample_random_image_crops(images, height, width, 2)
                for crop, (row, col) in zip(crops, indices):
                    expected = images[:, row:row + height, col:col + width]
                    self.assertTrue(torch.equal(crop, expected))


if __name__ == "__main__":
    unittest.main()
