"""Hardware-free regression tests for CropRandomizer's evaluation path."""
import unittest

import torch

from robomimic.models.obs_core import CropRandomizer, VisualCore


class TestCropRandomizer(unittest.TestCase):
    def test_eval_position_encoding_uses_source_coordinates(self):
        for channels in (1, 3):
            for leading_shape in ((), (2,), (2, 3)):
                for height, width, crop_height, crop_width in ((8, 10, 4, 6), (9, 12, 4, 5)):
                    with self.subTest(channels=channels, leading_shape=leading_shape, height=height):
                        images = torch.arange(channels * height * width, dtype=torch.float32)
                        images = images.reshape(channels, height, width).expand(*leading_shape, -1, -1, -1)
                        randomizer = CropRandomizer(
                            (channels, height, width), crop_height, crop_width, pos_enc=True,
                        ).eval()
                        output = randomizer.forward_in(images)
                        top = (height - crop_height) // 2
                        left = (width - crop_width) // 2
                        expected_image = images[..., top:top + crop_height, left:left + crop_width]
                        expected_y = torch.tensor([
                            [(top + row) / height] * crop_width for row in range(crop_height)
                        ])
                        expected_x = torch.tensor([
                            [(left + col) / width for col in range(crop_width)]
                        ] * crop_height)
                        self.assertEqual(tuple(output.shape), (*leading_shape, channels + 2, crop_height, crop_width))
                        torch.testing.assert_close(output[..., :channels, :, :], expected_image)
                        torch.testing.assert_close(output[..., channels, :, :], expected_y.expand(*leading_shape, -1, -1))
                        torch.testing.assert_close(output[..., channels + 1, :, :], expected_x.expand(*leading_shape, -1, -1))
                        torch.testing.assert_close(randomizer.forward_in(images), output)

    def test_eval_without_position_encoding_preserves_center_crop(self):
        images = torch.randn(2, 3, 9, 12)
        randomizer = CropRandomizer((3, 9, 12), 4, 5, num_crops=3).eval()
        output = randomizer.forward_in(images)
        torch.testing.assert_close(output, images[..., 2:6, 3:8])
        self.assertEqual(list(output.shape[1:]), randomizer.output_shape_in())

    def test_train_and_eval_encodings_use_the_same_convention(self):
        height, width = 8, 10
        # The image itself encodes its source coordinates, independently of the randomizer.
        rows = torch.arange(height).reshape(height, 1).expand(height, width) / height
        cols = torch.arange(width).reshape(1, width).expand(height, width) / width
        images = torch.stack((rows, cols, torch.zeros_like(rows))).unsqueeze(0).expand(2, -1, -1, -1)
        randomizer = CropRandomizer((3, height, width), 4, 6, num_crops=3, pos_enc=True)
        for training in (True, False):
            with self.subTest(training=training):
                randomizer.train(training)
                output = randomizer.forward_in(images)
                self.assertEqual(output.shape[0], 6 if training else 2)
                self.assertEqual(list(output.shape[1:]), randomizer.output_shape_in())
                torch.testing.assert_close(output[:, 0], output[:, 3])
                torch.testing.assert_close(output[:, 1], output[:, 4])

    def test_eval_preserves_image_gradients_and_matches_training_dtype(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                images = torch.randn(2, 3, 8, 10, dtype=dtype, requires_grad=True)
                randomizer = CropRandomizer((3, 8, 10), 4, 6, pos_enc=True)
                training_output = randomizer.forward_in(images)
                randomizer.eval()
                output = randomizer.forward_in(images)
                self.assertEqual(output.dtype, training_output.dtype)
                self.assertEqual(output.device, images.device)
                output.sum().backward()
                expected_grad = torch.zeros_like(images)
                expected_grad[..., 2:6, 2:8] = 1
                torch.testing.assert_close(images.grad, expected_grad)

    def test_visual_core_accepts_train_and_eval_crops(self):
        torch.manual_seed(0)
        randomizer = CropRandomizer((3, 72, 80), 64, 64, num_crops=2, pos_enc=True)
        encoder = VisualCore(
            input_shape=randomizer.output_shape_in(),
            backbone_kwargs={"pretrained": False},
            feature_dimension=8,
        )
        images = torch.randn(2, 3, 72, 80)
        features = randomizer.forward_out(encoder(randomizer.forward_in(images)))
        self.assertEqual(tuple(features.shape), (2, 8))
        features.square().mean().backward()
        self.assertIsNotNone(encoder.backbone.nets[0].weight.grad)
        self.assertGreater(encoder.backbone.nets[0].weight.grad.abs().sum().item(), 0)
        randomizer.eval()
        encoder.eval()
        with torch.no_grad():
            features = randomizer.forward_out(encoder(randomizer.forward_in(images)))
        self.assertEqual(tuple(features.shape), (2, 8))
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
