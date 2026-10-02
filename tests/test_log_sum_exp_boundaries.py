"""Log-space reductions preserve zero and infinite probability limits."""
import math
import unittest
import numpy as np
import torch
from robomimic.utils.loss_utils import log_sum_exp, log_mean_exp, log_normal_mixture


class LogSumExpBoundaryTest(unittest.TestCase):
    def test_infinite_limits(self):
        values = torch.tensor([[-float("inf"), -float("inf")],
                               [float("inf"), 0.],
                               [-float("inf"), math.log(3.)]], dtype=torch.float64)
        result = log_sum_exp(values, dim=1)
        self.assertEqual(result[0].item(), -float("inf"))
        self.assertEqual(result[1].item(), float("inf"))
        self.assertAlmostEqual(result[2].item(), math.log(3.))
        mean = log_mean_exp(values, dim=1)
        self.assertEqual(mean[0].item(), -float("inf"))
        self.assertEqual(mean[1].item(), float("inf"))
        self.assertAlmostEqual(mean[2].item(), math.log(1.5))

    def test_finite_forward_and_gradient_match_probabilities(self):
        values = torch.tensor([[1000., 1001., 999.], [-1000., -998., -999.]],
                              dtype=torch.float64, requires_grad=True)
        result = log_sum_exp(values, dim=-1)
        expected = []
        weights = []
        for row in values.detach().numpy():
            shift = float(max(row))
            masses = np.array([math.exp(float(x) - shift) for x in row])
            expected.append(shift + math.log(float(masses.sum())))
            weights.append(masses / masses.sum())
        torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
        result.sum().backward()
        np.testing.assert_allclose(values.grad.numpy(), np.array(weights), atol=1e-12)

    def test_overflowed_gaussian_tail_remains_negative_infinity(self):
        x = torch.tensor([[1e20]])
        means = torch.tensor([[[0.], [1.]]])
        variances = torch.ones_like(means)
        result = log_normal_mixture(x, means, variances)
        self.assertEqual(result.item(), -float("inf"))

    def test_reduction_dimensions(self):
        values = torch.tensor([[math.log(2.), math.log(3.)],
                               [math.log(5.), math.log(7.)]], dtype=torch.float64)
        torch.testing.assert_close(log_sum_exp(values, dim=0),
                                   torch.tensor([math.log(7.), math.log(10.)], dtype=torch.float64))
        torch.testing.assert_close(log_sum_exp(values, dim=1),
                                   torch.tensor([math.log(5.), math.log(12.)], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()

