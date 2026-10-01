import unittest

import torch

from robomimic.models.distributions import DiscreteValueDistribution


class DiscreteValueSamplingTests(unittest.TestCase):
    def test_shared_support_and_multidimensional_samples(self):
        values = torch.tensor([-7., 3., 11.], dtype=torch.float64)
        probs = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
        dist = DiscreteValueDistribution(values, probs=probs)
        for shape in (torch.Size(), torch.Size([5]), torch.Size([2, 3])):
            with self.subTest(shape=shape):
                samples = dist.sample(shape)
                expected = torch.tensor([3., 11.], dtype=values.dtype).expand(shape + (2,))
                torch.testing.assert_close(samples, expected)

    def test_batched_supports_follow_broadcast_mean(self):
        values = torch.tensor([[[-4., 8.]], [[2., 9.]]])
        probs = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        dist = DiscreteValueDistribution(values, probs=probs)
        expected = dist.mean()
        samples = dist.sample(torch.Size([2, 4]))
        torch.testing.assert_close(samples, expected.expand(2, 4, 2, 3))

    def test_unbatched_logits_broadcast_over_value_supports(self):
        values = torch.tensor([[10., 20.], [-10., -20.]])
        dist = DiscreteValueDistribution(values, logits=torch.tensor([-torch.inf, 0.]))
        torch.testing.assert_close(dist.sample(), torch.tensor([20., -20.]))

    def test_sample_moments_match_value_distribution(self):
        torch.manual_seed(1234)
        dist = DiscreteValueDistribution(
            torch.tensor([-2., 1., 5.], dtype=torch.float64),
            probs=torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64),
        )
        samples = dist.sample(torch.Size([100000]))
        self.assertTrue(torch.isin(samples, dist.values).all())
        torch.testing.assert_close(samples.mean(), dist.mean(), atol=0.03, rtol=0)
        torch.testing.assert_close(samples.var(unbiased=False), dist.variance(), atol=0.08, rtol=0)


if __name__ == "__main__":
    unittest.main()
