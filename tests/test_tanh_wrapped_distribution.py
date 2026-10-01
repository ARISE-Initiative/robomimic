import math
import unittest

import torch
from torch import distributions as D

from robomimic.models.distributions import TanhWrappedDistribution


class TanhJacobianTests(unittest.TestCase):
    def test_matches_transformed_distribution_with_scaled_actions(self):
        z = torch.tensor([[-1.2, -0.2, 0.7], [0.4, 1.1, -0.9]], dtype=torch.float64)
        for event_dims in (0, 1, 2):
            for scale in (0.25, 1., 3., -2.):
                with self.subTest(event_dims=event_dims, scale=scale):
                    base = D.Normal(torch.zeros_like(z), torch.ones_like(z))
                    if event_dims:
                        base = D.Independent(base, event_dims)
                    dist = TanhWrappedDistribution(base, scale=scale)
                    oracle = D.TransformedDistribution(
                        base, [D.TanhTransform(cache_size=1), D.AffineTransform(0., scale)],
                    )
                    value = scale * z.tanh()
                    expected = oracle.log_prob(value)
                    torch.testing.assert_close(dist.log_prob(value, z), expected)
                    torch.testing.assert_close(dist.log_prob(value), expected)

    def test_saturated_latents_keep_exact_density_and_gradient(self):
        z = torch.tensor([-40., -15., 15., 40.], dtype=torch.float64, requires_grad=True)
        base = D.Normal(torch.zeros_like(z), torch.ones_like(z))
        dist = TanhWrappedDistribution(base, scale=3.)
        loss = dist.log_prob(3. * z.tanh(), pre_tanh_value=z)
        # log(sech(z)^2) = 2 log(2) - 2 |z| - 2 log(1 + exp(-2 |z|)).
        log_jac = math.log(3.) + 2. * math.log(2.) - 2. * z.abs() - 2. * torch.log1p(torch.exp(-2. * z.abs()))
        torch.testing.assert_close(loss, base.log_prob(z) - log_jac)
        loss.sum().backward()
        torch.testing.assert_close(z.grad, -z.detach() + 2. * z.detach().tanh())

    def test_action_rescaling_obeys_density_change(self):
        base = D.Independent(D.Normal(torch.zeros(4), torch.ones(4)), 1)
        z = torch.tensor([0.1, -0.4, 0.7, 0.3])
        unit = TanhWrappedDistribution(base)
        scaled = TanhWrappedDistribution(base, scale=2.)
        torch.testing.assert_close(
            scaled.log_prob(2. * z.tanh(), z),
            unit.log_prob(z.tanh(), z) - 4. * math.log(2.),
        )

    def test_reparameterized_sampling_preserves_latent_gradients(self):
        torch.manual_seed(12)
        loc = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        dist = TanhWrappedDistribution(D.Independent(D.Normal(loc, torch.ones_like(loc)), 1), scale=2.)
        value, z = dist.rsample(torch.Size([5]), return_pretanh_value=True)
        self.assertEqual(dist.log_prob(value, z).shape, (5, 2))
        dist.log_prob(value, z).sum().backward()
        self.assertTrue(torch.isfinite(loc.grad).all())
        self.assertTrue((loc.grad.abs() > 0).any())


if __name__ == "__main__":
    unittest.main()
