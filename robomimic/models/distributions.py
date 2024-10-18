"""
Contains distribution models used as parts of other networks. These
classes usually inherit or emulate torch distributions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class TanhWrappedDistribution(D.Distribution):
    """
    Class that wraps another valid torch distribution, such that sampled values from the base distribution are
    passed through a tanh layer. The corresponding (log) probabilities are also modified accordingly.
    Tanh Normal distribution - adapted from rlkit and CQL codebase
    (https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/distributions.py#L6).
    """
    def __init__(self, base_dist, scale=1.0, epsilon=1e-6):
        """
        Args:
            base_dist (Distribution): Distribution to wrap with tanh output
            scale (float): Scale of output
            epsilon (float): Numerical stability epsilon when computing log-prob.
        """
        self.base_dist = base_dist
        self.scale = scale
        self.tanh_epsilon = epsilon
        super(TanhWrappedDistribution, self).__init__()

    def log_prob(self, value, pre_tanh_value=None):
        """
        Args:
            value (torch.Tensor): some tensor to compute log probabilities for
            pre_tanh_value: If specified, will not calculate atanh manually from @value. More numerically stable
        """
        value = value / self.scale
        if pre_tanh_value is None:
            one_plus_x = (1. + value).clamp(min=self.tanh_epsilon)
            one_minus_x = (1. - value).clamp(min=self.tanh_epsilon)
            pre_tanh_value = 0.5 * torch.log(one_plus_x / one_minus_x)
        lp = self.base_dist.log_prob(pre_tanh_value)
        tanh_lp = torch.log(1 - value * value + self.tanh_epsilon)
        # In case the base dist already sums up the log probs, make sure we do the same
        return lp - tanh_lp if len(lp.shape) == len(tanh_lp.shape) else lp - tanh_lp.sum(-1)

    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.base_dist.sample(sample_shape=sample_shape).detach()

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Sampling in the reparameterization case - for differentiable samples.
        """
        z = self.base_dist.rsample(sample_shape=sample_shape)

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev


class DiscreteValueDistribution(object):
    """
    Extension to torch categorical probability distribution in order to keep track
    of the support (categorical values, or in this case, value atoms). This is
    used for distributional value networks.
    """
    def __init__(self, values, probs=None, logits=None):
        """
        Creates a categorical distribution parameterized by either @probs or
        @logits (but not both). Expects inputs to be consistent in shape
        for broadcasting operations (e.g. multiplication).
        """
        self._values = values
        self._categorical_dist = D.Categorical(probs=probs, logits=logits)

    @property
    def values(self):
        return self._values

    @property
    def probs(self):
        return self._categorical_dist.probs

    @property
    def logits(self):
        return self._categorical_dist.logits

    def mean(self):
        """
        Categorical distribution mean, taking the value support into account.
        """
        return (self._categorical_dist.probs * self._values).sum(dim=-1)

    def variance(self):
        """
        Categorical distribution variance, taking the value support into account.
        """
        dist_squared = (self.mean().unsqueeze(-1) - self.values).pow(2)
        return (self._categorical_dist.probs * dist_squared).sum(dim=-1)
    
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the distribution. Make sure to return value atoms, not categorical class indices.
        """
        inds = self._categorical_dist.sample(sample_shape=sample_shape)
        return torch.gather(self.values, inds, dim=-1)
