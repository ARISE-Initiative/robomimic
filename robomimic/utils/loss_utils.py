"""
This file contains a collection of useful loss functions for use with torch tensors.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

def gmm_score(x, means, sigmas, weights):
    """
    Compute the score function (gradient of log probability) for a Gaussian Mixture Model.
    Assumes diagonal covariance matrices (independent dimensions).
    
    ∇ log p(x) = ∑_k π_k N(x|μ_k,Σ_k) ∇ log N(x|μ_k,Σ_k) / p(x)
               = ∑_k w_k(x) Σ_k^{-1} (μ_k - x)
    
    where w_k(x) = π_k N(x|μ_k,Σ_k) / p(x) is the responsibility of component k.
    
    Args:
        x: Input samples [batch, dim]
        means: Component means [batch, num_components, dim]
        sigmas: Component standard deviations [batch, num_components, dim] (diagonal covariance)
        weights: Component weights (mixing coefficients) [batch, num_components]
              Should sum to 1.
    
    Returns:
        score: ∇ log p(x) [batch, dim]
    """
    batch_size, dim = x.shape
    num_components = means.shape[1]
    
    # Normalize weights to ensure they sum to 1 along component dimension
    weights = weights / weights.sum(dim=1, keepdim=True)  # [batch, num_components]
    
    # Vectorized computation for all components at once
    # Expand x to [batch, num_components, dim] for broadcasting
    x_expanded = x.unsqueeze(1)  # [batch, 1, dim]
    
    # Difference from all means: [batch, num_components, dim]
    diff = x_expanded - means  # means is already [batch, num_components, dim]
    
    # For diagonal covariance: Σ^{-1} = diag(1/σ²)
    # sigmas: [batch, num_components, dim]
    sigma_sq = sigmas ** 2  # [batch, num_components, dim]
    sigma_sq_inv = 1.0 / sigma_sq  # [batch, num_components, dim]
    
    # Determinant for diagonal covariance: |Σ| = ∏_i σ_i²
    log_det = torch.sum(torch.log(sigma_sq), dim=2)  # [batch, num_components]
    
    # Mahalanobis distance for diagonal covariance
    # (x - μ)^T Σ^{-1} (x - μ) = ∑_i (x_i - μ_i)² / σ_i²
    mahal = torch.sum(diff ** 2 * sigma_sq_inv, dim=2)  # [batch, num_components]
    
    # Log probabilities for all components
    # log N(x|μ_k,Σ_k) = -0.5 * [(x-μ_k)^T Σ_k^{-1} (x-μ_k) + log|Σ_k| + d*log(2π)]
    log_probs = -0.5 * (mahal + log_det + dim * np.log(2 * np.pi))  # [batch, num_components]
    
    # Score for each component: Σ_k^{-1} (μ_k - x) = -(x - μ_k) / σ²
    component_scores = -diff * sigma_sq_inv  # [batch, num_components, dim]
    
    # Add log weights
    log_weights = torch.log(weights)  # [batch, num_components]
    log_weighted_probs = log_probs + log_weights  # [batch, num_components]
    
    # Compute log p(x) using log-sum-exp trick for numerical stability
    log_p_x = torch.logsumexp(log_weighted_probs, dim=1, keepdim=True)  # [batch, 1]
    
    # Compute responsibilities: w_k(x) = π_k N(x|μ_k,Σ_k) / p(x)
    log_responsibilities = log_weighted_probs - log_p_x  # [batch, num_components]
    responsibilities = torch.exp(log_responsibilities)  # [batch, num_components]
    
    # Weighted sum of component scores
    # score = ∑_k w_k(x) * score_k(x)
    score = torch.sum(responsibilities.unsqueeze(2) * component_scores, dim=1)  # [batch, dim]
    
    return score


def divergence_loss(dx_t_pred, dx_t, x_t, div_u_t, mean_t, sigma_t, weights_t=None):
    """
    Compute flow matching loss with divergence regularization
    
    L_CDM(θ) = E[ |(∇·u_t - ∇·v_t) + (u_t - v_t)·∇log p_t| ]
    
    Args:
        dx_t_pred: Prediected flow field sampels
        dx_t: Target flow field samples [batch, dim]
        x_t: Interpolated samples [batch, dim]
        div_u_t: Divergence of target flow field [batch]
        mean_t: Mean of p_t at x_t [batch, n_components, dim]
        sigma_t: Stddev of p_t at x_t [batch, n_components, dim]
        weights_t: the weights of the gmm [batch, n_components]
    """
    x_t.requires_grad_(True)
        
    # CONDITIONAL DIVERGENCE MATCHING LOSS
    # Sample random vector ε ~ N(0, I) for Hutchinson's estimator
    epsilon = torch.randn_like(x_t)
    
    # TERM 1: (∇·u_t - ∇·v_t)
    # ---------------------------
    # Divergence of predicted velocity v_t using Hutchinson's estimator
    v_dot_eps = (dx_t_pred * epsilon).sum()
    div_v_grad = torch.autograd.grad(
        outputs=v_dot_eps,
        inputs=x_t,
        grad_outputs=torch.ones_like(v_dot_eps),
        create_graph=True,
        retain_graph=True
    )[0]
    div_v_t = (epsilon * div_v_grad).sum(dim=1)  # [batch]
    
    # Difference of divergences
    divergence_diff = div_u_t - div_v_t 
    
    # TERM 2: (u_t - v_t)·∇log p_t
    # ------------------------------
    if weights_t is None:
        # Score (single gauss): ∇ log p(x_t | x_1) = -(x_t - t*x_1) / σ²(t)
        score = -(x_t - mean_t) / (sigma_t ** 2)  # [batch, dim]
    else:
        score = gmm_score(x_t, mean_t, sigma_t, weights_t)
    
    # Velocity difference
    velocity_diff = dx_t - dx_t_pred  # [batch, dim]
    
    # Dot product with score (sum over dimensions for each batch)
    velocity_score_dot = (velocity_diff * score).sum(dim=1)  # [batch]
    
    # COMBINED DIVERGENCE LOSS
    # L_CDM = E[ |(∇·u - ∇·v) + (u - v)·score| ]
    return torch.abs(divergence_diff + velocity_score_dot).mean()
    

def cosine_loss(preds, labels):
    """
    Cosine loss between two tensors.

    Args:
        preds (torch.Tensor): torch tensor
        labels (torch.Tensor): torch tensor

    Returns:
        loss (torch.Tensor): cosine loss
    """
    sim = torch.nn.CosineSimilarity(dim=len(preds.shape) - 1)(preds, labels)
    return -torch.mean(sim - 1.0)


def KLD_0_1_loss(mu, logvar):
    """
    KL divergence loss. Computes D_KL( N(mu, sigma) || N(0, 1) ). Note that 
    this function averages across the batch dimension, but sums across dimension.

    Args:
        mu (torch.Tensor): mean tensor of shape (B, D)
        logvar (torch.Tensor): logvar tensor of shape (B, D)

    Returns:
        loss (torch.Tensor): KL divergence loss between the input gaussian distribution
            and N(0, 1)
    """
    return -0.5 * (1. + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def KLD_gaussian_loss(mu_1, logvar_1, mu_2, logvar_2):
    """
    KL divergence loss between two Gaussian distributions. This function 
    computes the average loss across the batch.

    Args:
        mu_1 (torch.Tensor): first means tensor of shape (B, D)
        logvar_1 (torch.Tensor): first logvars tensor of shape (B, D)
        mu_2 (torch.Tensor): second means tensor of shape (B, D)
        logvar_2 (torch.Tensor): second logvars tensor of shape (B, D)

    Returns:
        loss (torch.Tensor): KL divergence loss between the two gaussian distributions
    """
    return -0.5 * (1. + \
        logvar_1 - logvar_2 \
        - ((mu_2 - mu_1).pow(2) / logvar_2.exp()) \
        - (logvar_1.exp() / logvar_2.exp()) \
        ).sum(dim=1).mean()


def log_normal(x, m, v):
    """
    Log probability of tensor x under diagonal multivariate normal with
    mean m and variance v. The last dimension of the tensors is treated
    as the dimension of the Gaussian distribution - all other dimensions
    are treated as independent Gaussians. Adapted from CS 236 at Stanford.

    Args:
        x (torch.Tensor): tensor with shape (B, ..., D)
        m (torch.Tensor): means tensor with shape (B, ..., D) or (1, ..., D)
        v (torch.Tensor): variances tensor with shape (B, ..., D) or (1, ..., D)

    Returns:
        log_prob (torch.Tensor): log probabilities of shape (B, ...)
    """
    element_wise = -0.5 * (torch.log(v) + (x - m).pow(2) / v + np.log(2 * np.pi))
    log_prob = element_wise.sum(-1)
    return log_prob


def log_normal_mixture(x, m, v, w=None, log_w=None):
    """
    Log probability of tensor x under a uniform mixture of Gaussians. 
    Adapted from CS 236 at Stanford.

    Args:
        x (torch.Tensor): tensor with shape (B, D)
        m (torch.Tensor): means tensor with shape (B, M, D) or (1, M, D), where 
            M is number of mixture components
        v (torch.Tensor): variances tensor with shape (B, M, D) or (1, M, D) where 
            M is number of mixture components
        w (torch.Tensor): weights tensor - if provided, should be 
            shape (B, M) or (1, M)
        log_w (torch.Tensor): log-weights tensor - if provided, should be 
            shape (B, M) or (1, M)

    Returns:
        log_prob (torch.Tensor): log probabilities of shape (B,)
    """

    # (B , D) -> (B , 1, D)
    x = x.unsqueeze(1)
    # (B, 1, D) -> (B, M, D) -> (B, M)
    log_prob = log_normal(x, m, v)
    if w is not None or log_w is not None:
        # this weights the log probabilities by the mixture weights so we have log(w_i * N(x | m_i, v_i))
        if w is not None:
            assert log_w is None
            log_w = torch.log(w)
        log_prob += log_w
        # then compute log sum_i exp [log(w_i * N(x | m_i, v_i))]
        # (B, M) -> (B,)
        log_prob = log_sum_exp(log_prob , dim=1)
    else:
        # (B, M) -> (B,)
        log_prob = log_mean_exp(log_prob , dim=1) # mean accounts for uniform weights
    return log_prob


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner.
    Adapted from CS 236 at Stanford.

    Args:
        x (torch.Tensor): a tensor 
        dim (int): dimension along which mean is computed

    Returns:
        y (torch.Tensor): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner.
    Adapted from CS 236 at Stanford.

    Args:
        x (torch.Tensor): a tensor 
        dim (int): dimension along which sum is computed

    Returns:
        y (torch.Tensor): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def project_values_onto_atoms(values, probabilities, atoms):
    """
    Project the categorical distribution given by @probabilities on the
    grid of values given by @values onto a grid of values given by @atoms.
    This is useful when computing a bellman backup where the backed up
    values from the original grid will not be in the original support,
    requiring L2 projection. 

    Each value in @values has a corresponding probability in @probabilities -
    this probability mass is shifted to the closest neighboring grid points in
    @atoms in proportion. For example, if the value in question is 0.2, and the
    neighboring atoms are 0 and 1, then 0.8 of the probability weight goes to 
    atom 0 and 0.2 of the probability weight will go to 1.

    Adapted from https://github.com/deepmind/acme/blob/master/acme/tf/losses/distributional.py#L42
    
    Args:
        values: value grid to project, of shape (batch_size, n_atoms)
        probabilities: probabilities for categorical distribution on @values, shape (batch_size, n_atoms)
        atoms: value grid to project onto, of shape (n_atoms,) or (1, n_atoms)

    Returns:
        new probability vectors that correspond to the L2 projection of the categorical distribution
        onto @atoms
    """

    # make sure @atoms is shape (n_atoms,)
    if len(atoms.shape) > 1:
        atoms = atoms.squeeze(0)

    # helper tensors from @atoms
    vmin, vmax = atoms[0], atoms[1]
    d_pos = torch.cat([atoms, vmin[None]], dim=0)[1:]
    d_neg = torch.cat([vmax[None], atoms], dim=0)[:-1]

    # ensure that @values grid is within the support of @atoms
    clipped_values = values.clamp(min=vmin, max=vmax)[:, None, :] # (batch_size, 1, n_atoms)
    clipped_atoms = atoms[None, :, None] # (1, n_atoms, 1)

    # distance between atom values in support
    d_pos = (d_pos - atoms)[None, :, None] # atoms[i + 1] - atoms[i], shape (1, n_atoms, 1)
    d_neg = (atoms - d_neg)[None, :, None] # atoms[i] - atoms[i - 1], shape (1, n_atoms, 1)

    # distances between all pairs of grid values
    deltas = clipped_values - clipped_atoms # (batch_size, n_atoms, n_atoms)

    # computes eqn (7) in distributional RL paper by doing the following - for each
    # output atom in @atoms, consider values that are close enough, and weight their
    # probability mass contribution by the normalized distance in [0, 1] given 
    # by (1. - (z_j - z_i) / (delta_z)).
    d_sign = (deltas >= 0.).float()
    delta_hat = (d_sign * deltas / d_pos) - ((1. - d_sign) * deltas / d_neg)
    delta_hat = (1. - delta_hat).clamp(min=0., max=1.)
    probabilities = probabilities[:, None, :]
    return (delta_hat * probabilities).sum(dim=2)
