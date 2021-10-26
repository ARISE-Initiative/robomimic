"""
This file contains some PyTorch utilities.
"""
import numpy as np
import torch
import torch.optim as optim
import inspect
from copy import deepcopy


def get_class_init_kwargs(cls):
    """
    Helper function to return a list of all valid keyword arguments (excluding "self") for the given @cls class.

    Args:
        cls (object): Class from which to grab __init__ kwargs

    Returns:
        list: All keyword arguments (excluding "self") specified by @cls __init__ constructor method
    """
    return list(inspect.signature(cls.__init__).parameters.keys())[1:]


def extract_subset_dict(dic, keys, copy=False):
    """
    Helper function to extract a subset of dictionary key-values from a current dictionary. Optionally (deep)copies
    the values extracted from the original @dic if @copy is True.

    Args:
        dic (dict): Dictionary containing multiple key-values
        keys (Iterable): Specific keys to extract from @dic. If the key doesn't exist in @dic, then the key is skipped
        copy (bool): If True, will deepcopy all values corresponding to the specified @keys

    Returns:
        dict: Extracted subset dictionary containing only the specified @keys and their corresponding values
    """
    subset = {k: dic[k] for k in keys if k in dic}
    return deepcopy(subset) if copy else subset


def extract_class_init_kwargs_from_dict(cls, dic, copy=False):
    """
    Helper function to return a dictionary of key-values that specifically correspond to @cls class's __init__
    constructor method, from @dic which may more may not contain additional, irrelevant kwargs.

    Note that @dic may possibly be missing certain kwargs as specified by cls.__init__. No error will be raised.

    Args:
        cls (object): Class from which to grab __init__ kwargs that will be be used as filtering keys for @dic
        dic (dict): Dictionary containing multiple key-values
        copy (bool): If True, will deepcopy all values corresponding to the specified @keys

    Returns:
        dict: Extracted subset dictionary possibly containing only the specified keys from cls.__init__ and their
            corresponding values
    """
    # extract only relevant kwargs for this specific backbone
    return extract_subset_dict(
        dic=dic,
        keys=get_class_init_kwargs(cls),
        copy=copy,
    )


def soft_update(source, target, tau):
    """
    Soft update from the parameters of a @source torch module to a @target torch module
    with strength @tau. The update follows target = target * (1 - tau) + source * tau.

    Args:
        source (torch.nn.Module): source network to push target network parameters towards
        target (torch.nn.Module): target network to update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.copy_(
            target_param * (1.0 - tau) + param * tau
        )


def hard_update(source, target):
    """
    Hard update @target parameters to match @source.

    Args:
        source (torch.nn.Module): source network to provide parameters
        target (torch.nn.Module): target network to update parameters for
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.copy_(param)


def get_torch_device(try_to_use_cuda):
    """
    Return torch device. If using cuda (GPU), will also set cudnn.benchmark to True
    to optimize CNNs.

    Args:
        try_to_use_cuda (bool): if True and cuda is available, will use GPU

    Returns:
        device (torch.Device): device to use for models
    """
    if try_to_use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def reparameterize(mu, logvar):
    """
    Reparameterize for the backpropagation of z instead of q.
    This makes it so that we can backpropagate through the sampling of z from
    our encoder when feeding the sampled variable to the decoder.

    (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)

    Args:
        mu (torch.Tensor): batch of means from the encoder distribution
        logvar (torch.Tensor): batch of log variances from the encoder distribution

    Returns:
        z (torch.Tensor): batch of sampled latents from the encoder distribution that
            support backpropagation
    """
    # logvar = \log(\sigma^2) = 2 * \log(\sigma)
    # \sigma = \exp(0.5 * logvar)

    # clamped for numerical stability
    logstd = (0.5 * logvar).clamp(-4, 15)
    std = torch.exp(logstd)

    # Sample \epsilon from normal distribution
    # use std to create a new tensor, so we don't have to care
    # about running on GPU or not
    eps = std.new(std.size()).normal_()

    # Then multiply with the standard deviation and add the mean
    z = eps.mul(std).add_(mu)

    return z


def optimizer_from_optim_params(net_optim_params, net):
    """
    Helper function to return a torch Optimizer from the optim_params 
    section of the config for a particular network.

    Args:
        optim_params (Config): optim_params part of algo_config corresponding
            to @net. This determines the optimizer that is created.

        net (torch.nn.Module): module whose parameters this optimizer will be
            responsible

    Returns:
        optimizer (torch.optim.Optimizer): optimizer
    """
    return optim.Adam(
        params=net.parameters(),
        lr=net_optim_params["learning_rate"]["initial"],
        weight_decay=net_optim_params["regularization"]["L2"],
    )


def lr_scheduler_from_optim_params(net_optim_params, net, optimizer):
    """
    Helper function to return a LRScheduler from the optim_params 
    section of the config for a particular network. Returns None
    if a scheduler is not needed.

    Args:
        optim_params (Config): optim_params part of algo_config corresponding
            to @net. This determines whether a learning rate scheduler is created.

        net (torch.nn.Module): module whose parameters this optimizer will be
            responsible

        optimizer (torch.optim.Optimizer): optimizer for this net

    Returns:
        lr_scheduler (torch.optim.lr_scheduler or None): learning rate scheduler
    """
    lr_scheduler = None
    if len(net_optim_params["learning_rate"]["epoch_schedule"]) > 0:
        # decay LR according to the epoch schedule
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=net_optim_params["learning_rate"]["epoch_schedule"],
            gamma=net_optim_params["learning_rate"]["decay_factor"],
        )
    return lr_scheduler


def backprop_for_loss(net, optim, loss, max_grad_norm=None, retain_graph=False):
    """
    Backpropagate loss and update parameters for network with
    name @name.

    Args:
        net (torch.nn.Module): network to update

        optim (torch.optim.Optimizer): optimizer to use

        loss (torch.Tensor): loss to use for backpropagation

        max_grad_norm (float): if provided, used to clip gradients

        retain_graph (bool): if True, graph is not freed after backward call

    Returns:
        grad_norms (float): average gradient norms from backpropagation
    """

    # backprop
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # gradient clipping
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

    # compute grad norms
    grad_norms = 0.
    for p in net.parameters():
        # only clip gradients for parameters for which requires_grad is True
        if p.grad is not None:
            grad_norms += p.grad.data.norm(2).pow(2).item()

    # step
    optim.step()

    return grad_norms


class dummy_context_mgr():
    """
    A dummy context manager - useful for having conditional scopes (such
    as @maybe_no_grad). Nothing happens in this scope.
    """
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def maybe_no_grad(no_grad):
    """
    Args:
        no_grad (bool): if True, the returned context will be torch.no_grad(), otherwise
            it will be a dummy context
    """
    return torch.no_grad() if no_grad else dummy_context_mgr()
