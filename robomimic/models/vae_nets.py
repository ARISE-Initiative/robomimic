"""
Contains an implementation of Variational Autoencoder (VAE) and other
variants, including other priors, and RNN-VAEs.
"""
import textwrap
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.models.base_nets import Module
from robomimic.models.obs_nets import MIMO_MLP


def vae_args_from_config(vae_config):
    """
    Generate a set of VAE args that are read from the VAE-specific part
    of a config (for example see `config.algo.vae` in BCConfig).
    """
    vae_args = dict(
        encoder_layer_dims=vae_config.encoder_layer_dims,
        decoder_layer_dims=vae_config.decoder_layer_dims,
        latent_dim=vae_config.latent_dim,
        decoder_is_conditioned=vae_config.decoder.is_conditioned,
        decoder_reconstruction_sum_across_elements=vae_config.decoder.reconstruction_sum_across_elements,
        latent_clip=vae_config.latent_clip,
        prior_learn=vae_config.prior.learn,
        prior_is_conditioned=vae_config.prior.is_conditioned,
        prior_layer_dims=vae_config.prior_layer_dims,
        prior_use_gmm=vae_config.prior.use_gmm,
        prior_gmm_num_modes=vae_config.prior.gmm_num_modes,
        prior_gmm_learn_weights=vae_config.prior.gmm_learn_weights,
        prior_use_categorical=vae_config.prior.use_categorical,
        prior_categorical_dim=vae_config.prior.categorical_dim,
        prior_categorical_gumbel_softmax_hard=vae_config.prior.categorical_gumbel_softmax_hard,
    )
    return vae_args


class Prior(Module):
    """
    Base class for VAE priors. It's basically the same as a @MIMO_MLP network (it
    instantiates one) but it supports additional methods such as KL loss computation 
    and sampling, and also may learn prior parameters as observation-independent 
    torch Parameters instead of observation-dependent mappings.
    """
    def __init__(
        self,
        param_shapes,
        param_obs_dependent,
        obs_shapes=None,
        mlp_layer_dims=(),
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            param_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for parameters that determine the prior
                distribution.

            param_obs_dependent (OrderedDict): a dictionary with boolean
                values consistent with @param_shapes which determines whether
                to learn parameters as part of the (obs-dependent) network or 
                directly as learnable parameters.

            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layer sizes

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(Prior, self).__init__()

        assert isinstance(param_shapes, OrderedDict) and isinstance(param_obs_dependent, OrderedDict)
        assert set(param_shapes.keys()) == set(param_obs_dependent.keys())
        self.param_shapes = param_shapes
        self.param_obs_dependent = param_obs_dependent

        net_kwargs = dict(
            obs_shapes=obs_shapes,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        self._create_layers(net_kwargs)

    def _create_layers(self, net_kwargs):
        """
        Create networks and parameters needed by the prior.
        """
        self.prior_params = nn.ParameterDict()

        self._is_obs_dependent = False
        mlp_output_shapes = OrderedDict()
        for pp in self.param_shapes:
            if self.param_obs_dependent[pp]:
                # prior parameters will be a function of observations using a network
                mlp_output_shapes[pp] = self.param_shapes[pp]
            else:
                # learnable prior parameters independent of observation
                param_init = torch.randn(*self.param_shapes[pp]) / np.sqrt(np.prod(self.param_shapes[pp]))
                self.prior_params[pp] = torch.nn.Parameter(param_init)

        # only make networks if we have obs-dependent prior parameters
        self.prior_module = None
        if len(mlp_output_shapes) > 0:
            # create @MIMO_MLP that takes obs and goal dicts and returns prior params
            self._is_obs_dependent = True
            obs_shapes = net_kwargs["obs_shapes"]
            goal_shapes = net_kwargs["goal_shapes"]
            obs_group_shapes = OrderedDict()
            assert isinstance(obs_shapes, OrderedDict)
            obs_group_shapes["obs"] = OrderedDict(obs_shapes)
            if goal_shapes is not None and len(goal_shapes) > 0:
                assert isinstance(goal_shapes, OrderedDict)
                obs_group_shapes["goal"] = OrderedDict(goal_shapes)
            self.prior_module = MIMO_MLP(
                input_obs_group_shapes=obs_group_shapes,
                output_shapes=mlp_output_shapes,
                layer_dims=net_kwargs["mlp_layer_dims"],
                encoder_kwargs=net_kwargs["encoder_kwargs"],
            )

    def sample(self, n, obs_dict=None, goal_dict=None):
        """
        Returns a batch of samples from the prior distribution.

        Args:
            n (int): this argument is used to specify the number
                of samples to generate from the prior.

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent. Leading dimension should
                be consistent with @n, the number of samples to generate.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            z (torch.Tensor): batch of sampled latent vectors.
        """
        raise NotImplementedError

    def kl_loss(self, posterior_params, z=None, obs_dict=None, goal_dict=None):
        """
        Computes sample-based KL divergence loss between the Gaussian distribution
        given by @mu, @logvar and the prior distribution. 

        Args:
            posterior_params (dict): dictionary with keys "mu" and "logvar" corresponding
                to torch.Tensor batch of means and log-variances of posterior Gaussian
                distribution.

            z (torch.Tensor): samples from the Gaussian distribution parametrized by
                @mu and @logvar. May not be needed depending on the prior.

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            kl_loss (torch.Tensor): KL divergence loss
        """
        raise NotImplementedError

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        if self.prior_module is not None:
            return self.prior_module.output_shape(input_shape)
        return { k : list(self.param_shapes[k]) for k in self.param_shapes }

    def forward(self, batch_size, obs_dict=None, goal_dict=None):
        """
        Computes prior parameters.

        Args:
            batch_size (int): batch size - this is needed for parameters that are
                not obs-dependent, to make sure the leading dimension is correct
                for downstream sampling and loss computation purposes

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            prior_params (dict): dictionary containing prior parameters
        """
        prior_params = dict()
        if self._is_obs_dependent:
            # forward through network for obs-dependent params
            prior_params = self.prior_module.forward(obs=obs_dict, goal=goal_dict)

        # return params that do not depend on obs as well
        for pp in self.param_shapes:
            if not self.param_obs_dependent[pp]:
                # ensure leading dimension will be consistent with other params
                prior_params[pp] = TensorUtils.expand_at(self.prior_params[pp], size=batch_size, dim=0)

        # ensure leading dimensions are all consistent
        TensorUtils.assert_size_at_dim(prior_params, size=batch_size, dim=0, 
                msg="prior params dim 0 mismatch in forward")

        return prior_params


class GaussianPrior(Prior):
    """
    A class that holds functionality for learning both unimodal Gaussian priors and
    multimodal Gaussian Mixture Model priors for use in VAEs.
    """
    def __init__(
        self,
        latent_dim,
        device,
        latent_clip=None,
        learnable=False,
        use_gmm=False,
        gmm_num_modes=10,
        gmm_learn_weights=False,
        obs_shapes=None,
        mlp_layer_dims=(),
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            latent_dim (int): size of latent dimension for the prior

            device (torch.Device): where the module should live (i.e. cpu, gpu)

            latent_clip (float): if provided, clip all latents sampled at
                test-time in each dimension to (-@latent_clip, @latent_clip)

            learnable (bool): if True, learn the parameters of the prior (as opposed
                to a default N(0, 1) prior)

            use_gmm (bool): if True, learn a Gaussian Mixture Model (GMM)
                prior instead of a unimodal Gaussian prior. To use this option,
                @learnable must be set to True.

            gmm_num_modes (int): number of GMM modes to learn. Only
                used if @use_gmm is True.

            gmm_learn_weights (bool): if True, learn the weights of the GMM
                model instead of setting them to be uniform across all the modes.
                Only used if @use_gmm is True.

            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations. If provided, assumes that
                the prior should depend on observation inputs, and networks 
                will be created to output prior parameters.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layer sizes

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.device = device
        self.latent_dim = latent_dim
        self.latent_clip = latent_clip
        self.learnable = learnable

        self.use_gmm = use_gmm
        if self.use_gmm:
            self.num_modes = gmm_num_modes
        else:
            # unimodal Gaussian prior
            self.num_modes = 1
        self.gmm_learn_weights = gmm_learn_weights

        self._input_dependent = (obs_shapes is not None) and (len(obs_shapes) > 0)

        if self._input_dependent:
            assert learnable
            assert isinstance(obs_shapes, OrderedDict)

            # network will generate mean and logvar
            param_shapes = OrderedDict(
                mean=(self.num_modes, self.latent_dim,),
                logvar=(self.num_modes, self.latent_dim,),
            )
            param_obs_dependent = OrderedDict(mean=True, logvar=True)

            if self.use_gmm and self.gmm_learn_weights:
                # network generates GMM weights
                param_shapes["weight"] = (self.num_modes,)
                param_obs_dependent["weight"] = True
        else:
            # learn obs-indep mean / logvar
            param_shapes = OrderedDict(
                mean=(1, self.num_modes, self.latent_dim),
                logvar=(1, self.num_modes, self.latent_dim),
            )
            param_obs_dependent = OrderedDict(mean=False, logvar=False)

            if self.use_gmm and self.gmm_learn_weights:
                # learn obs-indep GMM weights
                param_shapes["weight"] = (1, self.num_modes)
                param_obs_dependent["weight"] = False

        super(GaussianPrior, self).__init__(
            param_shapes=param_shapes,
            param_obs_dependent=param_obs_dependent,
            obs_shapes=obs_shapes,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _create_layers(self, net_kwargs):
        """
        Update from superclass to only create parameters / networks if not using
        N(0, 1) Gaussian prior.
        """
        if self.learnable:
            super(GaussianPrior, self)._create_layers(net_kwargs)

    def sample(self, n, obs_dict=None, goal_dict=None):
        """
        Returns a batch of samples from the prior distribution.

        Args:
            n (int): this argument is used to specify the number
                of samples to generate from the prior.

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent. Leading dimension should
                be consistent with @n, the number of samples to generate.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            z (torch.Tensor): batch of sampled latent vectors.
        """

        # check consistency between n and obs_dict
        if self._input_dependent:
            TensorUtils.assert_size_at_dim(obs_dict, size=n, dim=0, 
                msg="obs dict and n mismatch in @sample")

        if self.learnable:

            # forward to get parameters
            out = self.forward(batch_size=n, obs_dict=obs_dict, goal_dict=goal_dict)
            prior_means, prior_logvars, prior_logweights = out["means"], out["logvars"], out["logweights"]

            if prior_logweights is not None:
                prior_weights = torch.exp(prior_logweights)

            if self.use_gmm:
                # learned GMM

                # make uniform weights (in the case that weights were not learned)
                if not self.gmm_learn_weights:
                    prior_weights = torch.ones(n, self.num_modes).to(prior_means.device) / self.num_modes

                # sample modes
                gmm_mode_indices = D.Categorical(prior_weights).sample()
                
                # get GMM centers and sample using reparametrization trick
                selected_means = TensorUtils.gather_sequence(prior_means, indices=gmm_mode_indices)
                selected_logvars = TensorUtils.gather_sequence(prior_logvars, indices=gmm_mode_indices)
                z = TorchUtils.reparameterize(selected_means, selected_logvars)

            else:
                # learned unimodal Gaussian - remove mode dim and sample from Gaussian using reparametrization trick
                z = TorchUtils.reparameterize(prior_means[:, 0, :], prior_logvars[:, 0, :])

        else:
            # sample from N(0, 1)
            z = torch.randn(n, self.latent_dim).float().to(self.device)

        if self.latent_clip is not None:
            z = z.clamp(-self.latent_clip, self.latent_clip)

        return z

    def kl_loss(self, posterior_params, z=None, obs_dict=None, goal_dict=None):
        """
        Computes sample-based KL divergence loss between the Gaussian distribution
        given by @mu, @logvar and the prior distribution. 

        Args:
            posterior_params (dict): dictionary with keys "mu" and "logvar" corresponding
                to torch.Tensor batch of means and log-variances of posterior Gaussian
                distribution.

            z (torch.Tensor): samples from the Gaussian distribution parametrized by
                @mu and @logvar. Only needed if @self.use_gmm is True.

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            kl_loss (torch.Tensor): KL divergence loss
        """
        mu = posterior_params["mean"]
        logvar = posterior_params["logvar"]

        if not self.learnable:
            # closed-form Gaussian KL from N(0, 1) prior
            return LossUtils.KLD_0_1_loss(mu=mu, logvar=logvar)

        # forward to get parameters
        out = self.forward(batch_size=mu.shape[0], obs_dict=obs_dict, goal_dict=goal_dict)
        prior_means, prior_logvars, prior_logweights = out["means"], out["logvars"], out["logweights"]

        if not self.use_gmm:
            # collapse mode dimension and compute Gaussian KL in closed-form
            prior_means = prior_means[:, 0, :]
            prior_logvars = prior_logvars[:, 0, :]
            return LossUtils.KLD_gaussian_loss(
                mu_1=mu, 
                logvar_1=logvar, 
                mu_2=prior_means, 
                logvar_2=prior_logvars,
            )

        # GMM KL loss computation
        var = torch.exp(logvar.clamp(-8, 30)) # clamp for numerical stability
        prior_vars = torch.exp(prior_logvars.clamp(-8, 30))
        kl_loss = LossUtils.log_normal(x=z, m=mu, v=var) \
            - LossUtils.log_normal_mixture(x=z, m=prior_means, v=prior_vars, log_w=prior_logweights)
        return kl_loss.mean()

    def forward(self, batch_size, obs_dict=None, goal_dict=None):
        """
        Computes means, logvars, and GMM weights (if using GMM and learning weights).

        Args:
            batch_size (int): batch size - this is needed for parameters that are
                not obs-dependent, to make sure the leading dimension is correct
                for downstream sampling and loss computation purposes

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            prior_params (dict): dictionary containing prior parameters
        """
        assert self.learnable
        prior_params = super(GaussianPrior, self).forward(
            batch_size=batch_size, obs_dict=obs_dict, goal_dict=goal_dict)

        if self.use_gmm and self.gmm_learn_weights:
            # normalize learned weight outputs to sum to 1
            logweights = F.log_softmax(prior_params["weight"], dim=-1)
        else:
            logweights = None
            assert "weight" not in prior_params

        out = dict(means=prior_params["mean"], logvars=prior_params["logvar"], logweights=logweights)
        return out

    def __repr__(self):
        """Pretty print network"""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        msg += textwrap.indent("latent_dim={}\n".format(self.latent_dim), indent)
        msg += textwrap.indent("latent_clip={}\n".format(self.latent_clip), indent)
        msg += textwrap.indent("learnable={}\n".format(self.learnable), indent)
        msg += textwrap.indent("input_dependent={}\n".format(self._input_dependent), indent)
        msg += textwrap.indent("use_gmm={}\n".format(self.use_gmm), indent)
        if self.use_gmm:
            msg += textwrap.indent("gmm_num_nodes={}\n".format(self.num_modes), indent)
            msg += textwrap.indent("gmm_learn_weights={}\n".format(self.gmm_learn_weights), indent)
        if self.learnable:
            if self.prior_module is not None:
                msg += textwrap.indent("\nprior_module={}\n".format(self.prior_module), indent)
            msg += textwrap.indent("prior_params={}\n".format(self.prior_params), indent)
        msg = header + '(\n' + msg + ')'
        return msg


class CategoricalPrior(Prior):
    """
    A class that holds functionality for learning categorical priors for use
    in VAEs.
    """
    def __init__(
        self,
        latent_dim,
        categorical_dim,
        device,
        learnable=False,
        obs_shapes=None,
        mlp_layer_dims=(),
        goal_shapes=None,
        encoder_kwargs=None,

    ):
        """
        Args:
            latent_dim (int): size of latent dimension for the prior

            categorical_dim (int): size of categorical dimension (number of classes
                for each dimension of latent space)

            device (torch.Device): where the module should live (i.e. cpu, gpu)

            learnable (bool): if True, learn the parameters of the prior (as opposed
                to a default N(0, 1) prior)

            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations. If provided, assumes that
                the prior should depend on observation inputs, and networks 
                will be created to output prior parameters.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layer sizes

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.device = device
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.learnable = learnable

        self._input_dependent = (obs_shapes is not None) and (len(obs_shapes) > 0)

        if self._input_dependent:
            assert learnable
            assert isinstance(obs_shapes, OrderedDict)

            # network will generate logits for categorical distributions
            param_shapes = OrderedDict(
                logit=(self.latent_dim, self.categorical_dim,)
            )
            param_obs_dependent = OrderedDict(logit=True)
        else:
            # learn obs-indep mean / logvar
            param_shapes = OrderedDict(
                logit=(1, self.latent_dim, self.categorical_dim),
            )
            param_obs_dependent = OrderedDict(logit=False)

        super(CategoricalPrior, self).__init__(
            param_shapes=param_shapes,
            param_obs_dependent=param_obs_dependent,
            obs_shapes=obs_shapes,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _create_layers(self, net_kwargs):
        """
        Update from superclass to only create parameters / networks if not using
        uniform categorical prior.
        """
        if self.learnable:
            super(CategoricalPrior, self)._create_layers(net_kwargs)

    def sample(self, n, obs_dict=None, goal_dict=None):
        """
        Returns a batch of samples from the prior distribution.

        Args:
            n (int): this argument is used to specify the number
                of samples to generate from the prior.

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent. Leading dimension should
                be consistent with @n, the number of samples to generate.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            z (torch.Tensor): batch of sampled latent vectors.
        """

        # check consistency between n and obs_dict
        if self._input_dependent:
            TensorUtils.assert_size_at_dim(obs_dict, size=n, dim=0, 
                msg="obs dict and n mismatch in @sample")

        if self.learnable:

            # forward to get parameters
            out = self.forward(batch_size=n, obs_dict=obs_dict, goal_dict=goal_dict)
            prior_logits = out["logit"]

            # sample one-hot latents from categorical distribution
            dist = D.Categorical(logits=prior_logits)
            z = TensorUtils.to_one_hot(dist.sample(), num_class=self.categorical_dim)

        else:
            # try to include a categorical sample for each class if possible (ensuring rough uniformity)
            if (self.latent_dim == 1) and (self.categorical_dim <= n):
                # include samples [0, 1, ..., C - 1] and then repeat until batch is filled
                dist_samples = torch.arange(n).remainder(self.categorical_dim).unsqueeze(-1).to(self.device)
            else:
                # sample one-hot latents from uniform categorical distribution for each latent dimension
                probs = torch.ones(n, self.latent_dim, self.categorical_dim).float().to(self.device)
                dist_samples = D.Categorical(probs=probs).sample()
            z = TensorUtils.to_one_hot(dist_samples, num_class=self.categorical_dim)

        # reshape [B, D, C] to [B, D * C] to be consistent with other priors that return flat latents
        z = z.reshape(*z.shape[:-2], -1)
        return z

    def kl_loss(self, posterior_params, z=None, obs_dict=None, goal_dict=None):
        """
        Computes KL divergence loss between the Categorical distribution
        given by the unnormalized logits @logits and the prior distribution. 

        Args:
            posterior_params (dict): dictionary with key "logits" corresponding
                to torch.Tensor batch of unnormalized logits of shape [B, D * C] 
                that corresponds to the posterior categorical distribution

            z (torch.Tensor): samples from encoder - unused for this prior

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            kl_loss (torch.Tensor): KL divergence loss
        """
        logits = posterior_params["logit"].reshape(-1, self.latent_dim, self.categorical_dim)
        if not self.learnable:
            # prior logits correspond to uniform categorical distribution
            prior_logits = torch.zeros_like(logits)
        else:
            # forward to get parameters
            out = self.forward(batch_size=posterior_params["logit"].shape[0], obs_dict=obs_dict, goal_dict=goal_dict)
            prior_logits = out["logit"]

        prior_dist = D.Categorical(logits=prior_logits)
        posterior_dist = D.Categorical(logits=logits)

        # sum over latent dimensions, but average over batch dimension
        kl_loss = D.kl_divergence(posterior_dist, prior_dist)
        assert len(kl_loss.shape) == 2
        return kl_loss.sum(-1).mean()

    def forward(self, batch_size, obs_dict=None, goal_dict=None):
        """
        Computes prior logits (unnormalized log-probs).

        Args:
            batch_size (int): batch size - this is needed for parameters that are
                not obs-dependent, to make sure the leading dimension is correct
                for downstream sampling and loss computation purposes

            obs_dict (dict): inputs according to @obs_shapes. Only needs to be provided
                if any prior parameters are obs-dependent.

            goal_dict (dict): inputs according to @goal_shapes (only if using goal observations)

        Returns:
            prior_params (dict): dictionary containing prior parameters
        """
        assert self.learnable
        return super(CategoricalPrior, self).forward(
            batch_size=batch_size, obs_dict=obs_dict, goal_dict=goal_dict)

    def __repr__(self):
        """Pretty print network"""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        msg += textwrap.indent("latent_dim={}\n".format(self.latent_dim), indent)
        msg += textwrap.indent("categorical_dim={}\n".format(self.categorical_dim), indent)
        msg += textwrap.indent("learnable={}\n".format(self.learnable), indent)
        msg += textwrap.indent("input_dependent={}\n".format(self._input_dependent), indent)
        if self.learnable:
            if self.prior_module is not None:
                msg += textwrap.indent("\nprior_module={}\n".format(self.prior_module), indent)
            msg += textwrap.indent("prior_params={}\n".format(self.prior_params), indent)
        msg = header + '(\n' + msg + ')'
        return msg


class VAE(torch.nn.Module):
    """
    A Variational Autoencoder (VAE), as described in https://arxiv.org/abs/1312.6114.

    Models a distribution p(X) or a conditional distribution p(X | Y), where each
    variable can consist of multiple modalities. The target variable X, whose
    distribution is modeled, is specified through the @input_shapes argument,
    which is a map between modalities (strings) and expected shapes. In this way,
    a variable that consists of multiple kinds of data (e.g. image and flat-dimensional)
    can be modeled as well. A separate @output_shapes argument is used to specify the
    expected reconstructions - this allows for asymmetric reconstruction (for example,
    reconstructing low-resolution images).

    This implementation supports learning conditional distributions as well (cVAE). 
    The conditioning variable Y is specified through the @condition_shapes argument,
    which is also a map between modalities (strings) and expected shapes. In this way,
    variables with multiple kinds of data (e.g. image and flat-dimensional) can 
    jointly be conditioned on. By default, the decoder takes the conditioning 
    variable Y as input. To force the decoder to reconstruct from just the latent,
    set @decoder_is_conditioned to False (in this case, the prior must be conditioned).

    The implementation also supports learning expressive priors instead of using
    the usual N(0, 1) prior. There are three kinds of priors supported - Gaussian,
    Gaussian Mixture Model (GMM), and Categorical. For each prior, the parameters can 
    be learned as independent parameters, or be learned as functions of the conditioning
    variable Y (by setting @prior_is_conditioned).
    """
    def __init__(
        self,
        input_shapes,
        output_shapes,
        encoder_layer_dims,
        decoder_layer_dims,
        latent_dim,
        device,
        condition_shapes=None,
        decoder_is_conditioned=True,
        decoder_reconstruction_sum_across_elements=False,
        latent_clip=None,
        output_squash=(),
        output_scales=None,
        output_ranges=None,
        prior_learn=False,
        prior_is_conditioned=False,
        prior_layer_dims=(),
        prior_use_gmm=False,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=False,
        prior_use_categorical=False,
        prior_categorical_dim=10,
        prior_categorical_gumbel_softmax_hard=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for all encoder-specific inputs. This corresponds
                to the variable X whose distribution we are learning.

            output_shapes (OrderedDict): a dictionary that maps modality to 
                expected shape for outputs to reconstruct. Usually, this is
                the same as @input_shapes but this argument allows
                for asymmetries, such as reconstructing low-resolution
                images.

            encoder_layer_dims ([int]): sequence of integers for the encoder hidden 
                layer sizes.

            decoder_layer_dims ([int]): sequence of integers for the decoder hidden
                layer sizes.

            latent_dim (int): dimension of latent space for the VAE

            device (torch.Device): where the module should live (i.e. cpu, gpu)

            condition_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for all conditioning inputs. If this is provided,
                a conditional distribution is modeled (cVAE). Conditioning takes
                place in the decoder by default, and optionally, the prior.

            decoder_is_conditioned (bool): whether to condition the decoder
                on the conditioning variables. True by default. Only used if
                @condition_shapes is not empty.

            decoder_reconstruction_sum_across_elements (bool): by default, VAEs
                average across modality elements and modalities when computing
                reconstruction loss. If this is True, sum across all dimensions
                and modalities instead.

            latent_clip (float): if provided, clip all latents sampled at
                test-time in each dimension to (-@latent_clip, @latent_clip)

            output_squash ([str]): an iterable of modalities that should be 
                a subset of @output_shapes. The decoder outputs for these
                modalities will be squashed into a symmetric range [-a, a]
                by using a tanh layer and then scaling the output with the
                corresponding value in the @output_scales dictionary.

            output_scales (dict): a dictionary that maps modality to a
                scaling value. Used in conjunction with @output_squash.

            output_ranges (dict): a dictionary of [a, b] specifying the output range.
                when output_ranges is specified (not None), output_scales should be None

            prior_learn (bool): if True, the prior distribution parameters
                are also learned through the KL-divergence loss (instead 
                of being constrained to a N(0, 1) Gaussian distribution).
                If @prior_is_conditioned is True, a global set of parameters
                are learned, otherwise, a prior network that maps between 
                modalities in @condition_shapes and prior parameters is 
                learned. By default, a Gaussian prior is learned, unless 
                @prior_use_gmm is True, in which case a Gaussian Mixture 
                Model (GMM) prior is learned.

            prior_is_conditioned (bool): whether to condition the prior
                on the conditioning variables. False by default. Only used if
                @condition_shapes is not empty. If this is set to True,
                @prior_learn must be True.
            
            prior_layer_dims ([int]): sequence of integers for the prior hidden layer
                sizes. Only used for learned priors that take condition variables as
                input (i.e. when @prior_learn and @prior_is_conditioned are set to True,
                and @condition_shapes is not empty).

            prior_use_gmm (bool): if True, learn a Gaussian Mixture Model (GMM)
                prior instead of a unimodal Gaussian prior. To use this option,
                @prior_learn must be set to True.

            prior_gmm_num_modes (int): number of GMM modes to learn. Only
                used if @prior_use_gmm is True.

            prior_gmm_learn_weights (bool): if True, learn the weights of the GMM
                model instead of setting them to be uniform across all the modes.
                Only used if @prior_use_gmm is True.

            prior_use_categorical (bool): if True, use a categorical prior instead of
                a unimodal Gaussian prior. This will also cause the encoder to output
                a categorical distribution, and will use the Gumbel-Softmax trick
                for reparametrization.

            prior_categorical_dim (int): categorical dimension - each latent sampled
                from the prior will be of shape (@latent_dim, @prior_categorical_dim)
                and will be "one-hot" in the latter dimension. Only used if 
                @prior_use_categorical is True.

            prior_categorical_gumbel_softmax_hard (bool): if True, use the "hard" version of
                Gumbel Softmax for reparametrization. Only used if @prior_use_categorical is True.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations. Goals are treates as additional
                conditioning inputs. They are usually specified separately because
                they have duplicate modalities as the conditioning inputs (otherwise
                they could just be added to the set of conditioning inputs).

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.latent_clip = latent_clip
        self.device = device

        # encoder and decoder input dicts and output shapes dict for reconstruction
        assert isinstance(input_shapes, OrderedDict)
        assert isinstance(output_shapes, OrderedDict)
        self.input_shapes = deepcopy(input_shapes)
        self.output_shapes = deepcopy(output_shapes)

        # check for conditioning (cVAE)
        self._is_cvae = False
        self.condition_shapes = deepcopy(condition_shapes) if condition_shapes is not None else OrderedDict()
        if len(self.condition_shapes) > 0:
            # this is a cVAE - we learn a conditional distribution p(X | Y)
            assert isinstance(self.condition_shapes, OrderedDict)
            self._is_cvae = True
            self.decoder_is_conditioned = decoder_is_conditioned
            self.prior_is_conditioned = prior_is_conditioned
            assert self.decoder_is_conditioned or self.prior_is_conditioned, \
                "cVAE must be conditioned in decoder and/or prior"
            if self.prior_is_conditioned:
                assert prior_learn, "to pass conditioning inputs to prior, prior must be learned"

        # check for goal conditioning
        self._is_goal_conditioned = False
        self.goal_shapes = deepcopy(goal_shapes) if goal_shapes is not None else OrderedDict()
        if len(self.goal_shapes) > 0:
            assert self._is_cvae, "to condition VAE on goals, it must be a cVAE"
            assert isinstance(self.goal_shapes, OrderedDict)
            self._is_goal_conditioned = True

        self.encoder_layer_dims = encoder_layer_dims
        self.decoder_layer_dims = decoder_layer_dims

        # determines whether outputs are squashed with tanh and if so, to what scaling
        assert not (output_scales is not None and output_ranges is not None)
        self.output_squash = output_squash
        self.output_scales = output_scales if output_scales is not None else OrderedDict()
        self.output_ranges = output_ranges if output_ranges is not None else OrderedDict()

        assert set(self.output_squash) == set(self.output_scales.keys())
        assert set(self.output_squash).issubset(set(self.output_shapes))

        # decoder settings
        self.decoder_reconstruction_sum_across_elements = decoder_reconstruction_sum_across_elements

        # prior parameters
        self.prior_learn = prior_learn
        self.prior_layer_dims = prior_layer_dims
        self.prior_use_gmm = prior_use_gmm
        self.prior_gmm_num_modes = prior_gmm_num_modes
        self.prior_gmm_learn_weights = prior_gmm_learn_weights
        self.prior_use_categorical = prior_use_categorical
        self.prior_categorical_dim = prior_categorical_dim
        self.prior_categorical_gumbel_softmax_hard = prior_categorical_gumbel_softmax_hard
        assert np.sum([self.prior_use_gmm, self.prior_use_categorical]) <= 1

        # for obs core
        self._encoder_kwargs = encoder_kwargs

        if self.prior_use_gmm:
            assert self.prior_learn, "GMM must be learned"

        if self.prior_use_categorical:
            # initialize temperature for Gumbel-Softmax
            self.set_gumbel_temperature(1.0)

        # create encoder, decoder, prior
        self._create_layers()

    def _create_layers(self):
        """
        Creates the encoder, decoder, and prior networks.
        """
        self.nets = nn.ModuleDict()

        # VAE Encoder
        self._create_encoder()

        # VAE Decoder
        self._create_decoder()

        # VAE Prior.
        self._create_prior()

    def _create_encoder(self):
        """
        Helper function to create encoder.
        """

        # encoder takes "input" dictionary and possibly "condition" (if cVAE) and "goal" (if goal-conditioned)
        encoder_obs_group_shapes = OrderedDict()
        encoder_obs_group_shapes["input"] = OrderedDict(self.input_shapes)
        if self._is_cvae:
            encoder_obs_group_shapes["condition"] = OrderedDict(self.condition_shapes)
            if self._is_goal_conditioned:
                encoder_obs_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        
        # encoder outputs posterior distribution parameters
        if self.prior_use_categorical:
            encoder_output_shapes = OrderedDict(
                logit=(self.latent_dim * self.prior_categorical_dim,),
            )
        else:
            encoder_output_shapes = OrderedDict(
                mean=(self.latent_dim,), 
                logvar=(self.latent_dim,),
            )

        self.nets["encoder"] = MIMO_MLP(
            input_obs_group_shapes=encoder_obs_group_shapes,
            output_shapes=encoder_output_shapes, 
            layer_dims=self.encoder_layer_dims,
            encoder_kwargs=self._encoder_kwargs,
        )

    def _create_decoder(self):
        """
        Helper function to create decoder.
        """

        # decoder takes latent (included as "input" observation group) and possibly "condition" (if cVAE) and "goal" (if goal-conditioned)
        decoder_obs_group_shapes = OrderedDict()
        latent_shape = (self.latent_dim,)
        if self.prior_use_categorical:
            latent_shape = (self.latent_dim * self.prior_categorical_dim,)
        decoder_obs_group_shapes["input"] = OrderedDict(latent=latent_shape)
        if self._is_cvae:
            decoder_obs_group_shapes["condition"] = OrderedDict(self.condition_shapes)
            if self._is_goal_conditioned:
                decoder_obs_group_shapes["goal"] = OrderedDict(self.goal_shapes)

        self.nets["decoder"] = MIMO_MLP(
            input_obs_group_shapes=decoder_obs_group_shapes,
            output_shapes=self.output_shapes, 
            layer_dims=self.decoder_layer_dims,
            encoder_kwargs=self._encoder_kwargs,
        )

    def _create_prior(self):
        """
        Helper function to create prior.
        """

        # prior possibly takes "condition" (if cVAE) and "goal" (if goal-conditioned)
        prior_obs_group_shapes = OrderedDict(condition=None, goal=None)
        if self._is_cvae and self.prior_is_conditioned:
            prior_obs_group_shapes["condition"] = OrderedDict(self.condition_shapes)
            if self._is_goal_conditioned:
                prior_obs_group_shapes["goal"] = OrderedDict(self.goal_shapes)

        if self.prior_use_categorical:
            self.nets["prior"] = CategoricalPrior(
                latent_dim=self.latent_dim,
                categorical_dim=self.prior_categorical_dim,
                device=self.device,
                learnable=self.prior_learn,
                obs_shapes=prior_obs_group_shapes["condition"],
                mlp_layer_dims=self.prior_layer_dims,
                goal_shapes=prior_obs_group_shapes["goal"],
                encoder_kwargs=self._encoder_kwargs,
            )
        else:
            self.nets["prior"] = GaussianPrior(
                latent_dim=self.latent_dim,
                device=self.device,
                latent_clip=self.latent_clip,
                learnable=self.prior_learn,
                use_gmm=self.prior_use_gmm,
                gmm_num_modes=self.prior_gmm_num_modes,
                gmm_learn_weights=self.prior_gmm_learn_weights,
                obs_shapes=prior_obs_group_shapes["condition"],
                mlp_layer_dims=self.prior_layer_dims,
                goal_shapes=prior_obs_group_shapes["goal"],
                encoder_kwargs=self._encoder_kwargs,
            )

    def encode(self, inputs, conditions=None, goals=None):
        """
        Args:
            inputs (dict): a dictionary that maps input modalities to torch.Tensor
                batches. These should correspond to the encoder-only modalities
                (i.e. @self.encoder_only_shapes).

            conditions (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to the modalities used for conditioning
                in either the decoder or the prior (or both). Only for cVAEs.

            goals (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities. Only for cVAEs.

        Returns:
            posterior params (dict): dictionary with posterior parameters
        """
        return self.nets["encoder"](
            input=inputs,
            condition=conditions,
            goal=goals,
        )

    def reparameterize(self, posterior_params):
        """
        Args:
            posterior params (dict): dictionary from encoder forward pass that
                parametrizes the encoder distribution

        Returns:
            z (torch.Tensor): sampled latents that are also differentiable
        """
        if self.prior_use_categorical:
            # reshape to [B, D, C] to take softmax across categorical classes
            logits = posterior_params["logit"].reshape(-1, self.latent_dim, self.prior_categorical_dim)
            z = F.gumbel_softmax(
                logits=logits,
                tau=self._gumbel_temperature,
                hard=self.prior_categorical_gumbel_softmax_hard,
                dim=-1,
            )
            # reshape to [B, D * C], since downstream networks expect flat latents
            return TensorUtils.flatten(z)

        return TorchUtils.reparameterize(
            mu=posterior_params["mean"], 
            logvar=posterior_params["logvar"],
        )

    def decode(self, conditions=None, goals=None, z=None, n=None):
        """
        Pass latents through decoder. Latents should be passed in to
        this function at train-time for backpropagation, but they
        can be left out at test-time. In this case, latents will
        be sampled using the VAE prior.

        Args:
            conditions (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to the modalities used for conditioning
                in either the decoder or the prior (or both). Only for cVAEs.

            goals (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities. Only for cVAEs.

            z (torch.Tensor): if provided, these latents are used to generate
                reconstructions from the VAE, and the prior is not sampled.

            n (int): this argument is used to specify the number of samples to 
                generate from the prior. Only required if @z is None - i.e.
                sampling takes place

        Returns:
            recons (dict): dictionary of reconstructed inputs
        """

        if z is None:
            # sample latents from prior distribution
            assert n is not None
            z = self.sample_prior(n=n, conditions=conditions, goals=goals)

        # decoder takes latents as input, and maybe condition variables 
        # and goal variables
        inputs = dict(
            input=dict(latent=z), 
            condition=conditions, 
            goal=goals,
        )

        # pass through decoder to reconstruct variables in @self.output_shapes
        recons = self.nets["decoder"](**inputs)

        # apply tanh squashing to output modalities
        for k in self.output_squash:
            recons[k] = self.output_scales[k] * torch.tanh(recons[k])

        for k, v_range in self.output_ranges.items():
            assert v_range[1] > v_range[0]
            recons[k] = torch.sigmoid(recons[k]) * (v_range[1] - v_range[0]) + v_range[0]
        return recons

    def sample_prior(self, n, conditions=None, goals=None):
        """
        Samples from the prior using the prior parameters.

        Args:
            n (int): this argument is used to specify the number
                of samples to generate from the prior.

            conditions (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to the modalities used for conditioning
                in either the decoder or the prior (or both). Only for cVAEs.

            goals (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities. Only for cVAEs.

        Returns:
            z (torch.Tensor): sampled latents from the prior
        """
        return self.nets["prior"].sample(n=n, obs_dict=conditions, goal_dict=goals)

    def kl_loss(self, posterior_params, encoder_z=None, conditions=None, goals=None):
        """
        Computes KL divergence loss given the results of the VAE encoder forward
        pass and the conditioning and goal modalities (if the prior is input-dependent).

        Args:
            posterior_params (dict): dictionary with keys "mu" and "logvar" corresponding
                to torch.Tensor batch of means and log-variances of posterior Gaussian
                distribution. This is the output of @self.encode.

            encoder_z (torch.Tensor): samples from the Gaussian distribution parametrized by
                @mu and @logvar. Only required if using a GMM prior.

            conditions (dict): inputs according to @self.condition_shapes. Only needs to be provided
                if any prior parameters are input-dependent.

            goal_dict (dict): inputs according to @self.goal_shapes (only if using goal observations)

        Returns:
            kl_loss (torch.Tensor): VAE KL divergence loss
        """
        return self.nets["prior"].kl_loss(
            posterior_params=posterior_params,
            z=encoder_z,
            obs_dict=conditions, 
            goal_dict=goals,
        )

    def reconstruction_loss(self, reconstructions, targets):
        """
        Reconstruction loss. Note that we compute the average per-dimension error
        in each modality and then average across all the modalities.

        The beta term for weighting between reconstruction and kl losses will
        need to be tuned in practice for each situation (see
        https://twitter.com/memotv/status/973323454350090240 for more 
        discussion).

        Args:
            reconstructions (dict): reconstructed inputs, consistent with
                @self.output_shapes
            targets (dict): reconstruction targets, consistent with
                @self.output_shapes

        Returns:
            reconstruction_loss (torch.Tensor): VAE reconstruction loss
        """
        random_key = list(reconstructions.keys())[0]
        batch_size = reconstructions[random_key].shape[0]
        num_mods = len(reconstructions.keys())

        # collect errors per modality, while preserving shapes in @reconstructions
        recons_errors = []
        for k in reconstructions:
            L2_loss = (reconstructions[k] - targets[k]).pow(2)
            recons_errors.append(L2_loss)

        # reduce errors across modalities and dimensions
        if self.decoder_reconstruction_sum_across_elements:
            # average across batch but sum across modalities and dimensions
            loss = sum([x.sum() for x in recons_errors])
            loss /= batch_size
        else:
            # compute mse loss in each modality and average across modalities
            loss = sum([x.mean() for x in recons_errors])
            loss /= num_mods
        return loss

    def forward(self, inputs, outputs, conditions=None, goals=None, freeze_encoder=False):
        """
        A full pass through the VAE network to construct KL and reconstruction
        losses.

        Args:
            inputs (dict): a dictionary that maps input modalities to torch.Tensor
                batches. These should correspond to the encoder-only modalities
                (i.e. @self.encoder_only_shapes).

            outputs (dict): a dictionary that maps output modalities to torch.Tensor
                batches. These should correspond to the modalities used for
                reconstruction (i.e. @self.output_shapes).

            conditions (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to the modalities used for conditioning
                in either the decoder or the prior (or both). Only for cVAEs.

            goals (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities. Only for cVAEs.

            freeze_encoder (bool): if True, don't backprop into encoder by detaching
                encoder outputs. Useful for doing staged VAE training.

        Returns:
            vae_outputs (dict): a dictionary that contains the following outputs.

                encoder_params (dict): parameters for the posterior distribution
                    from the encoder forward pass

                encoder_z (torch.Tensor): latents sampled from the encoder posterior

                decoder_outputs (dict): reconstructions from the decoder

                kl_loss (torch.Tensor): KL loss over the batch of data

                reconstruction_loss (torch.Tensor): reconstruction loss over the batch of data
        """

        # In the comments below, X = inputs, Y = conditions, and we seek to learn P(X | Y).
        # The decoder and prior only have knowledge about Y and try to reconstruct X.
        # Notice that when Y is the empty set, this reduces to a normal VAE.

        # mu, logvar <- Enc(X, Y)
        posterior_params = self.encode(
            inputs=inputs, 
            conditions=conditions,
            goals=goals,
        )

        if freeze_encoder:
            posterior_params = TensorUtils.detach(posterior_params)

        # z ~ Enc(z | X, Y)
        encoder_z = self.reparameterize(posterior_params)

        # hat(X) = Dec(z, Y)
        reconstructions = self.decode(
            conditions=conditions, 
            goals=goals,
            z=encoder_z,
        )
        
        # this will also train prior network z ~ Prior(z | Y)
        kl_loss = self.kl_loss(
            posterior_params=posterior_params,
            encoder_z=encoder_z,
            conditions=conditions,
            goals=goals,
        )

        reconstruction_loss = self.reconstruction_loss(
            reconstructions=reconstructions, 
            targets=outputs,
        )

        return {
            "encoder_params" : posterior_params,
            "encoder_z" : encoder_z,
            "decoder_outputs" : reconstructions,
            "kl_loss" : kl_loss,
            "reconstruction_loss" : reconstruction_loss,
        }

    def set_gumbel_temperature(self, temperature):
        """
        Used by external algorithms to schedule Gumbel-Softmax temperature,
        which is used during reparametrization at train-time. Should only
        be used if @self.prior_use_categorical is True.
        """
        assert self.prior_use_categorical
        self._gumbel_temperature = temperature

    def get_gumbel_temperature(self):
        """
        Return current Gumbel-Softmax temperature. Should only be used if
        @self.prior_use_categorical is True.
        """
        assert self.prior_use_categorical
        return self._gumbel_temperature
