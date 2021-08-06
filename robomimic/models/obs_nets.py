"""
Contains torch Modules that help deal with inputs consisting of multiple
modalities. This is extremely common when networks must deal with one or 
more observation dictionaries, where each input dictionary can have
modality keys of a certain type and shape. 

As an example, an observation could consist of a flat "robot0_eef_pos" modality, 
and a 3-channel RGB "agentview_image" modality.
"""
import sys
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import Module, Sequential, MLP, RNN_Base, ResNet18Conv, SpatialSoftmax, \
    FeatureAggregator, VisualCore, Randomizer, CropRandomizer


def obs_encoder_args_from_config(obs_encoder_config):
    """
    Generate a set of args used to create visual backbones for networks
    from the obseration encoder config.
    """
    return dict(
        visual_feature_dimension=obs_encoder_config.visual_feature_dimension,
        visual_core_class=obs_encoder_config.visual_core,
        visual_core_kwargs=dict(obs_encoder_config.visual_core_kwargs),
        obs_randomizer_class=obs_encoder_config.obs_randomizer_class,
        obs_randomizer_kwargs=dict(obs_encoder_config.obs_randomizer_kwargs),
        use_spatial_softmax=obs_encoder_config.use_spatial_softmax,
        spatial_softmax_kwargs=dict(obs_encoder_config.spatial_softmax_kwargs),
    )


def obs_encoder_factory(
        obs_shapes,
        visual_feature_dimension,
        visual_core_class,
        visual_core_kwargs=None,
        obs_randomizer_class=None,
        obs_randomizer_kwargs=None,
        use_spatial_softmax=False,
        spatial_softmax_kwargs=None,
        feature_activation=nn.ReLU,
    ):
    """
    Utility function to create an @ObservationEncoder from kwargs specified in config.

    Args:
        obs_shapes (OrderedDict): a dictionary that maps modality to
            expected shapes for observations.

        visual_feature_dimension (int): feature dimension to encode images into

        visual_core_class (str): specifies Visual Backbone network for encoding images

        visual_core_kwargs (dict): arguments to pass to @visual_core_class

        obs_randomizer_class (str): specifies a Randomizer class for the input modality

        obs_randomizer_kwargs (dict): kwargs for the observation randomizer

        use_spatial_softmax (bool): if True, introduce a spatial softmax layer at
            the end of the visual backbone network, resulting in a sharp bottleneck
            representation for visual inputs.

        spatial_softmax_kwargs (dict): arguments to pass to spatial softmax layer

        feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
            None to apply no activation.
    """

    ### TODO: clean this part up in the config and args to this function ###
    if visual_core_kwargs is None:
        visual_core_kwargs = dict()
    visual_core_kwargs = deepcopy(visual_core_kwargs)

    if obs_randomizer_class is not None:
        obs_randomizer_class = eval(obs_randomizer_class)
    if obs_randomizer_kwargs is None:
        obs_randomizer_kwargs = dict()

    # use a special class to wrap the visual core and pooling together
    visual_core_kwargs_template = dict(
        visual_core_class=visual_core_class,
        visual_core_kwargs=deepcopy(visual_core_kwargs),
        visual_feature_dimension=visual_feature_dimension
    )
    if use_spatial_softmax:
        visual_core_kwargs_template["pool_class"] = "SpatialSoftmax"
        visual_core_kwargs_template["pool_kwargs"] = deepcopy(spatial_softmax_kwargs)
    else:
        visual_core_kwargs_template["pool_class"] = "SpatialMeanPool"

    enc = ObservationEncoder(feature_activation=feature_activation)
    for k in obs_shapes:
        mod_net_class = None
        mod_net_kwargs = None
        mod_randomizer = None
        if ObsUtils.has_image([k]):
            mod_net_class = "VisualCore"
            mod_net_kwargs = deepcopy(visual_core_kwargs_template)
            # need input shape to create visual core
            mod_net_kwargs["input_shape"] = obs_shapes[k]
            if obs_randomizer_class is not None:
                mod_obs_randomizer_kwargs = deepcopy(obs_randomizer_kwargs)
                mod_obs_randomizer_kwargs["input_shape"] = obs_shapes[k]
                mod_randomizer = obs_randomizer_class(**mod_obs_randomizer_kwargs)

        enc.register_modality(
            mod_name=k,
            mod_shape=obs_shapes[k],
            mod_net_class=mod_net_class,
            mod_net_kwargs=mod_net_kwargs,
            mod_randomizer=mod_randomizer
        )

    enc.make()
    return enc


class ObservationEncoder(Module):
    """
    Module that processes inputs by modality and then concatenates the processed
    modalities together. Each modality is processed with an encoder head network.
    Call @register_modality to register modalities with the encoder and then
    finally call @make to create the encoder networks. 
    """
    def __init__(self, feature_activation=nn.ReLU):
        """
        Args:
            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation. 
        """
        super(ObservationEncoder, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_share_mods = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation
        self._locked = False

    def register_modality(
        self, 
        mod_name,
        mod_shape, 
        mod_net_class=None, 
        mod_net_kwargs=None, 
        mod_net=None, 
        mod_randomizer=None,
        share_mod_net_from=None,
    ):
        """
        Register a modality that this encoder should be responsible for.

        Args:
            mod_name (str): modality name
            mod_shape (int tuple): shape of modality
            mod_net_class (str): name of class in base_nets.py that should be used
                to process this modality before concatenation. Pass None to flatten
                and concatenate the modality directly.
            mod_net_kwargs (dict): arguments to pass to @mod_net_class
            mod_net (Module instance): if provided, use this Module to process the modality
                instead of creating a different net
            mod_randomizer (Randomizer instance): if provided, use this Module to augment modalities
                coming in to the encoder, and possibly augment the processed output as well
            share_mod_net_from (str): if provided, use the same instance of @mod_net_class 
                as another modality. This modality must already exist in this encoder.
                Warning: Note that this does not share the modality randomizer
        """
        assert not self._locked, "ObservationEncoder: @register_modality called after @make"
        assert mod_name not in self.obs_shapes, "ObservationEncoder: modality {} already exists".format(mod_name)

        if mod_net is not None:
            assert isinstance(mod_net, Module), "ObservationEncoder: @mod_net must be instance of Module class"
            assert (mod_net_class is None) and (mod_net_kwargs is None) and (share_mod_net_from is None), \
                "ObservationEncoder: @mod_net provided - ignore other net creation options"

        if share_mod_net_from is not None:
            # share processing with another modality
            assert (mod_net_class is None) and (mod_net_kwargs is None)
            assert share_mod_net_from in self.obs_shapes

        if mod_net_class is not None:
            # convert string into class
            if sys.version_info.major == 3:
                assert isinstance(mod_net_class, str)
            else:
                assert isinstance(mod_net_class, (str, unicode))
            mod_net_class = eval(mod_net_class)

        mod_net_kwargs = deepcopy(mod_net_kwargs) if mod_net_kwargs is not None else {}
        if mod_randomizer is not None:
            assert isinstance(mod_randomizer, Randomizer)
            if mod_net_kwargs is not None:
                # update input shape to visual core
                mod_net_kwargs["input_shape"] = mod_randomizer.output_shape_in(mod_shape)

        self.obs_shapes[mod_name] = mod_shape
        self.obs_nets_classes[mod_name] = mod_net_class
        self.obs_nets_kwargs[mod_name] = mod_net_kwargs
        self.obs_nets[mod_name] = mod_net
        self.obs_randomizers[mod_name] = mod_randomizer
        self.obs_share_mods[mod_name] = share_mod_net_from

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for k in self.obs_shapes:
            if self.obs_nets_classes[k] is not None:
                # create net to process this modality
                self.obs_nets[k] = self.obs_nets_classes[k](**self.obs_nets_kwargs[k])
            elif self.obs_share_mods[k] is not None:
                # make sure net is shared with another modality
                self.obs_nets[k] = self.obs_nets[self.obs_share_mods[k]]

        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.obs_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.obs_shapes. All modalities in
                @self.obs_shapes must be present, but additional modalities
                can also be present.

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
        """
        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        assert set(self.obs_shapes.keys()).issubset(obs_dict), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        # process modalities by order given by @self.obs_shapes
        feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_in(x)
            # maybe process with obs net
            if self.obs_nets[k] is not None:
                x = self.obs_nets[k](x)
                if self.activation is not None:
                    x = self.activation(x)
            # maybe process encoder output with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_out(x)
            # flatten to [B, D]
            x = TensorUtils.flatten(x, begin_axis=1)
            feats.append(x)

        # concatenate all features together
        return torch.cat(feats, dim=-1)

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        """
        feat_dim = 0
        for k in self.obs_shapes:
            feat_shape = self.obs_shapes[k]
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_in(feat_shape)
            if self.obs_nets[k] is not None:
                feat_shape = self.obs_nets[k].output_shape(feat_shape)
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_out(feat_shape)
            feat_dim += int(np.prod(feat_shape))
        return [feat_dim]

    def __repr__(self):
        """
        Pretty print the encoder.
        """
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nModality(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("randomizer={}\n".format(self.obs_randomizers[k]), indent)
            msg += textwrap.indent("net={}\n".format(self.obs_nets[k]), indent)
            msg += textwrap.indent("sharing_from={}\n".format(self.obs_share_mods[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationDecoder(Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. Subclass this
    module in order to implement more complex schemes for generating each
    modality.
    """
    def __init__(
        self,
        decode_shapes,
        input_feat_dim,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation modality to 
                expected shape. This is used to generate output modalities from the
                input features.

            input_feat_dim (int): flat input dimension size
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.obs_shapes[k]) for k in self.obs_shapes }

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](feats)
            output[k] = out.reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nModality(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationGroupEncoder(Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and 
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    """
    def __init__(
        self,
        observation_group_shapes,
        visual_feature_dimension=64,
        visual_core_class='ResNet18Conv',
        visual_core_kwargs=None,
        obs_randomizer_class=None,
        obs_randomizer_kwargs=None,
        use_spatial_softmax=True,
        spatial_softmax_kwargs=None,
        feature_activation=nn.ReLU,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            visual_feature_dimension (int): feature dimension to encode images into

            visual_core_class (str): specifies Visual Backbone network for encoding images

            visual_core_kwargs (dict): arguments to pass to @visual_core_class

            obs_randomizer_class (str): specifies a Randomizer class for the input modality

            obs_randomizer_kwargs (dict): kwargs for the observation randomizer

            use_spatial_softmax (bool): if True, introduce a spatial softmax layer at
                the end of the visual backbone network, resulting in a sharp bottleneck
                representation for visual inputs.

            spatial_softmax_kwargs (dict): arguments to pass to spatial softmax layer

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation. 
        """
        super(ObservationGroupEncoder, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        assert np.all([isinstance(observation_group_shapes[k], OrderedDict) for k in observation_group_shapes])
        
        self.observation_group_shapes = observation_group_shapes

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = obs_encoder_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                visual_feature_dimension=visual_feature_dimension,
                visual_core_class=visual_core_class,
                visual_core_kwargs=visual_core_kwargs,
                obs_randomizer_class=obs_randomizer_class,
                obs_randomizer_kwargs=obs_randomizer_kwargs,
                use_spatial_softmax=use_spatial_softmax,
                spatial_softmax_kwargs=spatial_softmax_kwargs,
                feature_activation=feature_activation,
            )

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with 
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """

        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(inputs), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            outputs.append(
                self.nets[obs_group].forward(inputs[obs_group])
            )

        return torch.cat(outputs, dim=-1)

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.observation_group_shapes:
            msg += '\n'
            indent = ' ' * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class MIMO_MLP(Module):
    """
    Extension to MLP to accept multiple observation dictionaries as input and
    to output dictionaries of tensors. Inputs are specified as a dictionary of 
    observation dictionaries, with each key corresponding to an observation group.

    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating 
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """
    def __init__(
        self, 
        input_obs_group_shapes,
        output_shapes, 
        layer_dims,
        layer_func=nn.Linear, 
        activation=nn.ReLU,
        visual_feature_dimension=64,
        visual_core_class='ResNet18Conv',
        visual_core_kwargs=None,
        obs_randomizer_class=None,
        obs_randomizer_kwargs=None,
        use_spatial_softmax=False,
        spatial_softmax_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            layer_dims ([int]): sequence of integers for the MLP hidden layer sizes

            layer_func: mapping per MLP layer - defaults to Linear

            activation: non-linearity per MLP layer - defaults to ReLU

            visual_feature_dimension (int): feature dimension to encode images into

            visual_core_class (str): specifies Visual Backbone network for encoding images

            visual_core_kwargs (dict): arguments to pass to @visual_core_class

            obs_randomizer_class (str): specifies a Randomizer class for the input modality

            obs_randomizer_kwargs (dict): kwargs for the observation randomizer

            use_spatial_softmax (bool): if True, introduce a spatial softmax layer at
                the end of the visual backbone network, resulting in a sharp bottleneck
                representation for visual inputs.

            spatial_softmax_kwargs (dict): arguments to pass to spatial softmax layer
        """
        super(MIMO_MLP, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            visual_feature_dimension=visual_feature_dimension,
            visual_core_class=visual_core_class,
            visual_core_kwargs=visual_core_kwargs,
            obs_randomizer_class=obs_randomizer_class,
            obs_randomizer_kwargs=obs_randomizer_kwargs,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_kwargs=spatial_softmax_kwargs, 
        )

        # flat encoder output dimension
        mlp_input_dim = self.nets["encoder"].output_shape()[0]

        # intermediate MLP layers
        self.nets["mlp"] = MLP(
            input_dim=mlp_input_dim,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=activation,
            output_activation=activation, # make sure non-linearity is applied before decoder
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=layer_dims[-1],
        )

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes.

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes
        """
        enc_outputs = self.nets["encoder"](**inputs)
        mlp_out = self.nets["mlp"](enc_outputs)
        return self.nets["decoder"](mlp_out)

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nmlp={}".format(self.nets["mlp"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class RNN_MIMO_MLP(Module):
    """
    A wrapper class for a multi-step RNN and a per-step MLP and a decoder.

    Structure: [encoder -> rnn -> mlp -> decoder]

    All temporal inputs are processed by a shared @ObservationGroupEncoder,
    followed by an RNN, and then a per-step multi-output MLP. 
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        mlp_activation=nn.ReLU,
        mlp_layer_func=nn.Linear,
        per_step=True,
        visual_feature_dimension=64,
        visual_core_class='ResNet18Conv',
        visual_core_kwargs=None,
        obs_randomizer_class=None,
        obs_randomizer_kwargs=None,
        use_spatial_softmax=False,
        spatial_softmax_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the rnn model

            per_step (bool): if True, apply the MLP and observation decoder into @output_shapes
                at every step of the RNN. Otherwise, apply them to the final hidden state of the 
                RNN. 

            visual_feature_dimension (int): feature dimension to encode images into

            visual_core_class (str): specifies Visual Backbone network for encoding images

            visual_core_kwargs (dict): arguments to pass to @visual_core_class

            obs_randomizer_class (str): specifies a Randomizer class for the input modality

            obs_randomizer_kwargs (dict): kwargs for the observation randomizer

            use_spatial_softmax (bool): if True, introduce a spatial softmax layer at
                the end of the visual backbone network, resulting in a sharp bottleneck
                representation for visual inputs.

            spatial_softmax_kwargs (dict): arguments to pass to spatial softmax layer
        """
        super(RNN_MIMO_MLP, self).__init__()
        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)
        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes
        self.per_step = per_step

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            visual_feature_dimension=visual_feature_dimension,
            visual_core_class=visual_core_class,
            visual_core_kwargs=visual_core_kwargs,
            obs_randomizer_class=obs_randomizer_class,
            obs_randomizer_kwargs=obs_randomizer_kwargs,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_kwargs=spatial_softmax_kwargs,
        )

        # flat encoder output dimension
        rnn_input_dim = self.nets["encoder"].output_shape()[0]

        # bidirectional RNNs mean that the output of RNN will be twice the hidden dimension
        rnn_is_bidirectional = rnn_kwargs.get("bidirectional", False)
        num_directions = int(rnn_is_bidirectional) + 1 # 2 if bidirectional, 1 otherwise
        rnn_output_dim = num_directions * rnn_hidden_dim

        per_step_net = None
        self._has_mlp = (len(mlp_layer_dims) > 0)
        if self._has_mlp:
            self.nets["mlp"] = MLP(
                input_dim=rnn_output_dim,
                output_dim=mlp_layer_dims[-1],
                layer_dims=mlp_layer_dims[:-1],
                output_activation=mlp_activation,
                layer_func=mlp_layer_func
            )
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=mlp_layer_dims[-1],
            )
            if self.per_step:
                per_step_net = Sequential(self.nets["mlp"], self.nets["decoder"])
        else:
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=rnn_output_dim,
            )
            if self.per_step:
                per_step_net = self.nets["decoder"]

        # core network
        self.nets["rnn"] = RNN_Base(
            input_dim=rnn_input_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            per_step_net=per_step_net,
            rnn_kwargs=rnn_kwargs
        )

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)

        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        return self.nets["rnn"].get_rnn_init_state(batch_size, device=device)

    def output_shape(self, input_shape):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.

        Args:
            input_shape (dict): dictionary of dictionaries, where each top-level key
                corresponds to an observation group, and the low-level dictionaries
                specify the shape for each modality in an observation dictionary
        """

        # infers temporal dimension from input shape
        obs_group = list(self.input_obs_group_shapes.keys())[0]
        mod = list(self.input_obs_group_shapes[obs_group].keys())[0]
        T = input_shape[obs_group][mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="RNN_MIMO_MLP: input_shape inconsistent in temporal dimension")
        # returns a dictionary instead of list since outputs are dictionaries
        return { k : [T] + list(self.output_shapes[k]) for k in self.output_shapes }

    def forward(self, rnn_init_state=None, return_state=False, **inputs):
        """
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.

            rnn_init_state: rnn hidden state, initialize to zero state if set to None

            return_state (bool): whether to return hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.

            rnn_state (torch.Tensor or tuple): return the new rnn state (if @return_state)
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        # use encoder to extract flat rnn inputs
        rnn_inputs = TensorUtils.time_distributed(inputs, self.nets["encoder"], inputs_as_kwargs=True)
        assert rnn_inputs.ndim == 3  # [B, T, D]
        if self.per_step:
            return self.nets["rnn"].forward(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state)
        
        # apply MLP + decoder to last RNN output
        outputs = self.nets["rnn"].forward(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state)
        if return_state:
            outputs, rnn_state = outputs

        assert outputs.ndim == 3 # [B, T, D]
        if self._has_mlp:
            outputs = self.nets["decoder"](self.mlp(outputs[:, -1]))
        else:
            outputs = self.nets["decoder"](outputs[:, -1])

        if return_state:
            return outputs, rnn_state
        return outputs

    def forward_step(self, rnn_state, **inputs):
        """
        Unroll network over a single timestep.

        Args:
            inputs (dict): expects same modalities as @self.input_shapes, with
                additional batch dimension (but NOT time), since this is a 
                single time step.

            rnn_state (torch.Tensor): rnn hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Does not contain time dimension.

            rnn_state: return the new rnn state
        """
        # ensure that the only extra dimension is batch dim, not temporal dim 
        assert np.all([inputs[k].ndim - 1 == len(self.input_shapes[k]) for k in self.input_shapes])

        inputs = TensorUtils.to_sequence(inputs)
        outputs, rnn_state = self.forward(
            inputs, 
            rnn_init_state=rnn_state,
            return_state=True,
        )
        if self.per_step:
            # if outputs are not per-step, the time dimension is already reduced
            outputs = outputs[:, 0]
        return outputs, rnn_state

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\n\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nrnn={}".format(self.nets["rnn"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg
