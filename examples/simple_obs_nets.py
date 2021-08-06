"""
A simple example showing how to construct an ObservationEncoder for processing multiple input modalities.
This is purely for instructional purposes, in case others would like to make use of or extend the
functionality.
"""

from collections import OrderedDict

import torch
from robomimic.models.obs_nets import ObservationEncoder, CropRandomizer, MLP, VisualCore, ObservationDecoder
import robomimic.utils.tensor_utils as TensorUtils


def simple_obs_example():
    obs_encoder = ObservationEncoder(feature_activation=torch.nn.ReLU)

    # There are two ways to construct the network for processing a input modality.

    # 1. Construct through keyword args and class name

    # Assume we are processing image input of shape (3, 224, 224).
    camera1_shape = [3, 224, 224]

    # We will use a reconfigurable image processing backbone VisualCore to process the input image modality
    mod_net_class = "VisualCore"  # this is defined in models/base_nets.py

    # kwargs for VisualCore network
    mod_net_kwargs = {
        "input_shape": camera1_shape,
        "visual_core_class": "ResNet18Conv",  # use ResNet18 as the visualcore backbone
        "visual_core_kwargs": {"pretrained": False, "input_coord_conv": False},
        "pool_class": "SpatialSoftmax",  # use spatial softmax to regularize the model output
        "pool_kwargs": {"num_kp": 32}
    }

    # register the network for processing the modality
    obs_encoder.register_modality(
        mod_name="camera1",
        mod_shape=camera1_shape,
        mod_net_class=mod_net_class,
        mod_net_kwargs=mod_net_kwargs
    )

    # 2. Alternatively, we could initialize the modality network outside of the ObservationEncoder

    # The image doesn't have to be of the same shape
    camera2_shape = [3, 160, 240]

    # We could also attach an observation randomizer to perturb the input modality before sending to the network
    image_randomizer = CropRandomizer(input_shape=camera2_shape, crop_height=140, crop_width=220)

    # the cropper will alter the input shape
    mod_net_kwargs["input_shape"] = image_randomizer.output_shape_in(camera2_shape)
    mod_net = eval(mod_net_class)(**mod_net_kwargs)

    obs_encoder.register_modality(
        mod_name="camera2",
        mod_shape=camera2_shape,
        mod_net=mod_net,
        mod_randomizer=image_randomizer
    )

    # ObservationEncoder also supports weight sharing between modalities
    camera3_shape = [3, 224, 224]
    obs_encoder.register_modality(
        mod_name="camera3",
        mod_shape=camera3_shape,
        share_mod_net_from="camera1"
    )

    # We could mix low-dimensional observation, e.g., proprioception signal, in the encoder
    proprio_shape = [12]
    mod_net = MLP(input_dim=12, output_dim=32, layer_dims=(128,), output_activation=None)
    obs_encoder.register_modality(
        mod_name="proprio",
        mod_shape=proprio_shape,
        mod_net=mod_net
    )

    # Finally, construct the observation encoder
    obs_encoder.make()

    # pretty-print the observation encoder
    print(obs_encoder)

    # Construct fake inputs
    inputs = {
        "camera1": torch.randn(camera1_shape),
        "camera2": torch.randn(camera2_shape),
        "camera3": torch.randn(camera3_shape),
        "proprio": torch.randn(proprio_shape)
    }

    # Add a batch dimension
    inputs = TensorUtils.to_batch(inputs)

    # Send to GPU if applicable
    if torch.cuda.is_available():
        inputs = TensorUtils.to_device(inputs, torch.device("cuda:0"))
        obs_encoder.cuda()

    # output from each modality network is concatenated as a flat vector.
    # The concatenation order is the same as the modalities are registered
    obs_feature = obs_encoder(inputs)

    print(obs_feature.shape)

    # A convenient wrapper for decoding the feature vector to named output is ObservationDecoder
    obs_decoder = ObservationDecoder(
        input_feat_dim=obs_encoder.output_shape()[0],
        decode_shapes=OrderedDict({"action": (7,)})
    )

    print(obs_decoder(obs_feature))


if __name__ == "__main__":
    simple_obs_example()
