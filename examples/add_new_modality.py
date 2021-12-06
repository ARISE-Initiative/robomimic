"""
A simple example showing how to add custom observation modalities, and custom
observation networks (EncoderCore, ObservationRandomizer, etc.) as well.
We also show how to use your custom classes directly in a config, and link them to
your environment's observations
"""

import numpy as np
import torch
import robomimic
from robomimic.models import EncoderCore, Randomizer
from robomimic.utils.obs_utils import Modality, ScanModality
from robomimic.config.bc_config import BCConfig
import robomimic.utils.tensor_utils as TensorUtils


# Let's create a new modality to handle observation modalities, which will be interpreted as
# single frame images, with raw shape (H, W) in range [0, 255]
class CustomImageModality(Modality):
    # We must define the class string name to reference this modality with the @name attribute
    name = "custom_image"

    # We must define two class methods: a processor and an unprocessor method. The processor
    # method should map the raw observations (a numpy array OR torch tensor) into a form / shape suitable for learning,
    # and the unprocess method should do the inverse operation
    @classmethod
    def _default_obs_processor(cls, obs):
        # We add a channel dimension and normalize them to be in range [-1, 1]
        return (obs / 255.0 - 0.5) * 2

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        # We do the reverse
        return ((obs / 2) + 0.5) * 255.0


# You can also modify pre-existing modalities as well. Let's say you have scan data that pads the ends with a 0, so we
# want to pre-process those scans in a different way. We can specify a custom processor / unprocessor
# method that will override the default one (assumes obs are a flat 1D array):
def custom_scan_processor(obs):
    # Trim the padded ends
    return obs[1:-1]


def custom_scan_unprocessor(obs):
    # Re-add the padding
    # Note: need to check type
    return np.concatenate([np.zeros(1), obs, np.zeros(1)]) if isinstance(obs, np.ndarray) else \
        torch.concat([torch.zeros(1), obs, torch.zeros(1)])


# Override the default functions for ScanModality
ScanModality.set_obs_processor(processor=custom_scan_processor)
ScanModality.set_obs_unprocessor(unprocessor=custom_scan_unprocessor)


# Let's now create a custom encoding class for the custom image modality
class CustomImageEncoderCore(EncoderCore):
    # For simplicity, this will be a pass-through with some simple kwargs
    def __init__(
            self,
            input_shape,        # Required, will be inferred automatically at runtime

            # Any args below here you can specify arbitrarily
            welcome_str,
    ):
        # Always need to run super init first and pass in input_shape
        super().__init__(input_shape=input_shape)

        # Anything else should can be custom to your class
        # Let's print out the welcome string
        print(f"Welcome! {welcome_str}")

    # We need to always specify the output shape from this model, based on a given input_shape
    def output_shape(self, input_shape=None):
        # this is just a pass-through, so we return input_shape
        return input_shape

    # we also need to specify the forward pass for this network
    def forward(self, inputs):
        # just a pass through again
        return inputs


# Let's also create a custom randomizer class for randomizing our observations
class CustomImageRandomizer(Randomizer):
    """
    A simple example of a randomizer - we make @num_rand copies of each image in the batch,
    and add some small uniform noise to each. All randomized images will then get passed
    through the network, resulting in outputs corresponding to each copy - we will pool
    these outputs across the copies with a simple average.
    """
    def __init__(
        self,
        input_shape,
        num_rand=1,
        noise_scale=0.01,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            num_rand (int): number of random images to create on each forward pass
            noise_scale (float): magnitude of uniform noise to apply
        """
        super(CustomImageRandomizer, self).__init__()

        assert len(input_shape) == 3 # (C, H, W)

        self.input_shape = input_shape
        self.num_rand = num_rand
        self.noise_scale = noise_scale

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # @forward_in takes (B, C, H, W) -> (B, N, C, H, W) -> (B * N, C, H, W).
        # since only the batch dimension changes, and @input_shape does not include batch
        # dimension, we indicate that the non-batch dimensions don't change
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        
        # since the @forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Make N copies of each image, add random noise to each, and move
        copies into batch dimension to ensure compatibility with rest
        of network.
        """

        # note the use of @self.training to ensure no randomization at test-time
        if self.training:

            # make N copies of the images [B, C, H, W] -> [B, N, C, H, W]
            out = TensorUtils.unsqueeze_expand_at(inputs, size=self.num_rand, dim=1)

            # add random noise to each copy
            out = out + self.noise_scale * (2. * torch.rand_like(out) - 1.)

            # reshape [B, N, C, H, W] -> [B * N, C, H, W] to ensure network forward pass is unchanged
            return TensorUtils.join_dimensions(out, 0, 1)
        return inputs

    def forward_out(self, inputs):
        """
        Pools outputs across the copies by averaging them. It does this by splitting
        the outputs from shape [B * N, ...] -> [B, N, ...] and then averaging across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """

        # note the use of @self.training to ensure no randomization at test-time
        if self.training:
            batch_size = (inputs.shape[0] // self.num_rand)
            out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0, 
                target_dims=(batch_size, self.num_rand))
            return out.mean(dim=1)
        return inputs

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, num_rand={}, noise_scale={})".format(
            self.input_shape, self.num_rand, self.noise_scale)
        return msg


if __name__ == "__main__":
    # Now, we can directly reference the classes in our config!
    config = BCConfig()
    config.observation.encoder.custom_image.core_class = "CustomImageEncoderCore"       # Custom class, in string form
    config.observation.encoder.custom_image.core_kwargs.welcome_str = "hi there!"       # Any custom arguments, of any primitive type that is json-able
    config.observation.encoder.custom_image.obs_randomizer_class = "CustomImageRandomizer"
    config.observation.encoder.custom_image.obs_randomizer_kwargs.num_rand = 3
    config.observation.encoder.custom_image.obs_randomizer_kwargs.noise_scale = 0.05

    # We can also directly use this new modality and associate dataset / observation keys with it!
    config.observation.modalities.obs.custom_image = ["my_image1", "my_image2"]
    config.observation.modalities.goal.custom_image = ["my_image2", "my_image3"]

    # Let's view our config
    print(config)

    # That's it! Now we can pass this config into our training script, and robomimic will directly use our
    # custom modality + encoder network
