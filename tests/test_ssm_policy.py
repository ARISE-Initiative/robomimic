"""
Unit and integration tests for SSM (State-Space Model) policy components.

Tests cover:
- SelectiveSSMBlock forward pass
- SSM_Backbone instantiation and forward pass
- MIMO_SSM integration
- SSMActorNetwork and SSMGMMActorNetwork
- BC_SSM and BC_SSM_GMM algorithms
"""

import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from robomimic.models.ssm_nets import SelectiveSSMBlock, SSM_Backbone
from robomimic.models.obs_nets import MIMO_SSM
from robomimic.models.policy_nets import SSMActorNetwork, SSMGMMActorNetwork
from robomimic.config.bc_config import BCConfig


class TestSelectiveSSMBlock(unittest.TestCase):
    """Test SelectiveSSMBlock functionality."""

    def setUp(self):
        self.embed_dim = 128
        self.batch_size = 4
        self.seq_len = 10
        self.block = SelectiveSSMBlock(
            embed_dim=self.embed_dim,
            state_dim=16,
            conv_dim=4,
            expand_factor=2,
            dropout=0.1,
        )

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = self.block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_output_shape_method(self):
        """Test output_shape method."""
        input_shape = [self.seq_len, self.embed_dim]
        output_shape = self.block.output_shape(input_shape)
        
        self.assertEqual(output_shape, input_shape)


class TestSSMBackbone(unittest.TestCase):
    """Test SSM_Backbone functionality."""

    def setUp(self):
        self.embed_dim = 128
        self.context_length = 10
        self.batch_size = 4
        self.backbone = SSM_Backbone(
            embed_dim=self.embed_dim,
            context_length=self.context_length,
            num_layers=4,
            state_dim=16,
            conv_dim=4,
            expand_factor=2,
            dropout=0.1,
        )

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        x = torch.randn(self.batch_size, self.context_length, self.embed_dim)
        output = self.backbone(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.context_length, self.embed_dim))

    def test_parameter_count(self):
        """Test that backbone has trainable parameters."""
        param_count = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        self.assertGreater(param_count, 0)


class TestMIMO_SSM(unittest.TestCase):
    """Test MIMO_SSM multi-input multi-output SSM."""

    def setUp(self):
        self.obs_shapes = OrderedDict(
            image=(3, 84, 84),
            proprio=(10,),
        )
        self.output_shapes = OrderedDict(
            action=(7,),
        )
        self.input_obs_group_shapes = OrderedDict(
            obs=self.obs_shapes,
        )
        self.batch_size = 4
        self.context_length = 10

    def test_instantiation(self):
        """Test MIMO_SSM can be instantiated."""
        net = MIMO_SSM(
            input_obs_group_shapes=self.input_obs_group_shapes,
            output_shapes=self.output_shapes,
            ssm_embed_dim=256,
            ssm_num_layers=4,
            ssm_context_length=self.context_length,
            ssm_state_dim=16,
            ssm_conv_dim=4,
            ssm_dropout=0.1,
        )
        
        self.assertIsInstance(net, MIMO_SSM)

    def test_forward_pass(self):
        """Test forward pass with multi-modal observations."""
        net = MIMO_SSM(
            input_obs_group_shapes=self.input_obs_group_shapes,
            output_shapes=self.output_shapes,
            ssm_embed_dim=256,
            ssm_num_layers=4,
            ssm_context_length=self.context_length,
            ssm_state_dim=16,
            ssm_conv_dim=4,
            ssm_dropout=0.1,
        )

        obs_dict = dict(
            image=torch.randn(self.batch_size, self.context_length, 3, 84, 84),
            proprio=torch.randn(self.batch_size, self.context_length, 10),
        )

        outputs = net(obs=obs_dict)
        
        self.assertIn("action", outputs)
        self.assertEqual(outputs["action"].shape, (self.batch_size, self.context_length, 7))


class TestSSMActorNetwork(unittest.TestCase):
    """Test SSMActorNetwork policy network."""

    def setUp(self):
        self.obs_shapes = OrderedDict(
            image=(3, 84, 84),
            proprio=(10,),
        )
        self.ac_dim = 7
        self.batch_size = 4
        self.context_length = 10

    def test_instantiation(self):
        """Test SSMActorNetwork can be instantiated."""
        net = SSMActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            ssm_embed_dim=256,
            ssm_num_layers=4,
            ssm_context_length=self.context_length,
            ssm_state_dim=16,
            ssm_conv_dim=4,
            ssm_dropout=0.1,
        )
        
        self.assertIsInstance(net, SSMActorNetwork)

    def test_forward_pass(self):
        """Test forward pass produces actions in [-1, 1]."""
        net = SSMActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            ssm_embed_dim=256,
            ssm_num_layers=4,
            ssm_context_length=self.context_length,
            ssm_state_dim=16,
            ssm_conv_dim=4,
            ssm_dropout=0.1,
        )

        obs_dict = dict(
            image=torch.randn(self.batch_size, self.context_length, 3, 84, 84),
            proprio=torch.randn(self.batch_size, self.context_length, 10),
        )

        actions = net(obs_dict)
        
        self.assertEqual(actions.shape, (self.batch_size, self.context_length, self.ac_dim))
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))


class TestSSMGMMActorNetwork(unittest.TestCase):
    """Test SSMGMMActorNetwork GMM policy network."""

    def setUp(self):
        self.obs_shapes = OrderedDict(
            image=(3, 84, 84),
            proprio=(10,),
        )
        self.ac_dim = 7
        self.num_modes = 5
        self.batch_size = 4
        self.context_length = 10

    def test_instantiation(self):
        """Test SSMGMMActorNetwork can be instantiated."""
        net = SSMGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            ssm_embed_dim=256,
            ssm_num_layers=4,
            ssm_context_length=self.context_length,
            ssm_state_dim=16,
            ssm_conv_dim=4,
            ssm_dropout=0.1,
            num_modes=self.num_modes,
            min_std=0.01,
            std_activation="softplus",
            low_noise_eval=True,
        )
        
        self.assertIsInstance(net, SSMGMMActorNetwork)

    def test_forward_train(self):
        """Test forward_train produces GMM distribution."""
        net = SSMGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            ssm_embed_dim=256,
            ssm_num_layers=4,
            ssm_context_length=self.context_length,
            ssm_state_dim=16,
            ssm_conv_dim=4,
            ssm_dropout=0.1,
            num_modes=self.num_modes,
            min_std=0.01,
            std_activation="softplus",
            low_noise_eval=True,
        )

        obs_dict = dict(
            image=torch.randn(self.batch_size, self.context_length, 3, 84, 84),
            proprio=torch.randn(self.batch_size, self.context_length, 10),
        )

        dists = net.forward_train(obs_dict)
        
        # Distribution should have batch shape [B, T]
        self.assertEqual(len(dists.batch_shape), 2)
        self.assertEqual(dists.batch_shape[0], self.batch_size)
        self.assertEqual(dists.batch_shape[1], self.context_length)

    def test_forward_sample(self):
        """Test forward sampling produces valid actions."""
        net = SSMGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            ssm_embed_dim=256,
            ssm_num_layers=4,
            ssm_context_length=self.context_length,
            ssm_state_dim=16,
            ssm_conv_dim=4,
            ssm_dropout=0.1,
            num_modes=self.num_modes,
            min_std=0.01,
            std_activation="softplus",
            low_noise_eval=True,
        )
        net.eval()

        obs_dict = dict(
            image=torch.randn(self.batch_size, self.context_length, 3, 84, 84),
            proprio=torch.randn(self.batch_size, self.context_length, 10),
        )

        actions = net(obs_dict)
        
        self.assertEqual(actions.shape, (self.batch_size, self.context_length, self.ac_dim))


class TestBCConfigSSM(unittest.TestCase):
    """Test BCConfig SSM configuration."""

    def test_ssm_config_exists(self):
        """Test that SSM config section exists in BCConfig."""
        config = BCConfig()
        
        self.assertTrue(hasattr(config.algo, 'ssm'))
        self.assertTrue(hasattr(config.algo.ssm, 'enabled'))
        self.assertTrue(hasattr(config.algo.ssm, 'context_length'))
        self.assertTrue(hasattr(config.algo.ssm, 'embed_dim'))
        self.assertTrue(hasattr(config.algo.ssm, 'num_layers'))
        self.assertTrue(hasattr(config.algo.ssm, 'state_dim'))
        self.assertTrue(hasattr(config.algo.ssm, 'conv_dim'))
        self.assertTrue(hasattr(config.algo.ssm, 'dropout'))

    def test_ssm_config_defaults(self):
        """Test SSM config default values."""
        config = BCConfig()
        
        self.assertFalse(config.algo.ssm.enabled)
        self.assertEqual(config.algo.ssm.context_length, 10)
        self.assertEqual(config.algo.ssm.embed_dim, 256)
        self.assertEqual(config.algo.ssm.num_layers, 4)
        self.assertEqual(config.algo.ssm.state_dim, 16)
        self.assertEqual(config.algo.ssm.conv_dim, 4)
        self.assertEqual(config.algo.ssm.dropout, 0.1)


if __name__ == "__main__":
    unittest.main()
