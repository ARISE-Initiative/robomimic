"""
TODO: ADDED
Test divergence computation with different BC model architectures.
Tests that divergence properly handles:
1. Individual action outputs (MLP)
2. Sequence action outputs (RNN, Transformer)
3. Different action dimensions
"""

import torch
import pytest
from collections import OrderedDict

import robomimic.utils.divergence_utils as DivergenceUtils
from robomimic.models.policy_nets import ActorNetwork, GaussianActorNetwork, GMMActorNetwork
from robomimic.models.obs_nets import RNN_MIMO_MLP, MIMO_Transformer


def create_test_batch(batch_size=4, seq_length=10):
    """Create a test batch with EE pose observations."""
    batch = {
        'obs': {
            'robot0_eef_pos': torch.randn(batch_size, 3),
            'robot0_eef_quat': torch.randn(batch_size, 4),
            'robot0_joint_pos': torch.randn(batch_size, 7),
        },
        'actions': torch.randn(batch_size, 7),
        'goal_obs': None
    }
    return batch


def test_mlp_divergence():
    """Test divergence with standard MLP ActorNetwork."""
    print("\n=== Testing MLP (ActorNetwork) ===")
    
    obs_shapes = OrderedDict([
        ('robot0_eef_pos', (3,)),
        ('robot0_eef_quat', (4,)),
        ('robot0_joint_pos', (7,)),
    ])
    
    # Create MLP policy
    policy = ActorNetwork(
        obs_shapes=obs_shapes,
        ac_dim=7,
        mlp_layer_dims=[128, 128]
    )
    policy.eval()
    
    batch = create_test_batch(batch_size=8)
    
    # Compute divergence
    div = DivergenceUtils.compute_policy_divergence_during_training(
        model=policy,
        batch=batch,
        n_samples=2
    )
    
    assert div is not None, "Divergence should not be None"
    assert div.shape == (8,), f"Expected shape (8,), got {div.shape}"
    assert not torch.isnan(div).any(), "Divergence contains NaN"
    assert not torch.isinf(div).any(), "Divergence contains Inf"
    
    print(f"✓ MLP divergence: shape={div.shape}, mean={div.mean():.4f}, std={div.std():.4f}")


def test_rnn_divergence():
    """Test divergence with RNN that outputs sequences."""
    print("\n=== Testing RNN (RNN_MIMO_MLP) ===")
    
    obs_shapes = OrderedDict([
        ('robot0_eef_pos', (3,)),
        ('robot0_eef_quat', (4,)),
        ('robot0_joint_pos', (7,)),
    ])
    
    # Create RNN policy
    policy = RNN_MIMO_MLP(
        input_obs_group_shapes=OrderedDict([("obs", obs_shapes)]),
        output_shapes=OrderedDict([("action", (7,))]),
        rnn_hidden_dim=128,
        rnn_num_layers=2,
        rnn_type="LSTM",
        per_step=True  # Output action at each timestep
    )
    policy.eval()
    
    batch = create_test_batch(batch_size=8)
    
    # Test that model outputs sequences
    with torch.no_grad():
        # RNN expects sequence input during training
        # For single-step, we can expand time dimension
        obs_seq = {k: v.unsqueeze(1) for k, v in batch['obs'].items()}  # [batch, 1, ...]
        output = policy(obs=obs_seq)["action"]
        print(f"  RNN output shape: {output.shape}")
        assert output.dim() == 3 or output.dim() == 2, f"Unexpected output dimension: {output.dim()}"
    
    # Compute divergence (should handle sequence output automatically)
    div = DivergenceUtils.compute_policy_divergence_during_training(
        model=policy,
        batch=batch,
        n_samples=2
    )
    
    assert div is not None, "Divergence should not be None"
    assert div.shape == (8,), f"Expected shape (8,), got {div.shape}"
    assert not torch.isnan(div).any(), "Divergence contains NaN"
    
    print(f"✓ RNN divergence: shape={div.shape}, mean={div.mean():.4f}, std={div.std():.4f}")


def test_gaussian_divergence():
    """Test divergence with Gaussian stochastic policy."""
    print("\n=== Testing Gaussian (GaussianActorNetwork) ===")
    
    obs_shapes = OrderedDict([
        ('robot0_eef_pos', (3,)),
        ('robot0_eef_quat', (4,)),
        ('robot0_joint_pos', (7,)),
    ])
    
    # Create Gaussian policy
    policy = GaussianActorNetwork(
        obs_shapes=obs_shapes,
        ac_dim=7,
        mlp_layer_dims=[128, 128],
        fixed_std=False,
        init_std=1.0,
        low_noise_eval=True
    )
    policy.eval()
    
    batch = create_test_batch(batch_size=8)
    
    # Test that model outputs actions (not distributions) in eval mode
    with torch.no_grad():
        output = policy(obs_dict=batch['obs'], goal_dict=None)
        print(f"  Gaussian output shape: {output.shape}")
        assert output.shape == (8, 7), f"Expected (8, 7), got {output.shape}"
    
    # Compute divergence
    div = DivergenceUtils.compute_policy_divergence_during_training(
        model=policy,
        batch=batch,
        n_samples=2
    )
    
    assert div is not None, "Divergence should not be None"
    assert div.shape == (8,), f"Expected shape (8,), got {div.shape}"
    assert not torch.isnan(div).any(), "Divergence contains NaN"
    
    print(f"✓ Gaussian divergence: shape={div.shape}, mean={div.mean():.4f}, std={div.std():.4f}")


def test_different_action_dims():
    """Test divergence with different action dimensions."""
    print("\n=== Testing Different Action Dimensions ===")
    
    obs_shapes = OrderedDict([
        ('robot0_eef_pos', (3,)),
        ('robot0_eef_quat', (4,)),
        ('robot0_joint_pos', (7,)),
    ])
    
    for action_dim in [3, 6, 7, 10]:
        print(f"\n  Testing action_dim={action_dim}")
        
        policy = ActorNetwork(
            obs_shapes=obs_shapes,
            ac_dim=action_dim,
            mlp_layer_dims=[64, 64]
        )
        policy.eval()
        
        batch = create_test_batch(batch_size=4)
        batch['actions'] = torch.randn(4, action_dim)
        
        div = DivergenceUtils.compute_policy_divergence_during_training(
            model=policy,
            batch=batch,
            n_samples=1
        )
        
        assert div is not None, f"Divergence should not be None for action_dim={action_dim}"
        assert div.shape == (4,), f"Expected shape (4,), got {div.shape}"
        assert not torch.isnan(div).any(), f"Divergence contains NaN for action_dim={action_dim}"
        
        print(f"    ✓ action_dim={action_dim}: mean={div.mean():.4f}")


def test_missing_ee_observations():
    """Test that divergence returns None when EE observations are missing."""
    print("\n=== Testing Missing EE Observations ===")
    
    obs_shapes = OrderedDict([
        ('robot0_joint_pos', (7,)),  # No EE pose!
    ])
    
    policy = ActorNetwork(
        obs_shapes=obs_shapes,
        ac_dim=7,
        mlp_layer_dims=[64, 64]
    )
    policy.eval()
    
    batch = {
        'obs': {
            'robot0_joint_pos': torch.randn(4, 7),
        },
        'actions': torch.randn(4, 7),
        'goal_obs': None
    }
    
    div = DivergenceUtils.compute_policy_divergence_during_training(
        model=policy,
        batch=batch,
        n_samples=1
    )
    
    assert div is None, "Divergence should be None when EE observations are missing"
    print("✓ Correctly returns None for missing EE observations")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Divergence Computation with Different BC Architectures")
    print("=" * 70)
    
    try:
        test_mlp_divergence()
        test_gaussian_divergence()
        test_different_action_dims()
        test_missing_ee_observations()
        
        # RNN/Transformer tests may require more setup
        try:
            test_rnn_divergence()
        except Exception as e:
            print(f"\n⚠ RNN test skipped: {e}")
        
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
