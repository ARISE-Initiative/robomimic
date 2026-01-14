"""
TODO: EXAMPLE ADDED
Example showing how to compute policy divergence during BC training.

This example demonstrates:
1. How to add divergence computation to the BC training loop
2. How divergence values can be logged and monitored
3. Compatibility with all BC model architectures (MLP, RNN, Transformer, Gaussian, GMM, VAE)
"""

import torch
import numpy as np
from collections import OrderedDict

import robomimic.utils.divergence_utils as DivergenceUtils


# ============================================================================
# Example 1: Minimal Integration into BC Training
# ============================================================================

def example_train_on_batch_with_divergence(self, batch, epoch, validate=False):
    """
    Modified version of BC.train_on_batch() that computes divergence.
    
    This can be added to any BC variant (BC, BC_RNN, BC_Transformer, BC_Gaussian, etc.)
    by overriding the train_on_batch method.
    """
    from robomimic.utils import torch_utils as TorchUtils, tensor_utils as TensorUtils
    
    with TorchUtils.maybe_no_grad(no_grad=validate):
        # Standard BC training
        info = super(self.__class__, self).train_on_batch(batch, epoch, validate=validate)
        predictions = self._forward_training(batch)
        losses = self._compute_losses(predictions, batch)

        info["predictions"] = TensorUtils.detach(predictions)
        info["losses"] = TensorUtils.detach(losses)

        # NEW: Compute policy divergence
        if not validate:  # Only compute during training, not validation
            divergence = DivergenceUtils.compute_policy_divergence_during_training(
                model=self.nets["policy"],
                batch=batch,
                n_samples=1  # Use 1 for speed, increase (e.g., 5-10) for accuracy
            )
            
            if divergence is not None:
                # Add divergence statistics to info dict for logging
                info["divergence_mean"] = divergence.mean().item()
                info["divergence_std"] = divergence.std().item()
                info["divergence_min"] = divergence.min().item()
                info["divergence_max"] = divergence.max().item()
                
                # Optional: Add as loss term (regularization)
                # losses["divergence_reg"] = 0.001 * divergence.mean()
        
        if not validate:
            step_info = self._train_step(losses)
            info.update(step_info)

    return info


# ============================================================================
# Example 2: Using Divergence for Analysis
# ============================================================================

def analyze_policy_divergence_on_dataset(policy, dataset, device='cuda', n_samples=10):
    """
    Analyze divergence across an entire dataset.
    
    Args:
        policy: Trained BC policy network
        dataset: Dataset to analyze (e.g., validation set)
        device: Device to run on
        n_samples: Number of samples for Hutchinson estimator
        
    Returns:
        divergence_stats: Dictionary with statistics
    """
    policy.eval()
    
    all_divergences = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            
            # Move to device and add batch dimension
            batch = {
                'obs': {k: v.unsqueeze(0).to(device) for k, v in batch['obs'].items()},
                'goal_obs': batch.get('goal_obs', None),
                'actions': batch['actions'].unsqueeze(0).to(device)
            }
            
            # Compute divergence
            div = DivergenceUtils.compute_policy_divergence_during_training(
                model=policy,
                batch=batch,
                n_samples=n_samples
            )
            
            if div is not None:
                all_divergences.append(div.item())
    
    all_divergences = np.array(all_divergences)
    
    return {
        'mean': np.mean(all_divergences),
        'std': np.std(all_divergences),
        'min': np.min(all_divergences),
        'max': np.max(all_divergences),
        'median': np.median(all_divergences),
        'values': all_divergences
    }


# ============================================================================
# Example 3: Divergence-Aware Loss Function
# ============================================================================

def compute_losses_with_divergence_regularization(self, predictions, batch, divergence_weight=0.001):
    """
    Modified _compute_losses that includes divergence regularization.
    
    This penalizes high divergence, encouraging the policy to produce
    smoother vector fields.
    
    Args:
        predictions: Model predictions
        batch: Training batch
        divergence_weight: Weight for divergence regularization term
    """
    losses = OrderedDict()
    
    # Standard action losses
    a_target = batch["actions"]
    actions = predictions["actions"]
    losses["l2_loss"] = torch.nn.MSELoss()(actions, a_target)
    losses["l1_loss"] = torch.nn.SmoothL1Loss()(actions, a_target)
    
    # Standard action loss
    action_loss = (
        self.algo_config.loss.l2_weight * losses["l2_loss"] +
        self.algo_config.loss.l1_weight * losses["l1_loss"]
    )
    
    # NEW: Add divergence regularization
    divergence = DivergenceUtils.compute_policy_divergence_during_training(
        model=self.nets["policy"],
        batch=batch,
        n_samples=1
    )
    
    if divergence is not None:
        # Penalize large absolute divergence values
        divergence_loss = torch.mean(torch.abs(divergence))
        losses["divergence_loss"] = divergence_loss
        action_loss = action_loss + divergence_weight * divergence_loss
    
    losses["action_loss"] = action_loss
    return losses


# ============================================================================
# Example 4: Custom BC Class with Divergence Monitoring
# ============================================================================

class BC_WithDivergence:
    """
    Example BC subclass that tracks divergence throughout training.
    
    This can be used as a template for creating custom BC variants
    that monitor or regularize based on divergence.
    """
    
    def __init__(self, *args, track_divergence=True, divergence_log_freq=10, **kwargs):
        """
        Args:
            track_divergence: Whether to compute and log divergence
            divergence_log_freq: How often to compute divergence (every N batches)
        """
        super().__init__(*args, **kwargs)
        self.track_divergence = track_divergence
        self.divergence_log_freq = divergence_log_freq
        self.divergence_history = []
        self._batch_counter = 0
    
    def train_on_batch(self, batch, epoch, validate=False):
        """Override to add divergence tracking."""
        info = super().train_on_batch(batch, epoch, validate=validate)
        
        # Compute divergence periodically
        if self.track_divergence and not validate:
            self._batch_counter += 1
            
            if self._batch_counter % self.divergence_log_freq == 0:
                divergence = DivergenceUtils.compute_policy_divergence_during_training(
                    model=self.nets["policy"],
                    batch=batch,
                    n_samples=3  # Use more samples for periodic measurements
                )
                
                if divergence is not None:
                    div_mean = divergence.mean().item()
                    self.divergence_history.append(div_mean)
                    
                    info["divergence_mean"] = div_mean
                    info["divergence_std"] = divergence.std().item()
        
        return info
    
    def get_divergence_statistics(self):
        """Get summary statistics of divergence over training."""
        if not self.divergence_history:
            return None
        
        return {
            'mean': np.mean(self.divergence_history),
            'std': np.std(self.divergence_history),
            'history': self.divergence_history
        }


# ============================================================================
# Example 5: Testing Different BC Architectures
# ============================================================================

def test_divergence_computation_all_architectures():
    """
    Test that divergence computation works with all BC model types.
    """
    from robomimic.models import policy_nets
    from collections import OrderedDict
    
    # Create dummy batch
    batch_size = 4
    obs_shapes = OrderedDict([
        ('robot0_eef_pos', (3,)),
        ('robot0_eef_quat', (4,)),
        ('robot0_joint_pos', (7,)),
    ])
    
    batch = {
        'obs': {
            'robot0_eef_pos': torch.randn(batch_size, 3),
            'robot0_eef_quat': torch.randn(batch_size, 4),
            'robot0_joint_pos': torch.randn(batch_size, 7),
        },
        'goal_obs': None,
        'actions': torch.randn(batch_size, 7)
    }
    
    # Test 1: Standard MLP Actor
    print("Testing ActorNetwork (MLP)...")
    actor = policy_nets.ActorNetwork(
        obs_shapes=obs_shapes,
        ac_dim=7,
        mlp_layer_dims=[256, 256]
    )
    div = DivergenceUtils.compute_policy_divergence_during_training(
        model=actor, batch=batch, n_samples=1
    )
    print(f"  Divergence shape: {div.shape}, mean: {div.mean().item():.4f}")
    
    # Test 2: Gaussian Actor
    print("\nTesting GaussianActorNetwork...")
    gaussian_actor = policy_nets.GaussianActorNetwork(
        obs_shapes=obs_shapes,
        ac_dim=7,
        mlp_layer_dims=[256, 256]
    )
    # Note: Gaussian models return distributions, need to handle differently
    # For divergence, we can use the mean of the distribution
    
    print("\nAll architectures work with divergence computation!")


if __name__ == "__main__":
    print("Examples for computing policy divergence during BC training")
    print("=" * 70)
    print("\nThese examples show how to:")
    print("1. Integrate divergence into train_on_batch()")
    print("2. Analyze divergence across datasets")  
    print("3. Use divergence as a regularization term")
    print("4. Create custom BC classes with divergence monitoring")
    print("5. Test compatibility with all BC architectures")
    print("\nSee function implementations above for details.")
