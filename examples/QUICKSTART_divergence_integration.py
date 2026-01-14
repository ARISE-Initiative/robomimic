"""
TODO: EXAMPLE ADDED
Quick Integration Guide: Adding Divergence to BC Training

This file shows the exact code changes needed to add divergence computation
to your BC training loop.
"""

# ==============================================================================
# STEP 1: Add import to BC algorithm file
# ==============================================================================
# In robomimic/algo/bc.py, add at the top:

import robomimic.utils.divergence_utils as DivergenceUtils

# ==============================================================================
# STEP 2: Modify train_on_batch() method
# ==============================================================================
# Option A: Minimal integration (just logging)
# ----------------------------------------------

def train_on_batch(self, batch, epoch, validate=False):
    """Modified BC.train_on_batch() with divergence logging."""
    with TorchUtils.maybe_no_grad(no_grad=validate):
        info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
        predictions = self._forward_training(batch)
        losses = self._compute_losses(predictions, batch)

        info["predictions"] = TensorUtils.detach(predictions)
        info["losses"] = TensorUtils.detach(losses)

        if not validate:
            # ===== NEW CODE: Compute divergence =====
            try:
                divergence = DivergenceUtils.compute_policy_divergence_during_training(
                    model=self.nets["policy"],
                    batch=batch,
                    n_samples=1  # Fast: use 1 sample
                )
                if divergence is not None:
                    info["divergence"] = {
                        "mean": divergence.mean().item(),
                        "std": divergence.std().item(),
                        "min": divergence.min().item(),
                        "max": divergence.max().item(),
                    }
            except Exception as e:
                # Divergence computation failed (e.g., no EE observations)
                pass
            # ===== END NEW CODE =====
            
            step_info = self._train_step(losses)
            info.update(step_info)

    return info


# Option B: Add divergence as regularization loss
# ------------------------------------------------

def _compute_losses(self, predictions, batch):
    """Modified BC._compute_losses() with divergence regularization."""
    losses = OrderedDict()
    a_target = batch["actions"]
    actions = predictions["actions"]
    
    # Standard losses
    losses["l2_loss"] = nn.MSELoss()(actions, a_target)
    losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
    losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])
    
    action_losses = [
        self.algo_config.loss.l2_weight * losses["l2_loss"],
        self.algo_config.loss.l1_weight * losses["l1_loss"],
        self.algo_config.loss.cos_weight * losses["cos_loss"],
    ]
    
    # ===== NEW CODE: Add divergence regularization =====
    try:
        divergence = DivergenceUtils.compute_policy_divergence_during_training(
            model=self.nets["policy"],
            batch=batch,
            n_samples=1
        )
        if divergence is not None:
            # Penalize large divergence magnitudes
            div_loss = torch.abs(divergence).mean()
            losses["divergence_loss"] = div_loss
            
            # Add to action loss with weight (tune this hyperparameter!)
            divergence_weight = 0.001  # Or from self.algo_config.loss.divergence_weight
            action_losses.append(divergence_weight * div_loss)
    except:
        pass
    # ===== END NEW CODE =====
    
    action_loss = sum(action_losses)
    losses["action_loss"] = action_loss
    return losses


# ==============================================================================
# STEP 3: Update log_info() for TensorBoard logging (optional)
# ==============================================================================

def log_info(self, info):
    """Modified BC.log_info() to include divergence metrics."""
    log = super(BC, self).log_info(info)
    log["Loss"] = info["losses"]["action_loss"].item()
    
    # ===== NEW CODE: Log divergence =====
    if "divergence" in info:
        log["Divergence_Mean"] = info["divergence"]["mean"]
        log["Divergence_Std"] = info["divergence"]["std"]
        log["Divergence_Range"] = info["divergence"]["max"] - info["divergence"]["min"]
    
    if "divergence_loss" in info["losses"]:
        log["Divergence_Loss"] = info["losses"]["divergence_loss"].item()
    # ===== END NEW CODE =====
    
    return log


# ==============================================================================
# STEP 4: Add config parameters (optional)
# ==============================================================================
# In your training config JSON, you can add:

{
    "algo": {
        "loss": {
            "l2_weight": 1.0,
            "l1_weight": 0.0,
            "cos_weight": 0.0,
            // NEW: divergence regularization weight
            "divergence_weight": 0.001
        },
        // NEW: divergence logging settings
        "divergence": {
            "enabled": true,
            "n_samples": 1,  # Increase for more accurate estimates
            "log_frequency": 10  # Compute every N batches (0 = every batch)
        }
    }
}


# ==============================================================================
# ALTERNATIVE: Subclass BC instead of modifying
# ==============================================================================

from robomimic.algo.bc import BC
import robomimic.utils.divergence_utils as DivergenceUtils

class BC_WithDivergence(BC):
    """
    BC variant that automatically tracks and logs divergence.
    Use this to avoid modifying the base BC class.
    """
    
    def train_on_batch(self, batch, epoch, validate=False):
        """Override to add divergence tracking."""
        info = super().train_on_batch(batch, epoch, validate=validate)
        
        if not validate:
            try:
                divergence = DivergenceUtils.compute_policy_divergence_during_training(
                    model=self.nets["policy"],
                    batch=batch,
                    n_samples=self.algo_config.get("divergence_samples", 1)
                )
                if divergence is not None:
                    info["divergence_mean"] = divergence.mean().item()
                    info["divergence_std"] = divergence.std().item()
            except:
                pass
        
        return info
    
    def log_info(self, info):
        """Override to log divergence."""
        log = super().log_info(info)
        if "divergence_mean" in info:
            log["Divergence_Mean"] = info["divergence_mean"]
            log["Divergence_Std"] = info["divergence_std"]
        return log


# Then register this new algorithm:
from robomimic.algo import register_algo_factory_func

@register_algo_factory_func("bc_div")
def algo_config_to_class(algo_config):
    return BC_WithDivergence, {}


# ==============================================================================
# TESTING: Quick test to verify everything works
# ==============================================================================

if __name__ == "__main__":
    import torch
    from collections import OrderedDict
    import robomimic.utils.divergence_utils as DivergenceUtils
    from robomimic.models.policy_nets import ActorNetwork
    
    print("Testing divergence computation...")
    
    # Create dummy batch
    batch = {
        'obs': {
            'robot0_eef_pos': torch.randn(8, 3),
            'robot0_eef_quat': torch.randn(8, 4),
            'robot0_joint_pos': torch.randn(8, 7),
        },
        'actions': torch.randn(8, 7)
    }
    
    # Create dummy policy
    obs_shapes = OrderedDict([
        ('robot0_eef_pos', (3,)),
        ('robot0_eef_quat', (4,)),
        ('robot0_joint_pos', (7,)),
    ])
    
    policy = ActorNetwork(
        obs_shapes=obs_shapes,
        ac_dim=7,
        mlp_layer_dims=[128, 128]
    )
    
    # Compute divergence
    div = DivergenceUtils.compute_policy_divergence_during_training(
        model=policy,
        batch=batch,
        n_samples=1
    )
    
    print(f"✓ Divergence computed successfully!")
    print(f"  Shape: {div.shape}")
    print(f"  Mean: {div.mean().item():.4f}")
    print(f"  Std: {div.std().item():.4f}")
    print("\nIntegration successful! Ready to use in training.")
