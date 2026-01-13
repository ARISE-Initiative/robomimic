# Policy Divergence Computation for BC Models

## Overview

The updated `estimate_divergence_jvp()` function now works with **all BC model architectures** during training. It computes the divergence of the policy's action field with respect to end-effector pose perturbations on the SE(3) manifold.

## Supported Architectures

✅ **BC (MLP)** - `ActorNetwork`
✅ **BC_RNN** - `RNN_MIMO_MLP`
✅ **BC_Transformer** - `MIMO_Transformer` with `GPT_Backbone`
✅ **BC_Gaussian** - `GaussianActorNetwork`
✅ **BC_GMM** - `GMMActorNetwork`
✅ **BC_VAE** - `VAE`

## Key Features

### 1. **Unified Interface**
All BC models share the same interface: `model(obs_dict, goal_dict) -> actions`

### 2. **SE(3) Manifold Awareness**
- Perturbs end-effector pose properly using exponential map on SE(3)
- Computes twist-based perturbations (6D tangent space)
- Preserves unit quaternion constraints

### 3. **Efficient Computation**
- Uses Forward-Mode AD (JVP) instead of computing full Jacobian
- Hutchinson's trace estimator with configurable samples
- Single forward pass per sample

### 4. **Training Integration**
- Can be called directly in `train_on_batch()`
- Works with batched observations
- Compatible with goal-conditioned policies

## Usage

### Basic Usage in Training Loop

```python
from robomimic.utils import divergence_utils as DivergenceUtils

# In BC.train_on_batch():
divergence = DivergenceUtils.compute_policy_divergence_during_training(
    model=self.nets["policy"],
    batch=batch,
    n_samples=1  # Increase for better accuracy
)

if divergence is not None:
    info["divergence_mean"] = divergence.mean().item()
```

### Advanced Usage with Regularization

```python
# Add divergence as loss term
divergence = DivergenceUtils.compute_policy_divergence_during_training(
    model=self.nets["policy"],
    batch=batch,
    n_samples=5
)

if divergence is not None:
    divergence_loss = torch.abs(divergence).mean()
    losses["action_loss"] += 0.001 * divergence_loss  # Regularization weight
```

## Function Signatures

### Main Function

```python
def estimate_divergence_jvp(
    model,                          # BC policy network
    batch,                          # Training batch with 'obs' dict
    obs_key_pos='robot0_eef_pos',  # Key for EE position
    obs_key_quat='robot0_eef_quat', # Key for EE quaternion  
    goal_dict=None,                 # Optional goal observations
    n_samples=1                     # Hutchinson estimator samples
) -> torch.Tensor                   # [batch_size] divergence values
```

### Convenience Wrapper

```python
def compute_policy_divergence_during_training(
    model,      # self.nets["policy"]
    batch,      # from process_batch_for_training()
    n_samples=1 # number of samples
) -> torch.Tensor or None  # Returns None if EE pose not in observations
```

## Mathematical Background

### What is Divergence?

Divergence measures the "expansion" or "contraction" of a vector field:

$$\\nabla \\cdot f = \\text{Tr}(J_f)$$

where $J_f$ is the Jacobian matrix of the policy $f: \\text{SE}(3) \\to \\mathbb{R}^n$.

### Why on SE(3)?

The end-effector pose lives on the SE(3) manifold (3D position + SO(3) rotation). We compute divergence in the tangent space at each pose, which is the Lie algebra se(3) (6D twist space: linear velocity + angular velocity).

### Hutchinson's Estimator

To avoid computing the full Jacobian:

$$\\text{Tr}(J) \\approx \\mathbb{E}_{\\epsilon \\sim \\mathcal{N}(0,I)}[\\epsilon^T J \\epsilon]$$

We sample random directions $\\epsilon$ and use JVP to compute $J\\epsilon$ efficiently.

## Requirements

The function requires observations to contain:
- `robot0_eef_pos`: [batch, 3] - End-effector position
- `robot0_eef_quat`: [batch, 4] - End-effector quaternion (xyzw or wxyz)

If these keys are not present, the function returns `None`.

## Performance Notes

- **n_samples=1**: Fast, suitable for every training step
- **n_samples=5-10**: More accurate, suitable for periodic logging
- **Overhead**: ~2-3x forward pass time per sample

## Example Integration

See `examples/compute_divergence_during_training.py` for complete examples including:

1. Minimal integration into `train_on_batch()`
2. Dataset-wide divergence analysis
3. Divergence-based regularization
4. Custom BC class with divergence monitoring
5. Testing with all architectures

## Model-Specific Notes

### Deterministic Policies (BC, BC_RNN, BC_Transformer)
- Use `model.forward()` directly
- Returns actions: [batch, action_dim]

### Stochastic Policies (BC_Gaussian, BC_GMM)
- Use `model.forward()` (not `forward_train()`)
- Returns mean actions in eval mode
- For training mode, returns distribution means

### VAE-based (BC_VAE)
- Decodes latent samples to actions
- Returns reconstructed actions

All cases return action tensors that can be differentiated w.r.t. input observations.
