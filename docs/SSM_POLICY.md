# State-Space Model (SSM) Policy Architecture

## Overview

This feature implements a State-Space Model (SSM) policy architecture for robomimic, providing an efficient alternative to Transformer-based policies for sequential decision-making tasks. The implementation follows Mamba's selective state-space design, offering O(n) time complexity compared to Transformers' O(n²) complexity.

## Motivation

Long-horizon robotic manipulation tasks require processing extended observation sequences efficiently. While Transformer policies excel at capturing long-range dependencies, their quadratic complexity becomes computationally prohibitive for very long sequences. SSMs provide a compelling alternative:

- **Linear Complexity**: O(n) time and memory scaling with sequence length
- **Efficient Long-Horizon Modeling**: Can process hundreds of timesteps efficiently
- **Hardware Optimized**: Sequential scan operations are cache-friendly
- **Proven Performance**: Mamba has shown competitive results with Transformers across various domains

## Architecture

### Core Components

#### 1. SelectiveSSMBlock (`robomimic/models/ssm_nets.py`)

The fundamental building block implementing Mamba's selective scan mechanism:

```python
x ─┬─── Linear(d, 2*expand*d) ──┬── Conv1d + SiLU ── SSM Scan ── (*)──
   │                             │                                  │
   └─────────────────────────────┴── SiLU (gate) ──────────────────┘
                                                                   │
                                                             Linear ── y
```

**Key features**:
- Input-dependent dynamics (B, C, Δ parameters are functions of input)
- Local 1D convolution for capturing short-range patterns
- Gated activation for selective information flow
- Residual connections for gradient flow

#### 2. SSM_Backbone (`robomimic/models/ssm_nets.py`)

Stacks multiple `SelectiveSSMBlock` layers with layer normalization, analogous to `GPT_Backbone` for Transformers.

**Configuration**:
- `embed_dim`: Embedding dimension (default: 256)
- `num_layers`: Number of stacked SSM blocks (default: 4)
- `state_dim`: Hidden state dimension for SSM recurrence (default: 16)
- `conv_dim`: 1D convolution kernel size (default: 4)
- `expand_factor`: Inner dimension expansion (default: 2)
- `dropout`: Dropout probability (default: 0.1)

#### 3. MIMO_SSM (`robomimic/models/obs_nets.py`)

Multi-input multi-output wrapper connecting observation encoders → SSM backbone → action decoders. Mirrors `MIMO_Transformer` structure for consistency.

#### 4. Policy Networks (`robomimic/models/policy_nets.py`)

- **SSMActorNetwork**: Deterministic SSM policy
- **SSMGMMActorNetwork**: SSM policy with Gaussian Mixture Model outputs

### Integration with BC Algorithm

Two new algorithm classes in `robomimic/algo/bc.py`:

- **BC_SSM**: Behavioral cloning with deterministic SSM policy
- **BC_SSM_GMM**: Behavioral cloning with SSM-GMM policy

Both support:
- Partial sequence supervision (`supervise_all_steps`)
- Future action prediction (`pred_future_acs`)
- Goal-conditioned policies

## Configuration

Enable SSM policies via configuration file:

```python
# config.algo.ssm settings
config.algo.ssm.enabled = True                  # Enable SSM policy
config.algo.ssm.context_length = 10            # Sequence length (match train.frame_stack)
config.algo.ssm.embed_dim = 256                # Embedding dimension
config.algo.ssm.num_layers = 4                 # Number of SSM blocks
config.algo.ssm.state_dim = 16                 # SSM hidden state dimension
config.algo.ssm.conv_dim = 4                   # Conv kernel size
config.algo.ssm.dropout = 0.1                  # Dropout probability
config.algo.ssm.supervise_all_steps = False    # Supervise all/final actions
config.algo.ssm.pred_future_acs = False        # Predict future actions

# For GMM variant
config.algo.gmm.enabled = True
```

## Usage Example

### Training an SSM Policy

```python
from robomimic.config.bc_config import BCConfig

# Create config
config = BCConfig()

# Enable SSM
config.algo.ssm.enabled = True
config.algo.ssm.context_length = 10
config.algo.ssm.embed_dim = 256
config.algo.ssm.num_layers = 4

# Training settings
config.train.data = "/path/to/dataset.hdf5"
config.train.batch_size = 64
config.train.num_epochs = 500

# Train
# python scripts/train.py --config config.json
```

### Using SSM-GMM for Multi-Modal Actions

```python
# Enable both SSM and GMM
config.algo.ssm.enabled = True
config.algo.gmm.enabled = True
config.algo.gmm.num_modes = 5
config.algo.gmm.min_std = 0.01
```

## Implementation Details

### Design Decisions

1. **Self-Contained PyTorch Implementation**: No external dependencies on `mamba-ssm` library for maximum compatibility and transparency

2. **Mirroring Transformer Pattern**: All components follow the exact same structure as Transformer counterparts to ensure consistency and ease of use

3. **Modular Design**: Clear separation between core SSM blocks, backbone, MIMO wrapper, and policy networks

### Key Differences from Transformers

| Aspect | Transformer | SSM |
|--------|-------------|-----|
| Complexity | O(n²) | O(n) |
| Memory | O(n²) | O(n) |
| Parallelization | Fully parallel | Sequential (training), parallel (layers) |
| Position Encoding | Required | Not required |
| Long-Range Deps | Global attention | Recurrent state |

## Testing

Comprehensive test suite in `tests/test_ssm_policy.py`:

```bash
# Run SSM-specific tests
python -m pytest tests/test_ssm_policy.py -v

# Run all tests to ensure no regressions
python -m pytest tests/test_examples.py
```

Tests cover:
- Block-level forward passes
- Backbone instantiation and parameter counts
- MIMO integration with multi-modal observations
- Policy network output shapes and value ranges
- GMM distribution properties
- Configuration integration

## Performance Considerations

**When to Use SSM over Transformer**:
- Very long sequences (>50 timesteps)
- Memory-constrained environments
- Real-time inference requirements
- Tasks with strong sequential structure

**When to Use Transformer over SSM**:
- Short sequences (<20 timesteps)
- Tasks requiring global attention
- When pre-trained models are available

## Related Work

This implementation draws from:

- **Mamba**: Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023 ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752))
- **S4**: Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces," 2022 ([arXiv:2111.00396](https://arxiv.org/abs/2111.00396))

## Files Modified

### New Files
- `robomimic/models/ssm_nets.py`: Core SSM blocks and backbone

### Modified Files
- `robomimic/models/base_nets.py`: Added `ssm_args_from_config()`
- `robomimic/models/obs_nets.py`: Added `MIMO_SSM` class
- `robomimic/models/policy_nets.py`: Added `SSMActorNetwork` and `SSMGMMActorNetwork`
- `robomimic/algo/bc.py`: Added `BC_SSM` and `BC_SSM_GMM` classes + dispatch logic
- `robomimic/config/bc_config.py`: Added SSM configuration section

### Test Files
- `tests/test_ssm_policy.py`: Comprehensive unit and integration tests

## Future Extensions

Potential enhancements:
- Bidirectional SSM scanning for offline RL
- Hierarchical SSMs for multi-timescale modeling
- Integration with other algorithms (TD3-BC, IQL, etc.)
- Hardware-optimized CUDA kernels for faster training
