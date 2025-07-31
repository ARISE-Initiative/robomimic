# Implemented Algorithms

**robomimic** includes several high-quality implementations of offline learning algorithms, and offers tools to easily build [your own learning algorithms](../tutorials/custom_algorithms.html).
## Imitation Learning

### BC

- Vanilla Behavioral Cloning (see [this paper](https://papers.nips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf)), consisting of simple supervised regression from observations to actions. Implemented in the `BC` class in `algo/bc.py`, along with some variants such as `BC_GMM` (stochastic GMM policy) and `BC_VAE` (stochastic VAE policy)

### BC-RNN

- Behavioral Cloning with an RNN network. Implemented in the `BC_RNN` and `BC_RNN_GMM` (recurrent GMM policy) classes in `algo/bc.py`.

### BC-Transformer

- Behavioral Cloning with an Transformer network. Implemented in the `BC_Transformer` and `BC_Transformer_GMM` (transformer GMM policy) classes in `algo/bc.py`.

### Diffusion Policy

- Behavior cloning with a diffusion action head (see [this paper](https://arxiv.org/pdf/2303.04137v5)). Implemented in the `DiffusionPolicyUNet` class in `algo/diffusion_policy.py`.

### HBC

- Hierarchical Behavioral Cloning - the implementation is largely based off of [this paper](https://arxiv.org/abs/2003.06085). Implemented in the `HBC` class in `algo/hbc.py`.

## Offline Reinforcement Learning

### IRIS

- A recent batch offline RL algorithm from [this paper](https://arxiv.org/abs/1911.05321). Implemented in the `IRIS` class in `algo/iris.py`.

### BCQ

- A recent batch offline RL algorithm from [this paper](https://arxiv.org/abs/1812.02900). Implemented in the `BCQ` class in `algo/bcq.py`.

### CQL

- A recent batch offline RL algorithm from [this paper](https://arxiv.org/abs/2006.04779). Implemented in the `CQL` class in `algo/cql.py`.

### IQL

- A recent batch offline RL algorithm from [this paper](https://arxiv.org/abs/2110.06169). Implemented in the `IQL` class in `algo/iql.py`.

### TD3-BC

- A recent algorithm from [this paper](https://arxiv.org/abs/2106.06860). We implemented it as an example (see section below on building your own algorithm). Implemented in the `TD3_BC` class in `algo/td3_bc.py`.
