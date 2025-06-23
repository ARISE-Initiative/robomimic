# Language-conditioned Policy Learning

This tutorial will guide you through setting up language-conditioned policy learning in robomimic.

<div class="admonition note">
<p class="admonition-title">Note: Understand how to launch training runs and view results first!</p>

Before trying to train a language-conditioned policy, it might be useful to read the following tutorials:
- [how to launch training runs](./configs.html)
- [how to view training results](./viewing_results.html)
- [how to launch multiple training runs efficiently](./hyperparam_scan.html) 

</div>

## 1. Creating a Dataset Config

To create a dataset config with language conditioning, include the `lang` key under the dataset config dictionary. This key should specify the language annotations for all demos in this dataset.

Example:
```json
{
    ...
    "train": {
        "data": [
            {
                "path": "path/to/dataset.hdf5",
                "lang": "language instruction for your task"
            },
            ...
        ],
        ...
    },
    ...
}
```

## 2. Conditioning Policies on Language Embeddings

We support CLIP embeddings for encoding language. The pre-defined key for language embeddings is `lang_emb` (specified in `robomimic/utils/lang_utils.py`). You can condition your policy on `lang_emb` using 2 ways:

1. As feature input to action head
2. [FiLM](https://arxiv.org/pdf/1709.07871) over vision encoder

### Feature input to action head

This concatenates language embeddings with other low-dim observations input to the policy.

Example:
```json
{
    ...
    "observation": {
        "modalities": {
            "obs": {
                "low_dim": [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "lang_emb"
                ],
                ...
            },
        },
        ...
    },
    ...
}
```

### FiLM over vision encoder

This conditions the ResNet18 visual encoder with `lang_emb` using FiLM (see [paper](https://arxiv.org/pdf/1709.07871)).

Example:
```json
{
    ...
    "observation": {
        "rgb": {
            "core_class": "VisualCoreLanguageConditioned",
            "core_kwargs": {
                "feature_dimension": 64,
                "flatten": true,
                "backbone_class": "ResNet18ConvFiLM",
                "backbone_kwargs": {
                    "pretrained": false,
                    "input_coord_conv": false
                },
                "pool_class": null,
                "pool_kwargs": {}
            },
            "obs_randomizer_class": null,
            "obs_randomizer_kwargs": {}
        },
        ...
    }
}
```
