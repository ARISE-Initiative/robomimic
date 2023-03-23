# Pre-trained Visual Representations

**Robomimic** supports multiple pre-trained visual representations and offers integration for adapting observation encoders to the desired pre-trained visual representation encoders.

## Terminology

First, let's clarify the semantic distinctions when using different pre-trained visual representations:

- **Backbone Classes** refer to the various pre-trained visual encoders. For instance, `R3MConv` and `MVPConv` are the backbone classes for using [R3M](https://arxiv.org/abs/2203.12601) and [MVP](https://arxiv.org/abs/2203.06173) pre-trained representations, respectively.
- **Model Classes** pertain to the different sizes of the pretrained models within each selected backbone class. For example, `R3MConv` has three model classes - `resnet18`, `resnet34`, and `resnet50`, while `MVPConv` features five model classes - `vits-mae-hoi`, `vits-mae-in`, `vits-sup-in`, `vitb-mae-egosoup`, and `vitl-256-mae-egosoup`.

## Examples

Using pre-trained visual representations is simple. Each pre-trained encoder is defined by its `backbone_class`, `model_class`, and whether to `freeze` representations or finetune them. Please note that you may need to refer to the original library of the pre-trained representation for installation instructions.

If you are specifying your config with code (as in `examples/train_bc_rnn.py`), the following are example code blocks for using pre-trained representations:

```python
# R3M
config.observation.encoder.rgb.core_kwargs.backbone_class = 'R3MConv'                         # R3M backbone for image observations (unused if no image observations)
config.observation.encoder.rgb.core_kwargs.backbone_kwargs.r3m_model_class = 'resnet18'       # R3M model class (resnet18, resnet34, resnet50)
config.observation.encoder.rgb.core_kwargs.backbone_kwargs.freeze = True                      # whether to freeze network during training or allow finetuning
config.observation.encoder.rgb.core_kwargs.pool_class = None                                  # no pooling class for pretraining model

# MVP
config.observation.encoder.rgb.core_kwargs.backbone_class = 'MVPConv'                                   # MVP backbone for image observations (unused if no image observations)
config.observation.encoder.rgb.core_kwargs.backbone_kwargs.mvp_model_class = 'vitb-mae-egosoup'         # MVP model class (vits-mae-hoi, vits-mae-in, vits-sup-in, vitb-mae-egosoup, vitl-256-mae-egosoup)
config.observation.encoder.rgb.core_kwargs.backbone_kwargs.freeze = True                      # whether to freeze network during training or allow finetuning
config.observation.encoder.rgb.core_kwargs.pool_class = None                                            # no pooling class for pretraining model
```

Alternatively, if you are using a config json, you can set the appropriate keys in your json.
