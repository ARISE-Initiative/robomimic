# Pre-trained Visual Representations

**Robomimic** supports multiple pre-trained visual representations and offers integration for adapting observation encoders to the desired pre-trained visual representation encoders.

First, let's clarify the semantic distinctions when using different pre-trained visual representations:

- **Backbone Classes** refer to the various pre-trained visual encoders. For instance, `R3MConv` and `MVPConv` are the backbone classes for using [R3M](https://arxiv.org/abs/2203.12601) and [MVP](https://arxiv.org/abs/2203.06173) pre-trained representations, respectively.
- **Model Classes** pertain to the different sizes of the pretrained models within each selected backbone class. For example, `R3MConv` has three model classes - `resnet18`, `resnet34`, and `resnet50`, while `MVPConv` features five model classes - `vits-mae-hoi`, `vits-mae-in`, `vits-sup-in`, `vitb-mae-egosoup`, and `vitl-256-mae-egosoup`.

Examples of using pre-trained visual representation encoders can be found in this [example script](https://github.com/ARISE-Initiative/robomimic/blob/master/examples/train_bc_rnn.py#L137):
1. Each pre-trained encoder is defined by its `backbone_class`.
2. Model selection is specified in `backbone_kwargs.model_class`.

Please note that you may need to refer to the original library of the pre-trained representation for installation instructions.