# Contributing Guidelines

Our team wholeheartedly welcomes the community to contribute to robomimic. Contributions from members of the community will help ensure the long-term success of this project. Before you plan to make contributions, here are important resources to get started with:

- Read the robomimic [documentation](https://robomimic.github.io/docs/introduction/overview.html) and [paper](https://arxiv.org/abs/2108.03298)
- Check our latest status from existing [issues](https://github.com/ARISE-Initiative/robomimic/issues), [pull requests](https://github.com/ARISE-Initiative/robomimic/pulls), and [branches](https://github.com/ARISE-Initiative/robomimic/branches) and avoid duplicate efforts
- Join our [ARISE Slack](https://ariseinitiative.slack.com) workspace for technical discussions. Please [email us](mailto:yukez@cs.utexas.edu) to be added to the workspace.

We encourage the community to make four major types of contributions:

- **Bug fixes**: Address open issues and fix bugs presented in the `master` branch.
- **Additional datasets and tasks:** Make robomimic compatible with more kinds of datasets, simulators, and tasks.
- **New algorithms:** Develop new algorithms or re-implement existing algorithms for learning from robot manipulation datasets.
- **Implement new functionalities:** Implement new features, such as training with new kinds of observations (depth images, lidar, force/torque sensors).

Testing
-------
Before submitting your contributions, make sure that the changes do not break existing functionalities. We have a handful of tests for verifying the correctness of the code.

You can run all the tests with the following command in the tests folder of robomimic. Make sure that it does not throw any error before you proceed to the next step. Note that the tests can take a few minutes to run.

```sh
$ bash test.sh
```

Submission
----------
Please read the coding conventions below and make sure that your code is consistent with ours. When making a contribution, make a [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests) to robomimic with an itemized list of what you have done. When you submit a pull request, it is immensely helpful to include example script(s) that showcase the proposed changes and highlight any new APIs. We always love to see more test coverage. When it is appropriate, add a new test to the tests folder for checking the correctness of your code.

Coding Conventions
------------------
We value readability and adhere to the following coding conventions:
- Indent using four spaces (soft tabs)
- Always put spaces after list items and method parameters (e.g., `[1, 2, 3]` rather than `[1,2,3]`), and around operators and hash arrows (e.g., `x += 1` rather than `x+=1`)
- Use [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for the docstrings
- For scripts such as in the `scripts` and `examples` folders, include a docstring at the top of the file that describes the high-level purpose of the script and/or instructions on how to use the scripts (if relevant).

## Additional Coding Guidelines

We also list additional suggested contributing guidelines that we adhered to during development.

- When creating new networks (e.g. subclasses of `Module` in `models/base_nets.py`), always put sub-modules into a property called `self.nets`, and if there is more than one sub-module, make it a module collection (such as a `torch.nn.ModuleDict`). This is to ensure that the pattern `model.to(device)` works as expected with multiple levels of nested torch modules. As an example of nesting, see the `_create_networks` function in the `VAE` class (`models/vae_nets.py`) and the `MIMO_MLP` class (`models/obs_nets.py`).

- Do not use default mutable arguments -- they can lead to terrible bugs and unexpected behavior (see [this link](https://florimond.dev/blog/articles/2018/08/python-mutable-defaults-are-the-source-of-all-evil/) for more information). For this reason, in functions that expect optional dictionaries and lists (for example, the `core_kwargs` argument in the  `obs_encoder_factory` function, or the `layer_dims` argument in the `MLP` class constructor), we use a default argument of `core_kwargs=None` or an empty tuple (since tuples are immutable) `layer_dims=()`.

- Prefer `torch.expand` over `torch.repeat` wherever possible, for memory efficiency. See [this link](https://discuss.pytorch.org/t/expand-vs-repeat-semantic-difference/59789) for more details.

- When implementing new configs that specify kwargs that will be unpacked by a downstream python class (for example, the property `self.observation.encoder.rgb.core_kwargs` in the `BaseConfig` class, which is fed to the class specified by `self.observation.encoder.rgb.core_class`), the default config class should specify an empty config object (essentially an empty dictionary) for the kwargs. This is to make sure that external config jsons will be able to completely override both the class and the kwargs without worrying about existing default kwargs that could break the initialization of the class. For example, while the default `VisualCore` class takes a kwarg called `feature_dimension`, another class may not take this argument. If this kwarg already existed in the base config, the external json will just add additional kwargs.
  

We look forward to your contributions. Thanks!

**The robomimic core team**

