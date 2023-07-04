# Acknowledgments

### People

We would like to thank members of the [Stanford PAIR Group](http://pair.stanford.edu/) for their support and feedback on this project. These people in particular have made the following contributions at different stages of this project:

- [Rohun Kulkarni](https://www.linkedin.com/in/rohunkulkarni/) (assistance with collecting real robot datasets and running real robot experiments)
- [Albert Tung](https://www.linkedin.com/in/albert-tung3/) (assistance with collecting simulation datasets using the [RoboTurk](https://roboturk.stanford.edu/) system)
- [Fei Xia](http://fxia.me/) ([egl_probe](https://github.com/StanfordVL/egl_probe) library, which helped us run experiments on lab clusters)
- [Jim Fan](https://twitter.com/drjimfan?lang=en) (providing support for running experiments on lab clusters)

### Codebases

- Our Config class (see `config/config.py`) was adapted from [addict](https://github.com/mewwts/addict).
- The [BCQ](https://github.com/sfujim/BCQ),  [CQL](https://github.com/aviralkumar2907/CQL), and [TD3-BC](https://github.com/sfujim/TD3_BC) author-provided implementations were used as a reference for our implementations.
- The `TanhWrappedDistribution` class in `models/distributions.py` was adapted from [rlkit](TanhWrappedDistribution).
- Support for training distributional critics (see `BCQ_Distributional` in `algos/bcq.py`) was adapted from [Acme](https://github.com/deepmind/acme). It also served as a useful reference for implementing Gaussian Mixture Model (GMM) policies.
- Our transformer implementation was adapted from the excellent [minGPT](https://github.com/karpathy/minGPT) codebase.

We wholeheartedly welcome the community to contribute to our project through issues and pull requests. New contributors will be added to the list above.

