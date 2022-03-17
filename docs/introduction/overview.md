# Overview

<p align="center">
  <img width="24.0%" src="../images/task_lift.gif">
  <img width="24.0%" src="../images/task_can.gif">
  <img width="24.0%" src="../images/task_tool_hang.gif">
  <img width="24.0%" src="../images/task_square.gif">
  <img width="24.0%" src="../images/task_lift_real.gif">
  <img width="24.0%" src="../images/task_can_real.gif">
  <img width="24.0%" src="../images/task_tool_hang_real.gif">
  <img width="24.0%" src="../images/task_transport.gif">
 </p>

**robomimic** is a framework for robot learning from demonstration.
It offers a broad set of demonstration datasets collected on robot manipulation domains and learning algorithms to learn from these datasets.
While recent advances have been made in imitation learning and batch (offline) reinforcement learning, a lack of open-source human datasets and reproducible learning methods make assessing the state of the field difficult.
robomimic is an open-source effort that allows researchers and practitioners to benchmark tasks and algorithms to facilitate fair comparisons, with a focus on learning from human-provided demonstrations.

## Core Features
1. **Offline Learning Algorithms**
High-quality implementations of offline learning algorithms, including BC, BC-RNN, HBC, IRIS, BCQ, CQL, and TD3-BC
2. **Standardized Datasets**
Datasets collected from different sources (single proficient human, multiple humans, and machine-generated) across simulated and real-world tasks spanning multiple robots and environments
3. **Modular Design**
Support for learning both low-dimensional and visuomotor policies, diverse network architectures, interface to easily use external datasets
4. **Flexible Experiment Workflow**
Utilities for running hyperparameter sweeps, visualizing demonstration data and trained policies, and collecting new datasets using trained policies

## Reproducing benchmark study results

The robomimic framework also makes reproducing the results from this [benchmark study](https://arise-initiative.github.io/robomimic-web/study) easy. See the [reproducing results documentation](./results.html) for more information.

## Contributing to robomimic
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.
This framework originally began development in late 2018 by researchers in the [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL).
Now it is actively maintained and used for robotics research projects across multiple labs.
We welcome community contributions to this project.
For details please check our [contributing guidelines](../miscellaneous/contributing.html).

## Troubleshooting

Please see the [troubleshooting](../miscellaneous/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/ARISE-Initiative/robomimic/issues) on our github page.

## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```