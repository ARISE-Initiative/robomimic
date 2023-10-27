# MimicGen (CoRL 2023)

## Overview

The [MimicGen paper](https://arxiv.org/abs/2310.17596) released a large collection of task demonstrations across several different environments. 

The datasets contain over 48,000 task demonstrations across 12 tasks, grouped into the following categories:
- **source**: 120 human demonstrations across 12 tasks used to automatically generate the other datasets
- **core**: 26,000 task demonstrations across 12 tasks (26 task variants)
- **object**: 2000 task demonstrations on the Mug Cleanup task with different mugs
- **robot**: 16,000 task demonstrations across 4 different robot arms on 2 tasks (4 task variants)
- **large_interpolation**: 6000 task demonstrations across 6 tasks that pose significant challenges for modern imitation learning methods

See [this link](https://github.com/NVlabs/mimicgen_environments#dataset-types) for more information on the datasets.

<p align="center">
  <img width="100.0%" src="../images/mimicgen_mosaic.gif">
</p>

## Downloading

Please see [this link](https://github.com/NVlabs/mimicgen_environments#downloading-and-using-datasets) for instructions on downloading and using these datasets.

## Postprocessing

No postprocessing is needed for these datasets.

## Citation

```bibtex
@inproceedings{mandlekar2023mimicgen,
    title={MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations},
    author={Mandlekar, Ajay and Nasiriany, Soroush and Wen, Bowen and Akinola, Iretiayo and Narang, Yashraj and Fan, Linxi and Zhu, Yuke and Fox, Dieter},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023}
}
```