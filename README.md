
![License](https://img.shields.io/badge/license-MIT-blue.svg) ![EMNLP 2023](https://img.shields.io/badge/EMNLP-2023-orange.svg) ![Python Version](https://img.shields.io/badge/python-3.6-blue.svg)

# Syntax-Guided Transformers: Elevating Compositional Generalization and Grounding in Multimodal Environments

This is the official implementation of [Syntax-Guided Transformers: Elevating Compositional Generalization and Grounding in Multimodal Environments](https://arxiv.org/abs/2311.04364).

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Introduction

Compositional generalization, the ability of intelligent models to extrapolate understanding of components to novel compositions, is a fundamental yet challenging facet in AI research, especially within multimodal environments. In this work, we address this challenge by exploiting the syntactic structure of language to boost compositional generalization. This paper elevates the importance of syntactic grounding, particularly through attention masking techniques derived from text input parsing. We introduce and evaluate the merits of using syntactic information in the multimodal grounding problem. Our results on grounded compositional generalization underscore the positive impact of dependency parsing across diverse tasks when utilized with Weight Sharing across the Transformer encoder. The results push the state-of-the-art in multimodal grounding and parameter-efficient modeling and provide insights for future research.

## Dependencies

- Compatible with Python 3.6+
- Dependencies can be installed using `reascan/requirements.txt`
- Alternatively, dependencies can be installed using the Conda environment `new_env.yml`

### Setup

Install VirtualEnv (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Install all the required packages from `reascan/`:

```shell
$ pip install -r requirements.txt
```

## Data Preparation

Download and extract the ReaSCAN dataset in `reascan/data/ReaSCAN-v1.1` from [ReaSCAN](https://reascan.github.io/).

- For the constituency mask model, download and extract the ReaSCAN dataset in `reascan/data-with-mask/ReaSCAN-v1.1`.
- For the dependency mask model, download and extract the ReaSCAN dataset in `reascan/data-with-dep-mask/ReaSCAN-v1.1`.

To preprocess data, run the following commands at `reascan/src/utils/`:

```shell
$ python preprocess.py --dataset reascan
```

```shell
$ python preprocess.py --dataset reascan --mode dependency_mask
```

## Usage

The set of possible arguments is available in the `args.py` file. Here, we illustrate training a model on ReaSCAN (similar usage for refexp):

From `/reascan/`:

```shell
$ python src/main.py --mode train --test_split custom_comp --train_fname train.json --val_fname dev_comp_3500.json --load_config reascan_share_layers_dependency.json --run_name run_1 --batch_size 32 --gpu 0 --lr 0.0001 --epochs 120 --seed 3420
```

## Acknowledgments

- The base code for this repository was started from [here](https://github.com/ankursikarwar/Grounded-Compositional-Generalization/tree/main).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{kamali2023syntax,
  title={Syntax-Guided Transformers: Elevating Compositional Generalization and Grounding in Multimodal Environments},
  author={Kamali, Danial and Kordjamshidi, Parisa},
  booktitle={GenBench: The first workshop on generalisation (benchmarking) in NLP},
  pages={130},
  year={2023}
}

```
