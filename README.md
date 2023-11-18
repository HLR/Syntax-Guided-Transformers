
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

> Compositional generalization, the ability of intelligent models to extrapolate understanding of components to novel compositions, is a fundamental yet challenging facet in AI research, especially within multimodal environments. In this work, we address this challenge by exploiting the syntactic structure of language to boost compositional generalization. This paper elevates the importance of syntactic grounding, particularly through attention masking techniques derived from text input parsing. We introduce and evaluate the merits of using syntactic information in the multimodal grounding problem. Our results on grounded compositional generalization underscore the positive impact of dependency parsing across diverse tasks when utilized with Weight Sharing across the Transformer encoder. The results push the state-of-the-art in multimodal grounding and parameter-efficient modeling and provide insights for future research.

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

$virtualenv-ppython3venv

$sourcevenv/bin/activate

```

Install all the required packages from `reascan/`:

```shell

$pipinstall-rrequirements.txt

```

## Data Preparation

Download and extract the ReaSCAN dataset in `reascan/data/ReaSCAN-v1.1` from [ReaSCAN](https://reascan.github.io/).

- For the constituency mask model, download and extract the ReaSCAN dataset in `reascan/data-with-mask/ReaSCAN-v1.1`.
- For the dependency mask model, download and extract the ReaSCAN dataset in `reascan/data-with-dep-mask/ReaSCAN-v1.1`.

To preprocess data, run the following commands at `reascan/src/utils/`:

```shell

$pythonpreprocess.py--datasetreascan

```

```shell

$pythonpreprocess.py--datasetreascan--modedependency_mask

```

## Usage

The set of possible arguments is available in the `args.py` file. Here, we illustrate training a model on ReaSCAN (similar usage for refexp):

From `/reascan/`:

```shell

$pythonsrc/main.py--modetrain--test_splitcustom_comp--train_fnametrain.json--val_fnamedev_comp_3500.json--load_configreascan_share_layers_dependency.json--run_namerun_1--batch_size32--gpu0--lr0.0001--epochs120--seed3420

```

## Acknowledgments

- The base code for this repository was started from [here](https://github.com/ankursikarwar/Grounded-Compositional-Generalization/tree/main).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex

@misc{kamali2023syntaxguided,

title={Syntax-Guided Transformers: Elevating Compositional Generalization and Grounding in Multimodal Environments}, 

author={Danial Kamali and Parisa Kordjamshidi},

year={2023},

eprint={2311.04364},

archivePrefix={arXiv},

primaryClass={cs.CL}

}

```
