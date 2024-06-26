<div align="center">

# Concealing Sensitive Samples against Gradient Leakage in Federated Learning

[![Venue:AAAI 2024](https://img.shields.io/badge/Venue-AAAI%202024%20-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/30171)
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2301.11308&color=B31B1B)](https://arxiv.org/abs/2209.05724)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


## Abstract
Federated Learning (FL) is a distributed learning paradigm that enhances users’ privacy by eliminating the need for clients to share raw, private data with the server. Despite the success, recent studies expose the vulnerability of FL to model inversion attacks, where adversaries reconstruct users’ private data via eavesdropping on the shared gradient information. We hypothesize that a key factor in the success of such attacks is the low entanglement among gradients per data within the batch during stochastic optimization. This
creates a vulnerability that an adversary can exploit to reconstruct the sensitive data. Building upon this insight, we present a simple, yet effective defense strategy that obfuscates the gradients of the sensitive data with concealed samples. To achieve this, we propose synthesizing concealed samples to mimic the sensitive data at the gradient level while ensuring their visual dissimilarity from the actual sensitive data. Compared to the previous art, our empirical evaluations suggest that the proposed technique provides the strongest protection while simultaneously maintaining the FL performance.

## Getting Started

### 1. Requirements
Install the requirements using a `conda` environment:
```
conda env create -f environment.yml
```

### 2. Evaluate against MIAs
The script can be found under [dlg/scripts](dlg/scripts), below is an example without any defenses.

```
CUDA_VISIBLE_DEVICES=0 python dlg/main.py --demo --batch_idx=3 --output_dir='./logs/demo' --n_data=64 --dataset='MNIST' --defense='none'
```

### 3. Train in FL with defenses
The script can be found under [fl/scripts](fl/scripts).


## BibTeX
```
@inproceedings{wu2024concealing,
  title={Concealing Sensitive Samples against Gradient Leakage in Federated Learning},
  author={Wu, Jing and Hayat, Munawar and Zhou, Mingyi and Harandi, Mehrtash},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={19},
  pages={21717--21725},
  year={2024}
}
```
## Acknowledgements
This repository makes liberal use of code from [Breaching](https://github.com/JonasGeiping/breaching) and [Flower](https://github.com/adap/flower).

