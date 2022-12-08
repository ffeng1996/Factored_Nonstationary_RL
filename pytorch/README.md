# FansRL

Implementation in paper **Factored Adaptation for Non-Stationary Reinforcement Learning (NeurIPS'22)**.

---
Paper:
[[NeurIPS2022]](https://openreview.net/forum?id=VQ9fogN1q6e),
[[arXiv]](https://arxiv.org/abs/2203.16582)

## Introduction

Dealing with non-stationarity in environments (e.g., in the transition dynamics) and objectives (e.g., in the reward functions) is a challenging problem that is crucial in real-world applications of reinforcement learning (RL). While most current approaches model the changes as a single shared embedding vector, we leverage insights from the recent causality literature to model non-stationarity in terms of individual latent change factors, and causal graphs across different environments. In particular, we propose Factored Adaptation for Non-Stationary RL (FANS-RL), a factored adaption approach that learns jointly both the causal structure in terms of a factored MDP, and a factored representation of the individual time-varying change factors. We prove that under standard assumptions, we can completely recover the causal graph representing the factored transition and reward function, as well as a partial structure between the individual change factors and the state components. Through our general framework, we can consider general non-stationary scenarios with different function types and changing frequency, including changes across episodes and within episodes. Experimental results demonstrate that FANS-RL outperforms existing approaches in terms of return, compactness of the latent state representation, and robustness to varying degrees of non-stationarity.

## Requirements
The current version of the code has been tested with following libs:
* `cudatoolkit==10.0.130`
* `pytorch 1.2.0`
* `cudatoolkit 10.0.130`
* `gym 0.17.2`
* `Pillow`
* `numpy 1.22.0`
* `opencv-python 4.5.1.48`
* `mujoco-py 2.0.2.10`


## Running an experiment

As an example:
```
python main.py --env-type cheetah_dir
```

## Citation

If you find our work helpful to your research, please consider citing our paper:

```
@inproceedings{
feng2022factored,
title={Factored Adaptation for Non-Stationary Reinforcement Learning},
author={Fan Feng and Biwei Huang and Kun Zhang and Sara Magliacane},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=VQ9fogN1q6e}
}
```

## Acknowledgements
Parts of code were built upon [VariBAD](https://github.com/lmzintgraf/varibad).