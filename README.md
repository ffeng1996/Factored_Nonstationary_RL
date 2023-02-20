# Factored Nonstationary RL

---
Paper:
[[arXiv]](https://arxiv.org/abs/2203.16582)

## Introduction

Dealing with non-stationarity in environments (e.g., in the transition dynamics) and objectives (e.g., in the reward functions) is a challenging problem that is crucial in real-world applications of reinforcement learning (RL). While most current approaches model the changes as a single shared embedding vector, we leverage insights from the recent causality literature to model non-stationarity in terms of individual latent change factors, and causal graphs across different environments. In particular, we propose Factored Adaptation for Non-Stationary RL (FANS-RL), a factored adaption approach that learns jointly both the causal structure in terms of a factored MDP, and a factored representation of the individual time-varying change factors. We prove that under standard assumptions, we can completely recover the causal graph representing the factored transition and reward function, as well as a partial structure between the individual change factors and the state components. Through our general framework, we can consider general non-stationary scenarios with different function types and changing frequency, including changes across episodes and within episodes. Experimental results demonstrate that FANS-RL outperforms existing approaches in terms of return, compactness of the latent state representation, and robustness to varying degrees of non-stationarity.

**We are currently in the process of organizing the codebase and will release the final codebase with both PT and TF versions, stay tuned!**

TODO

- arrange codes (Pytorch and Tensorflow 2) and envs

- update README

## Acknowledgements
Parts of code were built upon [VariBAD](https://github.com/lmzintgraf/varibad).
