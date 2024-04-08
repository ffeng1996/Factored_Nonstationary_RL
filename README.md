# Factored Adaptation for Non-Stationary Reinforcement Learning

### Requirements

The main requirements can be found in `requirements.txt`. 

To install the requirements, you can follow the instructions below:

```
pip install -r requirements.txt
```

Additionally, Mujoco should be installed. 

### Overview

The main training loop is in `learner.py`, the VAE set-up and losses are in `vae.py`, the model design is in `models/`, the RL algorithms are in `algorithms/`, and the hyperparameters are in `config/`. You need to modify the hyperparameters for the specific environment you want to run.


### Running experiments

To evaluate, run

```
python main.py --env-type <env_name>
```

which will use hyperparameters from `config/args_<env_name>.py`.