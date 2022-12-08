from collections import OrderedDict

import gym
import numpy as np


def copy_obs_dict(obs):
    return {k: np.copy(v) for k, v in obs.items()}


def dict_to_obs(obs_dict):
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


def obs_space_info(obs_space):
    try:
        if isinstance(obs_space, gym.spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            subspaces = obs_space.spaces
        else:
            subspaces = {None: obs_space}
    except AttributeError:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = getattr(box, 'dtype', np.float32)
    return keys, shapes, dtypes


def obs_to_dict(obs):
    if isinstance(obs, dict):
        return obs
    return {None: obs}
