import random

import numpy as np
import torch

from environments.mujoco.half_cheetah import HalfCheetahEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HalfCheetahDirEnv(HalfCheetahEnv):

    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(HalfCheetahDirEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self.goal_direction * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self.get_task())
        return observation, reward, done, infos

    def sample_tasks(self, n_tasks):

        return [random.choice([-1.0, 1.0]) for _ in range(n_tasks, )]

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_direction = task

    def get_task(self):
        return np.array([self.goal_direction])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
