import random

import numpy as np

from .half_cheetah import HalfCheetahEnv


class HalfCheetahVelEnv(HalfCheetahEnv):

    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.goal_velocity)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self.get_task())
        return observation, reward, done, infos

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_velocity = task

    def get_task(self):
        return np.array([self.goal_velocity])

    def sample_tasks(self, n_tasks):
        return [random.uniform(0.0, 3.0) for _ in range(n_tasks)]

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)


class HalfCheetahRandVelOracleEnv(HalfCheetahVelEnv):

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
            [self.goal_velocity]
        ])
