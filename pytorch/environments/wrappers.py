import gym
import numpy as np
from gym import spaces
from gym.envs.registration import load

from environments.mujoco import rand_param_envs

try:

    gym.logger.set_level(40)
except AttributeError:
    pass


def mujoco_wrapper(entry_point, **kwargs):
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    return env


class RLWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 episodes_per_task,
                 add_done_info=None,
                 ):

        super().__init__(env)

        if not hasattr(self.env.unwrapped, 'task_dim'):
            self.env.unwrapped.task_dim = 0
        if not hasattr(self.env.unwrapped, 'belief_dim'):
            self.env.unwrapped.belief_dim = 0
        if not hasattr(self.env.unwrapped, 'get_belief'):
            self.env.unwrapped.get_belief = lambda: None
        if not hasattr(self.env.unwrapped, 'num_states'):
            self.env.unwrapped.num_states = None

        if add_done_info is None:
            if episodes_per_task > 1:
                self.add_done_info = True
            else:
                self.add_done_info = False
        else:
            self.add_done_info = add_done_info

        if self.add_done_info:
            if isinstance(self.observation_space, spaces.Box) or isinstance(self.observation_space,
                                                                            rand_param_envs.gym.spaces.box.Box):
                if len(self.observation_space.shape) > 1:
                    raise ValueError
                self.observation_space = spaces.Box(low=np.array([*self.observation_space.low, 0]),

                                                    high=np.array([*self.observation_space.high, 1])
                                                    )
            else:

                raise NotImplementedError

        self.episodes_per_task = episodes_per_task

        self.episode_count = 0

        self.step_count_bamdp = 0.0

        try:
            self.horizon_bamdp = self.episodes_per_task * self.env._max_episode_steps
        except AttributeError:
            self.horizon_bamdp = self.episodes_per_task * self.env.unwrapped._max_episode_steps

        self.horizon_bamdp += self.episodes_per_task - 1

        self.done_mdp = True

    def reset(self, task=None):

        self.env.reset_task(task)

        try:
            state = self.env.reset()
        except AttributeError:
            state = self.env.unwrapped.reset()

        self.episode_count = 0
        self.step_count_bamdp = 0
        self.done_mdp = False
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))

        return state

    def reset_mdp(self):
        state = self.env.reset()
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))
        self.done_mdp = False
        return state

    def step(self, action):

        state, reward, self.done_mdp, info = self.env.step(action)

        info['done_mdp'] = self.done_mdp

        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))

        self.step_count_bamdp += 1

        done_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_task:
                done_bamdp = True

        if self.done_mdp and not done_bamdp:
            info['start_state'] = self.reset_mdp()

        return state, reward, done_bamdp, info

    def __getattr__(self, attr):

        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr


class TimeLimitMask(gym.Wrapper):

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def __getattr__(self, attr):

        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr
