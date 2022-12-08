import numpy as np

from environments.env_utils.running_mean_std import RunningMeanStd
from . import VecEnvWrapper


class VecNormalize(VecEnvWrapper):

    def __init__(self, venv, clipobs=10., cliprew=10., gamma=0.99, epsilon=1e-8,
                 normalise_rew=False, ret_rms=None):
        VecEnvWrapper.__init__(self, venv)

        self.normalise_rew = normalise_rew

        self.clipobs = clipobs
        self.cliprew = cliprew

        if self.normalise_rew:
            if ret_rms is None:
                self.ret_rms = RunningMeanStd(shape=())
            else:
                self.ret_rms = ret_rms

        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def step_wait(self):

        obs, rews, news, infos = self.venv.step_wait()

        self.ret = self.ret * self.gamma + rews
        self.ret[news] = 0.

        rews = self._rewfilt(rews)
        return obs, rews, news, infos

    def _rewfilt(self, rews):
        if self.normalise_rew:

            if self.training:
                self.ret_rms.update(self.ret)

            rews_norm = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return [rews, rews_norm]
        else:
            return [rews, rews]

    def reset_mdp(self, index=None):
        if index is None:
            obs = self.venv.reset_mdp()
        else:
            self.venv.remotes[index].send(('reset_mdp', None))
            obs = self.venv.remotes[index].recv()
        return obs

    def reset(self, index=None, task=None):
        self.ret = np.zeros(self.num_envs)
        if index is None:
            obs = self.venv.reset(task=task)
        else:
            try:
                self.venv.remotes[index].send(('reset', task))
                obs = self.venv.remotes[index].recv()
            except AttributeError:
                obs = self.venv.envs[index].reset(task=task)
        return obs

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
