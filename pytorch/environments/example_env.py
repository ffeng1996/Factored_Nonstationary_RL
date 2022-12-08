import gym


class ExampleEnv(gym.Env):
    def __init__(self):
        super(ExampleEnv, self).__init__()

    def step(self, action):
        pass

    def reset(self):
        pass

    def get_task(self):
        pass

    def reset_task(self, task=None):
        pass

    def visualise_behaviour(self,
                            env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            reward_decoder=None,
                            state_decoder=None,
                            task_decoder=None,
                            image_folder=None,
                            **kwargs):
        pass
