import argparse
import warnings

import numpy as np
import torch

from config.mujoco import args_cheetah_dir, args_cheetah_vel
from environments.parallel_envs import make_vec_envs
from learner import Learner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='cheetah_dir')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    if env == 'cheetah_dir':
        args = args_cheetah_dir.get_args(rest_args)

    elif env == 'cheetah_vel':
        args = args_cheetah_vel.get_args(rest_args)

    else:
        raise Exception("Invalid Environment")

    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None,
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True

    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        learner = Learner(args)
        learner.train(offline=True)
        learner.train()


if __name__ == '__main__':
    main()
