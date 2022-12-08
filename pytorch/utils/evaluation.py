import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.parallel_envs import make_vec_envs
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(args,
             policy,
             ret_rms,
             iter_idx,
             tasks,
             encoder_s=None,
             encoder_r=None,
             num_episodes=None
             ):
    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = args.num_processes

    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    envs = make_vec_envs(env_name,
                         seed=args.seed * 42 + iter_idx,
                         num_processes=num_processes,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=num_processes + 1,
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy,
                         ret_rms=ret_rms,
                         tasks=tasks,
                         add_done_info=args.max_rollouts_per_task > 1,
                         )
    num_steps = envs._max_episode_steps

    state, belief, task = utl.reset_env(envs, args)

    task_count = torch.zeros(num_processes).long().to(device)

    if encoder_s is not None:

        latent_state_sample, latent_state_mean, latent_state_logvar, state_hidden_state = encoder_s.prior(num_processes)
    else:
        latent_state_sample = latent_state_mean = latent_state_logvar = state_hidden_state = None

    if encoder_r is not None:
        latent_rew_sample, latent_rew_mean, latent_rew_logvar, rew_hidden_state = encoder_r.prior(num_processes)
    else:
        latent_rew_sample = latent_rew_mean = latent_rew_logvar = rew_hidden_state = None

    for episode_idx in range(num_episodes):
        for step_idx in range(num_steps):
            with torch.no_grad():
                _, action = utl.select_action(args=args,
                                              policy=policy,
                                              state=state,
                                              belief=belief,
                                              task=task,
                                              latent_state_sample=latent_state_sample,
                                              latent_state_mean=latent_state_mean,
                                              latent_state_logvar=latent_state_logvar,
                                              latent_rew_sample=latent_rew_sample,
                                              latent_rew_mean=latent_rew_mean,
                                              latent_rew_logvar=latent_rew_logvar,
                                              deterministic=True)

            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
            done_mdp = [info['done_mdp'] for info in infos]

            if encoder is not None:
                latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                                              next_obs=state,
                                                                                              action=action,
                                                                                              reward=rew_raw,
                                                                                              done=None,
                                                                                              hidden_state=hidden_state)

            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                task_count[i] = min(task_count[i] + 1, num_episodes)
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task = utl.reset_env(envs, args, indices=done_indices, state=state)

    envs.close()

    return returns_per_episode[:, :num_episodes]


def get_test_rollout(args, env, policy, encoder_s=None, encoder_r=None):
    num_episodes = args.max_rollouts_per_task

    episode_prev_obs = [[] for _ in range(num_episodes)]
    episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]

    episode_returns = []
    episode_lengths = []

    if encoder_s is not None:
        episode_latent_samples_s = [[] for _ in range(num_episodes)]
        episode_latent_means_s = [[] for _ in range(num_episodes)]
        episode_latent_logvars_s = [[] for _ in range(num_episodes)]
    else:
        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
        episode_latent_means_s = episode_latent_logvars_s = None

    if encoder_r is not None:
        episode_latent_samples_r = [[] for _ in range(num_episodes)]
        episode_latent_means_r = [[] for _ in range(num_episodes)]
        episode_latent_logvars_r = [[] for _ in range(num_episodes)]
    else:
        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
        episode_latent_means_r = episode_latent_logvars_r = None

    env.reset_task()
    state, belief, task = utl.reset_env(env, args)
    state = state.reshape((1, -1)).to(device)
    task = task.view(-1) if task is not None else None

    for episode_idx in range(num_episodes):

        curr_rollout_rew = []

        if encoder_s is not None:
            if episode_idx == 0:
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder_s.prior(1)
                curr_latent_sample = curr_latent_sample[0].to(device)
                curr_latent_mean = curr_latent_mean[0].to(device)
                curr_latent_logvar = curr_latent_logvar[0].to(device)
            episode_latent_samples_s[episode_idx].append(curr_latent_sample[0].clone())
            episode_latent_means_s[episode_idx].append(curr_latent_mean[0].clone())
            episode_latent_logvars_s[episode_idx].append(curr_latent_logvar[0].clone())

        if encoder_r is not None:
            if episode_idx == 0:
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder_r.prior(1)
                curr_latent_sample = curr_latent_sample[0].to(device)
                curr_latent_mean = curr_latent_mean[0].to(device)
                curr_latent_logvar = curr_latent_logvar[0].to(device)
            episode_latent_samples_r[episode_idx].append(curr_latent_sample[0].clone())
            episode_latent_means_r[episode_idx].append(curr_latent_mean[0].clone())
            episode_latent_logvars_r[episode_idx].append(curr_latent_logvar[0].clone())

        for step_idx in range(1, env._max_episode_steps + 1):

            episode_prev_obs[episode_idx].append(state.clone())

            latent = utl.get_latent_for_policy(args,
                                               latent_sample=curr_latent_sample,
                                               latent_mean=curr_latent_mean,
                                               latent_logvar=curr_latent_logvar)
            _, action = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task, deterministic=True)
            action = action.reshape((1, *action.shape))

            (state, belief, task), (rew_raw, rew_normalised), done, infos = utl.env_step(env, action, args)
            state = state.reshape((1, -1)).to(device)
            task = task.view(-1) if task is not None else None

            if encoder_s is not None:
                curr_latent_sample_s, curr_latent_mean_s, curr_latent_logvar_s, hidden_state_s = encoder_s(
                    action.float().to(device),
                    state,
                    rew_raw.reshape((1, 1)).float().to(device),
                    hidden_state,
                    return_prior=False)

                episode_latent_samples_s[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means_s[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars_s[episode_idx].append(curr_latent_logvar[0].clone())

            episode_next_obs[episode_idx].append(state.clone())
            episode_rewards[episode_idx].append(rew_raw.clone())
            episode_actions[episode_idx].append(action.clone())

            if infos[0]['done_mdp']:
                break

        episode_returns.append(sum(curr_rollout_rew))
        episode_lengths.append(step_idx)

    if encoder_s is not None:
        episode_latent_means_s = [torch.stack(e) for e in episode_latent_means_s]
        episode_latent_logvars_s = [torch.stack(e) for e in episode_latent_logvars_s]

    episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
    episode_next_obs = [torch.cat(e) for e in episode_next_obs]
    episode_actions = [torch.cat(e) for e in episode_actions]
    episode_rewards = [torch.cat(r) for r in episode_rewards]

    return episode_latent_means_s, episode_latent_logvars_s, \
           episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
           episode_returns
