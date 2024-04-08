import os
import pickle
# import pickle5 as pickle
import random
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from environments.parallel_envs import make_vec_envs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reset_env(env, args, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset().float().to(device)
    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)

    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).float().to(device) if args.pass_task_to_policy else None
        
    return state, belief, task


def squash_action(action, args):
    if args.norm_actions_post_sampling:
        return torch.tanh(action)
    else:
        return action


def env_step(env, action, args):
    act = squash_action(action.detach(), args)
    next_obs, reward, done, infos = env.step(act)

    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)

    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).float().to(device) if (args.pass_task_to_policy or args.decode_task) else None

    return [next_obs, belief, task], reward, done, infos


def select_action(args,
                  policy,
                  deterministic,
                  state=None,
                  belief=None,
                  task=None,
                  latent_sample=None, latent_mean=None, latent_logvar=None):
    """ Select action using the policy. """
    latent = get_latent_for_policy(args=args, latent_sample=latent_sample, latent_mean=latent_mean,
                                   latent_logvar=latent_logvar)
    action = policy.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action = action
    else:
        value = None
    action = action.to(device)
    return value, action


def get_latent_for_policy(args, latent_sample=None, latent_mean=None, latent_logvar=None):

    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None

    if args.add_nonlinearity_to_latent:
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    if args.sample_embeddings:
        latent = latent_sample
    else:
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.shape[0] == 1:
        latent = latent.squeeze(0)

    return latent


def update_encoding(encoder, next_obs, action, reward, done, hidden_state):
    # reset hidden state of the recurrent net when we reset the task
    if done is not None:
        hidden_state_s = encoder.reset_hidden(hidden_state, done)
    with torch.no_grad():
        latent_sample_s, latent_mean_s, latent_logvar_s, hidden_state_s = encoder(actions=action.float(),
                                                                          states=next_obs,
                                                                          rewards=reward,
                                                                          hidden_state=hidden_state,
                                                                          return_prior=False)

    return latent_sample_s, latent_mean_s, latent_logvar_s, hidden_state_s


def seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'Use only one process and set --deterministic_execution to True if you want identical results '
              '(only recommended for debugging).')


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def recompute_embeddings(
        policy_storage,
        encoder_s,
        encoder_r,
        sample,
        update_idx,
        detach_every
):
    # get the prior
    tot_latent_samples = policy_storage.latent_samples[0].detach().clone()
    tot_latent_mean = policy_storage.latent_mean[0].detach().clone()
    tot_latent_logvar = policy_storage.latent_logvar[0].detach().clone()
    latent_sample_s = [torch.split(tot_latent_samples, tot_latent_samples.size(1) // 2, dim=-1)[0]]
    latent_mean_s = [torch.split(tot_latent_mean, tot_latent_mean.size(1) // 2, dim=-1)[0]]
    latent_logvar_s = [torch.split(tot_latent_logvar, tot_latent_logvar.size(1) // 2, dim=-1)[0]]
    latent_sample_r = [torch.split(tot_latent_samples, tot_latent_samples.size(1) // 2, dim=-1)[1]]
    latent_mean_r = [torch.split(tot_latent_mean, tot_latent_mean.size(1) // 2, dim=-1)[1]]
    latent_logvar_r = [torch.split(tot_latent_logvar, tot_latent_logvar.size(1) // 2, dim=-1)[1]]

    tot_latent_samples[0].requires_grad = True
    tot_latent_mean[0].requires_grad = True
    tot_latent_logvar[0].requires_grad = True

    # loop through experience and update hidden state
    # (we need to loop because we sometimes need to reset the hidden state)
    h = policy_storage.hidden_states[0].detach()
    h_s, h_r = torch.split(h, h.size(1) // 2, dim=-1)
    for i in range(policy_storage.actions.shape[0]):
        # reset hidden state of the GRU when we reset the task
        h_s = encoder_s.reset_hidden(h_s, policy_storage.done[i + 1])
        h_r = encoder_r.reset_hidden(h_r, policy_storage.done[i + 1])

        ts_s, tm_s, tl_s, h_s, _, _ = encoder_s(policy_storage.actions.float()[i:i + 1],
                                policy_storage.next_state[i:i + 1],
                                policy_storage.rewards_raw[i:i + 1],
                                h_s,
                                sample=sample,
                                return_prior=False,
                                detach_every=detach_every
                                )
        ts_r, tm_r, tl_r, h_r, _, _ = encoder_r(policy_storage.actions.float()[i:i + 1],
                                                policy_storage.next_state[i:i + 1],
                                                policy_storage.rewards_raw[i:i + 1],
                                                h_r,
                                                sample=sample,
                                                return_prior=False,
                                                detach_every=detach_every
                                                )


        latent_sample_s.append(ts_s)
        latent_mean_s.append(tm_s)
        latent_logvar_s.append(tl_s)
        latent_sample_r.append(ts_r)
        latent_mean_r.append(tm_r)
        latent_logvar_r.append(tl_r)

    if update_idx == 0:
        try:
            assert (torch.cat(policy_storage.latent_mean) - torch.cat(torch.cat(latent_mean_s, latent_mean_r), dim=-1)).sum() == 0
            assert (torch.cat(policy_storage.latent_logvar) - torch.cat(torch.cat((latent_logvar_s, latent_logvar_r), dim=-1))).sum() == 0
        except AssertionError:
            warnings.warn('You are not recomputing the embeddings correctly!')
            import pdb
            pdb.set_trace()

    policy_storage.latent_samples = torch.cat((latent_sample_s, latent_sample_r), dim=-1)
    policy_storage.latent_mean = torch.cat((latent_mean_s, latent_mean_r), dim=-1)
    policy_storage.latent_logvar = torch.cat((latent_logvar_s, latent_logvar_r), dim=-1)


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


def sample_gaussian(mu, logvar, num=None):
    std = torch.exp(0.5 * logvar)
    if num is not None:
        std = std.repeat(num, 1)
        mu = mu.repeat(num, 1)
    eps = torch.randn_like(std)
    return mu + std * eps


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def get_task_dim(args):
    env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                        gamma=args.policy_gamma, device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        tasks=None
                        )
    return env.task_dim


def get_num_tasks(args):
    env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                        gamma=args.policy_gamma, device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        tasks=None
                        )
    try:
        num_tasks = env.num_tasks
    except AttributeError:
        num_tasks = None
    return num_tasks


def clip(value, low, high):
    low, high = torch.tensor(low), torch.tensor(high)

    assert torch.all(low <= high), (low, high)

    clipped_value = torch.max(torch.min(value, high), low)
    return clipped_value
