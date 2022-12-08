import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])


class OnlineStorage(object):
    def __init__(self,
                 args, num_steps, num_processes,
                 state_dim, belief_dim, task_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards):

        self.args = args
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.task_dim = task_dim

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0

        self.normalise_rewards = normalise_rewards

        self.prev_state = torch.zeros(num_steps + 1, num_processes, state_dim)

        self.latent_dim = latent_dim
        self.latent_samples_s = []
        self.latent_mean_s = []
        self.latent_logvar_s = []
        self.latent_samples_r = []
        self.latent_mean_r = []
        self.latent_logvar_r = []
        self.hidden_size = hidden_size
        self.hidden_states_s = torch.zeros(num_steps + 1, num_processes, hidden_size)
        self.hidden_states_r = torch.zeros(num_steps + 1, num_processes, hidden_size)
        self.curr_state = torch.zeros(num_steps, num_processes, state_dim)
        self.next_state = torch.zeros(num_steps, num_processes, state_dim)

        if self.args.pass_belief_to_policy:
            self.beliefs = torch.zeros(num_steps + 1, num_processes, belief_dim)
        else:
            self.beliefs = None
        if self.args.pass_task_to_policy:
            self.tasks = torch.zeros(num_steps + 1, num_processes, task_dim)
        else:
            self.tasks = None

        self.curr_rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.curr_rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.next_rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.next_rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.prev_actions = torch.zeros(num_steps + 1, num_processes, action_shape)
        self.curr_actions = torch.zeros(num_steps, num_processes, action_shape)
        self.next_actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = None

        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.to_device()

    def to_device(self):
        if self.args.pass_state_to_policy:
            self.prev_state = self.prev_state.to(device)
        if self.args.pass_latent_to_policy:
            self.latent_samples = [t.to(device) for t in self.latent_samples]
            self.latent_mean = [t.to(device) for t in self.latent_mean]
            self.latent_logvar = [t.to(device) for t in self.latent_logvar]
            self.hidden_states = self.hidden_states.to(device)
            self.next_state = self.next_state.to(device)
        if self.args.pass_belief_to_policy:
            self.beliefs = self.beliefs.to(device)
        if self.args.pass_task_to_policy:
            self.tasks = self.tasks.to(device)
        self.rewards_raw = self.rewards_raw.to(device)
        self.rewards_normalised = self.rewards_normalised.to(device)
        self.done = self.done.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)

    def insert(self,
               state,
               belief,
               task,
               actions,
               rewards_raw,
               rewards_normalised,
               value_preds,
               masks,
               bad_masks,
               done,

               hidden_states_s=None,
               latent_sample_s=None,
               latent_mean_s=None,
               latent_logvar_s=None,
               hidden_states_r=None,
               latent_sample_r=None,
               latent_mean_r=None,
               latent_logvar_r=None,
               ):
        self.curr_state[self.step + 1].copy_(state)
        if self.args.pass_belief_to_policy:
            self.beliefs[self.step + 1].copy_(belief)
        if self.args.pass_task_to_policy:
            self.tasks[self.step + 1].copy_(task)
        if self.args.pass_latent_to_policy:
            self.latent_samples_s.append(latent_sample_s.detach().clone())
            self.latent_samples_r.append(latent_sample_r.detach().clone())
            self.latent_mean_s.append(latent_mean_s.detach().clone())
            self.latent_mean_r.append(latent_mean_r.detach().clone())
            self.latent_logvar_s.append(latent_logvar_s.detach().clone())
            self.latent_logvar_r.append(latent_logvar_r.detach().clone())
            self.hidden_states_s[self.step + 1].copy_(hidden_states_s.detach())
            self.hidden_states_r[self.step + 1].copy_(hidden_states_r.detach())
        self.curr_actions[self.step] = actions.detach().clone()
        self.curr_rewards_raw[self.step].copy_(rewards_raw)
        self.curr_rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.done[self.step + 1].copy_(done)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.prev_state[0].copy_(self.prev_state[-1])
        if self.args.pass_belief_to_policy:
            self.beliefs[0].copy_(self.beliefs[-1])
        if self.args.pass_task_to_policy:
            self.tasks[0].copy_(self.tasks[-1])
        if self.args.pass_latent_to_policy:
            self.latent_samples_s = []
            self.latent_mean_s = []
            self.latent_logvar_s = []
            self.hidden_states_s[0].copy_(self.hidden_states[-1])
            self.latent_samples_r = []
            self.latent_mean_r = []
            self.latent_logvar_r = []
            self.hidden_states_r[0].copy_(self.hidden_states[-1])
        self.done[0].copy_(self.done[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.action_log_probs = None

    def compute_returns(self, next_value, use_gae, gamma, tau, use_proper_time_limits=True):

        if self.normalise_rewards:
            rewards = self.rewards_normalised.clone()
        else:
            rewards = self.rewards_raw.clone()

        self._compute_returns(next_value=next_value, rewards=rewards, value_preds=self.value_preds,
                              returns=self.returns,
                              gamma=gamma, tau=tau, use_gae=use_gae, use_proper_time_limits=use_proper_time_limits)

    def _compute_returns(self, next_value, rewards, value_preds, returns, gamma, tau, use_gae, use_proper_time_limits):

        if use_proper_time_limits:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = (returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]) * self.bad_masks[
                        step + 1] + (1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]

    def num_transitions(self):
        return len(self.prev_state) * self.num_processes

    def before_update(self, policy):
        latent = utl.get_latent_for_policy(self.args,
                                           latent_sample=torch.stack(
                                               self.latent_samples[:-1]) if self.latent_samples is not None else None,
                                           latent_mean=torch.stack(
                                               self.latent_mean[:-1]) if self.latent_mean is not None else None,
                                           latent_logvar=torch.stack(
                                               self.latent_logvar[:-1]) if self.latent_mean is not None else None)
        _, action_log_probs, _ = policy.evaluate_actions(self.prev_state[:-1],
                                                         latent,
                                                         self.beliefs[:-1] if self.beliefs is not None else None,
                                                         self.tasks[:-1] if self.tasks is not None else None,
                                                         self.actions)
        self.action_log_probs = action_log_probs.detach()

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if self.args.pass_state_to_policy:
                state_batch = self.prev_state[:-1].reshape(-1, *self.prev_state.size()[2:])[indices]
            else:
                state_batch = None
            if self.args.pass_latent_to_policy:
                latent_sample_batch = torch.cat(self.latent_samples[:-1])[indices]
                latent_mean_batch = torch.cat(self.latent_mean[:-1])[indices]
                latent_logvar_batch = torch.cat(self.latent_logvar[:-1])[indices]
            else:
                latent_sample_batch = latent_mean_batch = latent_logvar_batch = None
            if self.args.pass_belief_to_policy:
                belief_batch = self.beliefs[:-1].reshape(-1, *self.beliefs.size()[2:])[indices]
            else:
                belief_batch = None
            if self.args.pass_task_to_policy:
                task_batch = self.tasks[:-1].reshape(-1, *self.tasks.size()[2:])[indices]
            else:
                task_batch = None

            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield state_batch, belief_batch, task_batch, \
                  actions_batch, \
                  latent_sample_batch, latent_mean_batch, latent_logvar_batch, \
                  value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
