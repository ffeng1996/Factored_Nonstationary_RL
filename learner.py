import os
import time

import gym
import numpy as np
import torch

from algorithms.online_storage import OnlineStorage
from algorithms.sac import SAC
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from vae import VAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Learner:
    def __init__(self, args):

        self.args = args
        utl.seed(self.args.seed, self.args.deterministic_execution)

        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = -1

        self.logger = TBLogger(self.args, self.args.exp_label)

        self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                  gamma=args.policy_gamma, device=device,
                                  episodes_per_task=self.args.max_rollouts_per_task,
                                  normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                  tasks=None
                                  )

        if self.args.single_task_mode:

            self.train_tasks = self.envs.get_task()

            self.train_tasks[1:] = self.train_tasks[0]

            self.train_tasks = [t for t in self.train_tasks]

            self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                      gamma=args.policy_gamma, device=device,
                                      episodes_per_task=self.args.max_rollouts_per_task,
                                      normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                      tasks=self.train_tasks
                                      )

            utl.save_obj(self.train_tasks, self.logger.full_output_folder, "train_tasks")
        else:
            self.train_tasks = None

        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.max_rollouts_per_task

        self.args.state_dim = self.envs.observation_space.shape[0]
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states

        self.args.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        self.vae = VaribadVAE(self.args, self.logger, lambda: self.iter_idx)
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()

    def initialise_policy_storage(self):
        return OnlineStorage(args=self.args,
                             num_steps=self.args.policy_num_steps,
                             num_processes=self.args.num_processes,
                             state_dim=self.args.state_dim,
                             latent_dim=self.args.latent_dim,
                             belief_dim=self.args.belief_dim,
                             task_dim=self.args.task_dim,
                             action_space=self.args.action_space,
                             hidden_size=self.args.encoder_gru_hidden_size,
                             normalise_rewards=self.args.norm_rew_for_policy,
                             )

    def initialise_policy(self):
        if self.args.policy == 'sac':
            policy = SAC(
                input_dim,
                action_dim,
                self.args
            )
        else:
            raise NotImplementedError

        return policy

    def train(self, offline=True):

        start_time = time.time()
        self.vae.compute_vae_loss(update=True, offline=True, offline_data=offline_data)

        if not offline:
            self.vae.thetas2s.requires_grad = False
            self.vae.thetaP2thetaC.requires_grad = False
            self.vae.sP2sC.requires_grad = False
            self.vae.aP2sC.requires_grad = False
            self.vae.aP2rC.requires_grad = False
            self.vae.sP2rC.requires_grad = False
            self.vae.optimiser_vae = torch.optim.Adam(
                [*self.vae.encoder_r.parameters(), *self.vae.encoder_s.parameters(), *self.vae.decoder_params],
                lr=self.args.lr_vae)

        prev_state, belief, task = utl.reset_env(self.envs, self.args)

        self.policy_storage.prev_state[0].copy_(prev_state)

        with torch.no_grad():
            self.log(None, None, start_time)

        for self.iter_idx in range(self.num_updates):

            with torch.no_grad():
                latent_sample_s, latent_mean_s, latent_logvar_s, hidden_state_s, \
                latent_sample_r, latent_mean_r, latent_logvar_r, hidden_state_r = self.encode_running_trajectory()

            assert len(self.policy_storage.latent_mean_s) == 0 and len(self.policy_storage.latent_mean_r) == 0
            self.policy_storage.hidden_states[0].copy_(hidden_state_s)
            self.policy_storage.hidden_states[0].copy_(hidden_state_r)
            self.policy_storage.latent_samples.append(latent_sample_s.clone())
            self.policy_storage.latent_samples.append(latent_sample_r.clone())
            self.policy_storage.latent_mean.append(latent_mean_s.clone())
            self.policy_storage.latent_mean.append(latent_mean_r.clone())
            self.policy_storage.latent_logvar.append(latent_logvar_s.clone())
            self.policy_storage.latent_logvar.append(latent_logvar_r.clone())

            for step in range(self.args.policy_num_steps):

                with torch.no_grad():
                    value, action = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=prev_state,
                        belief=belief,
                        task=task,
                        deterministic=False,
                        latent_state_sample=latent_sample_s,
                        latent_state_mean=latent_mean_s,
                        latent_state_logvar=latent_logvar_s,
                        latent_rew_sample=latent_sample_r,
                        latent_rew_mean=latent_mean_r,
                        latent_rew_logvar=latent_logvar_r
                    )

                [next_state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs, action,
                                                                                                  self.args)

                done = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))

                masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                with torch.no_grad():

                    latent_sample_s, latent_mean_s, latent_logvar_s, hidden_state_s = utl.update_encoding(
                        encoder=self.vae.encoder_s,
                        next_obs=next_state,
                        action=action,
                        reward=rew_raw,
                        done=done,
                        hidden_state=hidden_state_s,
                    )
                    latent_sample_r, latent_mean_r, latent_logvar_r, hidden_state_r = utl.update_encoding(
                        encoder=self.vae.encoder_r,
                        next_obs=next_state,
                        action=action,
                        reward=rew_raw,
                        done=done,
                        hidden_state=hidden_state_r,
                    )

                if not (self.args.disable_decoder and self.args.disable_kl_term):
                    self.vae.rollout_storage.insert(prev_state.clone(),
                                                    action.detach().clone(),
                                                    next_state.clone(),
                                                    rew_raw.clone(),
                                                    done.clone(),
                                                    task.clone() if task is not None else None)

                self.policy_storage.next_state[step] = next_state.clone()

                done_indices = np.argwhere(done.cpu().flatten()).flatten()
                if len(done_indices) > 0:
                    next_state, belief, task = utl.reset_env(self.envs, self.args,
                                                             indices=done_indices, state=next_state)

                self.policy_storage.insert(
                    state=next_state,
                    belief=belief,
                    task=task,
                    actions=action,
                    rewards_raw=rew_raw,
                    rewards_normalised=rew_normalised,
                    value_preds=value,
                    masks=masks_done,
                    bad_masks=bad_masks,
                    done=done,
                    hidden_states_s=hidden_state_s.squeeze(0),
                    latent_sample_s=latent_sample_s,
                    latent_mean_s=latent_mean_s,
                    latent_logvar_s=latent_logvar_s,
                    hidden_states_r=hidden_state_r.squeeze(0),
                    latent_sample_r=latent_sample_r,
                    latent_mean_r=latent_mean_r,
                    latent_logvar_r=latent_logvar_r,
                )

                prev_state = next_state

                self.frames += self.args.num_processes

            if self.args.precollect_len <= self.frames:

                if self.args.pretrain_len > self.iter_idx:
                    for p in range(self.args.num_vae_updates_per_pretrain):
                        self.vae.compute_vae_loss(update=True,
                                                  pretrain_index=self.iter_idx * self.args.num_vae_updates_per_pretrain + p)

                else:
                    train_stats = self.update(state=prev_state,
                                              belief=belief,
                                              task=task,
                                              latent_sample_s=latent_sample_s,
                                              latent_mean_s=latent_mean_s,
                                              latent_logvar_s=latent_logvar_s,
                                              latent_sample_r=latent_sample_r,
                                              latent_mean_r=latent_mean_r,
                                              latent_logvar_r=latent_logvar_r)

                    run_stats = [action, self.policy_storage.action_log_probs, value]
                    with torch.no_grad():
                        self.log(run_stats, train_stats, start_time)

            self.policy_storage.after_update()

        self.envs.close()

    def encode_running_trajectory(self):

        prev_obs, curr_obs, next_obs, prev_act, curr_act, next_act, curr_rew, next_rew, lens = self.vae.rollout_storage.get_running_batch()

        latent_sample_state_curr, latent_mean_state_curr, latent_logvar_state_curr, hidden_state_state = self.vae.encoder_s(
            actions=curr_act,
            states=curr_obs,
            rewards=None,
            hidden_state=None,
            return_prior=True,
            detach_every=None,
            )

        latent_sample_reward_curr, latent_mean_rew_curr, latent_logvar_rew_curr, hidden_state_rew = self.vae.encoder_r(
            actions=curr_act,
            states=curr_obs,
            rewards=curr_rew,
            hidden_state=None,
            return_prior=False,
            detach_every=None,
            )

        latent_sample_s = (torch.stack([latent_sample_state_curr[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_mean_s = (torch.stack([latent_mean_state_curr[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_logvar_s = (torch.stack([latent_logvar_state_curr[lens[i]][i] for i in range(len(lens))])).to(device)
        hidden_state_s = (torch.stack([hidden_state_state[lens[i]][i] for i in range(len(lens))])).to(device)

        latent_sample_r = (torch.stack([latent_sample_reward_curr[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_mean_r = (torch.stack([latent_mean_rew_curr[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_logvar_r = (torch.stack([latent_logvar_rew_curr[lens[i]][i] for i in range(len(lens))])).to(device)
        hidden_state_r = (torch.stack([hidden_state_rew[lens[i]][i] for i in range(len(lens))])).to(device)

        return latent_sample_s, latent_mean_s, latent_logvar_s, hidden_state_s, \
               latent_sample_r, latent_mean_r, latent_logvar_r, hidden_state_r

    def get_value(self, state, belief, task, latent_sample_s, latent_mean_s, latent_logvar_s, latent_sample_r,
                  latent_mean_r, latent_logvar_r):
        latent = utl.get_latent_for_policy(self.args,
                                           latent_state_sample=latent_sample_s, latent_state_mean=latent_mean_s,
                                           latent_state_logvar=latent_logvar_s,
                                           latent_rew_sample=latent_sample_r, latent_rew_mean=latent_mean_r,
                                           latent_rew_logvar=latent_logvar_r
                                           )
        return self.policy.actor_critic.get_value(state=state, belief=belief, task=task, latent=latent).detach()

    def update(self, state, belief, task, latent_sample_s, latent_mean_s, latent_logvar_s, latent_sample_r,
               latent_mean_r, latent_logvar_r):

        if self.iter_idx >= self.args.pretrain_len and self.iter_idx > 0:

            with torch.no_grad():
                next_value = self.get_value(state=state,
                                            belief=belief,
                                            task=task,
                                            latent_sample_s=latent_sample_s,
                                            latent_mean_s=latent_mean_s,
                                            latent_logvar_s=latent_logvar_s,
                                            latent_sample_r=latent_sample_r,
                                            latent_mean_r=latent_mean_r,
                                            latent_logvar_r=latent_logvar_r
                                            )

            self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                                self.args.policy_tau,
                                                use_proper_time_limits=self.args.use_proper_time_limits)

            policy_train_stats = self.policy.update(
                policy_storage=self.policy_storage,
                encoder_s=self.vae.encoder_s,
                encoder_r=self.vae.encoder_r,
                rlloss_through_encoder=self.args.rlloss_through_encoder,
                compute_vae_loss=self.vae.compute_vae_loss)
        else:
            policy_train_stats = 0, 0, 0, 0

            if self.iter_idx < self.args.pretrain_len:
                self.vae.compute_vae_loss(update=True)

        return policy_train_stats

    def log(self, run_stats, train_stats, start_time):

        if (self.iter_idx + 1) % self.args.eval_interval == 0:

            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            returns_per_episode = utl_eval.evaluate(args=self.args,
                                                    policy=self.policy,
                                                    ret_rms=ret_rms,
                                                    encoder_s=self.vae.encoder_s,
                                                    encoder_r=self.vae.encoder_r,
                                                    iter_idx=self.iter_idx,
                                                    tasks=self.train_tasks,
                                                    )

            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            for k in range(len(returns_avg)):
                self.logger.add('return_avg_per_iter/episode_{}'.format(k + 1), returns_avg[k], self.iter_idx)
                self.logger.add('return_avg_per_frame/episode_{}'.format(k + 1), returns_avg[k], self.frames)
                self.logger.add('return_std_per_iter/episode_{}'.format(k + 1), returns_std[k], self.iter_idx)
                self.logger.add('return_std_per_frame/episode_{}'.format(k + 1), returns_std[k], self.frames)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - start_time))}, "
                  f"\n Mean return (train): {returns_avg[-1].item()} \n"
                  )

        if (self.iter_idx + 1) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:

                torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy{idx_label}.pt"))
                torch.save(self.vae.encoder, os.path.join(save_path, f"encoder{idx_label}.pt"))
                if self.vae.state_decoder is not None:
                    torch.save(self.vae.state_decoder, os.path.join(save_path, f"state_decoder{idx_label}.pt"))
                if self.vae.reward_decoder is not None:
                    torch.save(self.vae.reward_decoder, os.path.join(save_path, f"reward_decoder{idx_label}.pt"))
                if self.vae.task_decoder is not None:
                    torch.save(self.vae.task_decoder, os.path.join(save_path, f"task_decoder{idx_label}.pt"))

                if self.args.norm_rew_for_policy:
                    rew_rms = self.envs.venv.ret_rms
                    utl.save_obj(rew_rms, save_path, f"env_rew_rms{idx_label}")

        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (train_stats is not None):

            self.logger.add('environment/state_max', self.policy_storage.prev_state.max(), self.iter_idx)
            self.logger.add('environment/state_min', self.policy_storage.prev_state.min(), self.iter_idx)

            self.logger.add('environment/rew_max', self.policy_storage.rewards_raw.max(), self.iter_idx)
            self.logger.add('environment/rew_min', self.policy_storage.rewards_raw.min(), self.iter_idx)

            self.logger.add('policy_losses/value_loss', train_stats[0], self.iter_idx)
            self.logger.add('policy_losses/action_loss', train_stats[1], self.iter_idx)
            self.logger.add('policy_losses/dist_entropy', train_stats[2], self.iter_idx)
            self.logger.add('policy_losses/sum', train_stats[3], self.iter_idx)

            self.logger.add('policy/action', run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)

            self.logger.add('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), self.iter_idx)
            self.logger.add('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), self.iter_idx)

            for [model, name] in [
                [self.policy.actor_critic, 'policy'],
                [self.vae.encoder, 'encoder'],
                [self.vae.reward_decoder, 'reward_decoder'],
                [self.vae.state_decoder, 'state_transition_decoder'],
                [self.vae.task_decoder, 'task_decoder']
            ]:
                if model is not None:
                    param_list = list(model.parameters())
                    param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                    self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
                    if param_list[0].grad is not None:
                        param_grad_mean = np.mean(
                            [param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
                        self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)
