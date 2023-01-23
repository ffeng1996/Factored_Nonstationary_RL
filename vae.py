import warnings

import gym
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.decoder import StateTransitionDecoder, RewardDecoder, TaskDecoder
from models.encoder import RNNEncoder
from models.dynamics import Dynamics
from utils.helpers import get_task_dim, get_num_tasks
from utils.storage_vae import RolloutStorageVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAE:

    def __init__(self, args, logger, get_iter_idx):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx
        self.task_dim = get_task_dim(self.args) if self.args.decode_task else None
        self.num_tasks = get_num_tasks(self.args) if self.args.decode_task else None

        self.thetas2s = torch.rand((self.args.latent_dim, self.args.state_dim), requires_grad=True, device=device)
        self.thetaP2thetaC = torch.rand((self.args.latent_dim, self.args.latent_dim), requires_grad=True, device=device)
        self.sP2sC = torch.rand((self.args.state_dim, self.args.state_dim), requires_grad=True, device=device)
        self.aP2sC = torch.rand((self.args.action_dim, self.args.state_dim), requires_grad=True, device=device)
        self.aP2rC = torch.rand((self.args.action_dim, 1), requires_grad=True, device=device)
        self.sP2rC = torch.rand((self.args.state_dim, 1), requires_grad=True, device=device)

        self.encoder_s = self.initialise_encoder(rew=False)
        self.encoder_r = self.initialise_encoder(rew=True)

        self.state_decoder_rec, self.state_decoder_pred, \
        self.reward_decoder_rec, self.reward_decoder_pred, self.task_decoder = self.initialise_decoder()

        self.latent_dyn_state = self.initialise_dyn()
        self.latent_dyn_rew = self.initialise_dyn()

        self.rollout_storage = RolloutStorageVAE(num_processes=self.args.num_processes,
                                                 max_trajectory_len=self.args.max_trajectory_len,
                                                 zero_pad=True,
                                                 max_num_rollouts=self.args.size_vae_buffer,
                                                 state_dim=self.args.state_dim,
                                                 action_dim=self.args.action_dim,
                                                 vae_buffer_add_thresh=self.args.vae_buffer_add_thresh,
                                                 task_dim=self.task_dim
                                                 )

        self.decoder_params = []
        if not self.args.disable_decoder:
            if self.args.decode_reward:
                self.decoder_params.extend(self.reward_decoder_rec.parameters())
                self.decoder_params.extend(self.reward_decoder_pred.parameters())
            if self.args.decode_state:
                self.decoder_params.extend(self.state_decoder_rec.parameters())
                self.decoder_params.extend(self.state_decoder_pred.parameters())
            if self.args.decode_task:
                self.decoder_params.extend(self.task_decoder.parameters())
        self.optimiser_vae = torch.optim.Adam(
            [*self.encoder_r.parameters(), *self.encoder_s.parameters(), *self.decoder_params,
             self.thetas2s, self.thetaP2thetaC, self.sP2sC, self.aP2sC, self.aP2rC, self.sP2rC],
            lr=self.args.lr_vae)

    def initialise_encoder(self, rew=True):
        if rew:
            rew_size = 1
            rew_embed_size = self.args.reward_embedding_size
        else:
            rew_size = None
            rew_embed_size = None
        encoder = RNNEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=rew_size,
            reward_embed_size=rew_embed_size,
        ).to(device)
        return encoder

    def initialise_decoder(self):
        if self.args.disable_decoder:
            return None, None, None, None, None

        latent_dim = self.args.latent_dim

        if self.args.disable_stochasticity_in_latent:
            latent_dim *= 2

        if self.args.decode_state:
            state_decoder_rec = StateTransitionDecoder(
                args=self.args,
                layers=self.args.state_decoder_layers,
                latent_dim=latent_dim,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                pred_type=self.args.state_pred_type,
            ).to(device)
            state_decoder_pred = StateTransitionDecoder(
                args=self.args,
                layers=self.args.state_decoder_layers,
                latent_dim=latent_dim,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                pred_type=self.args.state_pred_type,
                clatent=False
            ).to(device)
        else:
            state_decoder_rec = None
            state_decoder_pred = None

        if self.args.decode_reward:
            reward_decoder_rec = RewardDecoder(
                args=self.args,
                layers=self.args.reward_decoder_layers,
                latent_dim=latent_dim,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                num_states=self.args.num_states,
                multi_head=self.args.multihead_for_reward,
                pred_type=self.args.rew_pred_type,
                input_prev_state=self.args.input_prev_state,
                input_action=self.args.input_action,
            ).to(device)
            reward_decoder_pred = RewardDecoder(
                args=self.args,
                layers=self.args.reward_decoder_layers,
                latent_dim=latent_dim,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                num_states=self.args.num_states,
                multi_head=self.args.multihead_for_reward,
                pred_type=self.args.rew_pred_type,
                input_prev_state=self.args.input_prev_state,
                input_action=self.args.input_action,
                clatent=False,
            ).to(device)
        else:
            reward_decoder_rec = None
            reward_decoder_pred = None

        if self.args.decode_task:
            assert self.task_dim != 0
            task_decoder = TaskDecoder(
                latent_dim=latent_dim,
                layers=self.args.task_decoder_layers,
                task_dim=self.task_dim,
                num_tasks=self.num_tasks,
                pred_type=self.args.task_pred_type,
            ).to(device)
        else:
            task_decoder = None

        return state_decoder_rec, state_decoder_pred, reward_decoder_rec, reward_decoder_pred, task_decoder

    def initialise_dyn(self):
        dyn = Dynamics(
            args=self.args,
            dim_latent=self.args.latent_dim,
            layers=self.args.dynamics_layers
        ).to(device)
        return dyn

    def compute_state_rec_loss(self, latent, prev_obs, obs, prev_action, return_rec=False):
        state_rec = self.state_decoder_rec(latent, prev_obs, prev_action, latent2s=self.thetas2s, s2s=self.sP2sC,
                                           a2s=self.aP2sC)
        loss_state_rec = (state_rec - obs).pow(2).mean(dim=-1)
        if return_rec:
            return loss_state_rec, state_rec
        else:
            return loss_state_rec

    def compute_state_pred_loss(self, latent, curr_obs, next_obs, curr_action, return_predictions=False):
        state_pred = self.state_decoder_pred(latent, curr_obs, curr_action)

        if self.args.state_pred_type == 'deterministic':
            loss_state_pred = (state_pred - next_obs).pow(2).mean(dim=-1)
        elif self.args.state_pred_type == 'gaussian':
            state_pred_mean = state_pred[:, :state_pred.shape[1] // 2]
            state_pred_std = torch.exp(0.5 * state_pred[:, state_pred.shape[1] // 2:])
            m = torch.distributions.normal.Normal(state_pred_mean, state_pred_std)
            loss_state = -m.log_prob(next_obs).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_state_pred, state_pred
        else:
            return loss_state_pred

    def compute_rew_rec_loss(self, latent, obs, action, reward, return_rec=False):
        rew_rec = self.reward_decoder_rec(latent, obs, action, s2r=self.sP2rC, a2r=self.aP2rC)
        if self.args.rew_pred_type == 'bernoulli':
            rew_rec = torch.sigmoid(rew_rec)
            rew_target = (reward == 1).float()
            loss_rew_rec = F.binary_cross_entropy(rew_rec, rew_target, reduction='none').mean(dim=-1)
        elif self.args.rew_pred_type == 'deterministic':
            loss_rew_rec = (rew_rec - reward).pow(2).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_rec:
            return loss_rew_rec, rew_rec
        else:
            return loss_rew_rec

    def compute_rew_pred_loss(self, latent, obs, action, reward, return_predictions=False):
        rew_pred = self.reward_decoder_pred(latent, obs, action.float())
        if self.args.rew_pred_type == 'bernoulli':
            rew_pred = torch.sigmoid(rew_pred)
            rew_target = (reward == 1).float()
            loss_rew_pred = F.binary_cross_entropy(rew_pred, rew_target, reduction='none').mean(dim=-1)
        elif self.args.rew_pred_type == 'deterministic':
            loss_rew_pred = (rew_pred - reward).pow(2).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_rew_pred, rew_pred
        else:
            return loss_rew_pred

    def compute_task_reconstruction_loss(self, latent, task, return_predictions=False):

        task_pred = self.task_decoder(latent)

        if self.args.task_pred_type == 'task_id':
            env = gym.make(self.args.env_name)
            task_target = env.task_to_id(task).to(device)

            task_target = task_target.expand(task_pred.shape[:-1]).reshape(-1)
            loss_task = F.cross_entropy(task_pred.view(-1, task_pred.shape[-1]),
                                        task_target, reduction='none').view(task_pred.shape[:-1])
        elif self.args.task_pred_type == 'task_description':
            loss_task = (task_pred - task).pow(2).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_task, task_pred
        else:
            return loss_task

    def compute_kl_latent_loss(self, latent_mean_traj, latent_logvar_traj, latent_mean_pred, latent_logvar_pred,
                               elbo_indices):
        all_means = torch.cat((latent_mean_traj, latent_mean_pred))
        all_logvars = torch.cat((latent_logvar_traj, latent_logvar_pred))
        mu = all_means[1:]
        m = all_means[:-1]
        logE = all_logvars[1:]
        logS = all_logvars[:-1]
        kl_divergences = 0.5 * (
                    torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - latent_mean_traj.shape[-1] + torch.sum(
                1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

        if elbo_indices is not None:
            batchsize = kl_divergences.shape[-1]
            task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)
            kl_divergences = kl_divergences[elbo_indices, task_indices].reshape(
                (self.args.vae_subsample_elbos, batchsize))

        return kl_divergences

    def compute_kl_prior_loss(self, latent_mean, latent_logvar, elbo_indices):

        if self.args.kl_to_gauss_prior:
            kl_divergences = (- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1))
        else:
            gauss_dim = latent_mean.shape[-1]

            all_means = torch.cat((torch.zeros(1, *latent_mean.shape[1:]).to(device), latent_mean))
            all_logvars = torch.cat((torch.zeros(1, *latent_logvar.shape[1:]).to(device), latent_logvar))

            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
                1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

        if elbo_indices is not None:
            batchsize = kl_divergences.shape[-1]
            task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)
            kl_divergences = kl_divergences[elbo_indices, task_indices].reshape(
                (self.args.vae_subsample_elbos, batchsize))

        return kl_divergences

    def compute_loss(self, latent_mean_s, latent_logvar_s, latent_mean_r, latent_logvar_r, vae_prev_obs, vae_next_obs,
                     vae_actions,
                     vae_rewards, vae_tasks, trajectory_lens):

        num_unique_trajectory_lens = len(np.unique(trajectory_lens))

        assert (num_unique_trajectory_lens == 1) or (self.args.vae_subsample_elbos and self.args.vae_subsample_decodes)
        assert not self.args.decode_only_past

        max_traj_len = np.max(trajectory_lens)
        latent_mean_s = latent_mean_s[:max_traj_len + 1]
        latent_logvar_s = latent_logvar_s[:max_traj_len + 1]
        latent_mean_r = latent_mean_r[:max_traj_len + 1]
        latent_logvar_r = latent_logvar_r[:max_traj_len + 1]
        vae_prev_obs = vae_prev_obs[:max_traj_len]
        vae_next_obs = vae_next_obs[:max_traj_len]
        vae_actions = vae_actions[:max_traj_len]
        vae_rewards = vae_rewards[:max_traj_len]

        if not self.args.disable_stochasticity_in_latent:
            latent_samples_s = self.encoder_s._sample_gaussian(latent_mean_s, latent_logvar_s)
            latent_samples_r = self.encoder_r._sample_gaussian(latent_mean_r, latent_logvar_r)
        else:
            latent_samples_s = torch.cat((latent_mean_s, latent_logvar_s), dim=-1)
            latent_samples_r = torch.cat((latent_mean_r, latent_logvar_r), dim=-1)

        num_elbos = latent_samples_s.shape[0]
        num_decodes = vae_prev_obs.shape[0]
        batchsize = latent_samples_s.shape[1]

        if self.args.vae_subsample_elbos is not None:

            if num_unique_trajectory_lens == 1:
                elbo_indices = torch.LongTensor(self.args.vae_subsample_elbos * batchsize).random_(0, num_elbos)
            else:

                elbo_indices = np.concatenate([np.random.choice(range(0, t + 1), self.args.vae_subsample_elbos,
                                                                replace=self.args.vae_subsample_elbos > (t + 1)) for t
                                               in trajectory_lens])
                if max_traj_len < self.args.vae_subsample_elbos:
                    warnings.warn('The required number of ELBOs is larger than the shortest trajectory, '
                                  'so there will be duplicates in your batch.'
                                  'To avoid this use --split_batches_by_elbo or --split_batches_by_task.')
            task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)
            latent_samples_s = latent_samples_s[elbo_indices, task_indices, :].reshape(
                (self.args.vae_subsample_elbos, batchsize, -1))
            latent_samples_r = latent_samples_r[elbo_indices, task_indices, :].reshape(
                (self.args.vae_subsample_elbos, batchsize, -1))
            num_elbos = latent_samples_s.shape[0]
        else:
            elbo_indices = None

        dec_prev_obs = vae_prev_obs.unsqueeze(0).expand((num_elbos, *vae_prev_obs.shape))
        dec_next_obs = vae_next_obs.unsqueeze(0).expand((num_elbos, *vae_next_obs.shape))
        dec_actions = vae_actions.unsqueeze(0).expand((num_elbos, *vae_actions.shape))
        dec_rewards = vae_rewards.unsqueeze(0).expand((num_elbos, *vae_rewards.shape))

        if self.args.vae_subsample_decodes is not None:

            indices0 = torch.arange(num_elbos).repeat(self.args.vae_subsample_decodes * batchsize)
            if num_unique_trajectory_lens == 1:
                indices1 = torch.LongTensor(num_elbos * self.args.vae_subsample_decodes * batchsize).random_(0,
                                                                                                             num_decodes)
            else:
                indices1 = np.concatenate([np.random.choice(range(0, t), num_elbos * self.args.vae_subsample_decodes,
                                                            replace=True) for t in trajectory_lens])
            indices2 = torch.arange(batchsize).repeat(num_elbos * self.args.vae_subsample_decodes)
            dec_prev_obs = dec_prev_obs[indices0, indices1, indices2, :].reshape(
                (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            dec_next_obs = dec_next_obs[indices0, indices1, indices2, :].reshape(
                (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            dec_actions = dec_actions[indices0, indices1, indices2, :].reshape(
                (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            dec_rewards = dec_rewards[indices0, indices1, indices2, :].reshape(
                (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            num_decodes = dec_prev_obs.shape[1]

        dec_embedding_s = latent_samples_s.unsqueeze(0).expand((num_decodes, *latent_samples_s.shape)).transpose(1, 0)
        dec_embedding_r = latent_samples_r.unsqueeze(0).expand((num_decodes, *latent_samples_r.shape)).transpose(1, 0)

        if self.args.decode_reward:

            rew_reconstruction_loss = self.compute_rew_rec_loss(dec_embedding_r, dec_prev_obs, dec_next_obs,
                                                                dec_actions, dec_rewards)

            if self.args.vae_avg_elbo_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)

            if self.args.vae_avg_reconstruction_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)

            rew_reconstruction_loss = rew_reconstruction_loss.mean()
        else:
            rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = self.compute_state_rec_loss(dec_embedding_s, dec_prev_obs,
                                                                    dec_next_obs, dec_actions)

            if self.args.vae_avg_elbo_terms:
                state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)
            else:
                state_reconstruction_loss = state_reconstruction_loss.sum(dim=0)

            if self.args.vae_avg_reconstruction_terms:
                state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)
            else:
                state_reconstruction_loss = state_reconstruction_loss.sum(dim=0)

            state_reconstruction_loss = state_reconstruction_loss.mean()
        else:
            state_reconstruction_loss = 0

        task_reconstruction_loss = 0

        if not self.args.disable_kl_term:

            kl_loss_s = self.compute_kl_prior_loss(latent_mean_s, latent_logvar_s, elbo_indices)
            kl_loss_r = self.compute_kl_prior_loss(latent_mean_r, latent_logvar_r, elbo_indices)

            if self.args.vae_avg_elbo_terms:
                kl_loss = kl_loss_s.mean(dim=0) + kl_loss_r.mean(dim=0)
            else:
                kl_loss = kl_loss_s.sum(dim=0) + kl_loss_r.sum(dim=0)

            kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss

    def compute_loss_split_batches_by_elbo(self, latent_mean_s, latent_logvar_s, latent_mean_r, latent_logvar_r,
                                           vae_prev_obs, vae_next_obs, vae_actions,
                                           vae_rewards, vae_tasks, trajectory_lens):

        rew_reconstruction_loss = []
        state_reconstruction_loss = []
        task_reconstruction_loss = []

        assert len(np.unique(trajectory_lens)) == 1
        n_horizon = np.unique(trajectory_lens)[0]
        n_elbos = latent_mean_s.shape[0]

        for idx_elbo in range(n_elbos):

            curr_means_s = latent_mean_s[idx_elbo]
            curr_logvars_s = latent_logvar_s[idx_elbo]

            curr_means_r = latent_mean_r[idx_elbo]
            curr_logvars_r = latent_logvar_r[idx_elbo]

            if not self.args.disable_stochasticity_in_latent:
                curr_samples_s = self.encoder_s._sample_gaussian(curr_means_s, curr_logvars_s)
                curr_samples_r = self.encoder_r._sample_gaussian(curr_means_r, curr_logvars_r)
            else:
                curr_samples_s = torch.cat((latent_mean_s, latent_logvar_s))
                curr_samples_r = torch.cat((latent_mean_r, latent_logvar_r))

            if not self.args.decode_only_past:

                dec_embedding_s = curr_samples_s.unsqueeze(0).expand((n_horizon, *curr_samples_s.shape))
                dec_embedding_r = curr_samples_r.unsqueeze(0).expand((n_horizon, *curr_samples_r.shape))
                dec_prev_obs = vae_prev_obs
                dec_next_obs = vae_next_obs
                dec_actions = vae_actions
                dec_rewards = vae_rewards




            else:

                if self.args.decode_only_past:
                    dec_from = 0
                    dec_until = idx_elbo
                else:
                    dec_from = 0
                    dec_until = n_horizon

                if dec_from == dec_until:
                    continue

                dec_embedding_s = curr_samples_s.unsqueeze(0).expand(dec_until - dec_from, *curr_samples_s.shape)
                dec_embedding_r = curr_samples_r.unsqueeze(0).expand(dec_until - dec_from, *curr_samples_r.shape)

                dec_prev_obs = vae_prev_obs[dec_from:dec_until]
                dec_next_obs = vae_next_obs[dec_from:dec_until]
                dec_actions = vae_actions[dec_from:dec_until]
                dec_rewards = vae_rewards[dec_from:dec_until]

            if self.args.decode_reward:
                rrc = self.compute_rew_rec_loss(dec_embedding_r, dec_prev_obs, dec_next_obs, dec_actions,
                                                dec_rewards)

                rrc = rrc.sum(dim=0).mean()
                rew_reconstruction_loss.append(rrc)

            if self.args.decode_state:
                src = self.compute_state_rec_loss(dec_embedding_s, dec_prev_obs, dec_next_obs, dec_actions)

                src = src.sum(dim=0).mean()
                state_reconstruction_loss.append(src)

        if self.args.decode_reward:
            rew_reconstruction_loss = torch.stack(rew_reconstruction_loss)
            rew_reconstruction_loss = rew_reconstruction_loss.sum()
        else:
            rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = torch.stack(state_reconstruction_loss)
            state_reconstruction_loss = state_reconstruction_loss.sum()
        else:
            state_reconstruction_loss = 0

        if self.args.decode_task:
            task_reconstruction_loss = torch.stack(task_reconstruction_loss)
            task_reconstruction_loss = task_reconstruction_loss.sum()
        else:
            task_reconstruction_loss = 0

        if not self.args.disable_kl_term:

            kl_loss_s = self.compute_kl_prior_loss(latent_mean_s, latent_logvar_s, None)
            kl_loss_r = self.compute_kl_prior_loss(latent_mean_r, latent_logvar_r, None)

            kl_loss = kl_loss_s.sum(dim=0).mean() + kl_loss_r.sum(dim=0).mean()
        else:
            kl_loss = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss

    def compute_vae_loss(self, update=False, pretrain_index=None, offline=False, offline_data=None):
        if offline and offline_data is not None:
            prev_state = offline_data['prev_state']
            curr_state = offline_data['curr_state']
            next_state = offline_data['next_state']

            prev_action = offline_data['prev_action']
            curr_action = offline_data['curr_action']
            next_action = offline_data['next_action']

            curr_rew = offline_data['curr_rew']
            next_rew = offline_data['next_rew']

            latent_sample_state_curr, \
            latent_mean_state_curr, latent_logvar_state_curr, _ = self.encoder_s(actions=curr_action,
                                                                                 states=curr_state,
                                                                                 rewards=None,
                                                                                 hidden_state=None,
                                                                                 return_prior=False,
                                                                                 detach_every=self.args.tbptt_stepsize if hasattr(
                                                                                     self.args,
                                                                                     'tbptt_stepsize') else None,
                                                                                 )
            _, latent_mean_state_next_traj, latent_logvar_state_next_traj, _ = self.encoder_s(actions=next_action,
                                                                                              states=next_state,
                                                                                              rewards=None,
                                                                                              hidden_state=None,
                                                                                              return_prior=False,
                                                                                              sample=False,
                                                                                              detach_every=None)
            latent_mean_state_next_pred, latent_logvar_state_next_pred = self.latent_dyn_state(latent_mean_state_curr,
                                                                                               latent_logvar_state_curr,
                                                                                               latent2latent=self.thetaP2thetaC)
            curr_state_rec_loss = self.compute_state_rec_loss(latent_sample_state_curr, prev_state, curr_state,
                                                              prev_action)
            next_state_pred_loss = self.compute_state_pred_loss(latent_sample_state_curr, curr_state, next_state,
                                                                curr_action)
            kl_latent_state_loss = self.compute_kl_latent_loss(latent_mean_state_next_traj,
                                                               latent_logvar_state_next_traj,
                                                               latent_mean_state_next_pred,
                                                               latent_logvar_state_next_pred, elbo_indices=None)
            kl_pri_state_loss = self.compute_kl_prior_loss(latent_mean_state_curr, latent_logvar_state_curr,
                                                           elbo_indices=None)
            latent_sample_reward_curr, \
            latent_mean_rew_curr, latent_logvar_rew_curr, _ = self.encoder_r(actions=curr_action,
                                                                             states=curr_state,
                                                                             rewards=curr_rew,
                                                                             hidden_state=None,
                                                                             return_prior=False,
                                                                             detach_every=self.args.tbptt_stepsize if hasattr(
                                                                                 self.args, 'tbptt_stepsize') else None,
                                                                             )
            _, latent_mean_rew_next_traj, latent_logvar_rew_next_traj, _ = self.encoder_r(actions=next_action,
                                                                                          states=next_state,
                                                                                          rewards=next_rew,
                                                                                          hidden_state=None,
                                                                                          return_prior=False,
                                                                                          detach_every=None)
            latent_mean_rew_next_pred, latent_logvar_rew_next_pred = self.latent_dyn_rew(latent_mean_rew_curr,
                                                                                         latent_logvar_rew_curr,
                                                                                         latent2latent=self.thetaP2thetaC)
            curr_rew_rec_loss = self.compute_rew_rec_loss(latent_sample_reward_curr, curr_state, curr_action, curr_rew)
            next_rew_pred_loss = self.compute_rew_pred_loss(latent_sample_reward_curr, next_state, next_action,
                                                            next_rew)
            kl_latent_rew_loss = self.compute_kl_latent_loss(latent_mean_rew_next_traj, latent_logvar_rew_next_traj,
                                                             latent_mean_rew_next_pred, latent_logvar_rew_next_pred,
                                                             elbo_indices=None)
            kl_pri_rew_loss = self.compute_kl_prior_loss(latent_mean_rew_curr, latent_logvar_rew_curr,
                                                         elbo_indices=None)
        else:
            if not self.rollout_storage.ready_for_update():
                return 0

            if self.args.disable_decoder and self.args.disable_kl_term:
                return 0

            vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
            trajectory_lens = self.rollout_storage.get_batch(batchsize=self.args.vae_batch_num_trajs)

            _, latent_mean_s, latent_logvar_s, _ = self.encoder_s(actions=vae_actions,
                                                                  states=vae_next_obs,
                                                                  rewards=vae_rewards,
                                                                  hidden_state=None,
                                                                  return_prior=True,
                                                                  detach_every=self.args.tbptt_stepsize if hasattr(
                                                                      self.args, 'tbptt_stepsize') else None,
                                                                  )
            _, latent_mean_r, latent_logvar_r, _ = self.encoder_r(actions=vae_actions,
                                                                  states=vae_next_obs,
                                                                  rewards=vae_rewards,
                                                                  hidden_state=None,
                                                                  return_prior=True,
                                                                  detach_every=self.args.tbptt_stepsize if hasattr(
                                                                      self.args, 'tbptt_stepsize') else None,
                                                                  )

            if self.args.split_batches_by_elbo:
                losses = self.compute_loss_split_batches_by_elbo(latent_mean_s, latent_logvar_s, latent_mean_r,
                                                                 latent_logvar_r, vae_prev_obs, vae_next_obs,
                                                                 vae_actions, vae_rewards, vae_tasks,
                                                                 trajectory_lens)
            else:
                losses = self.compute_loss(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                                           vae_rewards, vae_tasks, trajectory_lens)
            rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss = losses

        loss = (self.args.rew_rec_loss_coeff * curr_rew_rec_loss +
                self.args.rew_pred_loss_coeff * next_rew_pred_loss +
                self.args.state_rec_loss_coeff * curr_state_rec_loss +
                self.args.state_pred_loss_coeff * next_state_pred_loss +
                self.args.kl_pri_state_weight * kl_pri_state_loss +
                self.args.kl_dyn_state_weight * kl_latent_state_loss +
                self.args.kl_pri_rew_weight * kl_pri_rew_loss +
                self.args.kl_dyn_rew_weight * kl_latent_rew_loss
                ).mean()

        if not self.args.disable_kl_term:
            assert kl_pri_rew_loss.requires_grad
            assert kl_pri_state_loss.requires_grad
            assert kl_latent_rew_loss.requires_grad
            assert kl_latent_state_loss.requires_grad
        if self.args.decode_reward:
            assert curr_rew_rec_loss.requires_grad
            assert next_rew_pred_loss.requires_grad
        if self.args.decode_state:
            assert curr_state_rec_loss.requires_grad
            assert next_state_pred_loss.requires_grad

        elbo_loss = loss.mean()

        if update:
            self.optimiser_vae.zero_grad()
            elbo_loss.backward()

            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder_s.parameters(), self.args.encoder_max_grad_norm)
                nn.utils.clip_grad_norm_(self.encoder_r.parameters(), self.args.encoder_max_grad_norm)
            if self.args.decoder_max_grad_norm is not None:
                if self.args.decode_reward:
                    nn.utils.clip_grad_norm_(self.reward_decoder_rec.parameters(), self.args.decoder_max_grad_norm)
                    nn.utils.clip_grad_norm_(self.reward_decoder_pred.parameters(), self.args.decoder_max_grad_norm)
                if self.args.decode_state:
                    nn.utils.clip_grad_norm_(self.state_decoder_rec.parameters(), self.args.decoder_max_grad_norm)
                    nn.utils.clip_grad_norm_(self.state_decoder_pred.parameters(), self.args.decoder_max_grad_norm)

            self.optimiser_vae.step()
        if offline:
            self.log(elbo_loss, curr_rew_rec_loss, next_rew_pred_loss, curr_state_rec_loss, next_state_pred_loss,
                     kl_pri_rew_loss, kl_latent_rew_loss, kl_pri_state_loss, kl_latent_state_loss)
        else:
            self.log(elbo_loss, curr_rew_rec_loss, next_rew_pred_loss, curr_state_rec_loss, next_state_pred_loss,
                     kl_pri_rew_loss, kl_latent_rew_loss, kl_pri_state_loss, kl_latent_state_loss, pretrain_index)
        return elbo_loss

    def log(self, elbo_loss, rew_rec_loss, rew_pred_loss, state_rec_loss, state_pred_loss, kl_pri_state_loss,
            kl_latent_rew_loss, kl_pri_rew_loss, kl_latent_state_loss,
            pretrain_index=None):

        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = - self.args.pretrain_len * self.args.num_vae_updates_per_pretrain + pretrain_index

        if curr_iter_idx % self.args.log_interval == 0:
            if self.args.decode_reward:
                self.logger.add('vae_losses/reward_reconstr_err', rew_rec_loss.mean(), curr_iter_idx)
                self.logger.add('vae_losses/reward_predstr_err', rew_pred_loss.mean(), curr_iter_idx)
            if self.args.decode_state:
                self.logger.add('vae_losses/state_reconstr_err', state_rec_loss.mean(), curr_iter_idx)
                self.logger.add('vae_losses/state_predstr_err', state_pred_loss.mean(), curr_iter_idx)
            if not self.args.disable_kl_term:
                self.logger.add('vae_losses/kl_reward_prior', kl_pri_rew_loss.mean(), curr_iter_idx)
                self.logger.add('vae_losses/kl_reward_latent', kl_latent_rew_loss.mean(), curr_iter_idx)
                self.logger.add('vae_losses/kl_state_prior', kl_pri_state_loss.mean(), curr_iter_idx)
                self.logger.add('vae_losses/kl_state_latent', kl_latent_state_loss.mean(), curr_iter_idx)
            self.logger.add('vae_losses/sum', elbo_loss, curr_iter_idx)
