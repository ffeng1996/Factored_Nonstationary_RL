import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNNEncoder(nn.Module):
    def __init__(self,
                 args,

                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,

                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 ):
        super(RNNEncoder, self).__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        if reward_size is not None and reward_embed_size is not None:
            self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)
            curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        else:
            self.reward_encoder = None
            curr_input_dim = action_embed_dim + state_embed_dim

        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        self.gru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, latent_dim)

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        return hidden_state

    def prior(self, batch_size, sample=True):

        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        h = hidden_state

        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True, detach_every=None):

        actions = utl.squash_action(actions, self.args)

        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))

        if hidden_state is not None:
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        if return_prior:
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(actions.shape[1])
            hidden_state = prior_hidden_state.clone()

        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        if self.reward_encoder is not None:
            rewards = rewards.reshape((-1, *rewards.shape[-2:]))
            hr = self.reward_encoder(rewards)
            h = torch.cat((ha, hs, hr), dim=2)
        else:
            h = torch.cat((ha, hs), dim=2)

        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))
        if detach_every is None:

            output, _ = self.gru(h, hidden_state)
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i * detach_every:i * detach_every + detach_every]
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                output.append(curr_output)

                hidden_state = hidden_state.detach()
            output = torch.cat(output, dim=0)
        gru_h = output.clone()

        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output = torch.cat((prior_hidden_state, output))

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]
        return latent_sample, latent_mean, latent_logvar, output
