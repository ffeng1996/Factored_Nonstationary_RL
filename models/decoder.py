import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StateTransitionDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 pred_type='deterministic',
                 clatent=True,
                 ):
        super(StateTransitionDecoder, self).__init__()

        self.args = args

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)

        if clatent:
            self.action_encoder = utl.FeatureExtractor(state_dim, action_embed_dim, F.relu)
            curr_input_dim = state_dim + state_embed_dim + action_embed_dim
        else:
            self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
            curr_input_dim = latent_dim + state_embed_dim + action_embed_dim

        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2 * state_dim)
        else:
            self.fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, actions, latent2s=None, s2s=None, a2s=None):

        actions = utl.squash_action(actions, self.args)
        if a2s is not None:
            actions = torch.matmul(actions, a2s)
        ha = self.action_encoder(actions)
        if s2s is not None:
            state = torch.matmul(state, s2s)
        hs = self.state_encoder(state)
        if latent2s is not None:
            latent_state = torch.matmul(latent_state, latent2s)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        return self.fc_out(h)


class RewardDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=True,
                 input_action=True,
                 clatent=True
                 ):
        super(RewardDecoder, self).__init__()

        self.args = args

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        if clatent:
            self.state_encoder = utl.FeatureExtractor(1, state_embed_dim, F.relu)
            self.action_encoder = utl.FeatureExtractor(1, action_embed_dim, F.relu)
        else:
            self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
            self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = latent_dim + state_embed_dim
        if input_action:
            curr_input_dim += action_embed_dim

        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2)
        else:
            self.fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_rew, state, actions, s2r=None, a2r=None):

        if actions is not None:
            actions = utl.squash_action(actions, self.args)
            if a2r is not None:
                actions = torch.matmul(actions, a2r)
        if self.multi_head:
            h = latent_rew.clone()
        else:
            if s2r is not None:
                state = torch.matmul(state, s2r)
            hns = self.state_encoder(state)
            h = torch.cat((latent_rew, hns), dim=-1)
            if self.input_action:
                ha = self.action_encoder(actions)
                h = torch.cat((h, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class TaskDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 pred_type,
                 task_dim,
                 num_tasks,
                 ):
        super(TaskDecoder, self).__init__()

        self.pred_type = pred_type

        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim if pred_type == 'task_description' else num_tasks
        self.fc_out = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent_state):

        h = latent_state

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)
