import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dynamics(nn.Module):
    def __init__(self, args, dim_latent, layers):
        super(Dynamics, self).__init__()

        self.args = args
        self.fc_layers_mean = nn.ModuleList([])
        self.fc_layers_var = nn.ModuleList([])
        curr_input_dim = dim_latent
        for i in range(len(layers)):
            self.fc_layers_mean.append(nn.Linear(curr_input_dim, layers[i]))
            self.fc_layers_var.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]
        self.fc_out_mean = nn.Linear(curr_input_dim, dim_latent)
        self.fc_out_var = nn.Linear(curr_input_dim, dim_latent)

    def forward(self, latent_mean, latent_var, latent2latent=None):
        if latent2latent is not None:
            latent_mean = torch.matmul(latent_mean, latent2latent)
            latent_var = torch.matmul(latent_var, latent2latent)

        for i in range(len(self.fc_layers_mean)):
            latent_mean = F.relu(self.fc_layers_mean[i](latent_mean))
        for i in range(len(self.fc_layers_var)):
            latent_var = F.relu(self.fc_layers_var[i](latent_var))
        next_mean = self.fc_out_mean(latent_mean)
        next_var = self.fc_out_var(latent_var)

        return next_mean, next_var
