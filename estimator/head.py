import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimatorHead(nn.Module):
    """
    Head module for the Estimator model.

    Args:
        dim_h: hidden dimension
        act: activation function
    """
    def __init__(self, dim_h, act='leaky_relu'):
        super().__init__()
        self.set_activation(act)

        self.w_mlp = nn.Sequential(
            nn.Linear(dim_h, dim_h // 2),
            self.activation(),
            nn.Linear(dim_h // 2, 1)
        )
        self.mean_mlp = nn.Sequential(
            nn.Linear(dim_h, dim_h // 2),
            self.activation(),
            nn.Linear(dim_h // 2, 1)
        )
        self.var_mlp = nn.Sequential(
            nn.Linear(dim_h, dim_h // 2),
            self.activation(),
            nn.Linear(dim_h // 2, 1)
        )
        self.loc_mlp = nn.Sequential(
            nn.Linear(dim_h, dim_h // 2),
            self.activation(),
            nn.Linear(dim_h // 2, 2)
        )
        self.con_mlp = nn.Sequential(
            nn.Linear(dim_h, dim_h // 2),
            self.activation(),
            nn.Linear(dim_h // 2, 1)
        )
    
    def set_activation(self, act):
        if act == 'relu':
            self.activation = nn.ReLU
        elif act == 'leaky_relu':
            self.activation = nn.LeakyReLU
        elif act == 'elu':
            self.activation = nn.ELU
        elif act == 'gelu':
            self.activation = nn.GELU
        else:
            raise ValueError(f'Invalid activation function: {act}')

    def forward(self, batch):
        x = batch.x
        log_w = torch.log_softmax(self.w_mlp(x), dim=0)
        mean = F.softplus(self.mean_mlp(x))
        var = F.softplus(self.var_mlp(x))
        loc = self.loc_mlp(x)
        loc = torch.atan2(loc[..., 1], loc[..., 0]).unsqueeze(-1)
        con = torch.reciprocal(F.softplus(self.con_mlp(x)) + 1e-9)

        output_param = torch.cat([log_w, mean, var, loc, con], dim=-1) # [num_nodes, 5]
        batch.output = output_param

        state = torch.max(x, dim=0)[0] # max pooling
        batch.prev_state = state
        
        return batch
    