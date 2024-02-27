import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding propsed by the NeRF paper (https://arxiv.org/abs/2003.08934)
    Implementation based on https://github.com/jiupinjia/rocket-recycling/blob/main/policy.py#L22

    Args:
        in_dim: input dimension
        K: number of frequencies
        scale: scale of the positional encoding
        include_input: whether to include the input in the positional encoding
    
    Returns:
        h: positional encoding of the input
    """
    def __init__(self, dim_in, K=5, scale=1.0, include_input=True):
        super().__init__()
        self.K = K
        self.scale = scale
        self.include_input = include_input
        self.dim_out = dim_in * (K*2 + include_input)

    def forward(self, x):
        x = x * self.scale
        if self.K == 0:
            return x

        h = [x] if self.include_input else []
        for i in range(self.K):
            h.append(torch.sin(2**i * torch.pi * x))
            h.append(torch.cos(2**i * torch.pi * x))
        h = torch.cat(h, dim=-1) / self.scale
        return h


class EstimatorEmbedding(nn.Module):
    """
    Embedding layer for the Estimator model.

    Args:
        dim_in: input dimension
        dim_h: hidden dimension
        K: number of frequencies for positional encoding
    """
    def __init__(self, dim_in, dim_h, K):
        super().__init__()

        self.pos_encoding = PositionalEncoding(dim_in=2, K=K)
        self.dim_h_prime = 4*dim_h + self.pos_encoding.dim_out
        
        self.viz_norm = nn.LayerNorm(dim_in)
        self.viz_projection = nn.Linear(dim_in, dim_h, bias=False)

        self.ref_norm = nn.LayerNorm(dim_in)
        self.ref_projection = nn.Linear(dim_in, dim_h, bias=False)

        self.pred_norm = nn.LayerNorm(dim_in)
        self.pred_projection = nn.Linear(dim_in, dim_h, bias=False)

        self.state_projection = nn.Linear(self.dim_h_prime, dim_h, bias=False)

    
    def forward(self, batch):
        num_nodes = batch.x.size(0) # number of nodes in the graph

        # Get positional encoding
        coord = batch.pos # [num_nodes, 2]
        coord_features = self.pos_encoding(coord) # [num_nodes, pos_encoding.dim_out]
            
        # Get image features and project to dim_h
        viz_features = batch.viz_features # [num_nodes, dim_in]
        viz_features = self.viz_norm(viz_features) # [num_nodes, dim_in]
        viz_features = self.viz_projection(viz_features) # [num_nodes, dim_h]

        # Get ref features and project to dim_h
        ref_features = batch.ref_features # [1, dim_in]
        ref_features = self.ref_norm(ref_features) # [1, dim_in]
        ref_features = self.ref_projection(ref_features) # [1, dim_h]
        ref_features = ref_features.expand(num_nodes, -1) # [num_nodes, dim_h]
        
        # Get pred features and project to dim_h
        pred_features = batch.pred_features # [1, dim_in]
        pred_features = self.pred_norm(pred_features) # [1, dim_in]
        pred_features = self.pred_projection(pred_features) # [1, dim_h]
        pred_features = pred_features.expand(num_nodes, -1) # [num_nodes, dim_h]

        # Get prev output features and project to dim_h
        if hasattr(batch, 'prev_state'):
            state_features = batch.prev_state # [dim_h_prime, ]
        else:
            state_features = torch.zeros(
                self.dim_h_prime, device=viz_features.device) # [dim_h_prime, ]
        
        state_features = self.state_projection(state_features) # [dim_h_prime, ]
        state_features = state_features.expand(num_nodes, -1) # [num_nodes, dim_h_prime]
        
        x = torch.cat(
            [coord_features, viz_features, ref_features, pred_features, state_features],
            dim=-1,
            )
        batch.x = x
        
        return batch