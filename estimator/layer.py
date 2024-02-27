import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import to_dense_batch


class EstimatorLayer(nn.Module):
    """ 
    Layer module for the Estimator model.
    Implementation based on the GPS layer (https://arxiv.org/abs/2205.12454)
    
    Args:
        dim_h: hidden dimension
        dim_edge: edge dimension
        num_heads: number of attention heads
        act: activation function
        dropout: dropout rate
        attn_dropout: attention dropout rate    
    """
    def __init__(self, dim_h, dim_edge, num_heads, act, dropout, attn_dropout):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.set_activation(act)

        # GINE as local message-passing model
        self.local_model = gnn.GINEConv(
            nn.Sequential(
                gnn.Linear(dim_h, dim_h),
                self.activation(),
                gnn.Linear(dim_h, dim_h),
            ),
            edge_dim=dim_edge
        )

        # Transformer as global attention transformer-style model
        self.self_attn = nn.MultiheadAttention(
            dim_h, num_heads, dropout=attn_dropout, batch_first=True
        )

        # Layer norm
        self.norm1_local = gnn.norm.LayerNorm(dim_h)
        self.norm1_attn = gnn.norm.LayerNorm(dim_h)

        # Dropout
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed forward network
        self.ff_linear1 = nn.Linear(dim_h, dim_h*2)
        self.ff_linear2 = nn.Linear(dim_h*2, dim_h)
        self.act_fn_ff = self.activation()
        self.norm2 = gnn.norm.LayerNorm(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

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
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
        h_local = self.dropout_local(h_local)
        h_local = h_in1 + h_local # Residual connection.
        h_local = self.norm1_local(h_local, batch.batch)
        h_out_list.append(h_local)

        # Multi-head attention.
        h_dense, mask = to_dense_batch(h, batch.batch)
        h_attn = self._sa_block(h_dense, None, ~mask)[mask]
        assert mask.all() # batch size must be 1
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn # Residual connection.
        h_attn = self.norm1_attn(h_attn, batch.batch)
        h_out_list.append(h_attn)

        # Combine local and global outputs.
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        h = self.norm2(h, batch.batch)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """ Self-attention block. """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def _ff_block(self, x):
        """ Feed Forward block. """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))