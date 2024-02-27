import torch.nn as nn

from estimator.embedding import EstimatorEmbedding
from estimator.layer import EstimatorLayer
from estimator.head import EstimatorHead


class EstimatorModel(nn.Module):
    """
    Spatial-distribution estimator model.

    Args:
        dim_in: input dimension
        dim_h: hidden dimension
        K: number of frequencies for positional encoding
        dim_edge: edge dimension
        num_layers: number of layers
        num_heads: number of attention heads
        act: activation function
        dropout: dropout rate
        attn_dropout: attention dropout rate
    """
    def __init__(
        self,
        dim_in=768,
        dim_h=256,
        K=4,
        dim_edge=768,
        num_layers=2,
        num_heads=2,
        act='leaky_relu',
        dropout=0.2,
        attn_dropout=0.2,
        **kwargs
        ):
        super().__init__()

        self.embedding = EstimatorEmbedding(
            dim_in=dim_in,
            dim_h=dim_h,
            K=K,
        )
        dim_h_prime = self.embedding.dim_h_prime

        layers = []
        for _ in range(num_layers):
            layers.append(EstimatorLayer(
                dim_h=dim_h_prime,
                dim_edge=dim_edge,
                num_heads=num_heads,
                act=act,
                dropout=dropout,
                attn_dropout=attn_dropout,
                ))
        self.layers = nn.Sequential(*layers)

        self.head = EstimatorHead(dim_h=dim_h_prime, act=act)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch