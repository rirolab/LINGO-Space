import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, VonMises, MultivariateNormal

from estimator.utils import cartesian2polar


class EstimatorLoss(nn.Module):
    """
    Loss module for the Estimator model.

    Args:
        alpha: weight for negative log-likelihood loss
        beta: weight for cross-entropy loss
        use_gt_w: whether to use ground-truth weight for negative log-likelihood loss
    """
    def __init__(self, alpha, beta, use_gt_w=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_gt_w = use_gt_w
        self.Z = None
    
    def forward(self, batch):
        loss_dict = {}
        log_w = batch.output[..., 0]
        mean = batch.output[..., 1]
        var = batch.output[..., 2]
        loc = batch.output[..., 3]
        con = batch.output[..., 4]

        # Negative log-likelihood loss
        gaussian = Normal(mean, torch.sqrt(var + 1e-9))
        von_mises = VonMises(loc, con)

        center_polar = cartesian2polar(batch.y - batch.pos)
        gaussia_log_prob = gaussian.log_prob(center_polar[..., 0])
        von_mises_log_prob = von_mises.log_prob(center_polar[..., 1])

        ref_idx = torch.zeros_like(log_w)
        ref_idx[batch.ref_idx.item()] = 1.0

        log_prob = gaussia_log_prob + von_mises_log_prob + \
            torch.log(ref_idx + 1e-9) if self.use_gt_w else log_w
        nll_loss = -torch.logsumexp(log_prob, dim=-1)
        nll_loss = nll_loss.mean()
        loss_dict['nll_loss'] = nll_loss

        # Cross-entropy loss for weight
        weight_loss = F.nll_loss(log_w, batch.ref_idx, reduction='mean')
        loss_dict['weight_loss'] = weight_loss

        loss = self.alpha * nll_loss + self.beta * weight_loss
        loss_dict['loss'] = loss

        return loss_dict
    