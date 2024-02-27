import os, tqdm, random
from collections import defaultdict
import hydra
import wandb
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from dataset import LINGOSpaceDataset
from estimator.model import EstimatorModel
from estimator.loss import EstimatorLoss
from estimator.utils import get_max_point, check_spatial_relation

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@hydra.main(config_path='./cfg', config_name='train', version_base=None)
def main(cfg):
    # Dataset
    train_dataset = LINGOSpaceDataset(
        cfg.dataset.root, cfg.dataset.type, split='train', n=cfg.dataset.n,
         device=cfg.train.device, encoder=cfg.dataset.encoder)
    val_dataset = LINGOSpaceDataset(
        cfg.dataset.root, cfg.dataset.type, split='val', n=cfg.dataset.n,
         device=cfg.train.device, encoder=cfg.dataset.encoder)

    # batch_size must be 1
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model
    model = EstimatorModel(
        dim_in=cfg.model.dim_in,
        dim_h=cfg.model.dim_h,
        K=cfg.model.K,
        dim_edge=cfg.model.dim_in,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        act=cfg.model.act,
        dropout=cfg.model.dropout,
        attn_dropout=cfg.model.attn_dropout
    )
    
    # Loss
    criterion = EstimatorLoss(
        alpha=cfg.loss.alpha, beta=cfg.loss.beta, use_gt_w=cfg.loss.use_gt_w)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.scheduler.T_max, eta_min=cfg.scheduler.min_lr, last_epoch=-1)

    # Checkpoint directory
    ckpt_root = cfg.train.ckpt_root
    ckpt_path = os.path.join(ckpt_root, cfg.dataset.type)
    os.makedirs(ckpt_path, exist_ok=True)

    # Logger
    if cfg.wandb.use:
        wandb.init(project=cfg.wandb.project, name=cfg.dataset.type)

    model = model.to(cfg.train.device)
    best_score = 0
    pbar = tqdm.tqdm(range(cfg.train.max_epoch))
    for epoch in pbar:
        # Train
        model.train()
        train_loss = defaultdict(list)
        train_scores = []
        for batch in train_dataloader:
            batch = batch.to(cfg.train.device)
            parameters = []
            M = len(batch.pred_features)
            for i in range(M):
                batch_clone = batch.clone()

                batch_clone.pred_features = batch.pred_features[i:i+1]
                batch_clone.ref_features = batch.ref_features[i:i+1]
                batch_clone.ref_idx = batch.ref_idxs[i]
                                
                optimizer.zero_grad()
                batch_clone = model(batch_clone)
                loss_dict = criterion(batch_clone)
                loss = loss_dict['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                
                parameters.append(batch_clone.output.detach())
                batch.prev_state = batch_clone.prev_state.detach()
                
                train_loss['loss'].append(loss.item())
                train_loss['nll_loss'].append(loss_dict['nll_loss'].item())
                train_loss['weight_loss'].append(loss_dict['weight_loss'].item())
            
            # Get the maximum point of the spatial distribution
            max_point = get_max_point(parameters, batch_clone.pos) # pixel coordinate
            max_point = torch.tensor(max_point, device=batch_clone.pos.device) # pixel coordinate

            # Check spatial relation
            total_score = 0
            for i in range(M):
                score = check_spatial_relation(
                    max_point, batch.predicates[0][i], batch.ref_bboxes[i])
                total_score = total_score + score / M
            train_scores.append(total_score)

        # Eval
        with torch.no_grad():
            model.eval()
            val_loss = defaultdict(list)
            val_scores = []
            for batch in val_dataloader:
                batch = batch.to(cfg.train.device)
                parameters = []
                M = len(batch.pred_features)
                for i in range(M):
                    batch_clone = batch.clone()

                    batch_clone.pred_features = batch.pred_features[i:i+1]
                    batch_clone.ref_features = batch.ref_features[i:i+1]
                    batch_clone.ref_idx = batch.ref_idxs[i]

                    batch_clone = model(batch_clone)
                    loss_dict = criterion(batch_clone)
                    loss = loss_dict['loss']

                    parameters.append(batch_clone.output)
                    batch.prev_state = batch_clone.prev_state

                    val_loss['loss'].append(loss.item())
                    val_loss['nll_loss'].append(loss_dict['nll_loss'].item())
                    val_loss['weight_loss'].append(loss_dict['weight_loss'].item())
            
                # Get the maximum point of the spatial distribution
                max_point = get_max_point(parameters, batch_clone.pos) # pixel coordinate
                max_point = torch.tensor(max_point, device=batch_clone.pos.device) # pixel coordinate

                # Check spatial relation
                total_score = 0
                for i in range(M):
                    score = check_spatial_relation(
                        max_point, batch.predicates[0][i], batch.ref_bboxes[i])
                    total_score = total_score + score / M
                val_scores.append(total_score)


        if cfg.wandb.use:
            wandb.log({
                'train/loss': sum(train_loss['loss']) / len(train_loss['loss']),
                'train/nll_loss': sum(train_loss['nll_loss']) / len(train_loss['nll_loss']),
                'train/weight_loss': sum(train_loss['weight_loss']) / len(train_loss['weight_loss']),
                'train/score': sum(train_scores) / len(train_scores),
                'val/loss': sum(val_loss['loss']) / len(val_loss['loss']),
                'val/nll_loss': sum(val_loss['nll_loss']) / len(val_loss['nll_loss']),
                'val/weight_loss': sum(val_loss['weight_loss']) / len(val_loss['weight_loss']),
                'val/score': sum(val_scores) / len(val_scores),
            }, step=epoch)

        pbar.set_description(f'Epoch: {epoch}, Train: {sum(train_scores) / len(train_scores):.2f}, Val: {sum(val_scores) / len(val_scores):.2f}')
        
        # Save model per epoch
        # torch.save(model.state_dict(), f'{ckpt_path}/{epoch}.pt')

        # Save the best model
        if sum(val_scores) / len(val_scores) > best_score:
            best_score = sum(val_scores) / len(val_scores)
            torch.save(model.state_dict(), f'{ckpt_path}/best.pt')


    if cfg.wandb.use:
        wandb.finish()


if __name__ == '__main__':
    main()