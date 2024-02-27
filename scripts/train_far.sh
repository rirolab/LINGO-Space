python train.py \
    dataset.type=far-seen-colors \
    dataset.n=100 \
    dataset.encoder=ViT-B/32 \
    model.dim_in=512 \
    model.dim_h=64 \
    model.dropout=0.1 \
    model.attn_dropout=0.1 \
    loss.alpha=0.02 \
    loss.beta=10.0