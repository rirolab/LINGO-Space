python train.py \
    dataset.type=composite \
    dataset.n=200 \
    dataset.encoder=ViT-L/14 \
    model.dim_in=768 \
    model.dim_h=256 \
    model.dropout=0.2 \
    model.attn_dropout=0.2 \
    loss.alpha=0.002 \
    loss.beta=4.0