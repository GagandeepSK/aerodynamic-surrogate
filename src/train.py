"""
Train the NACA aerodynamic surrogate model.

Usage
-----
From the project root on EC2:

    python -m src.train
    python -m src.train --epochs 300 --lr 3e-4 --batch 4096

After training, the best checkpoint is saved to model/surrogate_model.pt.
Run src/export_weights.py next to produce model_weights.json and embed it
into the dashboard.

Checkpoint contents
-------------------
    model_state_dict   trained weights
    input_mean / std   z-score normalisation for the 5 input features
    output_mean / std  z-score normalisation for the 2 outputs (Cl, Cd)
    history            per-epoch train / val loss
    test_r2/mae/rmse   final test-set metrics
    feature_cols       list of feature column names
    target_cols        list of target column names
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import AeroSurrogate
from src.physics_loss import physics_losses


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train NACA aerodynamic surrogate')
    p.add_argument('--data',       default='data/dataset.csv')
    p.add_argument('--output',     default='model/surrogate_model.pt')
    p.add_argument('--epochs',     type=int,   default=200)
    p.add_argument('--batch',      type=int,   default=2048)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--lambda-lin', type=float, default=0.05, dest='lam_lin',
                   help='Weight for the Cl-linearity physics loss')
    p.add_argument('--lambda-zl',  type=float, default=0.05, dest='lam_zl',
                   help='Weight for the zero-lift physics loss')
    p.add_argument('--lambda-sym', type=float, default=0.05, dest='lam_sym',
                   help='Weight for the symmetry physics loss')
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(y_pred: np.ndarray, y_true: np.ndarray):
    """MAE, RMSE, R² per output column."""
    mae  = np.abs(y_pred - y_true).mean(0)
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean(0))
    ss_res = ((y_pred - y_true) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    r2   = 1.0 - ss_res / (ss_tot + 1e-12)
    return mae, rmse, r2


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    print(f"  {len(df):,} samples")

    feature_cols = ['m_frac', 'p_frac', 't_frac', 'alpha_deg', 'log10_Re']
    target_cols  = ['Cl', 'Cd']

    source_model = {
        'id': 'external_reference_selected_2pi',
        'lift_curve_slope': '2*pi*(alpha-alpha_0)',
        'decision': (
            'Global Glauert thickness multiplier rejected after it worsened '
            'NASA TM 4074 NACA 0012 attached-flow lift MAE.'
        ),
    }

    X_np  = df[feature_cols].values.astype(np.float32)
    Y_np  = df[target_cols].values.astype(np.float32)
    A0_np = df['alpha0_deg'].values.astype(np.float32)

    # ── 70 / 15 / 15 split ────────────────────────────────────────────────────
    N     = len(X_np)
    idx   = np.random.permutation(N)
    n_tr  = int(0.70 * N)
    n_val = int(0.15 * N)
    i_tr, i_val, i_te = (idx[:n_tr],
                          idx[n_tr:n_tr + n_val],
                          idx[n_tr + n_val:])

    X_tr, Y_tr, A0_tr = X_np[i_tr],  Y_np[i_tr],  A0_np[i_tr]
    X_val, Y_val      = X_np[i_val], Y_np[i_val]
    X_te,  Y_te       = X_np[i_te],  Y_np[i_te]

    # ── Normalisation (fit on train split only) ────────────────────────────────
    in_mean  = X_tr.mean(0);   in_std  = X_tr.std(0)
    out_mean = Y_tr.mean(0);   out_std = Y_tr.std(0)

    def norm_X(X): return (X - in_mean)  / (in_std  + 1e-8)
    def norm_Y(Y): return (Y - out_mean) / (out_std + 1e-8)
    def denorm_Y(Y_n): return Y_n * out_std + out_mean

    # Torch tensors
    t  = lambda a: torch.from_numpy(a)
    Xt  = t(norm_X(X_tr));   Yt  = t(norm_Y(Y_tr));  A0t  = t(A0_tr)
    Xtr = t(X_tr)            # raw (unscaled) for physics loss
    Xv  = t(norm_X(X_val));  Yv  = t(norm_Y(Y_val))
    Xte = t(norm_X(X_te))

    im  = torch.from_numpy(in_mean);   is_ = torch.from_numpy(in_std)
    om  = torch.from_numpy(out_mean);  os_ = torch.from_numpy(out_std)

    ds  = TensorDataset(Xt, Yt, Xtr, A0t)
    dl  = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = AeroSurrogate()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters : {n_params:,}")
    print(f"  Train / val / test: {len(i_tr):,} / {len(i_val):,} / {len(i_te):,}")
    print(f"  Epochs {args.epochs}  Batch {args.batch}  LR {args.lr}")
    print(f"  lambda: lin={args.lam_lin}  zl={args.lam_zl}  sym={args.lam_sym}\n")

    best_val = math.inf
    history  = {'train_data': [], 'train_phys': [], 'val': []}
    t0       = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        ep_data = 0.0;  ep_phys = 0.0

        for Xb, Yb, Xb_raw, A0b in dl:
            optimizer.zero_grad()

            l_data = criterion(model(Xb), Yb)

            phys   = physics_losses(model, Xb_raw, A0b, im, is_, om, os_)
            l_phys = (args.lam_lin * phys['linearity']
                    + args.lam_zl  * phys['zero_lift']
                    + args.lam_sym * phys['symmetry'])

            (l_data + l_phys).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_data += l_data.item();  ep_phys += l_phys.item()

        ep_data /= len(dl);  ep_phys /= len(dl)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), Yv).item()

        scheduler.step(val_loss)

        history['train_data'].append(ep_data)
        history['train_phys'].append(ep_phys)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_mean':   in_mean.tolist(),
                'input_std':    in_std.tolist(),
                'output_mean':  out_mean.tolist(),
                'output_std':   out_std.tolist(),
                'history':      history,
                'feature_cols': feature_cols,
                'target_cols':  target_cols,
                'source_model': source_model,
            }, args.output)

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [{epoch:4d}/{args.epochs}]  "
                  f"data={ep_data:.5f}  phys={ep_phys:.5f}  "
                  f"val={val_loss:.5f}  best={best_val:.5f}  "
                  f"lr={lr_now:.1e}  "
                  f"[{time.time()-t0:.0f}s]")

    # ── Test evaluation ────────────────────────────────────────────────────────
    ckpt = torch.load(args.output, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        Y_te_pred = denorm_Y(model(Xte).numpy())

    mae, rmse, r2 = evaluate(Y_te_pred, Y_te)

    print(f"\n{'='*56}")
    print(f"  Test set ({len(i_te):,} samples)")
    print(f"{'='*56}")
    for i, col in enumerate(target_cols):
        print(f"  {col:4s}  R²={r2[i]:.4f}  MAE={mae[i]:.5f}  RMSE={rmse[i]:.5f}")

    gate_pass = all(r2[i] >= 0.95 for i in range(len(target_cols)))
    print(f"\n  Gate (R² >= 0.95): {'PASS' if gate_pass else 'FAIL — consider more epochs or data'}")
    print(f"{'='*56}")

    # Update checkpoint with test metrics
    ckpt.update({'test_r2': r2.tolist(), 'test_mae': mae.tolist(),
                 'test_rmse': rmse.tolist()})
    torch.save(ckpt, args.output)
    print(f"\n  Saved: {args.output}")

    return model, history, r2


if __name__ == '__main__':
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_model(args)
