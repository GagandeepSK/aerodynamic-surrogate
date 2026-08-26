"""
Generate the training dataset for the NACA aerodynamic surrogate.

Sampling strategy
-----------------
80,000 samples are drawn from the 5-dimensional parameter space using
stratified random sampling (integer m, p, t combined with continuous alpha
and log-uniform Re).  This gives denser coverage than a coarse uniform grid
while keeping the dataset at a manageable size for training on EC2 t3.micro.

Columns in output CSV
---------------------
m_frac      max camber fraction            (0.00 – 0.09)
p_frac      max-camber position fraction   (0.10 – 0.90)
t_frac      max thickness fraction         (0.06 – 0.24)
alpha_deg   angle of attack                (-10 – 20 deg)
log10_Re    log_10(Re)                     (5.0 – 7.0)
alpha0_deg  zero-lift angle                (degrees, precomputed for physics loss)
Cl          lift coefficient
Cd          drag coefficient
LD          lift-to-drag ratio

Usage
-----
    python -m data.generate_dataset
    python -m data.generate_dataset --n 100000 --out data/dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.airfoil_geometry import zero_lift_angle_batch
from src.thin_airfoil_theory import stall_angle_batch, lift_coefficient_batch
from src.drag_model import drag_coefficient_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n',    type=int, default=80_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out',  default='data/dataset.csv')
    return p.parse_args()


def generate(n: int = 80_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Parameter sampling ─────────────────────────────────────────────────
    # m, p, t: integers representing NACA digit values
    m_int = rng.integers(0, 10, n)    # 0–9
    p_int = rng.integers(1, 10, n)
    p_int = np.where(m_int == 0, 0, p_int)  # symmetric NACA sections use p=0
    t_int = rng.integers(6, 25, n)    # 6–24 %

    alpha = rng.uniform(-10.0, 20.0, n).astype(np.float32)
    log_Re = rng.uniform(5.0, 7.0, n).astype(np.float32)
    Re     = 10.0 ** log_Re

    m_frac = (m_int / 100.0).astype(np.float32)
    p_frac = (p_int / 10.0).astype(np.float32)
    t_frac = (t_int / 100.0).astype(np.float32)

    # ── Physics ────────────────────────────────────────────────────────────
    print("  Computing zero-lift angles …", flush=True)
    alpha0_rad = zero_lift_angle_batch(m_frac, p_frac, n_theta=100)
    alpha0_deg = np.degrees(alpha0_rad).astype(np.float32)

    print("  Computing lift coefficients …", flush=True)
    Cl = lift_coefficient_batch(alpha, m_frac, p_frac, t_frac, alpha0_rad)

    print("  Computing drag coefficients …", flush=True)
    alpha_stall = stall_angle_batch(m_frac, t_frac)
    Cd = drag_coefficient_batch(alpha, t_frac, Re, Cl, alpha_stall)

    LD = np.where(Cd > 0, Cl / Cd, 0.0)

    df = pd.DataFrame({
        'm_frac':    np.round(m_frac,   6),
        'p_frac':    np.round(p_frac,   6),
        't_frac':    np.round(t_frac,   6),
        'alpha_deg': np.round(alpha,    4),
        'log10_Re':  np.round(log_Re,   6),
        'alpha0_deg':np.round(alpha0_deg, 6),
        'Cl':        np.round(Cl,        6),
        'Cd':        np.round(Cd,        6),
        'LD':        np.round(LD,        4),
    })
    return df


def main():
    args = parse_args()
    print(f"Generating {args.n:,} samples (seed={args.seed}) …")
    df = generate(args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows  ->  {args.out}")
    print(f"\nSummary:")
    print(df[['Cl', 'Cd', 'LD']].describe().round(4).to_string())


if __name__ == '__main__':
    main()
