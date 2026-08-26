"""
Physics-informed loss terms for the aerodynamic surrogate.

Three constraints from thin airfoil theory are enforced on every training batch:

L_linearity  Cl should follow  2*pi*(alpha - alpha_0)  for |alpha| < 5 deg.
             This is the well-established linear regime of thin airfoil theory.

L_zero_lift  At alpha = alpha_0, the lift coefficient should be exactly 0.
             alpha_0 is the zero-lift angle precomputed for each sample.

L_symmetry   For symmetric airfoils (m = 0), Cl must be 0 at alpha = 0.

Implementation note
-------------------
Collocation points are taken directly from the training batch rather than from
a separate collocation grid.  At batch_size=2048 this is sufficient for the
constraint to be well-represented during training.
"""

import math
import torch


def physics_losses(
    model,
    X_raw: torch.Tensor,
    alpha0_raw: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
) -> dict:
    """
    Compute physics constraint losses for one training batch.

    Parameters
    ----------
    model        : AeroSurrogate (must be in training mode for autograd)
    X_raw        : (N, 5) unscaled inputs
                   [m_frac, p_frac, t_frac, alpha_deg, log10_Re]
    alpha0_raw   : (N,) zero-lift angles in degrees (stored in dataset)
    input_mean   : (5,) tensor — training-set feature means
    input_std    : (5,) tensor — training-set feature stds
    output_mean  : (2,) tensor — training-set target means  [Cl, Cd]
    output_std   : (2,) tensor — training-set target stds   [Cl, Cd]

    Returns
    -------
    dict with keys 'linearity', 'zero_lift', 'symmetry' — each a scalar tensor
    """
    PI = math.pi
    EPS = 1e-8

    def norm(X: torch.Tensor) -> torch.Tensor:
        return (X - input_mean) / (input_std + EPS)

    def predict_cl(X_n: torch.Tensor) -> torch.Tensor:
        y = model(X_n)
        return y[:, 0] * output_std[0] + output_mean[0]

    losses = {}
    device = X_raw.device
    zero   = torch.zeros(1, device=device)[0]

    # ── L_linearity ──────────────────────────────────────────────────────────
    alpha_deg = X_raw[:, 3]
    lin_mask  = alpha_deg.abs() < 5.0
    if lin_mask.sum() > 1:
        X_lin     = X_raw[lin_mask]
        cl_pred   = predict_cl(norm(X_lin))
        alpha_rad = X_lin[:, 3] * (PI / 180.0)
        a0_rad    = alpha0_raw[lin_mask] * (PI / 180.0)
        cl_theory = 2.0 * PI * (alpha_rad - a0_rad)
        losses['linearity'] = torch.mean((cl_pred - cl_theory) ** 2)
    else:
        losses['linearity'] = zero

    # ── L_zero_lift ───────────────────────────────────────────────────────────
    # Replace each sample's alpha with its own alpha_0; Cl must vanish there.
    X_zl      = X_raw.clone()
    X_zl[:, 3] = alpha0_raw
    cl_zl     = predict_cl(norm(X_zl))
    losses['zero_lift'] = torch.mean(cl_zl ** 2)

    # ── L_symmetry ────────────────────────────────────────────────────────────
    sym_mask = X_raw[:, 0] < 0.001   # m_frac = 0  →  symmetric airfoil
    if sym_mask.sum() > 1:
        X_sym      = X_raw[sym_mask].clone()
        X_sym[:, 3] = 0.0             # set alpha = 0
        cl_sym     = predict_cl(norm(X_sym))
        losses['symmetry'] = torch.mean(cl_sym ** 2)
    else:
        losses['symmetry'] = zero

    return losses
