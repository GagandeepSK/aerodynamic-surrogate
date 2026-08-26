"""
2D lift coefficient for NACA 4-digit sections.

The attached-flow region retains the thin-airfoil slope. A smooth transition
near the estimated stall angle then captures the finite maximum lift and the
post-stall decline. This is a compact engineering correlation, not a substitute
for viscous CFD or wind-tunnel data.
"""

import numpy as np
from src.airfoil_geometry import zero_lift_angle, zero_lift_angle_batch


def stall_angle(m: float, t: float) -> float:
    """Estimate positive stall onset in degrees for a NACA 4-digit section."""
    alpha_stall = 14.0 + 1.5 * (t / 0.12 - 1.0) + 0.5 * (m / 0.02)
    return float(np.clip(alpha_stall, 11.5, 17.0))


def stall_angle_batch(m_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """Vectorised positive stall estimate, shape (N,)."""
    return np.clip(14.0 + 1.5 * (t_arr / 0.12 - 1.0)
                   + 0.5 * (m_arr / 0.02), 11.5, 17.0)


def lift_slope(_t):
    """Externally selected thin-airfoil lift slope, independent of thickness.

    A global Glauert multiplier was tested as a rejected experiment. It
    worsened NASA TM 4074 attached-flow lift MAE for NACA 0012, so this
    source model deliberately retains the classical 2π/rad slope.
    """
    return 2.0 * np.pi


def _cl_max(m, t):
    """Geometry-dependent maximum section lift correlation."""
    return 1.25 + 8.0 * m + 0.12 * (t / 0.12 - 1.0)


def _attached_lift(cl_linear, alpha_deg, alpha_stall, m, t):
    """Keep linear theory, blending to Cl_max only in the stall approach."""
    cl_peak = _cl_max(m, t)
    blend_start = alpha_stall - 3.0
    blend = np.clip((alpha_deg - blend_start) / 3.0, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    return (1.0 - blend) * cl_linear + blend * cl_peak


def lift_coefficient(
    alpha_deg: float,
    m: float,
    p: float,
    t: float = 0.12,
    alpha_0_rad: float = None,
) -> float:
    """Return 2D section lift with attached-flow and post-stall behaviour."""
    if alpha_0_rad is None:
        alpha_0_rad = zero_lift_angle(m, p)

    alpha_stall = stall_angle(m, t)
    cl_linear = lift_slope(t) * (np.deg2rad(alpha_deg) - alpha_0_rad)
    cl_attached = _attached_lift(cl_linear, alpha_deg, alpha_stall, m, t)

    excess = max(0.0, alpha_deg - alpha_stall)
    decay = np.exp(-0.10 * excess)
    cl_post = _cl_max(m, t) * decay + 0.30 * (1.0 - decay)
    return float(cl_attached if alpha_deg <= alpha_stall else cl_post)


def lift_coefficient_batch(
    alpha_arr: np.ndarray,
    m_arr: np.ndarray,
    p_arr: np.ndarray,
    t_arr: np.ndarray,
    alpha0_rad_arr: np.ndarray = None,
) -> np.ndarray:
    """Vectorised section lift coefficient, shape (N,)."""
    if alpha0_rad_arr is None:
        alpha0_rad_arr = zero_lift_angle_batch(m_arr, p_arr)

    alpha_stall = stall_angle_batch(m_arr, t_arr)
    cl_linear = lift_slope(t_arr) * (np.radians(alpha_arr) - alpha0_rad_arr)
    cl_attached = _attached_lift(cl_linear, alpha_arr, alpha_stall, m_arr, t_arr)

    excess = np.maximum(0.0, alpha_arr - alpha_stall)
    decay = np.exp(-0.10 * excess)
    cl_post = _cl_max(m_arr, t_arr) * decay + 0.30 * (1.0 - decay)
    return np.where(alpha_arr <= alpha_stall, cl_attached, cl_post)
