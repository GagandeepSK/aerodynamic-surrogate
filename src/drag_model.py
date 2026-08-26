"""
Semi-empirical 2D profile-drag model for NACA 4-digit sections.

The correlations are intentionally low-order. They capture Reynolds-number and
thickness effects, a moderate lift-dependent viscous increment, and the gradual
pressure-drag rise approaching separation. They are not a substitute for
viscous CFD or experimental polar data.
"""

import numpy as np


def min_profile_drag(t_frac: np.ndarray, Re: np.ndarray) -> np.ndarray:
    """Minimum profile drag, calibrated to the order of NACA 0012 data."""
    return (0.0065 * (t_frac / 0.12) ** 0.6
            * (1e6 / np.maximum(Re, 1e3)) ** 0.30)


def drag_coefficient(
    alpha_deg: float,
    t_frac: float,
    Re: float,
    Cl: float = 0.0,
    alpha_stall: float = 14.0,
) -> float:
    """Return total 2D drag coefficient at one operating point."""
    cd_profile = float(min_profile_drag(np.array([t_frac]), np.array([Re]))[0])

    # Smaller than the previous coefficient, which overstated drag throughout
    # the attached-flow polar when compared with the reference points.
    cd_visc = (0.0040 * Cl ** 2 * (t_frac / 0.12) ** 0.5
               * (1e6 / max(Re, 1e3)) ** 0.10)

    # A small angle-dependent term represents attached-flow pressure drag.
    cd_angle = 0.000020 * float(alpha_deg) ** 2

    # Pressure drag begins before nominal stall and rises smoothly.
    excess = max(0.0, alpha_deg - (alpha_stall - 3.0))
    cd_stall = 0.0040 * excess ** 1.30

    return float(max(0.002, cd_profile + cd_visc + cd_angle + cd_stall))


def drag_coefficient_batch(
    alpha_arr: np.ndarray,
    t_arr: np.ndarray,
    Re_arr: np.ndarray,
    Cl_arr: np.ndarray,
    alpha_stall_arr: np.ndarray,
) -> np.ndarray:
    """Vectorised drag coefficient, shape (N,)."""
    cd_profile = min_profile_drag(t_arr, Re_arr)
    cd_visc = (0.0040 * Cl_arr ** 2 * (t_arr / 0.12) ** 0.5
               * (1e6 / np.maximum(Re_arr, 1e3)) ** 0.10)
    cd_angle = 0.000020 * alpha_arr ** 2
    excess = np.maximum(0.0, alpha_arr - (alpha_stall_arr - 3.0))
    cd_stall = 0.0040 * excess ** 1.30
    return np.maximum(0.002, cd_profile + cd_visc + cd_angle + cd_stall)
