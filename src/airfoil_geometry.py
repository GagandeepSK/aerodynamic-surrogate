"""
NACA 4-digit airfoil geometry.

Conventions
-----------
m  max camber as fraction of chord (0.00 – 0.09)
p  chordwise position of max camber as fraction of chord (0.10 – 0.90)
t  max thickness as fraction of chord (0.06 – 0.24)

The zero-lift angle alpha_0 is computed by numerically integrating the
camber-slope integral from thin airfoil theory.  This is more accurate than
the crude linear approximation used in the preview dashboard.
"""

import numpy as np


# ── Thickness distribution ──────────────────────────────────────────────────

def naca_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """
    NACA 4-digit symmetric thickness half-distribution.
    Standard 5-coefficient series (open trailing edge).

    Parameters
    ----------
    x : chord-wise positions in [0, 1]
    t : max thickness as fraction of chord (e.g. 0.12 for NACA xx12)

    Returns
    -------
    yt(x) : half-thickness at each x
    """
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )


# ── Camber line ──────────────────────────────────────────────────────────────

def naca_camber(x: np.ndarray, m: float, p: float):
    """
    NACA 4-digit camber line and its slope.

    Parameters
    ----------
    x : chord-wise positions in [0, 1]
    m : max camber fraction (e.g. 0.02 for NACA 2x12)
    p : max-camber position fraction (e.g. 0.40 for NACA x4xx)

    Returns
    -------
    yc      : camber line ordinates
    dyc_dx  : camber slope
    """
    yc     = np.zeros_like(x, dtype=float)
    dyc_dx = np.zeros_like(x, dtype=float)

    if m == 0.0:
        return yc, dyc_dx

    front = x <= p
    back  = ~front

    if np.any(front) and p > 0:
        xf = x[front]
        yc[front]     = (m / p ** 2) * (2 * p * xf - xf ** 2)
        dyc_dx[front] = (2 * m / p ** 2) * (p - xf)

    if np.any(back) and (1.0 - p) > 0:
        xb = x[back]
        yc[back]     = (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * xb - xb ** 2)
        dyc_dx[back] = (2 * m / (1.0 - p) ** 2) * (p - xb)

    return yc, dyc_dx


# ── Zero-lift angle ──────────────────────────────────────────────────────────

def zero_lift_angle(m: float, p: float, n_theta: int = 200) -> float:
    """
    Zero-lift angle alpha_0 in radians via thin airfoil theory.

    alpha_0 = -(1/pi) * integral_0^pi  (dyc/dx)(cos(theta) - 1)  d_theta
    where  x = (1 - cos(theta)) / 2

    For symmetric airfoils (m = 0) returns exactly 0.

    Parameters
    ----------
    m, p      : NACA camber parameters (fractions)
    n_theta   : number of integration intervals (200 is more than sufficient)
    """
    if m == 0.0:
        return 0.0

    theta = np.linspace(0.0, np.pi, n_theta + 1)
    x     = 0.5 * (1.0 - np.cos(theta))

    _, dyc_dx = naca_camber(x, m, p)

    integrand = dyc_dx * (np.cos(theta) - 1.0)
    alpha_0   = -(1.0 / np.pi) * np.trapezoid(integrand, theta)
    return float(alpha_0)


def zero_lift_angle_batch(m_arr: np.ndarray, p_arr: np.ndarray,
                          n_theta: int = 100) -> np.ndarray:
    """
    Vectorised zero-lift angle for arrays of (m, p) pairs.
    Returns alpha_0 in radians, shape (N,).

    Uses broadcasting to compute all N integrals simultaneously — much
    faster than calling zero_lift_angle() in a Python loop.
    """
    theta = np.linspace(0.0, np.pi, n_theta + 1)   # (n_theta+1,)
    x     = 0.5 * (1.0 - np.cos(theta))             # (n_theta+1,)

    m = m_arr[:, None]   # (N, 1)
    p = p_arr[:, None]   # (N, 1)
    x_b = x[None, :]     # (1, n_theta+1)

    # Front region: x <= p
    p2  = p ** 2 + 1e-12
    dyc_front = (2.0 * m / p2) * (p - x_b)

    # Rear region: x > p
    q2  = (1.0 - p) ** 2 + 1e-12
    dyc_rear  = (2.0 * m / q2) * (p - x_b)

    front_mask = x_b <= p
    dyc_dx = np.where(front_mask, dyc_front, dyc_rear)

    # Symmetric airfoils: m = 0 → slope = 0 everywhere
    dyc_dx = np.where(m == 0.0, 0.0, dyc_dx)

    cos_minus_1 = (np.cos(theta) - 1.0)[None, :]    # (1, n_theta+1)
    integrand   = dyc_dx * cos_minus_1

    alpha0 = -(1.0 / np.pi) * np.trapezoid(integrand, theta, axis=1)
    return alpha0   # (N,) radians


# ── Surface coordinates ──────────────────────────────────────────────────────

def naca_coords(m: float, p: float, t: float, n: int = 100):
    """
    Upper and lower surface coordinates for a NACA 4-digit airfoil.

    Cosine spacing gives higher point density near the leading and
    trailing edges, where the geometry changes most rapidly.

    Parameters
    ----------
    m, p, t : NACA parameters (fractions)
    n       : number of points per surface

    Returns
    -------
    x_upper, y_upper, x_lower, y_lower : each shape (n+1,)
    """
    beta = np.linspace(0.0, np.pi, n + 1)
    x    = 0.5 * (1.0 - np.cos(beta))   # cosine spacing

    yt         = naca_thickness(x, t)
    yc, dyc_dx = naca_camber(x, m, p)
    theta_c    = np.arctan(dyc_dx)

    x_upper = x  - yt * np.sin(theta_c)
    y_upper = yc + yt * np.cos(theta_c)
    x_lower = x  + yt * np.sin(theta_c)
    y_lower = yc - yt * np.cos(theta_c)

    return x_upper, y_upper, x_lower, y_lower
