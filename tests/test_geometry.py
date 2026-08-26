"""
Geometry tests — verify NACA coordinate generation and zero-lift angle.

Known values
------------
NACA 2412 (m=0.02, p=0.4, t=0.12)
  alpha_0 ~ -2.08 deg  (NACA TR 460; Abbott & von Doenhoff Table 4-I)

NACA 0012 (m=0, t=0.12)
  alpha_0 = 0.00 deg  (symmetric)
  max thickness at x/c = 0.30 approximately
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.airfoil_geometry import (
    naca_thickness, naca_camber, zero_lift_angle, naca_coords
)


def test_symmetric_zero_lift():
    a0 = zero_lift_angle(m=0.0, p=0.4)
    assert a0 == 0.0, f"Expected 0, got {a0}"
    print("PASS  test_symmetric_zero_lift")


def test_naca2412_zero_lift():
    a0_rad = zero_lift_angle(m=0.02, p=0.4)
    a0_deg = np.degrees(a0_rad)
    # NACA report value: approximately -2.08 deg
    assert -2.5 < a0_deg < -1.5, f"Expected ~ -2.08 deg, got {a0_deg:.3f}"
    print(f"PASS  test_naca2412_zero_lift  (alpha_0 = {a0_deg:.3f} deg)")


def test_thickness_at_nose():
    # At x=0, thickness should be 0 for the standard formula
    yt = naca_thickness(np.array([0.0]), t=0.12)
    assert yt[0] == 0.0, f"Expected yt(0)=0, got {yt[0]}"
    print("PASS  test_thickness_at_nose")


def test_thickness_max_location():
    # NACA 4-digit thickness peaks near x/c ~ 0.30
    x = np.linspace(0.0, 1.0, 1000)
    yt = naca_thickness(x, t=0.12)
    x_max = x[np.argmax(yt)]
    assert 0.25 < x_max < 0.35, f"Max thickness at x={x_max:.3f}, expected ~0.30"
    print(f"PASS  test_thickness_max_location  (x_max = {x_max:.3f})")


def test_naca_coords_closed_at_te():
    # For symmetric airfoil, upper and lower should meet at trailing edge
    xu, yu, xl, yl = naca_coords(0.0, 0.0, 0.12, n=100)
    gap = abs(yu[-1] - yl[-1])
    assert gap < 0.005, f"TE gap = {gap:.5f}, should be < 0.005"
    print(f"PASS  test_naca_coords_closed_at_te  (TE gap = {gap:.5f})")


def test_naca_coords_le_at_zero():
    # Leading edge at x=0
    xu, yu, xl, yl = naca_coords(0.02, 0.4, 0.12, n=100)
    assert abs(xu[0]) < 0.01, f"LE x = {xu[0]:.5f}"
    print("PASS  test_naca_coords_le_at_zero")


if __name__ == '__main__':
    test_symmetric_zero_lift()
    test_naca2412_zero_lift()
    test_thickness_at_nose()
    test_thickness_max_location()
    test_naca_coords_closed_at_te()
    test_naca_coords_le_at_zero()
    print("\nAll geometry tests passed.")
