"""
Physics tests — verify Cl and Cd analytical model outputs.

Validation targets
------------------
Cl at alpha=0 for symmetric airfoil must be 0.
Cl should increase with alpha in the linear regime (d Cl/d alpha ~ 2*pi/rad).
Cd should decrease with increasing Re.
Cd should increase with thickness.
No induced-drag term (2D model).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.thin_airfoil_theory import lift_coefficient, stall_angle
from src.drag_model import drag_coefficient, min_profile_drag
from data.calibrate_drag import production_drag
from src.airfoil_geometry import zero_lift_angle


def test_symmetric_cl_at_zero_alpha():
    cl = lift_coefficient(alpha_deg=0.0, m=0.0, p=0.4, t=0.12)
    assert abs(cl) < 0.01, f"Expected Cl=0 for symmetric at alpha=0, got {cl:.4f}"
    print(f"PASS  test_symmetric_cl_at_zero_alpha  (Cl = {cl:.5f})")


def test_cl_linearity_slope():
    # Externally selected thin-airfoil slope: 2*pi/rad.
    m, p, t = 0.02, 0.4, 0.12
    a0 = zero_lift_angle(m, p)
    cl1 = lift_coefficient(1.0, m, p, t, a0)
    cl2 = lift_coefficient(3.0, m, p, t, a0)
    dcl_da = (cl2 - cl1) / 2.0         # per degree
    expected = 2 * np.pi * np.pi / 180
    assert abs(dcl_da - expected) / expected < 0.02, \
        f"dCl/da = {dcl_da:.4f}, expected {expected:.4f}"
    print(f"PASS  test_cl_linearity_slope  (dCl/dalpha = {dcl_da:.4f} /deg)")


def test_cl_slope_independent_of_thickness():
    """Reject a global thickness multiplier unless external data support it."""
    m, p = 0.0, 0.4
    slope_thin = (lift_coefficient(2.0, m, p, 0.06) - lift_coefficient(0.0, m, p, 0.06)) / 2.0
    slope_thick = (lift_coefficient(2.0, m, p, 0.24) - lift_coefficient(0.0, m, p, 0.24)) / 2.0
    assert abs(slope_thick - slope_thin) < 1e-10, (
        f"Global thickness slope dependence reintroduced: {slope_thin:.8f} vs {slope_thick:.8f}"
    )
    print(f"PASS  test_cl_slope_independent_of_thickness  (slope = {slope_thin:.4f} /deg)")


def test_cl_positive_for_cambered_at_zero():
    # NACA 2412 should have Cl > 0 at alpha=0 (positive camber)
    cl = lift_coefficient(0.0, m=0.02, p=0.4, t=0.12)
    assert cl > 0.1, f"Expected Cl > 0 for cambered airfoil at alpha=0, got {cl:.4f}"
    print(f"PASS  test_cl_positive_for_cambered_at_zero  (Cl = {cl:.4f})")


def test_cl_decreases_post_stall():
    m, p, t = 0.02, 0.4, 0.12
    st = stall_angle(m, t)
    cl_prestall  = lift_coefficient(st - 1.0, m, p, t)
    cl_poststall = lift_coefficient(st + 5.0, m, p, t)
    assert cl_poststall < cl_prestall, \
        f"Post-stall Cl ({cl_poststall:.3f}) should be < pre-stall ({cl_prestall:.3f})"
    print(f"PASS  test_cl_decreases_post_stall")


def test_cd_min_range():
    # NACA 0012 @ Re=1e6: Cd_min should be 0.005 – 0.009
    cd = drag_coefficient(alpha_deg=0.0, t_frac=0.12, Re=1e6, Cl=0.0, alpha_stall=14.0)
    assert 0.004 < cd < 0.012, \
        f"NACA 0012 Cd_min @ Re=1e6 = {cd:.5f}, expected 0.004–0.012"
    print(f"PASS  test_cd_min_range  (Cd = {cd:.5f})")


def test_cd_decreases_with_Re():
    cd_lo = drag_coefficient(0.0, 0.12, 1e5, 0.0, 14.0)
    cd_hi = drag_coefficient(0.0, 0.12, 1e7, 0.0, 14.0)
    assert cd_hi < cd_lo, f"Cd should decrease with Re: {cd_lo:.5f} -> {cd_hi:.5f}"
    print(f"PASS  test_cd_decreases_with_Re  "
          f"(Re=1e5: {cd_lo:.5f}, Re=1e7: {cd_hi:.5f})")


def test_cd_increases_with_thickness():
    cd_thin  = drag_coefficient(0.0, 0.06, 1e6, 0.0, 14.0)
    cd_thick = drag_coefficient(0.0, 0.24, 1e6, 0.0, 14.0)
    assert cd_thick > cd_thin, \
        f"Thicker airfoil should have more drag: {cd_thin:.5f} -> {cd_thick:.5f}"
    print(f"PASS  test_cd_increases_with_thickness  "
          f"(t=6%: {cd_thin:.5f}, t=24%: {cd_thick:.5f})")


def test_cd_increases_past_stall():
    cd_pre  = drag_coefficient(12.0, 0.12, 1e6, 1.0, 13.0)
    cd_post = drag_coefficient(16.0, 0.12, 1e6, 1.0, 13.0)
    assert cd_post > cd_pre, \
        f"Cd should rise past stall: {cd_pre:.4f} -> {cd_post:.4f}"
    print(f"PASS  test_cd_increases_past_stall")


def test_calibration_drag_form_matches_production_model():
    """The calibrated parameter form must preserve every production-drag term."""
    alpha, thickness, re, cl, alpha_stall = 12.0, 0.12, 6e6, 1.0, 14.0
    params = np.array([1.0, 0.0040, 0.000020])
    expected = drag_coefficient(alpha, thickness, re, cl, alpha_stall)
    actual = production_drag(alpha, cl, re, params)
    assert np.isclose(actual, expected, rtol=0.0, atol=1e-12), (
        f"Calibration form {actual:.12f} differs from production {expected:.12f}"
    )
    print("PASS  test_calibration_drag_form_matches_production_model")


def test_no_induced_drag_term():
    # In 2D, Cl^2 term should only contribute via viscous increment (very small)
    # The 3D induced drag Cl^2/(pi*e*AR) must NOT be present.
    # Check: Cd should not scale as ~Cl^2 for small Cl in the pre-stall regime.
    cd0   = drag_coefficient(0.0, 0.12, 1e6, 0.0,  14.0)
    cd_cl = drag_coefficient(0.0, 0.12, 1e6, 1.0,  14.0)
    # Viscous increment at Cl=1: 0.004 * 1^2 * 1 * 1 = 0.004
    # 3D induced (AR=7, e=0.87): Cl^2/(pi*0.87*7) ~ 0.052  — much larger
    increment = cd_cl - cd0
    assert increment < 0.025, \
        (f"Cl^2 increment = {increment:.4f} is too large; "
         "check that 3D induced drag was not included")
    print(f"PASS  test_no_induced_drag_term  (Cl^2 increment = {increment:.5f})")


if __name__ == '__main__':
    test_symmetric_cl_at_zero_alpha()
    test_cl_linearity_slope()
    test_cl_slope_independent_of_thickness()
    test_cl_positive_for_cambered_at_zero()
    test_cl_decreases_post_stall()
    test_cd_min_range()
    test_cd_decreases_with_Re()
    test_cd_increases_with_thickness()
    test_cd_increases_past_stall()
    test_calibration_drag_form_matches_production_model()
    test_no_induced_drag_term()
    print("\nAll physics tests passed.")
