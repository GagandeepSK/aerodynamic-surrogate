"""
Bounded attached-flow calibration of the production semi-empirical drag model.

Calibration case:
  NACA 0012, Re=6e6, fixed-transition / grit condition
  Ladson, NASA TM 4074 (1988), Table VII.

Only alpha <= 14.02 degrees is fitted. The two separated-flow points are
explicitly excluded. This is a calibration result, not held-out validation.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REFERENCE = Path("data/validation/naca0012_ladson_tm4074_re6e6.json")
OUTPUT = Path("analysis/naca0012_tm4074_drag_calibration.json")


def production_drag(alpha, cl, re, params):
    """
    Parameterised production drag form from src/drag_model.py.

    Parameters:
      cd0_scale: multiplies the Reynolds/thickness baseline
      cl2_coeff: non-negative lift-dependent viscous increment
      alpha2_coeff: attached-flow angle-dependent pressure increment
    """
    cd0_scale, cl2_coeff, alpha2_coeff = params

    thickness = 0.12
    alpha_stall = 14.0

    cd_profile = (
        cd0_scale
        * 0.0065
        * (thickness / 0.12) ** 0.60
        * (1e6 / re) ** 0.30
    )

    cd_visc = (
        cl2_coeff
        * cl**2
        * (thickness / 0.12) ** 0.50
        * (1e6 / re) ** 0.10
    )
    cd_angle = alpha2_coeff * alpha**2

    # Matches the production stall onset and rise in src/drag_model.py.
    excess = np.maximum(0.0, alpha - (alpha_stall - 3.0))
    cd_stall = 0.0040 * excess**1.30

    return np.maximum(0.002, cd_profile + cd_visc + cd_angle + cd_stall)


def main():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))

    attached = [p for p in ref["points"] if p["alpha"] <= 14.02]
    alpha = np.array([p["alpha"] for p in attached], dtype=float)
    cl = np.array([p["Cl"] for p in attached], dtype=float)
    cd_ref = np.array([p["Cd"] for p in attached], dtype=float)
    re = float(ref["reynolds"])

    initial = np.array([1.0, 0.0010, 0.000020])
    lower = np.array([0.5, 0.0, 0.000001])
    upper = np.array([4.0, 0.0300, 0.000500])

    result = least_squares(
        lambda params: production_drag(alpha, cl, re, params) - cd_ref,
        x0=initial,
        bounds=(lower, upper),
        method="trf",
    )

    fitted_cd = production_drag(alpha, cl, re, result.x)
    error = fitted_cd - cd_ref

    payload = {
        "purpose": (
            "Attached-flow drag calibration. Not held-out validation and "
            "not intended for separated flow."
        ),
        "source": ref["source"],
        "reference_url": ref["url"],
        "condition": {
            "airfoil": "NACA 0012",
            "reynolds": re,
            "surface_condition": ref["surface_condition"],
            "fitted_alpha_range_deg": [float(alpha.min()), float(alpha.max())],
            "excluded_post_stall_alpha_deg": [
                p["alpha"] for p in ref["points"] if p["alpha"] > 14.02
            ],
        },
        "parameters": {
            "cd0_scale": float(result.x[0]),
            "cl2_coeff": float(result.x[1]),
            "alpha2_coeff": float(result.x[2]),
            "fixed_stall_drag_coeff": 0.0040,
            "fixed_stall_drag_exponent": 1.30,
            "fixed_stall_onset_deg": 11.0,
        },
        "metrics": {
            "cd_mae": float(np.mean(np.abs(error))),
            "cd_rmse": float(np.sqrt(np.mean(error**2))),
            "cd_max_abs_error": float(np.max(np.abs(error))),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("NASA TM 4074 attached-flow drag calibration")
    print(f"Points fitted: {len(alpha)} from {alpha.min():.2f}° to {alpha.max():.2f}°")
    print("Post-stall points at 15.06° and 16.16° excluded.")
    print()
    print("Fitted production-model parameters")
    print(f"  cd0_scale     = {result.x[0]:.6f}")
    print(f"  cl2_coeff     = {result.x[1]:.6f}")
    print(f"  alpha2_coeff  = {result.x[2]:.8f}")
    print()
    print("Attached-flow calibration residual")
    print(f"  Cd MAE        = {payload['metrics']['cd_mae']:.6f}")
    print(f"  Cd RMSE       = {payload['metrics']['cd_rmse']:.6f}")
    print(f"  Cd max error  = {payload['metrics']['cd_max_abs_error']:.6f}")
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
