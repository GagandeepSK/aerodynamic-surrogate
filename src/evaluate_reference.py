"""
Evaluate a trained surrogate against a provenance-pinned external reference polar.

The NASA TM 4074 NACA 0012 case is a reference/calibration case, not a
held-out validation set after any fitting is performed against it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.model import AeroSurrogate


def mae(values):
    return float(np.mean(np.abs(values)))


def evaluate(model_path, reference_path, output_path):
    reference_path = Path(reference_path)
    output_path = Path(output_path)

    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location="cpu")

    model = AeroSurrogate()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    in_mean = np.asarray(checkpoint["input_mean"], dtype=np.float32)
    in_std = np.asarray(checkpoint["input_std"], dtype=np.float32)
    out_mean = np.asarray(checkpoint["output_mean"], dtype=np.float32)
    out_std = np.asarray(checkpoint["output_std"], dtype=np.float32)

    rows = ref["points"]
    alpha = np.asarray([row["alpha"] for row in rows], dtype=np.float32)
    y_true = np.asarray(
        [[row["Cl"], row["Cd"]] for row in rows], dtype=np.float32
    )

    # NACA 0012: symmetric, 12% thickness.
    x_raw = np.column_stack([
        np.zeros_like(alpha),
        np.zeros_like(alpha),
        np.full_like(alpha, 0.12),
        alpha,
        np.full_like(alpha, np.log10(ref["reynolds"])),
    ]).astype(np.float32)

    x_norm = (x_raw - in_mean) / (in_std + 1e-8)

    with torch.no_grad():
        y_pred = model(torch.from_numpy(x_norm)).numpy()
    y_pred = y_pred * out_std + out_mean

    abs_error = np.abs(y_pred - y_true)
    attached = alpha <= 14.02
    post_stall = alpha > 14.02

    def metrics(mask):
        return {
            "points": int(mask.sum()),
            "cl_mae": mae(y_pred[mask, 0] - y_true[mask, 0]),
            "cl_max_error": float(abs_error[mask, 0].max()),
            "cd_mae": mae(y_pred[mask, 1] - y_true[mask, 1]),
            "cd_max_error": float(abs_error[mask, 1].max()),
        }

    result = {
        "reference_source": ref["source"],
        "reference_url": ref["url"],
        "reynolds": ref["reynolds"],
        "surface_condition": ref["surface_condition"],
        "scope_note": (
            "This is an external reference/calibration case. "
            "It is not a held-out validation set after calibration."
        ),
        "overall": metrics(np.ones_like(alpha, dtype=bool)),
        "attached_flow_through_14_02_deg": metrics(attached),
        "post_stall_above_14_02_deg": metrics(post_stall),
        "points": [
            {
                "alpha_deg": float(a),
                "cl_prediction": float(pred[0]),
                "cl_reference": float(actual[0]),
                "cl_abs_error": float(err[0]),
                "cd_prediction": float(pred[1]),
                "cd_reference": float(actual[1]),
                "cd_abs_error": float(err[1]),
                "regime": "attached" if a <= 14.02 else "post_stall",
            }
            for a, pred, actual, err in zip(alpha, y_pred, y_true, abs_error)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nExternal reference comparison")
    print(f"Source: {ref['source']}")
    print(f"Condition: NACA 0012, Re={ref['reynolds']:.0e}")
    print(f"Surface: {ref['surface_condition']}")

    for name, values in [
        ("Overall", result["overall"]),
        ("Attached flow, α ≤ 14.02°", result["attached_flow_through_14_02_deg"]),
        ("Post-stall, α > 14.02°", result["post_stall_above_14_02_deg"]),
    ]:
        print(f"\n{name} ({values['points']} points)")
        print(f"  Cl MAE: {values['cl_mae']:.4f}")
        print(f"  Cd MAE: {values['cd_mae']:.5f}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/surrogate_model.pt")
    parser.add_argument(
        "--reference",
        default="data/validation/naca0012_ladson_tm4074_re6e6.json",
    )
    parser.add_argument(
        "--output",
        default="analysis/naca0012_tm4074_metrics.json",
    )
    args = parser.parse_args()

    evaluate(args.model, args.reference, args.output)
