"""
Export trained PyTorch model weights to JSON for browser-side inference.

Usage
-----
    python -m src.export_weights
    python -m src.export_weights --model model/surrogate_model.pt \
                                  --output model/model_weights.json \
                                  --embed  dashboard/index.html

The JSON file is loaded by the vanilla-JS MLP forward pass in the dashboard.
When --embed is given, the weights are inlined directly into the HTML as the
MODEL_WEIGHTS JavaScript constant, making the dashboard fully self-contained:
one HTML file, no server, works from file:// and from S3 static hosting.

JSON schema
-----------
{
  "net.0.weight": [[...], ...],   // shape (128, 5)
  "net.0.bias":   [...],          // shape (128,)
  "net.2.weight": [[...], ...],   // shape (128, 128)
  ...
  "net.6.weight": [[...], ...],   // shape (2, 64)
  "net.6.bias":   [...],          // shape (2,)
  "input_mean":   [...],          // shape (5,)  -- feature z-score mean
  "input_std":    [...],          // shape (5,)  -- feature z-score std
  "output_mean":  [...],          // shape (2,)  -- target z-score mean
  "output_std":   [...],          // shape (2,)  -- target z-score std
  "architecture": [5, 128, 128, 64, 2],
  "feature_cols": [...],
  "target_cols":  [...]
}
"""

import argparse
import json
import re
from pathlib import Path

import torch

from src.model import AeroSurrogate


def export_weights(model_path: str, json_path: str) -> dict:
    """
    Load checkpoint, serialise weights + normalisation params to JSON.
    Returns the weights dict for optional downstream use.
    """
    ckpt  = torch.load(model_path, map_location='cpu')
    model = AeroSurrogate()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    weights = {}

    for name, param in model.state_dict().items():
        weights[name] = param.numpy().tolist()

    weights['input_mean']  = ckpt['input_mean']
    weights['input_std']   = ckpt['input_std']
    weights['output_mean'] = ckpt['output_mean']
    weights['output_std']  = ckpt['output_std']
    weights['architecture'] = [5, 128, 128, 64, 2]
    weights['feature_cols'] = ckpt.get('feature_cols',
                              ['m_frac','p_frac','t_frac','alpha_deg','log10_Re'])
    weights['target_cols']  = ckpt.get('target_cols', ['Cl','Cd'])
    weights['source_model'] = ckpt.get('source_model', {'id': 'unknown'})

    for k in ('test_r2', 'test_mae', 'test_rmse'):
        if k in ckpt:
            weights[k] = ckpt[k]

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(weights, f, separators=(',', ':'))

    n  = sum(p.numel() for p in model.parameters())
    r2 = ckpt.get('test_r2', ['n/a', 'n/a'])
    print(f"Exported {n:,} parameters  ->  {json_path}")
    print(f"  R2: Cl={r2[0]:.4f}  Cd={r2[1]:.4f}" if isinstance(r2[0], float)
          else "  (test metrics not available)")
    return weights


def embed_into_dashboard(weights: dict, dashboard_path: str) -> None:
    """
    Inline weights JSON into the dashboard HTML by replacing the placeholder
    marker  /*WEIGHTS_PLACEHOLDER*/null  with the actual JSON object.
    """
    path = Path(dashboard_path)
    if not path.exists():
        raise FileNotFoundError(f"Dashboard not found: {dashboard_path}")

    html = path.read_text(encoding='utf-8')
    placeholder = '/*WEIGHTS_PLACEHOLDER*/null'
    weights_json = json.dumps(weights, separators=(',', ':'))

    if placeholder in html:
        html = html.replace(placeholder, weights_json, 1)
    else:
        # Dashboard files are self-contained after the first export. Replace
        # an existing embedded object so re-exporting never requires manual edits.
        pattern = r'(const MODEL_WEIGHTS\s*=\s*)\{.*?\}(;\s*\n\s*function _linear)'
        html, replacements = re.subn(
            pattern,
            lambda match: match.group(1) + weights_json + match.group(2),
            html,
            count=1,
            flags=re.DOTALL,
        )
        if replacements != 1:
            raise ValueError(
                f"Could not find a weights placeholder or existing MODEL_WEIGHTS "
                f"object in {dashboard_path}."
            )

    path.write_text(html, encoding='utf-8')

    kb = len(html.encode('utf-8')) / 1024
    print(f"Embedded weights into {dashboard_path}  ({kb:.0f} KB total)")


def main():
    p = argparse.ArgumentParser(description='Export model weights to JSON')
    p.add_argument('--model',  default='model/surrogate_model.pt')
    p.add_argument('--output', default='model/model_weights.json')
    p.add_argument('--embed',  default=None, metavar='DASHBOARD_HTML',
                   help='Inline weights into this HTML file after export')
    args = p.parse_args()

    weights = export_weights(args.model, args.output)

    if args.embed:
        embed_into_dashboard(weights, args.embed)
        print(f"\nDashboard is now self-contained.")
        print(f"Open {args.embed} in a browser with internet access for the pinned Plotly CDN; no inference server is required.")


if __name__ == '__main__':
    main()
