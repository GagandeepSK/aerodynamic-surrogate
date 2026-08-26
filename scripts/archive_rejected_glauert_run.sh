#!/usr/bin/env sh
set -eu

root="${1:-$HOME/aero-surrogate}"
archive="$root/analysis/archive/rejected_glauert_v3"

mkdir -p "$archive"
for rel in \
  model/surrogate_model.pt \
  model/model_weights.json \
  dashboard/index.html \
  analysis/naca0012_tm4074_metrics.json \
  analysis/naca0012_tm4074_drag_calibration.json
do
  test -f "$root/$rel"
  cp -p "$root/$rel" "$archive/$(basename "$rel")"
done

(
  cd "$archive"
  sha256sum surrogate_model.pt model_weights.json index.html \
    naca0012_tm4074_metrics.json naca0012_tm4074_drag_calibration.json
) > manifest.sha256

printf 'Archived rejected global-thickness EC2 run to %s\n' "$archive"
