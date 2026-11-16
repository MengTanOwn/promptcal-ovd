#!/usr/bin/env bash
# Run the whole study, in order, and stop at the first failure.
#
#   ./scripts/reproduce_all.sh            # everything, including both extractions
#   SKIP_EXTRACT=1 ./scripts/reproduce_all.sh   # analysis only, on existing shards
#
# Expects PROMPTCAL_DATA (and optionally PROMPTCAL_FEATURES) to be set; see docs/data.md.
set -euo pipefail

cd "$(dirname "$0")/.."

PCC=${PCC:-python}
STEP() { printf '\n=== %s ===\n' "$*"; }

STEP "environment"
"$PCC" -c "from promptcal import config; print(config.describe())"

if [ -z "${SKIP_EXTRACT:-}" ]; then
  STEP "extract COCO"
  "$PCC" -m promptcal.cli extract --bench coco
fi

STEP "wording"
"$PCC" -m promptcal.cli wording

STEP "calibration"
"$PCC" -m promptcal.cli calibration

STEP "bootstrap"
"$PCC" -m promptcal.cli bootstrap

STEP "conformal"
"$PCC" -m promptcal.cli conformal

if [ -z "${SKIP_EXTRACT:-}" ]; then
  STEP "extract LVIS"
  "$PCC" -m promptcal.cli extract --bench lvis
fi

# the transfer needs the LVIS shards, whether they were just built or already on disk
if "$PCC" - <<'EOF'
import glob, sys
from promptcal import config
sys.exit(0 if glob.glob(str(config.features_dir() / "lvis_gd_part*.npz")) else 1)
EOF
then
  STEP "transfer COCO -> LVIS"
  "$PCC" -m promptcal.cli transfer
else
  echo "skipping transfer: no lvis_gd_part*.npz in the feature directory"
  echo "run: promptcal extract --bench lvis"
fi

STEP "macros and figures"
"$PCC" -m promptcal.cli macros
"$PCC" -m promptcal.cli figures

echo
echo "done. results/ holds the JSONs, paper/nums.tex the macros."
echo "compile the manuscript with: make paper"
