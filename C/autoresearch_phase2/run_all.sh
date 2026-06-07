#!/usr/bin/env bash
# run_all.sh
# ----------
# Reproduces the full Phase 1 + Phase 2 analysis from scratch.
# Run from the project root: bash run_all.sh
#
# Requirements: Python 3.10+, pip
# All dependencies are installed automatically from requirements.txt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"
PY="python3"

echo "========================================================"
echo "  autoresearch program.md — Phase 1 + Phase 2 pipeline"
echo "========================================================"
echo ""

# ── 1. Install dependencies ───────────────────────────────────────────────────
echo "[1/3] Installing Python dependencies..."
$PY -m pip install --quiet -r "$ROOT/requirements.txt"
echo "      Done."
echo ""

# ── 2. Phase 1: structural classification ────────────────────────────────────
echo "[2/3] Running Phase 1 (structural classification)..."
$PY "$SCRIPTS/phase1_pipeline.py"
echo ""
echo "      Outputs written to: phase1_output/"
echo "        classification.tsv   — σ-class and φ-class per file"
echo "        records.json         — full feature vectors + section trees"
echo "        zss_distances.tsv    — pairwise ZSS tree-edit distance matrix"
echo ""

# ── 3. Phase 2: simulation and statistical analysis ───────────────────────────
echo "[3/3] Running Phase 2 (simulation + analysis)..."
$PY "$SCRIPTS/phase2_pipeline.py"
echo ""
echo "      Outputs written to: phase2_output/"
echo "        outcomes.tsv         — aggregate metrics per variant"
echo "        trajectories.json    — full simulated experiment logs"
echo "        epsilon_sweep.tsv    — ~φ class count at each ε level"
echo "        phase2_report.txt    — full narrative + statistical results"
echo ""

echo "========================================================"
echo "  All done. See phase1_output/ and phase2_output/."
echo ""
echo "  To use real autoresearch data instead of simulation:"
echo "  Replace the simulate_run() function in"
echo "  scripts/phase2_pipeline.py with a real results.tsv"
echo "  reader (see PLUG_AND_REPLACE section in that file)."
echo "========================================================"
