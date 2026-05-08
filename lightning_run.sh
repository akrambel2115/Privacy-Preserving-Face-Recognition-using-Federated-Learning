#!/usr/bin/env bash
# lightning_run.sh — Full-speed simulation on a Lightning AI H100 studio.
#
# Usage (inside the Lightning studio terminal):
#   chmod +x lightning_run.sh
#   ./lightning_run.sh
#
# Assumptions:
#   - Working directory is the project root.
#   - Training data lives at ./data/train  (one sub-directory per client).
#   - PyTorch + CUDA drivers are pre-installed by the Lightning base image.
#
# What this script does:
#   1. Installs project-specific Python packages (skipping torch/CUDA).
#   2. Installs the project in editable mode.
#   3. Runs a quick 5-round profile to sanity-check throughput.
#   4. Launches the full 200-round simulation with all speed knobs on.
#
# Tune DATA_DIR, NUM_WORKERS, and BATCH_SIZE to your studio's specs.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (edit these as needed)
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data/train}"
NUM_ROUNDS="${NUM_ROUNDS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"      # H100 studios typically have 16+ cores
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./results/final_checkpoint.pt}"
USE_FUSED="${USE_FUSED:-true}"       # set to "false" for sequential mode
USE_AMP="${USE_AMP:-true}"           # mixed precision (fp16 fwd+bwd)

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  [1/4] Installing dependencies..."
echo "============================================================"
pip install --quiet --no-deps -r requirements_lightning.txt
pip install --quiet -e .

# ---------------------------------------------------------------------------
# 2. Verify CUDA
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [2/4] Verifying CUDA..."
echo "============================================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  GPU mem  : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ---------------------------------------------------------------------------
# 3. Quick profile (5 rounds)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [3/4] Quick 5-round profile..."
echo "============================================================"

PROFILE_FLAGS="--data-dir ${DATA_DIR} --num-rounds 5 --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS}"
if [ "${USE_FUSED}" = "true" ]; then
    PROFILE_FLAGS="${PROFILE_FLAGS} --use-fused"
fi
if [ "${USE_AMP}" = "true" ]; then
    PROFILE_FLAGS="${PROFILE_FLAGS} --use-amp"
fi

python scripts/profile_simulation.py ${PROFILE_FLAGS}

# ---------------------------------------------------------------------------
# 4. Full simulation
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [4/4] Full ${NUM_ROUNDS}-round simulation..."
echo "============================================================"

SIM_FLAGS="--data-dir ${DATA_DIR} --num-rounds ${NUM_ROUNDS} --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} --log-round-timing --checkpoint-path ${CHECKPOINT_PATH}"
if [ "${USE_FUSED}" = "true" ]; then
    SIM_FLAGS="${SIM_FLAGS} --use-fused"
fi
if [ "${USE_AMP}" = "true" ]; then
    SIM_FLAGS="${SIM_FLAGS} --use-amp"
fi

python scripts/run_simulation.py ${SIM_FLAGS}

echo ""
echo "============================================================"
echo "  Done! Checkpoint saved to ${CHECKPOINT_PATH}"
echo "============================================================"
