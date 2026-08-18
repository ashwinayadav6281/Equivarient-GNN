#!/bin/bash
# ============================================================
# Server Setup for Equivariant Elastic Tensor Model
#
# Usage:
#   bash setup_equivariant_server.sh              # Full: setup → download → train
#   bash setup_equivariant_server.sh download      # Only download 13k tensors
#   bash setup_equivariant_server.sh train          # Only train (data must exist)
# ============================================================

set -e

echo "============================================"
echo "  Equivariant Model — Server Setup"
echo "============================================"

# ── Step 1: Virtual environment ──
if [ ! -d "venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "[1/5] Virtual environment already exists."
fi

source venv/bin/activate
echo "  Python: $(python --version)"

# ── Step 2: Install dependencies ──
echo "[2/5] Installing dependencies..."
pip install --upgrade pip -q

# PyTorch with CUDA (change cu118 to match your server's CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

# e3nn (equivariant neural networks library)
pip install e3nn -q

# Other dependencies
pip install pymatgen scikit-learn numpy pandas requests -q

echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  e3nn:    $(python -c 'import e3nn; print(e3nn.__version__)')"

# ── Step 3: Ensure elasticity CSV exists ──
if [ ! -f "mp_latest_elasticity.csv" ]; then
    echo "[3/5] Downloading elasticity data from Materials Project..."
    python dataset.py
else
    LINES=$(wc -l < mp_latest_elasticity.csv)
    echo "[3/5] mp_latest_elasticity.csv already exists ($LINES lines)"
fi

# ── Step 4: Download elastic tensors + CIF structures ──
if [ "$1" = "train" ]; then
    echo "[4/5] SKIP download (train-only mode)"
else
    TENSOR_COUNT=0
    if [ -f "dataset_equivariant/elastic_tensors.json" ]; then
        TENSOR_COUNT=$(python -c "import json; print(len(json.load(open('dataset_equivariant/elastic_tensors.json'))))" 2>/dev/null || echo "0")
    fi
    
    if [ "$TENSOR_COUNT" -gt 13000 ]; then
        echo "[4/5] Dataset already complete ($TENSOR_COUNT tensors)"
    else
        echo "[4/5] Downloading elastic tensors for all materials..."
        echo "  Current: $TENSOR_COUNT tensors. Target: ~13,000"
        echo "  (This takes ~1-2 hours with batched API calls. Has resume support.)"
        python fetch_elastic_tensors_full.py
    fi
fi

# ── Step 5: Launch training ──
if [ "$1" = "download" ]; then
    echo "[5/5] SKIP training (download-only mode)"
    echo ""
    echo "To start training later:"
    echo "  source venv/bin/activate"
    echo "  nohup python train_equivariant_server.py --epochs 100 > equivariant_training.log 2>&1 &"
else
    echo "[5/5] Starting equivariant model training in background..."
    nohup python train_equivariant_server.py --epochs 100 > equivariant_training.log 2>&1 &
    TRAIN_PID=$!

    echo ""
    echo "============================================"
    echo "  Training started in background!"
    echo "  PID: $TRAIN_PID"
    echo "============================================"
    echo ""
    echo "Useful commands:"
    echo "  tail -f equivariant_training.log               # Watch live progress"
    echo "  tail -f checkpoints_equivariant/training_log.txt  # Detailed log"
    echo "  nvidia-smi                                     # Check GPU usage"
    echo "  kill $TRAIN_PID                                # Stop training"
fi
echo ""
