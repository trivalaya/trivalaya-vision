#!/bin/bash
# GPU deploy script for legend_cnn training
#
# On the GPU machine:
#   1. scp legend_cnn_gpu_deploy.tar.gz to the working directory
#   2. Run this script
#
# Prerequisites: Python 3.10+, pip, CUDA-capable GPU

set -e

echo "=== Legend CNN GPU Deploy ==="

# Unpack
if [ -f legend_cnn_gpu_deploy.tar.gz ]; then
    echo "Unpacking archive..."
    tar xzf legend_cnn_gpu_deploy.tar.gz
else
    echo "ERROR: legend_cnn_gpu_deploy.tar.gz not found"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision numpy pandas matplotlib scikit-learn 2>/dev/null || \
pip3 install torch torchvision numpy pandas matplotlib scikit-learn

# Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

from legend_cnn.model import LegendCNN
from legend_cnn.dataset import load_splits
m = LegendCNN(num_classes=10)
print(f'Model params: {sum(p.numel() for p in m.parameters()):,}')
splits = load_splits()
print(f'Splits: train={splits[\"n_train\"]} val={splits[\"n_val\"]} test={splits[\"n_test\"]}')
print('All imports OK')
"

echo ""
echo "=== Ready to train ==="
echo "Run:"
echo "  python -m legend_cnn.train \\"
echo "    --ribbon-stats legend_cnn/runs/run_20260410_013529_promoted_v1/ribbon_stats.csv \\"
echo "    --epochs 50 --batch-size 256 --lr 1e-3 --patience 5"
echo ""
echo "Then evaluate:"
echo "  python -m legend_cnn.evaluate \\"
echo "    --ribbon-stats legend_cnn/runs/run_20260410_013529_promoted_v1/ribbon_stats.csv"
