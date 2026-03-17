#!/bin/bash
# Phase 1 Setup: Create conda environment with all dependencies
set -e

CONDA=/home/jinxulin/miniconda3/bin/conda
ENV_NAME=sibyl_AURA
PROJECT_DIR=/home/jinxulin/sibyl_system/projects/AURA

# Create project directories
mkdir -p $PROJECT_DIR/exp/results
mkdir -p $PROJECT_DIR/exp/checkpoints
mkdir -p $PROJECT_DIR/exp/code
mkdir -p /home/jinxulin/sibyl_system/shared/datasets

# Create conda environment if it doesn't exist
if ! $CONDA env list | grep -q "$ENV_NAME"; then
    echo "[SETUP] Creating conda environment $ENV_NAME..."
    $CONDA create -n $ENV_NAME python=3.11 -y
else
    echo "[SETUP] Environment $ENV_NAME already exists."
fi

# Install dependencies
echo "[SETUP] Installing dependencies..."
$CONDA run -n $ENV_NAME pip install --upgrade pip
$CONDA run -n $ENV_NAME pip install \
    torch torchvision \
    scipy statsmodels scikit-learn \
    matplotlib seaborn \
    numpy pandas \
    tqdm

# Install TDA-specific libraries
echo "[SETUP] Installing TDA libraries..."
$CONDA run -n $ENV_NAME pip install pydvl dattri trak

echo "[SETUP] Verifying installations..."
$CONDA run -n $ENV_NAME python -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
import torchvision; print(f'torchvision={torchvision.__version__}')
import scipy; print(f'scipy={scipy.__version__}')
import statsmodels; print(f'statsmodels={statsmodels.__version__}')
import sklearn; print(f'sklearn={sklearn.__version__}')
try:
    import trak; print(f'trak={trak.__version__}')
except Exception as e:
    print(f'trak import error: {e}')
try:
    import pydvl; print(f'pydvl={pydvl.__version__}')
except Exception as e:
    print(f'pydvl import error: {e}')
try:
    import dattri; print(f'dattri loaded')
except Exception as e:
    print(f'dattri import error: {e}')
print('All imports OK')
"

echo "[SETUP] Environment setup complete."
