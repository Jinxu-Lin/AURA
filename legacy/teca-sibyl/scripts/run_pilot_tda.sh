#!/bin/bash
set -e
export PATH="/home/jinxulin/miniconda3/bin:$PATH"
eval "$(/home/jinxulin/miniconda3/bin/conda shell.bash hook)"
conda activate sibyl_TECA
cd /home/jinxulin/sibyl_system/projects/TECA
export CUDA_VISIBLE_DEVICES=2
python3 pilot_tda_gradient.py 2>&1
