#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ap1321
export PATH=/vol/bitbucket/ap1321/venv/bin/:$PATH
source activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
python main.py
