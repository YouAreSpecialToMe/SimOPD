#!/bin/bash
#SBATCH --job-name=opdtraj
#SBATCH --partition=rush
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zz865/opdtraj_%j.out
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0
python3 /home/zz865/fetch_opd.py /share/rush/zz865/opd
