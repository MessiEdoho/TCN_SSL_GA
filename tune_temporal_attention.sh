#!/bin/bash -l
#SBATCH --job-name=tune_attn
# One node with one GPU for PyTorch training + CPU cores for Optuna TPE
#SBATCH -N 1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 10

#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 gpus
#SBATCH --gres=gpu:1

# M2: temporal attention HP search over the M1 TCN backbone hyperparameters.
# Backbone held non-trainable during tuning for search-space reduction (only
# the attention module + classification head receive gradient updates). The
# 50-trial budget plus the frozen backbone makes this faster per-trial than
# M1 tuning, but allow generous walltime since trials still run up to ~20
# epochs each on the full 220K training partition.
#SBATCH -t 13-00:00:00

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

# Activate environment
module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Reads outputs/best_params.json (M1 backbone HPs) and writes
# outputs/best_attention_params.json (M2 attention HPs).
python tune_temporal_attention.py

echo "===== JOB END ====="
date
