#!/bin/bash -l
#SBATCH --job-name=tune_msattn
# One node with one GPU for PyTorch training + CPU cores for Optuna TPE
#SBATCH -N 1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 10

#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 gpus
#SBATCH --gres=gpu:1

# M4: temporal attention HP search over the M3 MultiScaleTCN backbone
# hyperparameters. Backbone held non-trainable during tuning for search-
# space reduction (only the attention module + classification head receive
# gradient updates). The 50-trial budget plus the frozen backbone is
# faster per-trial than M3 tuning, but allow generous walltime since trials
# still run up to ~20 epochs each on the full 220K training partition.
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

# Reads MODEL3_OUTPUT/.../best_multiscale_params.json (M3 backbone HPs)
# and writes MODEL4_OUTPUT/best_multiscale_attn_params.json (M4 attention HPs).
python tune_multiscale_attention.py

echo "===== JOB END ====="
date
