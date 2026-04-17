#!/bin/bash -l
#SBATCH --job-name=tcn_hpt_dry
# One node with one GPU for PyTorch training + CPU cores for Optuna TPE
#SBATCH -N 1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 10

#SBATCH --partition=csgpu
# Request 1 gpus
#SBATCH --gres=gpu:1

# Dry-run: TCN_HPT_DRY_RUN=1 caps N_TRIALS=2 and N_STARTUP=1 in the Python
# script. With the 1% stratified val subset (~43K segments), train+val per
# epoch is ~8.2K batches. Cold epoch ~24 min, warm epochs ~5-9 min. Two
# trials with ~10 epochs each fit comfortably in 4 hours.
#SBATCH -t 04:00:00

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== DRY-RUN JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

# Activate environment
module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Enable dry-run mode: script reads this env var and caps N_TRIALS at 2.
# Purpose: verify script starts, loads data, reaches training loop, and
# writes all outputs without consuming a full 50-trial budget.
export TCN_HPT_DRY_RUN=1

python tcn_HPT_binary.py

echo "===== DRY-RUN JOB END ====="
date
