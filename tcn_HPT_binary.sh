#!/bin/bash -l
#SBATCH --job-name=tcn_HPT_binary
# One node with one GPU for PyTorch training + CPU cores for Optuna TPE
#SBATCH -N 1
#SBATCH --ntasks-per-node 6
#SBATCH --constraint=highmem
#SBATCH --partition=csgpu
# 60 Optuna trials x up to 100 epochs each (early stopping typically fires ~30-50).
# 3-day walltime is generous but safe for large non_seizure partitions.
#SBATCH -t 10-00:00:00

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
conda activate uniqureSSLGA

cd ~/TCN_SSL_GA

# Run the converted Python script (tcn_HPT_binary.ipynb -> tcn_HPT_binary.py)
python tcn_HPT_binary.py

echo "===== JOB END ====="
date
