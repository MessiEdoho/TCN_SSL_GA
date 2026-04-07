#!/bin/bash -l
#SBATCH --job-name=tcn_debug
#SBATCH -N 1
#SBATCH --ntasks-per-node 10
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
# One trial only -- 3 hours should be more than enough
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== DEBUG JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

python tcn_HPT_binary_debug.py

echo "===== DEBUG JOB END ====="
date
