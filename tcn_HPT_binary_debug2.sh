#!/bin/bash -l
#SBATCH --job-name=tcn_debug2
#SBATCH -N 1
#SBATCH --ntasks-per-node 10
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== DEBUG2 JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

python tcn_HPT_binary_debug2.py

echo "===== DEBUG2 JOB END ====="
date
