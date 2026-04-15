#!/bin/bash -l
#SBATCH --job-name=verify_data_integrity
# Single node, no GPU needed -- this is a read-only I/O-bound scan
#SBATCH -N 1

# A few cores help with OS-level I/O scheduling on Lustre
#SBATCH --ntasks-per-node 4

# Scanning ~33M .npy files across 7 directories. Estimated 4-8 hours
# depending on /scratch load. 1-day walltime provides safety margin.
#SBATCH -t 5-00:00:00

# No GPU partition -- runs on any available CPU node
# (Do NOT request --partition=csgpu or --gres=gpu)

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Purpose: Post-incident data integrity verification"
echo "Directories: TRAIN_DATA (1-5), VAL_DATA, TEST_DATA"
echo "Checks: readable, shape, dtype, finite, amplitude"
echo "Mode: READ-ONLY (no files will be modified or deleted)"

# Activate environment (numpy is the only dependency)
module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Run the integrity check
python verify_data_integrity.py

echo "===== JOB END ====="
date
