#!/bin/bash -l
#SBATCH --job-name=create_balanced_splits
# Single node, no GPU needed -- reads EDF headers + .npy files, writes JSON
#SBATCH -N 1

# A few cores help with I/O scheduling on Lustre (reading EDF headers
# and scanning .npy files for extreme amplitude filtering)
#SBATCH --ntasks-per-node 4

# Estimated runtime: 1-3 hours depending on /scratch load.
# EDF header reads are fast (~10 ms each), but the extreme amplitude
# scan loads ~100K .npy files. 2-day walltime provides safety margin.
#SBATCH -t 2-00:00:00

# No GPU partition -- runs on any available CPU node
# (Do NOT request --partition=csgpu or --gres=gpu)

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Purpose: Proximity-aware downsampling of non-ictal segments"
echo "Output: data_splits_nonictal_sampled.json"
echo "Mode: READ-ONLY on .npy files (no data files modified or deleted)"

# Activate environment (requires numpy, mne, pandas, matplotlib)
module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Run the balanced splits generation
python create_balanced_splits.py

echo "===== JOB END ====="
date
