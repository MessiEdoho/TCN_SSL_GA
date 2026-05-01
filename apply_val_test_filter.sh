#!/bin/bash -l
#SBATCH --job-name=apply_val_test_filter
# specify number of nodes
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 3

# specify the walltime e.g 10 days
#SBATCH -t 10-00:00:00

# set to email at start, end, and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPU cores allocated: $(nproc)"

# Activate environment
module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# One-shot data-prep step. Reads data_splits_nonictal_sampled.json
# (filtered TRAIN, raw VAL/TEST) and writes
# data_splits_nonictal_sampled_filtered.json with VAL and TEST also
# filtered by the same |x| > 1000 / NaN / Inf criterion already applied
# to TRAIN. Persistent log goes to Data_diagnostic/.
python apply_val_test_filter.py \
    --input  /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json \
    --output /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered.json \
    --threshold 1000.0 \
    --log-path /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/apply_val_test_filter.log

echo "===== JOB END ====="
date
