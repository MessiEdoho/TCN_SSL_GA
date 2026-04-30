#!/bin/bash -l
#SBATCH --job-name=scan_extreme
# CPU-only diagnostic scan: count NaN/Inf and |x|>1000 segments in val
# and test partitions. No GPU required.
# speficity number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 3

# specify the walltime e.g 10 days 
#SBATCH -t 10-00:00:00

# set to email at start,end and failed jobs
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

# Run the scan. Threshold = 1000.0 matches the offline filter applied to
# TRAIN in create_balanced_splits.py. Both the JSON summary and the .log
# file are written to the Data_diagnostic directory so audit-trail outputs
# are kept separate from training output trees.
DIAG_DIR=/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic
mkdir -p "$DIAG_DIR"

python scan_val_test_extreme.py \
    --manifest /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json \
    --alt-manifest /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json \
    --workers 16 \
    --threshold 1000.0 \
    --output "$DIAG_DIR/scan_results.json" \
    --log-path "$DIAG_DIR/scan_val_test_extreme.log"

echo "===== JOB END ====="
date
