#!/bin/bash -l
#SBATCH --job-name=scan_extreme
# CPU-only diagnostic scan: count NaN/Inf and |x|>1000 segments in val
# and test partitions. No GPU required.
#SBATCH -N 1
#SBATCH --ntasks-per-node 16
#SBATCH --partition=csserial

# Conservative wall-time. Empirical estimate: ~5-15 min for 4.3M val
# segments at 16 workers; allow headroom for slow Lustre/GPFS.
#SBATCH -t 10:00:00

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

# Run the scan with 16 parallel workers, threshold = 1000.0 (matches the
# offline filter applied to TRAIN in create_balanced_splits.py).
# Outputs scan_results.json in the working directory.
python scan_val_test_extreme.py \
    --manifest /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json \
    --alt-manifest /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json \
    --workers 16 \
    --threshold 1000.0 \
    --output scan_results.json

echo "===== JOB END ====="
date
