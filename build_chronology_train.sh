#!/bin/bash -l
#SBATCH --job-name=build_chrono_tr
# Single node, no GPU needed -- only walks per-mouse segment grids and
# reads small Excel files. Pure CPU + light I/O.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# One-off chronology builder for the train mice in the UN-DOWNSAMPLED
# splits manifest (data_splits.json). Same per-mouse grid walk as the
# val/test build, just over the train partition. Required upstream of
# the full-train evaluation + post-processing sweep.
# Wall-time: ~30-90 min for ~30 train mice (grid walk scales with
# n_samples; full-train mice have the largest recordings).
#SBATCH -t 0-02:00:00

#SBATCH --output=/home/people/22206468/slurm-build_chrono_tr-%j.out
##SBATCH --error=/home/people/22206468/slurm-build_chrono_tr-%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset} (CPU-only chronology rebuild)"

echo "----- CPU allocation -----"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "nproc (visible)     : $(nproc)"
echo "Affinity (taskset)  : $(taskset -cp $$ 2>/dev/null || echo 'taskset unavailable')"
echo "--------------------------"

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Reads:
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json    (UN-DOWNSAMPLED)
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json
#   /home/people/22206468/scratch/seizure_times_updated/{mouse}_xlsx.xlsx  (train + val + test mice)
# Writes:
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies/{mouse_id}_chronology.npz
#                                                                          (overwrites if present;
#                                                                           chronology is deterministic
#                                                                           given metadata + annotations)
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/build_chronology.log  (appended)
python build_chronology.py \
    --include-train \
    --input-splits /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json

echo "===== JOB END ====="
date
