#!/bin/bash -l
#SBATCH --job-name=build_chrono
# Single node, no GPU needed -- this script only walks per-mouse segment
# grids and reads small Excel files. Pure CPU + light I/O.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# One-off chronology builder for the 30 val + test mice. For each mouse,
# replays the preprocessing grid walk against the same Excel annotations
# preprocessing consumed and writes a per-mouse {mouse_id}_chronology.npz
# under /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies/.
# Each downstream training/eval script consumes these to reorder cached
# y_true / y_prob into per-mouse chronological order. Re-run only if
# the splits manifest, EDF metadata, or annotations change.
# Wall-time: ~5-10 minutes is plenty for 30 mice (per-mouse grid walk
# is ~1 s for ~300k positions).
#SBATCH -t 0-01:00:00

#SBATCH --output=/home/people/22206468/slurm-build_chrono-%j.out
##SBATCH --error=/home/people/22206468/slurm-build_chrono-%j.err

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
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered.json
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json
#   /home/people/22206468/scratch/seizure_times_updated/{mouse}_xlsx.xlsx  (val + test mice)
# Writes:
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies/{mouse_id}_chronology.npz
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/build_chronology.log
python build_chronology.py

echo "===== JOB END ====="
date
