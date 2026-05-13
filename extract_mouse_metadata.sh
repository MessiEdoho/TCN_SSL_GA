#!/bin/bash -l
#SBATCH --job-name=extract_meta
# Single node, no GPU needed -- this script only reads EDF headers
# (preload=False) for ~30 mice. Pure I/O on the cluster filesystem.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# CPU partition: no GPU required for header-only EDF reads.
#SBATCH --partition=cs

# One-off metadata extraction for the val + test mice. Reads the splits
# manifest to identify the 30 mice, locates each mouse's EDF in
# /home/people/22206468/scratch/Raw EDF/, opens the header (no sample
# loading), and writes per-mouse n_samples / fs_hz / recording_start_dt
# to a single JSON consumed by the local recovery script.
# Wall-time: ~5-10 minutes is plenty for header-only reads on 30 EDFs.
#SBATCH -t 2-00:00:00

# SLURM stdout / stderr routed to home dir (NOT under any output folder).
# %j expands to the job ID at submission so re-runs do not overwrite.
#SBATCH --output=/home/people/22206468/slurm-extract_meta-%j.out
##SBATCH --error=/home/people/22206468/slurm-extract_meta-%j.err

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset} (CPU-only header reads)"

echo "----- CPU allocation -----"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "nproc (visible)     : $(nproc)"
echo "Affinity (taskset)  : $(taskset -cp $$ 2>/dev/null || echo 'taskset unavailable')"
echo "--------------------------"

# Activate environment
module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Reads:
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered.json
#   /home/people/22206468/scratch/Raw EDF/*.edf  (val + test mice only)
# Writes:
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/extract_mouse_metadata.log
python extract_mouse_metadata.py

echo "===== JOB END ====="
date
