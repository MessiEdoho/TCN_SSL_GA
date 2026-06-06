#!/bin/bash -l
#SBATCH --job-name=extract_meta_tr
# Single node, no GPU needed -- only reads EDF headers (preload=False)
# for ~102 mice (train + val + test). Pure I/O on the cluster filesystem.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# One-off metadata extraction for ALL mice (train + val + test) from the
# un-downsampled splits manifest. Overwrites
# mouse_recording_metadata.json with a strict superset of the val/test-
# only file -- EDF header fields (n_samples, fs_hz, recording_start_dt)
# are identical regardless of partition, so val/test entries reproduce
# byte-for-byte; only the `partitions` field expands for mice that
# appear in multiple partitions.
# Wall-time: ~10-30 min on ~102 EDF header reads.
#SBATCH -t 0-02:00:00

#SBATCH --output=/home/people/22206468/slurm-extract_meta_tr-%j.out
##SBATCH --error=/home/people/22206468/slurm-extract_meta_tr-%j.err

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

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Reads:
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json   (UN-DOWNSAMPLED)
#   /home/people/22206468/scratch/Raw EDF/*.edf                         (train + val + test mice)
# Writes:
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json
#                                                                       (overwrites; superset of prior file)
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/extract_mouse_metadata.log  (appended)
python extract_mouse_metadata.py \
    --include-train \
    --input-splits /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json

echo "===== JOB END ====="
date
