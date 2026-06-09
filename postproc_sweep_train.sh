#!/bin/bash -l
#SBATCH --job-name=postproc_sweep_tr
#SBATCH -N 1
#SBATCH --ntasks=1
# CPU-only -- no model forward pass, no GPU. Sweep is pure post-processing
# on the cached full-train predictions NPZ: smoothing convolutions,
# run-detection, event matching against Excel annotations.
#SBATCH --cpus-per-task=4

#SBATCH --partition=cs

# Sweep post-processing order x MIN_EVENT_SEC for the given variant
# (M3 or M4) on the FULL UN-DOWNSAMPLED TRAIN partition ONLY. Writes
# to the sibling output root post_process_varing_sec_train/. Does NOT
# touch the val/test artefacts -- run postproc_sweep.sh for those.
#
# Wall time estimate: ~10-15 h for the train sweep (10 configs x
# ~60-90 min/config at 28.7M segments).
#SBATCH -t 1-00:00:00

#SBATCH --output=/home/people/22206468/slurm-postproc_sweep_tr-%j.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset} (CPU-only)"

echo "----- CPU allocation -----"
echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"
echo "nproc (visible)     : $(nproc)"
echo "Affinity (taskset)  : $(taskset -cp $$ 2>/dev/null || echo 'taskset unavailable')"
echo "--------------------------"

# Variant is passed as the first positional argument.
# Available: TCN | TCNWithAttention | MultiScaleTCN | MultiScaleTCNWithAttention
VARIANT="${1:?Usage: sbatch -J <jobname> postproc_sweep_train.sh <VARIANT>}"
echo "Variant: $VARIANT"

mkdir -p "$HOME/slurm_logs"
ln -sf "/home/people/22206468/slurm-postproc_sweep_tr-${SLURM_JOB_ID}.out" \
       "$HOME/slurm_logs/postproc_sweep_tr-${VARIANT}-${SLURM_JOB_ID}.out"
echo "Per-variant log: $HOME/slurm_logs/postproc_sweep_tr-${VARIANT}-${SLURM_JOB_ID}.out"

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Reads:
#   <variant>/full_train_evaluation/<train_NPZ>
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_full_train_enriched.json
#   mouse_recording_metadata.json
#   /home/people/22206468/scratch/seizure_times_updated/{mouse}_xlsx.xlsx
# Writes:
#   <variant>/full_train_evaluation/post_process_varing_sec_train/
#       train/refractory_then_min/MIN_EVENT_SEC_{10,15,20,25,30}s/
#       train/min_then_refractory/MIN_EVENT_SEC_{10,15,20,25,30}s/
#       comparison_train.csv
#       impact_train.png
#       postproc_sweep.log
python postproc_sweep.py --variant "$VARIANT" --cluster --partitions train

echo "===== JOB END ====="
date
