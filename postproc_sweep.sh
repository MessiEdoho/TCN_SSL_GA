#!/bin/bash -l
#SBATCH --job-name=postproc_sweep
#SBATCH -N 1
#SBATCH --ntasks=1
# CPU-only -- no model forward pass, no GPU. Sweep is pure post-processing
# on cached _full predictions: smoothing convolutions, run-detection,
# event matching against Excel annotations. 2 CPUs is plenty.
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# Sweep post-processing order x MIN_EVENT_SEC for the given variant
# (M1 / M2 / M3 / M4), on both val and test partitions in one run.
# Wall time ~5-15 min: dominated by per-mouse Excel reads and chronology
# matching inside eval_utils.evaluate_event_level. CPU-only.
#SBATCH -t 0-02:00:00

#SBATCH --output=/home/people/22206468/slurm-postproc_sweep-%j.out

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
VARIANT="${1:?Usage: sbatch -J <jobname> postproc_sweep.sh <VARIANT>}"
echo "Variant: $VARIANT"

# Per-variant symlink to this job's SLURM stdout. Mirrors the Option-B
# pattern in raw_segment_level_3partitions.sh so `ls $HOME/slurm_logs/` immediately tells
# which variant a job ran without opening any file.
mkdir -p "$HOME/slurm_logs"
ln -sf "/home/people/22206468/slurm-postproc_sweep-${SLURM_JOB_ID}.out" \
       "$HOME/slurm_logs/postproc_sweep-${VARIANT}-${SLURM_JOB_ID}.out"
echo "Per-variant log: $HOME/slurm_logs/postproc_sweep-${VARIANT}-${SLURM_JOB_ID}.out"

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Reads cached _full predictions NPZ + mouse metadata + Excel annotations
# (see postproc_sweep.py header for exact paths per variant). Writes the
# 16-row sweep results under <variant_output>/post_process_varing_sec/.
python postproc_sweep.py --variant "$VARIANT" --cluster

echo "===== JOB END ====="
date
