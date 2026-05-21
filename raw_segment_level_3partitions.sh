#!/bin/bash -l
#SBATCH --job-name=raw_segment_level_3partitions
# One node with one GPU for the FP32 forward pass over the train set
# (~220k segments after proximity-aware downsampling).
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1

# Raw segment-level (Row 1) cross-partition comparison: load trained
# weights, run FP32 inference on the train partition, compute Row 1
# metrics, build a Train-vs-Val-vs-Test comparison (8 metrics:
# accuracy, recall macro, precision macro, specificity, AUROC, AP,
# F1 macro, MCC) for overfitting / generalisation diagnosis.
# Outputs land under {OUTPUT_ROOT}/train_evaluation/.
# Wall-time: ~5-15 min per model on V100/L40S; 1 hour budget is generous.
#SBATCH -t 13-00:00:00

#SBATCH --output=/home/people/22206468/slurm-raw_segment_level_3partitions-%j.out
##SBATCH --error=/home/people/22206468/slurm-raw_segment_level_3partitions-%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"

echo "----- GPU allocation -----"
echo "SLURM_JOB_GPUS         : ${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_PER_NODE    : ${SLURM_GPUS_PER_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES   : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPUDevice list (nvidia-smi -L):"
nvidia-smi -L 2>/dev/null || echo "  none detected"
echo "Per-GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,compute_cap,driver_version \
           --format=csv 2>/dev/null \
    || echo "  nvidia-smi unavailable"
echo "--------------------------"

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

# Variant is passed as the first positional argument.
# Available: TCN | TCNWithAttention | MultiScaleTCN | MultiScaleTCNWithAttention
VARIANT="${1:?Usage: sbatch -J <jobname> raw_segment_level_3partitions.sh <VARIANT>}"
echo "Variant: $VARIANT"

# Per-variant symlink to this job's SLURM stdout file, so that an `ls
# $HOME/slurm_logs/` immediately tells you which jobs were which variant
# without opening any file. The original
# /home/people/22206468/slurm-raw_segment_level_3partitions-${SLURM_JOB_ID}.out
# is the live target; the symlink stays valid as long as that file exists.
mkdir -p "$HOME/slurm_logs"
ln -sf "/home/people/22206468/slurm-raw_segment_level_3partitions-${SLURM_JOB_ID}.out" \
       "$HOME/slurm_logs/raw_segment_level_3partitions-${VARIANT}-${SLURM_JOB_ID}.out"
echo "Per-variant log: $HOME/slurm_logs/raw_segment_level_3partitions-${VARIANT}-${SLURM_JOB_ID}.out"

python raw_segment_level_3partitions.py --variant "$VARIANT"

echo "===== JOB END ====="
date
