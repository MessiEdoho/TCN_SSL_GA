#!/bin/bash -l
#SBATCH --job-name=ftrain_m4
# One node with one GPU for the FP32 forward pass over the FULL
# un-downsampled train partition (~28.7M segments).
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1

# M4 FULL-TRAIN evaluation: load ms_attn_final_weights.pt, run an FP32
# forward pass on the un-downsampled train partition, compute Row 1 /
# Row 2 segment-level + per-mouse chronological event-level metrics.
# Writes the same artefact bundle as the test-eval script, under
# MODEL4_OUTPUT/MultiScaleTCNAttention/full_train_evaluation/.
#
# Wall-time estimate: ~3x the test set (9.75M -> 28.7M segments).
# Test took ~22h on V100, so plan for ~60-70h plus ~1h post-processing.
# 5-day cap matches the agreed budget.
#SBATCH -t 5-00:00:00

#SBATCH --output=/home/people/22206468/slurm-ftrain_m4-%j.out
##SBATCH --error=/home/people/22206468/slurm-ftrain_m4-%j.err

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

# Reads:
#   MODEL4_OUTPUT/MultiScaleTCNAttention/ms_attn_final_weights.pt
#   MODEL3_OUTPUT/MultiScaleTCNtuning_outputs/best_multiscale_params.json
#   MODEL4_OUTPUT/multiscale_attention_tuning_outputs/best_multiscale_attn_params.json
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_full_train_enriched.json
# Writes:
#   MODEL4_OUTPUT/MultiScaleTCNAttention/full_train_evaluation/
#       ms_attn_full_train_predictions_full.npz
#       ms_attn_full_train_evaluation_report.json
#       ms_attn_full_train_three_row_summary.csv
#       ms_attn_full_train_classification_report_row{1,2}.json
#       ms_attn_full_train_event_details_row2.csv
#       figures/, Result_classReport/
#   MODEL4_OUTPUT/MultiScaleTCNAttention/logs/ms_attn_full_train_evaluation.log
python full_train_eval_MultiScaleTCNAttention.py

echo "===== JOB END ====="
date
