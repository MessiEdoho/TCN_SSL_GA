#!/bin/bash -l
#SBATCH --job-name=branch_shapley
# One node with one GPU for the forward-only Shapley computation.
#SBATCH -N 1
# Single Python process with 10 CPUs available for DataLoader workers,
# matching the allocation used by MultiScaleTCN_evaluation.sh.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# GPU partition: branch convolutions are evaluated in FP32 on GPU. CPU-only
# inference over the multi-million-segment test set would take days.
#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 GPU
#SBATCH --gres=gpu:1

# Per-prediction branch Shapley attribution for MultiScaleTCN (M3) and
# MultiScaleTCNWithAttention (M4). The script enumerates all 8 coalitions
# over the 3 branches per batch, but the branch convolutions run once
# per batch (their outputs are cached and reused across masks), so the
# wall-time cost is ~1.3x a single test-set forward pass per model.
#
# Test-set FP32 inference for MultiScaleTCN is documented at ~25 h in
# MultiScaleTCN_evaluation.sh; Shapley analysis is ~30-35 h per model.
# Running --model all covers both M3 and M4 sequentially, so the safe
# budget for the test partition is ~3 days. Override below if running
# only one model or only the (smaller) validation partition.
#SBATCH -t 3-00:00:00

# SLURM stdout / stderr routed to the home directory. %j expands to the
# job ID at submission time so re-runs do not overwrite each other.
#SBATCH --output=/home/people/22206468/slurm-branch_shapley-%j.out
##SBATCH --error=/home/people/22206468/slurm-branch_shapley-%j.err

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"

# GPU allocation diagnostics (matches MultiScaleTCN_evaluation.sh).
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

# CPU allocation diagnostics.
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

# Configurable runtime arguments. Override at submission time with e.g.
#   sbatch --export=ALL,PARTITION=val,MODEL=M3 branch_shapley_analysis.sh
PARTITION="${PARTITION:-test}"
MODEL="${MODEL:-all}"
echo "Partition argument : ${PARTITION}"
echo "Model argument     : ${MODEL}"

# Reads:
#   outputs/MultiScaleTCN/multiscale_tcn_final_weights.pt           (M3)
#   outputs/MultiScaleTCNAttention/multiscale_tcn_attention_final_weights.pt (M4)
#   outputs/best_multiscale_params.json
#   outputs/best_multiscale_attn_params.json
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered_enriched.json
#       (the canonical enriched manifest used by all training and evaluation
#        scripts; overrideable via --splits-path)
# Writes:
#   outputs/interpretability/${PARTITION}/branch_shapley/
#       {m3,m4}_shapley_${PARTITION}.csv
#       {m3,m4}_shapley_${PARTITION}_summary.json
#       logs/branch_shapley_{m3,m4}_${PARTITION}.log
python branch_shapley_analysis.py --partition "${PARTITION}" --model "${MODEL}"

echo "===== JOB END ====="
date
