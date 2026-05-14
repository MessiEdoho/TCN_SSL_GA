#!/bin/bash -l
#SBATCH --job-name=deploy_inf
# One node with one GPU for the FP32 forward pass over a single EDF.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# GPU partition: deploy_inference.py runs the trained model in FP32.
# A GPU is required -- CPU-only would take many hours per multi-day EDF.
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1

# Single-EDF deployment inference. Replace the python invocation below
# with the EDF path / variant / weights / params for the mouse you want
# to evaluate. Outputs land under the --output-dir; an annotation Excel
# may be passed via --annotations to also compute event-level metrics.
# Wall-time budget: depends on EDF length. Per ~200 h of recording at
# 500 Hz, expect a few hours of FP32 forward pass on V100 / L40S. Bump
# the time below for unusually long EDFs.
#SBATCH -t 1-00:00:00

#SBATCH --output=/home/people/22206468/slurm-deploy_inf-%j.out
##SBATCH --error=/home/people/22206468/slurm-deploy_inf-%j.err

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

# Edit these paths for the deployment run you want to perform.
# Detection only -- emits events.csv with absolute datetime stamps.
# Evaluation against ground truth (if needed) is a separate operation
# handled by the *_evaluation.py scripts.
EDF_PATH="/home/people/22206468/scratch/Raw EDF/m_new.edf"
VARIANT="MultiScaleTCN"
WEIGHTS="/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/multiscale_tcn_final_weights.pt"
PARAMS="/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCNtuning_outputs/best_multiscale_params.json"
OUTDIR="/home/people/22206468/scratch/OUTPUT/DEPLOY/m_new_MultiScaleTCN"
# Optional for *Attention variants:
#   --attn-params /home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT/multiscale_attention_tuning_outputs/best_multiscale_attn_params.json
ATTN_PARAMS=""

CMD="python deploy_inference.py
    --edf            $EDF_PATH
    --variant        $VARIANT
    --weights        $WEIGHTS
    --params         $PARAMS
    --output-dir     $OUTDIR"

if [ -n "$ATTN_PARAMS" ]; then CMD="$CMD --attn-params $ATTN_PARAMS"; fi

echo "Running: $CMD"
$CMD

echo "===== JOB END ====="
date
