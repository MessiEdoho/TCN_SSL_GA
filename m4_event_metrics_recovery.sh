#!/bin/bash -l
#SBATCH --job-name=m4_event_metrics_recovery
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1

# M4 (MultiScaleTCNAttention) val artefact recovery: load
# ms_attn_final_weights.pt, run an FP32 full-val forward pass with
# eval_utils' four-layer NaN protection, then write the chronology-aware
# event-level evaluation artefacts (Row 1 + Row 2, FAR/hr corrected)
# under OUTPUT_ROOT/event_metrics/. Required because the original M4
# training run crashed at AUROC due to FP16 attention-softmax NaN before
# the val artefacts were written.
#SBATCH -t 0-08:00:00

#SBATCH --output=/home/people/22206468/slurm-m4_event_metrics_recovery-%j.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

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

# Persistent log goes to:
#   /home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT/MultiScaleTCNAttention/event_metrics/logs/m4_event_metrics_recovery.log
# All recovery artefacts go to:
#   /home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT/MultiScaleTCNAttention/event_metrics/
python m4_event_metrics_recovery.py

echo "===== JOB END ====="
date
