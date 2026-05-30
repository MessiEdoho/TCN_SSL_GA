#!/bin/bash -l
#SBATCH --job-name=evt_saliency
# Tiny GPU job: forward-pass only the windows that overlap TP/FP/FN events
# (~20k windows for the full test set, ~0.2% of the partition), then run
# per-event overlap-add on CPU.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# GPU partition: needs CUDA for the M4 forward pass that produces the
# per-window attention vectors. The compute is small but the model still
# needs to be on the GPU.
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1

# Wall-time budget: typical run ~15-30 min on a V100. 4-hour cap gives
# headroom for both partitions in one job, slow Lustre I/O, or a
# higher-than-expected event count.
#SBATCH -t 0-04:00:00

#SBATCH --output=/home/people/22206468/slurm-evt_saliency-%j.out
##SBATCH --error=/home/people/22206468/slurm-evt_saliency-%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

# -----------------------------------------------------------------------------
# Partition selection. Pass via SLURM: sbatch --export=PARTITION=val event_attention_saliency.sh
# Defaults to test if unset. Accepts {val, test}.
# -----------------------------------------------------------------------------
PARTITION="${PARTITION:-test}"
if [[ "$PARTITION" != "val" && "$PARTITION" != "test" ]]; then
  echo "ERROR: PARTITION must be 'val' or 'test' (got '$PARTITION')" >&2
  exit 2
fi

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition (SLURM)  : ${SLURM_JOB_PARTITION:-unset}"
echo "Partition (script) : ${PARTITION}"

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
#   INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered_enriched.json
#   For PARTITION=val :  MODEL4_OUTPUT/MultiScaleTCNAttention/val_event_details.csv
#   For PARTITION=test:  MODEL4_OUTPUT/MultiScaleTCNAttention/evaluation/test_event_details.csv
#   INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json
#   seizure_times_updated/{mouse}_xlsx.xlsx
# Writes:
#   MODEL4_OUTPUT/MultiScaleTCNAttention/saliency_interprete/${PARTITION}/
#     event_attention_saliency.log
#     event_saliency_curves.png
#     tp_fp_fn_difference.png
#     per_event_saliency.npz
#     event_saliency_summary.json
python event_attention_saliency.py --partition "${PARTITION}"

echo "===== JOB END ====="
date
