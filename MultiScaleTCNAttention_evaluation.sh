#!/bin/bash -l
#SBATCH --job-name=eval_msattn
# One node with one GPU for the FP32 forward pass over the held-out test set
#SBATCH -N 1
# Single Python process with 10 CPUs available for DataLoader workers.
# MultiScaleTCNAttention_evaluation.py uses NUM_DATA_WORKERS=4 plus the main
# process, matching the training-script allocation gives I/O headroom on the
# parallel filesystem.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# GPU partition: the test-set forward pass runs the trained
# MultiScaleTCNWithAttention in FP32 (Layer 3 of the four-layer NaN
# protection workflow). A GPU is required -- CPU-only would take days.
#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 GPU
#SBATCH --gres=gpu:1

# M4 final TEST-set evaluation: load ms_attn_final_weights.pt, run an FP32
# forward pass over the held-out test partition (~9.75M segments), then
# compute Row 1 (raw t=0.5) and Row 2 (post-processed t=0.5) metrics via
# the same helpers as the training-script's post-training pass. Outputs
# JSON evaluation report, two-row CSV, per-row classification reports,
# event details, figures (including the M4-only attention saliency
# figure), and the Result_classReport bar plot under
#   /home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT/MultiScaleTCNAttention/evaluation/.
#
# Wall-time estimate: m3_post_eval.sh recorded an ~11h FP32 forward pass
# on the ~4.3M-segment val set; the test set is ~2.3x larger (~9.75M
# segments). M4 has slightly more compute per segment due to the temporal
# attention head, so plan for ~26-30h of inference plus ~30min for
# post-processing, figures, and attention saliency. A 13-day budget
# matches MultiScaleTCN_evaluation.sh and provides ample margin.
#SBATCH -t 13-00:00:00

# SLURM stdout / stderr routed to the home directory (NOT under the M4
# output folder). %j expands to the job ID at submission time so re-runs
# do not overwrite each other.
#SBATCH --output=/home/people/22206468/slurm-eval_msattn-%j.out
##SBATCH --error=/home/people/22206468/slurm-eval_msattn-%j.err

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"

# GPU allocation diagnostics. The other shell scripts in this repo only log
# the device name, which is not enough to debug shared-GPU contention or
# OOM. This block additionally records:
#   - SLURM_JOB_GPUS / CUDA_VISIBLE_DEVICES   (which GPU index was assigned)
#   - nvidia-smi -L                            (canonical "GPU N: <name> (UUID)")
#   - per-GPU index, name, total/free MiB, compute capability, driver version
# Useful when a job is unexpectedly slow, OOMs, or lands on a GPU shared
# with another job.
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

# CPU allocation diagnostics: confirm SLURM gave us 10 cores AND that the
# Python process can actually use all of them (cpuset / cgroup binding).
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
#   MODEL4_OUTPUT/MultiScaleTCNAttention/ms_attn_final_weights.pt
#   MODEL3_OUTPUT/MultiScaleTCNtuning_outputs/best_multiscale_params.json (backbone HPs)
#   MODEL4_OUTPUT/best_multiscale_attn_params.json                        (attention HPs)
#   data_splits_outputs/data_splits_nonictal_sampled_filtered.json (test)
# Writes:
#   MODEL4_OUTPUT/MultiScaleTCNAttention/evaluation/  (JSON, CSVs, NPZ,
#                                                      figures incl. attention
#                                                      saliency, Result_classReport)
#   MODEL4_OUTPUT/MultiScaleTCNAttention/logs/ms_attn_evaluation.log
python MultiScaleTCNAttention_evaluation.py

echo "===== JOB END ====="
date
