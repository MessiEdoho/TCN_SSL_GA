#!/bin/bash -l
#SBATCH --job-name=eval_mstcn
# One node with one GPU for the FP32 forward pass over the held-out test set
#SBATCH -N 1
# Single Python process with 10 CPUs available for DataLoader workers.
# MultiScaleTCN_evaluation.py uses NUM_DATA_WORKERS=4 plus the main process,
# matching the training-script allocation gives I/O headroom on the
# parallel filesystem.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# GPU partition: the test-set forward pass runs the trained MultiScaleTCN
# in FP32 (Layer 3 of the four-layer NaN protection workflow). A GPU is
# required -- CPU-only would take days.
#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 GPU
#SBATCH --gres=gpu:1

# M3 final TEST-set evaluation: load multiscale_tcn_final_weights.pt,
# run an FP32 forward pass over the held-out test partition (~9.75M
# segments), then compute Row 1 (raw t=0.5) and Row 2 (post-processed
# t=0.5) metrics via the same helpers as the training-script's
# post-training pass. Outputs JSON evaluation report, two-row CSV,
# per-row classification reports, event details, figures, and the
# Result_classReport bar plot under
#   /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/evaluation/.
#
# Wall-time estimate: m3_post_eval.sh recorded an ~11h FP32 forward pass
# on the ~4.3M-segment val set; the test set is ~2.3x larger (~9.75M
# segments), so plan for ~25h of inference plus ~30min for post-processing
# and figures. A 2-day budget provides comfortable margin; bump to 3 days
# if shared-GPU contention slows throughput.
#SBATCH -t 13-00:00:00

# SLURM stdout / stderr routed to the home directory (NOT under the M3
# output folder). %j expands to the job ID at submission time so re-runs
# do not overwrite each other.
#SBATCH --output=/home/people/22206468/slurm-eval_mstcn-%j.out
##SBATCH --error=/home/people/22206468/slurm-eval_mstcn-%j.err

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"
echo "GPU allocated: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

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
#   MODEL3_OUTPUT/MultiScaleTCN/multiscale_tcn_final_weights.pt
#   MODEL3_OUTPUT/MultiScaleTCNtuning_outputs/best_multiscale_params.json
#   data_splits_outputs/data_splits_nonictal_sampled_filtered.json (test)
# Writes:
#   MODEL3_OUTPUT/MultiScaleTCN/evaluation/  (JSON, CSVs, NPZ, figures,
#                                             Result_classReport)
#   MODEL3_OUTPUT/MultiScaleTCN/logs/multiscale_tcn_evaluation.log
python MultiScaleTCN_evaluation.py

echo "===== JOB END ====="
date
