#!/bin/bash -l
#SBATCH --job-name=m3_post_eval
# specify number of nodes
#SBATCH -N 1
# Single Python process with 10 CPUs available for DataLoader workers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

#SBATCH --partition=csgpu

# Request 1 GPU. FP32 final eval needs more memory headroom than AMP,
# but the 447,617-parameter MultiScaleTCN at batch_size=32 fits in any
# Tesla V100 / A100 / L40S allocation comfortably.
#SBATCH --gres=gpu:1

# M3 post-training evaluation: load multiscale_tcn_final_weights.pt and
# run the full-validation pass + three-row post-processing pipeline with
# four-layer NaN protection (input filter + dataset hardening + FP32
# forward + finiteness assert). This recovers the three-row report that
# the original training run aborted before producing. ~14-16 h wall time
# expected: ~11 h for the full-val FP32 pass plus ~30-60 min for post-
# processing and figure generation. Wall-time budget allows for slow
# Lustre/GPFS I/O.
#SBATCH -t 3-00:00:00

# SLURM stdout / stderr routed to the home directory (NOT under any
# individual output folder). %j expands to the job ID at submission time
# so re-runs do not overwrite each other.
#SBATCH --output=/home/people/22206468/slurm-m3_post_eval-%j.out
##SBATCH --error=/home/people/22206468/slurm-m3_post_eval-%j.err

# Email notifications at start, end, and failure
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
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

# Persistent log goes to:
#   /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/logs/m3_post_eval.log
# Outputs (report JSON, three-row CSV, optimal-threshold JSON, figures)
# go to /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/.
python m3_post_eval.py

echo "===== JOB END ====="
date
