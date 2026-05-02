#!/bin/bash -l
#SBATCH --job-name=train_mstcn_attn
# One node with one GPU for PyTorch end-to-end training
#SBATCH -N 1
# Single Python process with 10 CPUs available for DataLoader workers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 gpus
#SBATCH --gres=gpu:1

# M4 final training: Multi-Scale TCN backbone + temporal attention,
# trained from scratch with all parameters unfrozen (joint training).
# Backbone HPs are inherited from M3's tuning study; attention HPs from
# M4's tuning study. 100 epochs with early stopping patience 10. Produces
# ms_attn_final_weights.pt and the three-row evaluation report on the
# validation partition (test set reserved for final_evaluation.py). This
# is where the M3 vs M4 ablation comparison is performed (see
# STUDY_REPORT.txt Section 7.6.7).
#SBATCH -t 3-00:00:00

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

# Reads MODEL3_OUTPUT/.../best_multiscale_params.json (backbone HPs from
# M3) and MODEL4_OUTPUT/best_multiscale_attn_params.json (attention HPs
# from M4 tuning), then writes weights and evaluation outputs to
# outputs/MultiScaleTCNAttention/.
python MultiScaleTCNAttention.py

echo "===== JOB END ====="
date
