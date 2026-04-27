#!/bin/bash -l
#SBATCH --job-name=train_tcn_attn
# One node with one GPU for PyTorch end-to-end training
#SBATCH -N 1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 10

#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 gpus
#SBATCH --gres=gpu:1

# M2 final training: TCN backbone + temporal attention, trained from
# scratch with all parameters unfrozen (joint training). Backbone HPs are
# inherited from M1's tuning study; attention HPs from M2's tuning study.
# 100 epochs with early stopping patience 10. Produces
# tcn_attention_final_weights.pt and the three-row evaluation report on
# the validation partition (test set reserved for final_evaluation.py).
# This is where the M1 vs M2 ablation comparison is performed (see
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

# Activate environment
module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Reads outputs/best_params.json (backbone HPs from M1) and
# outputs/best_attention_params.json (attention HPs from M2 tuning), then
# writes weights and evaluation outputs to outputs/TCNAttention/.
python TCNTemporalAttention.py

echo "===== JOB END ====="
date
