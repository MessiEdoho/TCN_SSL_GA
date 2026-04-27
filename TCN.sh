#!/bin/bash -l
#SBATCH --job-name=train_tcn
# One node with one GPU for PyTorch end-to-end training
#SBATCH -N 1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 10

#SBATCH --partition=csgpu

##SBATCH --exclude=sonicgpu20
# Request 1 gpus
#SBATCH --gres=gpu:1

# M1 final training: train the single-branch TCN architecture using the
# best hyperparameters identified by tcn_HPT_binary.py. 100 epochs with
# early stopping patience 10. Produces tcn_final_weights.pt and the full
# three-row evaluation report on the validation partition (test set is
# reserved for final_evaluation.py and is not loaded here).
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

# Reads outputs/best_params.json (M1 best HPs from tuning) and writes
# weights, evaluation report, and post-processing config to outputs/TCN/.
python TCN.py

echo "===== JOB END ====="
date
