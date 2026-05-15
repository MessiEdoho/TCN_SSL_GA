#!/bin/bash -l
#SBATCH --job-name=count_train
# CPU-only, no GPU. JSON read + per-mouse counter pass.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# One-off helper. Reads splits["train"] from the splits manifest and
# writes a per-mouse segment-count CSV to
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/train_per_mouse_counts.csv
# Used downstream for stratifying a k-fold cross-validation by subject
# AND ictal prevalence.
# Wall-time: < 1 minute.
#SBATCH -t 0-00:15:00

#SBATCH --output=/home/people/22206468/slurm-count_train-%j.out
##SBATCH --error=/home/people/22206468/slurm-count_train-%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"

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

python count_train_per_mouse.py

echo "===== JOB END ====="
date
