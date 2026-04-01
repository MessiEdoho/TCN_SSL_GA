#!/bin/bash -l
#SBATCH --job-name=delete_m291
#SBATCH --output=delete_m291_%j.log
# speficity number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 5

# specify the walltime e.g 10 days 
#SBATCH -t 10-00:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate environment
module purge
module load anaconda3
conda activate uniqureSSLGA

cd ~/TCN_SSL_GA

# Run the Python script

# --- dry run first (no deletion) ---
#python delete_files_by_prefix.py /scratch/22206468/TRAIN_DATA_5/seizure /scratch/22206468/TRAIN_DATA_5/non_seizure --prefix m291

# --- once you have checked the log and are happy, comment out the line above
# --- and uncomment the line below, then resubmit ---
python delete_files_by_prefix.py /scratch/22206468/TRAIN_DATA_5/seizure /scratch/22206468/TRAIN_DATA_5/non_seizure --prefix m291 --confirm

