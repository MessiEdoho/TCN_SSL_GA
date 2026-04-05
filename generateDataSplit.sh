#!/bin/bash -l
#SBATCH --job-name=batch_generate_data_splits
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
python generate_data_splits.py --include-test