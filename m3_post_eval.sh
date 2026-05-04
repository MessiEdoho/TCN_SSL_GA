#!/bin/bash -l
#SBATCH --job-name=m3_post_eval
# specify number of nodes
#SBATCH -N 1
# Single Python process. The CPU-only post-eval pass needs only 1-2 cores
# (no DataLoader workers, no GPU), so cpus-per-task is dropped from 10 to 2.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# CPU partition: m3_post_eval.py now reads cached predictions
# (multiscale_tcn_predictions_raw.npz) instead of re-running model
# inference, so a GPU is no longer required. Run on the CPU partition.
#SBATCH --partition=cs

# (Retired) GPU partition + allocation. Re-enable together with the
# inference block in m3_post_eval.py if a fresh forward pass is ever
# required.
##SBATCH --partition=csgpu
##SBATCH --gres=gpu:1

# M3 post-training evaluation: load cached y_true / y_prob from
# multiscale_tcn_predictions_raw.npz, run Row 1 + Row 2 post-processing
# (Row 3 retired -- threshold optimisation disabled), and write the
# evaluation report, two-row CSV, classification reports, figures, and
# the new Result_classReport bar plot. The 11-hour FP32 forward pass is
# replaced by a ~1-second .npz load, so total wall time is dominated by
# post-processing + figure generation: ~30 min suffices.
#SBATCH -t 0-05:00:00

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
echo "Partition: ${SLURM_JOB_PARTITION:-unset} (CPU-only post-eval)"

# CPU allocation diagnostics.
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
# Outputs (report JSON, two-row CSV, classification reports, figures, and
# Result_classReport/multiscale_tcn_classreport_barplot.png) go under
#   /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/.
python m3_post_eval.py

echo "===== JOB END ====="
date
