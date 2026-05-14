#!/bin/bash -l
#SBATCH --job-name=enrich_npz
# Single node, no GPU. Read 2 NPZ files, look up per-segment chronology
# from per-mouse chronology NPZs, write enriched NPZs alongside originals.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# One-off enrichment of cached prediction NPZs (M3 val + M3 test) with
# per-segment mouse_id, chrono_idx, t_start_sec arrays. Original y_true
# and y_prob are preserved. No model inference is run.
# Wall-time: ~5-15 minutes (NPZ load is ~1-3 minutes for the 9.75M-segment
# test NPZ; the rest is fast dict lookups).
#SBATCH -t 0-01:00:00

#SBATCH --output=/home/people/22206468/slurm-enrich_npz-%j.out
##SBATCH --error=/home/people/22206468/slurm-enrich_npz-%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset} (CPU-only NPZ enrichment)"

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

# Reads:
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered.json
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies/{mouse}_chronology.npz
#   /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/multiscale_tcn_predictions_raw.npz
#   /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/evaluation/multiscale_tcn_test_predictions_raw.npz
# Writes:
#   <each input>.with_name("..._enriched.npz")
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/enrich_npz.log
python enrich_npz.py

echo "===== JOB END ====="
date
