#!/bin/bash -l
#SBATCH --job-name=enrich_manif
# Single node, no GPU. Reads JSON manifest + per-mouse chronology NPZs,
# writes an enriched manifest with chrono_idx / t_start_sec / mouse_id
# fields added to each val + test record.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# One-off manifest enrichment. Run once after build_chronology.py finishes.
# Re-run only if the splits manifest or chronologies change.
# Wall-time: ~2-5 minutes (JSON load + per-record dict lookups).
#SBATCH -t 0-00:30:00

#SBATCH --output=/home/people/22206468/slurm-enrich_manif-%j.out
##SBATCH --error=/home/people/22206468/slurm-enrich_manif-%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset} (CPU-only manifest enrichment)"

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
# Writes:
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered_enriched.json
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/enrich_manifest.log
python enrich_manifest.py

echo "===== JOB END ====="
date
