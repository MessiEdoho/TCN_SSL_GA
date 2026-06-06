#!/bin/bash -l
#SBATCH --job-name=enrich_manif_tr
# Single node, no GPU. Reads the UN-DOWNSAMPLED splits manifest plus
# per-mouse chronology NPZs (must already exist for train mice; produced
# by build_chronology.py --include-train) and writes a chronology-enriched
# copy with mouse_id / chrono_idx / t_start_sec on every train + val + test
# record.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# One-off enrichment of the un-downsampled manifest. Run once after
# build_chronology_train.sh finishes. Wall-time: ~10-30 min (~28.7M
# train records get a per-record dict lookup; JSON write dominates).
#SBATCH -t 0-01:00:00

#SBATCH --output=/home/people/22206468/slurm-enrich_manif_tr-%j.out
##SBATCH --error=/home/people/22206468/slurm-enrich_manif_tr-%j.err

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
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json     (UN-DOWNSAMPLED)
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies/{mouse}_chronology.npz
# Writes:
#   /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_full_train_enriched.json
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/enrich_manifest.log  (appended)
python enrich_manifest.py \
    --include-train \
    --input  /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json \
    --output /scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_full_train_enriched.json

echo "===== JOB END ====="
date
