#!/bin/bash -l
#SBATCH --job-name=m3_min_event_sec_sweep
#SBATCH -N 1
#SBATCH --ntasks=1
# CPU-only -- no model forward pass, no GPU. Sweep is pure post-processing
# on cached predictions: smoothing convolutions, run-detection, event
# matching against Excel annotations. 2 CPUs is plenty.
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# Sweep MIN_EVENT_SEC in {10, 15, 20, 30} on cached M3 test predictions
# to quantify FP-reduction tradeoff. Wall time ~5-15 min: the dominant
# cost is the one-time chronology rebuild (skipped if the enriched
# manifest is used); each sweep iteration thereafter is seconds.
#SBATCH -t 0-02:00:00

#SBATCH --output=/home/people/22206468/slurm-m3_min_event_sec_sweep-%j.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mercy.edoho@ucdconnect.ie

echo "===== JOB START ====="
date
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: ${SLURM_JOB_PARTITION:-unset} (CPU-only)"

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

# Reads (defaults -- override with CLI flags if needed):
#   /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/evaluation/multiscale_tcn_test_predictions_raw.npz
#   /home/people/22206468/scratch/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered_enriched.json
#   /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json
#   /home/people/22206468/scratch/seizure_times_updated/*.xlsx
#
# Writes:
#   /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/evaluation/post_process_varing_sec/
python m3_min_event_sec_sweep.py

echo "===== JOB END ====="
date
