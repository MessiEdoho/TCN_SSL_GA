#!/bin/bash -l
#SBATCH --job-name=fn_audit
# CPU-only. Reads cached predictions NPZ + Excel annotations and writes
# one diagnostic CSV row per missed seizure. Cheap.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=cs

# Wall-time estimate: ~2-5 min for val/test (~4M segments), ~15-25 min
# for the full-train partition (~28.7M segments). 1-hour cap is generous.
#SBATCH -t 0-01:00:00

#SBATCH --output=/home/people/22206468/slurm-fn_audit-%j.out
##SBATCH --error=/home/people/22206468/slurm-fn_audit-%j.err

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

# Variant + partition passed as positional arguments.
#   Variants : MultiScaleTCN | MultiScaleTCNWithAttention
#   Partitions: val | test | train
VARIANT="${1:?Usage: sbatch -J <jobname> false_negative_audit.sh <VARIANT> <PARTITION>}"
PARTITION="${2:?Usage: sbatch -J <jobname> false_negative_audit.sh <VARIANT> <PARTITION>}"
echo "Variant   : $VARIANT"
echo "Partition : $PARTITION"

mkdir -p "$HOME/slurm_logs"
ln -sf "/home/people/22206468/slurm-fn_audit-${SLURM_JOB_ID}.out" \
       "$HOME/slurm_logs/fn_audit-${VARIANT}-${PARTITION}-${SLURM_JOB_ID}.out"
echo "Per-variant/partition log: $HOME/slurm_logs/fn_audit-${VARIANT}-${PARTITION}-${SLURM_JOB_ID}.out"

module purge
module load anaconda3
conda activate torch_v100_py310

cd ~/TCN_SSL_GA

# Reads cached predictions NPZ + Excel annotations + mouse metadata.
# Writes:
#   <variant>/evaluation/false_negative_audit_<partition>.{csv,json,log}     (val, test)
#   <variant>/full_train_evaluation/false_negative_audit_train.{csv,json,log} (train)
python false_negative_audit.py --variant "$VARIANT" --partition "$PARTITION"

echo "===== JOB END ====="
date
