#!/bin/bash

# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Partition
#SBATCH --partition=M_AND_I_GPU

# nodelist
## SBATCH --nodelist levi

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=2

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=10-00:00:00


# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=tensor2struct
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# ===================
# Run Experiments
# ===================

repo_home=/home/${USER}/tensor2struct-public
cd ${repo_home}
echo "Prepare for runing in ${repo_home}"

# run.py will use this global variable
# export LOG_BASE_DIR=${SCRATCH_HOME}
echo "Logging to ${LOG_BASE_DIR}"

python experiments/semi_sup/run.py semi_train \
    configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
    --config_args "{\"target_domain\": \"calendar\"}" &
python experiments/semi_sup/run.py semi_train \
    configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
    --config_args "{\"target_domain\": \"publications\"}" &
python experiments/semi_sup/run.py semi_train \
    configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
    --config_args "{\"target_domain\": \"recipes\"}" &
python experiments/semi_sup/run.py semi_train \
    configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
    --config_args "{\"target_domain\": \"housing\"}" && fg

# python experiments/semi_sup/run.py semi_train \
#     configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
#     --config_args "{\"target_domain\": \"restaurants\"}" &
# python experiments/semi_sup/run.py semi_train \
#     configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
#     --config_args "{\"target_domain\": \"blocks\"}" &
# python experiments/semi_sup/run.py semi_train \
#     configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
#     --config_args "{\"target_domain\": \"socialnetwork\"}" &
# python experiments/semi_sup/run.py semi_train \
#     configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
#     --config_args "{\"target_domain\": \"basketball\"}" && fg

echo "Experiment finished!"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

src_path=${SCRATCH_HOME}/log
dest_path=${repo_home}/log
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
