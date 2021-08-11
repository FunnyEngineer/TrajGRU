#!/bin/bash
#SBATCH --job-name="TrajGRU"
#SBATCH --partition=rtx2080ti
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --time=3-0:0
#SBATCH --chdir=./
#SBATCH --output=./log/cout.txt
#SBATCH --error=./log/cerr.txt
echo
echo "============================ Messages from Goddess============================"
echo " * Job starting from: "`date`
echo " * Job ID : "$SLURM_JOBID
echo " * Job name : "$SLURM_JOB_NAME
echo " * Job partition : "$SLURM_JOB_PARTITION
echo " * Nodes : "$SLURM_JOB_NUM_NODES
echo " * Cores : "$SLURM_NTASKS
echo " * Working directory: "${SLURM_SUBMIT_DIR/$HOME/"~"}
echo "==============================================================================="
echo

module load python/3.8.10-gpu
#python3 dataset.py
#python3 main.py
python3 mse_main.py
echo
echo "============================ Messages from Goddess============================"
echo " * Jab ended at : "`date`
echo "==============================================================================="
