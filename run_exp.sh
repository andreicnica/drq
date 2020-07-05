#!/bin/bash
#SBATCH --array=1-NUM_EXPS
#SBATCH --partition=long  # Ask for unkillable job
#SBATCH --cpus-per-task=6                    # Ask for 2 CPUs
#SBATCH --gres=gpu:pascal:1                          # Ask for 1 GPU
#SBATCH --mem=8Gb                             # Ask for 10 GB of RAM
#SBATCH --time=23:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/nicaandr/slurm_logs/slurm-%j.out  # Write the log on tmp1

# 1. Load your environment
export MJLIB_PATH=/cvmfs/ai.mila.quebec/apps/x86_64/common/mujoco/2.0/bin/libmujoco200.so
export MJKEY_PATH=/cvmfs/config.mila.quebec/etc/mujoco/mjkey.txt
module load anaconda/3
module load pytorch/1.4.0
source $CONDA_ACTIVATE
conda-activate
source activate drq
conda deactivate
source activate drq

rel_path="/network/tmp1/username/PATH_TO_EXP_FOLDER"

echo "Running sbatch array job $SLURM_ARRAY_TASK_ID"

DISABLE_MUJOCO_RENDERING=1 liftoff train_lift.py --max-runs 1 --no-detach ${rel_path}