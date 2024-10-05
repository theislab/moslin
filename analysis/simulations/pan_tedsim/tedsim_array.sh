#!/bin/bash
#SBATCH --job-name=tedsim_arrayjob
#SBATCH --array=0-60%6
#SBATCH -c1
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=4-0
#SBATCH --mem=10gb
#SBATCH --output=tedsim_%a.log


PROJECT_DIR=""

# Specify the path to the config file
config=grid_config.txt

# Extract the seed for the current $SLURM_ARRAY_TASK_ID
seed=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

# Extract the ssr for the current $SLURM_ARRAY_TASK_ID
ssr=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

# Extract the depth for the current $SLURM_ARRAY_TASK_ID
depth=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

source venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/nvidia/cuda/11.8/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

echo "This is array task ${SLURM_ARRAY_TASK_ID}, the seed is ${seed} and ssr is ${ssr}."

python3 ${PROJECT_DIR}/tedsim_fit.py --seed ${seed} --ssr ${ssr} --depth ${depth}
