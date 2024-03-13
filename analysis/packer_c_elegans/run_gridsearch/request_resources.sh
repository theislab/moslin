#!/bin/bash

#SBATCH -n 4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=grid
#SBATCH --output=slurm_logs/grid_%A_%a.out
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --array=0-49

sweep_id="$1"
project_name="$2"

# load software modules
module load eth_proxy gcc/8.2.0 cuda/11.2.2 cudnn/8.8.1.3
source $HOME/.bashrc

# activate a mamba environment
echo 'Activating the environment'
mamba activate moslin

# start the wandb agent
echo "Starting wandb agend with the following sweep id: $sweep_id"
wandb agent "quadbio/$project_name/$sweep_id" --project $project_name

