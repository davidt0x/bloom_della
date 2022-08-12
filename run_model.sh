#!/bin/bash
#SBATCH --time 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=600G
#SBATCH --gres=gpu:2
#SBATCH --out=logs/%j.out
#SBATCH --reservation=gputest

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1

module load anaconda3/2021.11
. activate bloom
python embed.py

