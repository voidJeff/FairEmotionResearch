#!/bin/bash

#SBATCH -o      slurm.sh.out
#SBATCH -p      defq
#SBATCH --partition=k80
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

module purge
module load cuda/11.7

python train.py --train_dir ./train_set --val_dir ./val_set --lr 1e-4 -n emotion_baseline_7lab






