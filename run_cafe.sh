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

python train.py --dataset cafe --load_path ./save/train/emotion_baseline_7lab/best.pth.tar --num_epoch 10 --lr 1e-4 -n baseline_cafe_affectnet_transfer





