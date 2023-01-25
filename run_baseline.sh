#!/usr/bin/env/bash

#SBATCH	-o	slurm.sh.out
#SBATCH	-p	defq

python train.py --train_dir ./train_set --val_dir ./val_set --lr 1e-4 -n emotion_baseline