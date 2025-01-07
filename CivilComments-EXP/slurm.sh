#!/bin/bash

#SBATCH --job-name=wandb_sweep_dsbert
#SBATCH --partition=hpda
#SBATCH --mem=50gb      
#SBATCH -G 1
# SBATCH --gres=gpu:a100_3g.40gb
#SBATCH --time=2-00:00:00       
#SBATCH --output=output/tgt_model_train.out
#SBATCH --error=output/tgt_model_train.err

module load aidl/pytorch/2.2

CUDA_VISIBLE_DEVICES=0 python tcav_apply.py