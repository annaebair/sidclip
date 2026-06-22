#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

# Usage: sbatch distill.sh <model> <dataset> <shot> <seed> <teacher>
# Example: sbatch distill.sh TimEfNetb0 StanfordCars 8 0 ViT-L/14

python distill.py --epochs 40 --lr 8e-6 --dist_method kl --batch_size 64 \
  --syn_epochs 30 --model $1 --dataset $2 --shot $3 --syn_shot 300 \
  --seed $4 --teacher $5
