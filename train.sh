#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

# Usage: sbatch train.sh <dataset> <model> <shot> <seed>
# Example: sbatch train.sh StanfordCars efficientnet_b0 8 0

python train.py --lr 4e-4 --epochs 40 --dataset $1 --model $2 --shot $3 --seed $4
