#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

# Usage: sbatch finetune.sh <model> <dataset> <shot> <seed> <save_name>
# Example: sbatch finetune.sh ViT-L/14 StanfordCars 8 0 stanfordcars_vitl14

python finetune.py --mode last_layer --model $1 --dataset $2 --shot $3 \
  --seed $4 --save_name $5 --lr 1e-4 --epochs 40
