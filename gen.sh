#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

# Usage: sbatch gen.sh <dataset> <shot> <start_idx> <seed>
# Example: sbatch gen.sh StanfordCars 8 0 0

export HF_HOME=/path/to/hf_cache

python stable_diffusion_gen.py --dataset $1 --shot $2 --syn_shot 300 --start_idx $3 --seed $4
