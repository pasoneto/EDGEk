#!/bin/bash
#SBATCH --account=project_2009235
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --gres=gpu:v100:1

module load pytorch

source /projappl/project_2009235/edge/bin/activate

python3.9 train.py --batch_size 128 --epochs 2 --save_interval 1
