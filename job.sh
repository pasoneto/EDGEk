#!/bin/bash
#SBATCH --account=project_2009235
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:v100:1

module load pytorch

source /projappl/project_2009235/edge/bin/activate

python3.9 train.py --batch_size 128 --epochs 20000 --save_interval 100
