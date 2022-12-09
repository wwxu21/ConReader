#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10M
#SBATCH --partition=A6000
hostname
nvidia-smi

