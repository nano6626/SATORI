#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-7:00
python /home/chandoki/projects/def-mzhen/chandoki/rough_alignment/optim4.py
