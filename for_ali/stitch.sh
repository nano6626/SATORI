#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=64000M
#SBATCH --time=0-8:00
python /home/chandoki/projects/def-mzhen/chandoki/rough_alignment/find_matching_points_fullres_50.py
