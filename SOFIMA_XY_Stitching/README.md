# XY Stitching using SOFIMA

Based on the notebook [https://github.com/google-research/sofima/blob/main/notebooks/em_stitching.ipynb](https://github.com/google-research/sofima/blob/main/notebooks/em_stitching.ipynb). 

## Usage

The code is designed to be run on Compute Canada. To queue a job, use the following shell script.

```
#!/bin/bash
#SBATCH --array=0-1200
#SBATCH --gpus-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=0-00:30
python /home/chandoki/projects/def-mzhen/chandoki/stitching/stitching.py ${SLURM_ARRAY_TASK_ID}
```