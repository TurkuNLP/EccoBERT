#!/bin/bash
#SBATCH --job-name=tok
#SBATCH --account=project_2005072
#SBATCH --partition=medium
#SBATCH --time=7:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
##SBATCH --gres=gpu:a100:1,nvme:900
#
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

module load pytorch/1.9
singularity_wrapper exec python3 train_tokenizer.py --filelist $1 --N 10000 --out $2
