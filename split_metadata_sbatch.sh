#!/bin/bash
#SBATCH --account=project_2005072
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
##SBATCH --gres=gpu:a100:1,nvme:900
#
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

METADATA=$1
SOURCEDATA=$2
OUTDIR=$3

module load pytorch/1.9
singularity_wrapper exec python3 split_metadata.py $METADATA $SOURCEDATA $OUTDIR
