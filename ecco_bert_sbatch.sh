#!/bin/bash
#SBATCH --account=Project_2005072
#SBATCH --time=36:00:00
##SBATCH --time=00:15:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
##SBATCH --mem=32G
#SBATCH --partition=gpumedium
##SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:4,nvme:8
#SBATCH --cpus-per-task=32

module purge
module load pytorch/1.9

export TORCH_EXTENSIONS_DIR=$LOCAL_SCRATCH
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
export NCCL_IB_DISABLE=1

NGPUS=4
NNODES=4

TOKENIZER=$1 ## tokenizer vocabulary to use
TRAINDATA=$2 ## training data to use
DEVDATA=$3 ## evaluation data to use
OUTDIR=$4 ## directory to which the model is saved

echo "Slurm job ID: $SLURM_JOB_ID"

if [ $# -eq 4 ]
then
    srun python ecco_bert.py $TOKENIZER $TRAINDATA $DEVDATA $OUTDIR --model bert --gpus $NGPUS --nodes $NNODES 
elif [ $# -eq 5 ]
then
    srun python ecco_bert.py $TOKENIZER $TRAINDATA $DEVDATA $OUTDIR --model bert --gpus $NGPUS --nodes $NNODES --load_checkpoint $5 
fi
