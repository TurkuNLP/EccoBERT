#!/bin/bash
#SBATCH --account=Project_2005072
#SBATCH --time=01:00:00
##SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --mem=96G
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4,nvme:8
#SBATCH --cpus-per-task=32

module load pytorch/1.9
export TORCH_EXTENSIONS_DIR=$LOCAL_SRATCH
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
export NCCL_IB_DISABLE=1

TOKENIZER=$1 ## tokenizer vocabulary to use (x1, x2, x2CLS)
TRAINDATA=$2 ## training data to use
DEVDATA=$3 ## evaluation data to use
OUTDIR=$4 ## directory to which the model is saved

if [ $# -eq 4 ]
then
    srun python ecco_bert.py $TOKENIZER $TRAINDATA $DEVDATA $OUTDIR
elif [ $# -eq 5 ]
then
    srun python ecco_bert.py $TOKENIZER $TRAINDATA $DEVDATA $OUTDIR --load_checkpoint $5
fi
