#!/bin/bash
#SBATCH --account=Project_2005072
#SBATCH --time=6:00:00
##SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --mem=32G
#SBATCH --partition=gpusmall
##SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:1,nvme:8
#SBATCH --cpus-per-task=32

module purge
module load pytorch/1.9

export TORCH_EXTENSIONS_DIR=$LOCAL_SCRATCH
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
# export NCCL_IB_DISABLE=1

pip install captum --user

NGPUS=1
NNODES=1

LR=$1
TRAINSTEPS=$2
TOKENIZER=$3 ## tokenizer vocabulary to use
MODEL=$4 ## model to use ('bert', 'bert-very-large', or 'bigbird')
TRAINDATA=$5 ## training data to use
DEVDATA=$6 ## evaluation data to use
OUTDIR=$7 ## directory to which the model is saved

echo "Slurm job ID: $SLURM_JOB_ID"

if [ $# -eq 7 ]
then
    srun python ecco_predict.py $LR $TRAINSTEPS $TOKENIZER $MODEL $DEVDATA $OUTDIR --train $TRAINDATA  --gpus $NGPUS --nodes $NNODES
elif [ $# -eq 8 ]
then
    srun python ecco_predict.py $LR $TRAINSTEPS $TOKENIZER $MODEL $DEVDATA $OUTDIR --train $TRAINDATA  --gpus $NGPUS --nodes $NNODES --load_checkpoint $8
fi
