#!/bin/bash
#SBATCH --account=Project_2005072
#SBATCH --time=18:00:00
##SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --mem=32G
#SBATCH --partition=gpusmall
##SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:1,nvme:8
#SBATCH --cpus-per-task=32

module purge
# module load pytorch/1.9
module load pytorch/1.11

export TORCH_EXTENSIONS_DIR=$LOCAL_SCRATCH
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
# export NCCL_IB_DISABLE=1

python -m pip --version
python -m pip install captum --user
# pip install git+https://github.com/huggingface/transformers --user

NGPUS=1
NNODES=1

LR=$1
TRAINSTEPS=$2
MODELTYPE=$3
TOKENIZER=$4 ## tokenizer vocabulary to use
MODEL=$5 ## model to use ('bert', 'bert-very-large', or 'bigbird')
TRAINDATA=$6 ## training data to use
DEVDATA=$7 ## evaluation data to use
OUTDIR=$8 ## directory to which the model is saved

echo "Slurm job ID: $SLURM_JOB_ID"

if [ $# -eq 8 ]
then
    srun python ecco_predict.py $LR $TRAINSTEPS $MODELTYPE $TOKENIZER $MODEL $DEVDATA $OUTDIR --train $TRAINDATA  --gpus $NGPUS --nodes $NNODES
elif [ $# -eq 9 ]
then
    srun python ecco_predict.py $LR $TRAINSTEPS $MODELTYPE $TOKENIZER $MODEL $DEVDATA $OUTDIR --train $TRAINDATA  --gpus $NGPUS --nodes $NNODES --load_checkpoint $9
fi
