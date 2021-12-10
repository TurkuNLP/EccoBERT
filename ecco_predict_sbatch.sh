#!/bin/bash
#SBATCH --account=Project_2000539
#SBATCH --time=02:00:00
##SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=gpusmall
##SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:1,nvme:8
#SBATCH --cpus-per-task=8

module purge
module load pytorch/1.9

export TORCH_EXTENSIONS_DIR=$LOCAL_SCRATCH
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
# export NCCL_IB_DISABLE=1

NGPUS=1
NNODES=1

TOKENIZER=$1 ## tokenizer vocabulary to use
MODEL=$2 ## model to use ('bert', 'bert-very-large', or 'bigbird')
TRAINDATA=$3 ## training data to use
DEVDATA=$4 ## evaluation data to use
OUTDIR=$5 ## directory to which the model is saved

echo "Slurm job ID: $SLURM_JOB_ID"

if [ $# -eq 5 ]
then
    srun python ecco_predict.py $TOKENIZER $MODEL $DEVDATA $OUTDIR --train $TRAINDATA  --gpus $NGPUS --nodes $NNODES 
elif [ $# -eq 6 ]
then
    srun python ecco_predict.py $TOKENIZER $MODEL $DEVDATA $OUTDIR --train $TRAINDATA  --gpus $NGPUS --nodes $NNODES --load_checkpoint $6
fi
