#!/bin/bash
#SBATCH --account=Project_2005072
#SBATCH --time=36:00:00
##SBATCH --time=00:15:00
#SBATCH --nodes=2
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
NNODES=2

MODEL=$1 ## model to use ('bert', 'bert-very-large', or 'bigbird')
TOKENIZER=$2 ## tokenizer vocabulary to use
TRAINDATA=$3 ## training data to use
DEVDATA=$4 ## evaluation data to use
OUTDIR=$5 ## directory to which the model is saved

echo "Slurm job ID: $SLURM_JOB_ID"

if [ $# -eq 5 ]
then
    srun python ecco_bert.py $TOKENIZER $TRAINDATA $DEVDATA $OUTDIR --model $MODEL --gpus $NGPUS --nodes $NNODES --keep_structure
elif [ $# -eq 6 ]
then
    srun python ecco_bert.py $TOKENIZER $TRAINDATA $DEVDATA $OUTDIR --model $MODEL --gpus $NGPUS --nodes $NNODES --keep_structure --load_checkpoint $6
fi
