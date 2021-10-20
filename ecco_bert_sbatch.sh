#!/bin/bash
#SBATCH --account=Project_2005072
##SBATCH --time=00:30:00
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=96G
##SBATCH --partition=gpumedium
#SBATCH --partition=gputest
##SBATCH --gres=gpu:a100:4,nvme:8
#SBATCH --gres=gpu:a100:2,nvme:8
#SBATCH --cpus-per-task=32

module purge
module load pytorch/1.9
#export SING_IMAGE=/appl/soft/ai/singularity/images/pytorch_1.9.0_csc_custom.sif
export SING_IMAGE=/scratch/project_2004600/containers/ds-torch-xx0921.sif

export TORCH_EXTENSIONS_DIR=$LOCAL_SCRATCH
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR
export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1

NGPUS=2
NNODES=1

TOKENIZER=$1 ## tokenizer vocabulary to use (x1, x2, x2CLS)
TRAINDATA=$2 ## training data to use
DEVDATA=$3 ## evaluation data to use
OUTDIR=$4 ## directory to which the model is saved

if [ $# -eq 4 ]
then
    srun python ecco_bert.py --gpus $NGPUS --nodes $NNODES $TOKENIZER $TRAINDATA $DEVDATA $OUTDIR
elif [ $# -eq 5 ]
then
    srun python ecco_bert.py --gpus $NGPUS --nodes $NNODES $TOKENIZER $TRAINDATA $DEVDATA $OUTDIR --load_checkpoint $5
fi
