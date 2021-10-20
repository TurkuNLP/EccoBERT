#!/bin/bash
#SBATCH --account=Project_2005072

##### REAL RUN SMALL

#SBATCH --time=02:00:00
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2,nvme:8
NGPUS=2
NNODES=1


# ##### REAL RUN MEDIUM

# #SBATCH --time=01:00:00
# #SBATCH --partition=gpumedium
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-node=4
# #SBATCH --gres=gpu:a100:4,nvme:8
# NGPUS=4
# NNODES=2


###### GPUTEST RUN

# #SBATCH --time=00:15:00
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=2
# #SBATCH --partition=gputest
# #SBATCH --gres=gpu:a100:2,nvme:8
# NGPUS=2
# NNODES=1


#SBATCH --mem=96G


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

pip3 install --user pytorch-lightning 


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
