#!/bin/bash
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd

# Load Singularity
module load singularity

SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"
Execute_File="/home/vkamineni/Projects/RECC/code/dynamic_pipeline/OutputsMerging.py"

ARTIFACT_PATH=cifar10/resnet18/model_int8_ptq.pth
CODEWORD=63
Tvalue=2
Approch=no  #parfit replace no

# Execute your Python script inside the container
srun --cpu-bind=cores --mem-bind=local \
singularity exec \
  --env ARTIFACT_PATH=${ARTIFACT_PATH} \
  --env CODEWORD=${CODEWORD} \
  --env Tvalue=${Tvalue} \
  --env Approch=${Approch} \
  --nv "${SIF}" \
  python3 "${Execute_File}"