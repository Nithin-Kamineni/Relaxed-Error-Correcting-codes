#!/bin/bash
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd


# Load Singularity
module load singularity

SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"
Execute_File="/home/vkamineni/Projects/RECC/code/dynamic_pipeline/PreFormatData.py"

ARTIFACT_PATH=cifar10/resnet18/model_int8_ptq.pth

# Execute your Python script inside the container
srun --cpu-bind=cores --mem-bind=local \
singularity exec \
  --env ARTIFACT_PATH="${ARTIFACT_PATH}" \
  --nv "${SIF}" \
  python3 "${Execute_File}"