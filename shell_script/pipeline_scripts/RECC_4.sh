#!/bin/bash
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd


# Load Singularity
module load singularity

CPLEX_BIND_ROOT="$(dirname "$(dirname "$(dirname "${CPEX_BIN}")")")"   # -> /apps/cplex/12.9
SIF="/blue/rewetz/vkamineni/RECC_MIP_v16.sif"
Execute_File="/home/vkamineni/Projects/RECC/code/dynamic_pipeline/TestModel.py"
# Execute_File="/home/vkamineni/Projects/RECC/code/test2.py"

ARTIFACT_PATH=cifar10/resnet18/model_int8_ptq.pth
EvalModel=cifar10/resnet18/M63_t2/no/model_int8_ptq.pth

# Execute your Python script inside the container
singularity exec \
  --env ARTIFACT_PATH=${ARTIFACT_PATH} \
  --env EvalModel=${EvalModel} \
  --nv "${SIF}" \
  python3 "${Execute_File}"