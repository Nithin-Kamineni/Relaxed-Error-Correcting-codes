#!/bin/bash
#SBATCH --job-name=parfit-63-resnet18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=48   # 24 workers × 2 cores each
#SBATCH --mem=64gb           # scale RAM accordingly
#SBATCH --time=48:00:00
#SBATCH --array=1,3-6
#SBATCH --output=logs/%x.%A_%a.out

# logs/%x.%j.out

date;hostname;pwd


# Load Singularity
module load singularity

SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"
Execute_File="/home/vkamineni/Projects/RECC/code/dynamic_pipeline/dynamic_parallel_payload_process.py"
# Execute_File="/home/vkamineni/Projects/RECC/code/excecution/ortools-test.py"
# Execute_File="/home/vkamineni/Projects/RECC/code/test2.py"

ARTIFACT_PATH=cifar10/resnet18/model_int8_ptq.pth
CODEWORD=63
# Tvalue=2
Tvalue="${SLURM_ARRAY_TASK_ID}"
Approch=parfit  #parfit replace no

# Execute your Python script inside the container
srun --cpu-bind=cores --mem-bind=local \
singularity exec \
  --env ARTIFACT_PATH=${ARTIFACT_PATH} \
  --env CODEWORD=${CODEWORD} \
  --env Tvalue=${Tvalue} \
  --env Approch=${Approch} \
  --nv "${SIF}" \
  python3 "${Execute_File}"