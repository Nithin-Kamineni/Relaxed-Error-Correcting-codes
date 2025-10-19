#!/bin/bash
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72   # 24 workers × 3 cores each
#SBATCH --mem=128gb           # scale RAM accordingly
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd


# Load Singularity
module load singularity
module load cplex/12.9

# Verify CPU & binding for debugging:
# lscpu | egrep 'Model name|MHz|NUMA|Thread|CPU\(s\):'
# echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
# echo "SLURM_TRES_PER_TASK=${SLURM_TRES_PER_TASK:-<unset>}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4

CPLEX_BIND_ROOT="$(dirname "$(dirname "$(dirname "${CPEX_BIN}")")")"   # -> /apps/cplex/12.9
SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"
Execute_File="/home/vkamineni/Projects/RECC/code/dynamic_pipeline/dynamic_parallel_payload_process.py"
# Execute_File="/home/vkamineni/Projects/RECC/code/excecution/ortools-test.py"
# Execute_File="/home/vkamineni/Projects/RECC/code/test2.py"

# Execute your Python script inside the container
srun --cpu-bind=cores --mem-bind=local \
singularity exec \
  --env OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  --env MKL_NUM_THREADS=${MKL_NUM_THREADS} \
  --env OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} \
  --env NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS} \
  --env BLIS_NUM_THREADS=${BLIS_NUM_THREADS} \
  --nv "${SIF}" \
  python3 "${Execute_File}"