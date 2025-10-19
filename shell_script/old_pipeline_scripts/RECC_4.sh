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
module load cplex/12.9

# Figure out the CPLEX binary from the module's env
# Most CPLEX modules export something like CPLEX_STUDIO_DIR129
echo "CPLEX_STUDIO_DIR129=${CPLEX_STUDIO_DIR129:-<unset>}"
if [ -n "${CPLEX_STUDIO_DIR129:-}" ] && [ -x "${CPLEX_STUDIO_DIR129}/cplex/bin/x86-64_linux/cplex" ]; then
  CPEX_BIN="${CPLEX_STUDIO_DIR129}/cplex/bin/x86-64_linux/cplex"
else
  # Fallbacks: try the classic app tree or locate
  for p in \
    /apps/cplex/12.9/cplex/bin/x86-64_linux/cplex \
    /apps/ufrc/cplex/12.9/cplex/bin/x86-64_linux/cplex \
    $(command -v cplex || true); do
    if [ -x "$p" ]; then CPEX_BIN="$p"; break; fi
  done
fi

if [ -z "${CPEX_BIN:-}" ] || [ ! -x "${CPEX_BIN}" ]; then
  echo "ERROR: Cannot find CPLEX binary from the module. Run 'module show cplex/12.9' to see the env var it sets."
  exit 1
fi
echo "Using CPLEX binary: ${CPEX_BIN}"

CPLEX_BIND_ROOT="$(dirname "$(dirname "$(dirname "${CPEX_BIN}")")")"   # -> /apps/cplex/12.9
SIF="/blue/rewetz/vkamineni/RECC_MIP_v14.sif"
Execute_File="/home/vkamineni/Projects/RECC/code/dynamic_pipeline/TestModel.py"
# Execute_File="/home/vkamineni/Projects/RECC/code/test2.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

SLURM_CPUS_PER_TASK=32

# Execute your Python script inside the container
singularity exec \
  --bind "${CPLEX_BIND_ROOT}:${CPLEX_BIND_ROOT}" \
  --env CPEX_BIN="${CPEX_BIN}" \
  --env SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}" \
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 --env VECLIB_MAXIMUM_THREADS=1 \
  --nv "${SIF}" \
  python3 "${Execute_File}"