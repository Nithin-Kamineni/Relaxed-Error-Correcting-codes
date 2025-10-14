#!/bin/bash
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd


# Load Singularity
module load singularity
module load cplex/12.9

# Verify CPU & binding for debugging:
lscpu | egrep 'Model name|MHz|NUMA|Thread|CPU\(s\):'
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
# echo "SLURM_TRES_PER_TASK=${SLURM_TRES_PER_TASK:-<unset>}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"

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
SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"
Execute_File="/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"
# Execute_File="/home/vkamineni/Projects/RECC/code/excecution/ortools-test.py"
# Execute_File="/home/vkamineni/Projects/RECC/code/test2.py"

# SLURM_CPUS_PER_TASK=16

  # --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  # --env NUMEXPR_NUM_THREADS=1 --env VECLIB_MAXIMUM_THREADS=1 \

# list of envs varibles to run
# --- per-task knob you want to pass to Python ---
SUBTASKS_PART=( 1 2 3 4 5 6 7 8 9 10 11 12 )

# list of scripts to run
SCRIPTS=(
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #1
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #2
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #3
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #4
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #5
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #6
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #7
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #8
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #9
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #10
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #11
  "/home/vkamineni/Projects/RECC/code/test_1_Implementation.py"   #12
)

# Execute your Python script inside the container
srun --cpu-bind=cores --mem-bind=local \
singularity exec \
  --bind "${CPLEX_BIND_ROOT}:${CPLEX_BIND_ROOT}" \
  --env CPEX_BIN="${CPEX_BIN}" \
  --env SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}" \
  --env SUBTASKS_PART="${SUBTASKS_PART[$idx]}" \
  --nv "${SIF}" \
  python3 "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"