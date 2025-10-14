#!/bin/bash
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

set -euo pipefail
date; hostname; pwd

# Fresh environment
module purge
module load python/3.10         # or another version you prefer
module load cplex/12.9

# Create an ephemeral venv in the job dir (safe + repeatable)
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir pulp numpy pandas scipy networkx matplotlib

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

# Tiny smoke test (solves instantly and prints logs)
python - <<PY
import pulp as pl, os
print("PuLP version:", pl.__version__)
x = pl.LpVariable("x", lowBound=0, cat="Integer")
prob = pl.LpProblem("tiny", pl.LpMinimize); prob += x; prob += x >= 10
solver = pl.CPLEX_CMD(
    path=r"${CPEX_BIN}",
    msg=True,
    timeLimit=60,
    options=[
        "set logfile tiny.log",
        "set mip display 4",
        "set parallel 1",
        "set threads ${SLURM_CPUS_PER_TASK}"
    ],
)
stat = prob.solve(solver)
print("Status:", pl.LpStatus[stat], "x=", x.value(), "obj=", pl.value(prob.objective))
PY

echo "OK: CPLEX via CLI is working."
