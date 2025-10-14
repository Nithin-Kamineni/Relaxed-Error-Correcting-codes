# submit the array
jid=$(sbatch --parsable RECC_2.sh)

# one summary job runs after the array finishes (success or fail)
sbatch --dependency=afterany:$jid \
  --mail-type=END --mail-user=vkamineni@phhp.ufl.edu \
  --job-name="${jid}_summary" \
  --wrap "
set -euo pipefail
report=\$(sacct -j $jid --format=JobID,JobName%30,State%20,ExitCode,Elapsed,MaxRSS -np)
bad=\$(awk -F'|' '\$3 ~ /(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED)/' <<<\"\$report\")
if [[ -z \"\$bad\" ]]; then
  echo \"SUCCESS: all tasks in array $jid completed on \$(date)

\$report\"
else
  echo \"FAILURE: some tasks in array $jid did not complete OK on \$(date)

\$report\"
fi
"