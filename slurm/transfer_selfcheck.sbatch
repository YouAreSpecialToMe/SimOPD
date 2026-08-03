#!/usr/bin/env bash
#SBATCH -J simopd-selfcheck
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# Verify the code-benchmark harness on a compute node. No GPU: this scores each
# dataset's own canonical solutions, so it needs unit-test execution and nothing else.
#
#   sbatch slurm/transfer_selfcheck.sbatch
#
# Do NOT run this on a login node. It forks a worker per test, each of which pages in
# a share of MBPP+'s 793MB expected-output table -- ~12GB at parallel=8, which on a
# shared login node stalls rather than finishes. Memory is requested accordingly.

set -uo pipefail
cd /home/zz865/pythonProject/SimOPD
source simopd/bin/activate

echo "node $(hostname)  cores $SLURM_CPUS_PER_TASK  mem ${SLURM_MEM_PER_NODE}M"
free -g | head -2

time python scripts/transfer_eval.py --selfcheck --parallel "${PARALLEL:-4}"
echo "exit=$?"
