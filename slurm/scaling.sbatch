#!/usr/bin/env bash
# 4 concurrent exclusive 8-GPU nodes = 32 GPUs. Queue order is not a time estimate.
#SBATCH --job-name=simopd-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --array=0-17%4
#SBATCH --output=simopd-scale-%A_%a.log

set -euo pipefail
cd "${SIMOPD_ROOT:?export SIMOPD_ROOT to the reviewed checkout}"
source "${SIMOPD_VENV:-simopd}/bin/activate"
if [[ -f simopd_env.sh ]]; then source simopd_env.sh; fi
read -r teacher student method seed < <(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
sys.path.insert(0, 'scripts')
from scaling import matrix
r = matrix()[int(sys.argv[1])]
print(r['teacher'], r['student'], r['method'], r['seed'])
PY
)
python scripts/scaling.py run --teacher "$teacher" --student "$student" --method "$method" --seed "$seed" --execute
