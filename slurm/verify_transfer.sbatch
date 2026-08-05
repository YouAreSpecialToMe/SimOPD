#!/usr/bin/env bash
#SBATCH -J simopd-xferverify
#SBATCH -p nlplarge-sasha-highpri
#SBATCH -w nlplarge-compute-01
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH -o /home/zz865/pythonProject/SimOPD/logs/%x-%j.out

# Run the transfer column end to end, once, before sixteen arms depend on it.
#
# Its scoring halves are verified -- HumanEval+ 163/164 and MBPP+ 376/378, twice,
# deterministic -- but vLLM generate -> official harness -> parquet has never executed.
# Every GPU attempt was blocked: first by the account's 2-GPU cap while vanilla_s0 held
# both, then by the checkpoint having no loadable weights at all (verl's save_contents
# omitted 'hf_model', fixed 2026-08-05).
#
# The probe checkpoint predates that fix, so its FSDP shards are merged here first.
# That also exercises the recovery path eval_offline.py now prints when it meets an
# unmerged checkpoint -- worth knowing it actually works before recommending it.
#
# This is the last unexercised link in the eval pipeline. Finding it broken now costs
# an afternoon; finding it broken after the campaign costs the campaign.

set -uo pipefail
cd /home/zz865/pythonProject/SimOPD
source simopd/bin/activate
export VERL_USE_MODELSCOPE=False VLLM_USE_MODELSCOPE=False
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1

CKPT=$(ls -d /scratch/zz865/simopd/ckpt/simopd/probe_1p7b_from_8b/global_step_*/actor 2>/dev/null |
       sort -t_ -k3 -n | tail -1)
[ -n "$CKPT" ] || { echo "no probe checkpoint on this node's /scratch (it is node-local)" >&2; exit 1; }
HF="$CKPT/huggingface"
echo "checkpoint: $CKPT"

if ls "$HF"/*.safetensors >/dev/null 2>&1; then
    echo "already merged"
else
    echo "=== merging FSDP shards -> HF format ==="
    time python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$HF" || {
        echo "merge failed -- then the recovery advice eval_offline.py prints is wrong" >&2
        echo "and needs correcting before anyone follows it." >&2; exit 1; }
    ls -la "$HF" | tail -5
fi

echo
echo "=== transfer column, the whole path ==="
time python scripts/eval_offline.py \
    --model "$CKPT" --benchmarks humanevalplus,mbppplus,ifeval \
    --run-id probe_1p7b_from_8b --step 50 \
    --temperature 0 --n 1 --max-tokens 2048 --parallel 4 \
    --out-dir /home/zz865/data/simopd_evals
rc=$?

echo
echo "=== what landed ==="
python - <<'PY'
import glob, os, pandas as pd
for f in sorted(glob.glob("/home/zz865/data/simopd_evals/probe_1p7b_from_8b__*.parquet"),
                key=os.path.getmtime)[-3:]:
    df = pd.read_parquet(f)
    per = df.groupby("problem_id")["correct"].mean()
    extra = ""
    if "n_instructions" in df:
        extra = f"  strict_instr={df['n_instructions_followed'].sum()/df['n_instructions'].sum():.4f}"
    if "base_correct" in df:
        extra = f"  base={df.groupby('problem_id')['base_correct'].mean().mean():.4f}"
    print(f"  {df['benchmark'].iloc[0]:14s} n={len(per):4d}  score={per.mean():.4f}{extra}"
          f"  trunc={df['truncated'].mean():.3f}  cols={len(df.columns)}")
PY
echo "exit=$rc"
