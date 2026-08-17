#!/usr/bin/env bash
# 3-step rehearsal for one A-axis arm on a 2-GPU pair, MACHINE-VERDICTED:
# exit 0 = 每条判据通过(deploy/dlc/a_axis_fleet.sh 据此写 .OK 并放行发射),
# exit 1 = 任一判据失败,日志留在 $LOGD/rehearsal_<arm>.log 待查。
#
#   bash deploy/dsw/rehearse_a_axis.sh a4_dagger_anneal 0,1
#
# 判据(自动 grep/解析,与 configs/arms.yaml 各臂 note 的彩排要求一一对应):
#   全部       run_opd_baseline 退出 0;日志无 Traceback
#   a1/a3/a4   gkd_mix armed + cache loaded 横幅;侧带 jsonl 出行,lam_target
#              合法([0,1],a1≈0.5 a3≈1.0);a4 额外要求 lam_target 随步下降
#   a5         a5_aggrevate armed + keys loaded + teacher_registry published
#              横幅;无重复 request id 报错(复核 NEW-ISSUE 3);侧带行满足
#              六计数器和==n_seen(丢序列检测)
#
# 走 _lane.sh 同款 arm.py env 纪律:赋值后 eval,拒绝臂在此死掉,绝不 eval
# 空串跑成 vanilla。
set -euo pipefail
ARM=${1:?arm id}
GPUS=${2:?gpu pair, e.g. 0,1}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
D=/mgfs/shared/Group_GY/changhao/simopd_data
LOGD=$D/a_axis
mkdir -p "$LOGD"
cd "$ROOT"
source simopd/bin/activate
# 共享 HF 缓存 + HF_HUB_OFFLINE=1(429 疫苗)+ WANDB 凭证;详见 a_axis_fleet.sh 同款注释。
[ -f "$ROOT/simopd_env.sh" ] && source "$ROOT/simopd_env.sh"

_arm_env=$(python scripts/arm.py env "$ARM")
eval "$_arm_env"

export EXPERIMENT_NAME="rehearsal_${ARM}"
# run_opd_baseline 的短跑防呆闸(假绿灯事故的产物)拦一切 <250 步的未标记
# 运行;彩排的合法通行证正是 REHEARSAL=1 —— 闸的注释原文:"rehearsals all
# carry a TAG or REHEARSAL, and that is exactly the discriminator"。
# (2026-08-17 实测:缺它则四臂彩排在 step 0 前齐灭。)
export REHEARSAL=1
export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE=offline                 # 彩排不进仪表板,指标看控制台/侧带
export DATA_DIR=$D/simopd_math
export CKPT_ROOT=$D/ckpt                  # 3 步 < SAVE_FREQ,不落盘,只为路径合法
export TOTAL_TRAINING_STEPS=3
export MAX_RESPONSE_LENGTH=16384
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
export PYTHONUNBUFFERED=1

LOG=$LOGD/rehearsal_${ARM}.log
SIDEBAND=/tmp/simopd_gkd_stats_rehearsal_${ARM}.jsonl
rm -f "$SIDEBAND"
echo "rehearsal $ARM on GPUs $GPUS -> $LOG"
rc=0
bash scripts/run_opd_baseline.sh \
    data.seed=0 \
    actor_rollout_ref.rollout.seed=0 \
    > "$LOG" 2>&1 || rc=$?

fail() { echo "REHEARSAL FAIL [$ARM]: $1"; tail -5 "$LOG" | sed 's/^/    /'; exit 1; }

[ "$rc" -eq 0 ] || fail "run exited $rc"
! grep -q 'Traceback (most recent call last)' "$LOG" || fail "Traceback in log"

case "$ARM" in
  a5_aggrevate)
    grep -q 'a5_aggrevate armed' "$LOG"            || fail "a5 wrapper never armed"
    grep -q 'training-prompt keys loaded' "$LOG"   || fail "membership keys not loaded"
    grep -q 'teacher_registry: published' "$LOG"   || fail "teacher handles never published"
    ! grep -iEq 'request.*already|duplicate request' "$LOG" || fail "duplicate request id (NEW-ISSUE 3)"
    ;;
  *)
    grep -q 'gkd_mix armed' "$LOG"                 || fail "gkd_mix wrapper never armed"
    grep -q 'gkd_mix: cache loaded' "$LOG"         || fail "teacher cache not loaded"
    ;;
esac

python - "$ARM" "$SIDEBAND" <<'PY' || exit 1
import json, sys
arm, path = sys.argv[1], sys.argv[2]
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except OSError:
    sys.exit(f"REHEARSAL FAIL [{arm}]: sideband {path} never written")
if not rows:
    sys.exit(f"REHEARSAL FAIL [{arm}]: sideband empty")
if arm == "a5_aggrevate":
    for r in rows:
        s = sum(r.get(k, 0) for k in
                ("mixed", "pure_student", "full_teacher", "cap_full", "degraded", "aborted"))
        if s != r.get("n_seen", -1):
            sys.exit(f"REHEARSAL FAIL [{arm}]: outcome sum {s} != n_seen {r.get('n_seen')} @step {r.get('step')}")
else:
    lams = [r["lam_target"] for r in rows if "lam_target" in r]
    if not lams or not all(0.0 <= x <= 1.0 for x in lams):
        sys.exit(f"REHEARSAL FAIL [{arm}]: lam_target illegal: {lams}")
    if arm == "a4_dagger_anneal" and len(lams) >= 2 and not all(b < a for a, b in zip(lams, lams[1:])):
        sys.exit(f"REHEARSAL FAIL [{arm}]: schedule not descending: {lams}")
    if arm == "a1_gkd_mix0.5" and abs(lams[-1] - 0.5) > 1e-9:
        sys.exit(f"REHEARSAL FAIL [{arm}]: constant lambda drifted: {lams}")
    if arm == "a3_offpolicy" and abs(lams[-1] - 1.0) > 1e-9:
        sys.exit(f"REHEARSAL FAIL [{arm}]: constant lambda drifted: {lams}")
print(f"rehearsal sideband OK [{arm}]: {len(rows)} rows")
PY

echo "REHEARSAL PASS [$ARM]"
