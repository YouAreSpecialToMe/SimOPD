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
#              横幅;无重复 request id 报错(复核 NEW-ISSUE 3);侧带按步折叠后:
#              闭步末行六计数器和==n_seen(丢序列检测;末步容忍步中快照),
#              sum>n_seen 恒致命;教师参与度硬闸——mixed+full_teacher+tail_tokens
#              全零即 FAIL(第四轮实测:注册表被 Ray GC 后 331/331 静默降级成
#              香草训练还绿灯退出),degraded>10% 亦 FAIL
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
# :- 尊重臂设值(2026-08-18,h7/h8 彩排启用):固定深度臂经 arm.py env 携带自己的
# 响应帽,此前的硬 export 会把它们覆写成香草 16k——彩排跑了个寂寞。A 臂不设此
# 值,行为逐字节不变。
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
export PYTHONUNBUFFERED=1

LOG=$LOGD/rehearsal_${ARM}.log
SIDEBAND=/tmp/simopd_gkd_stats_rehearsal_${ARM}.jsonl
# VERDICT_ONLY=1 重判既有产物(日志+侧带),不重跑 3 步——判据代码修订后
# 对已完成的运行复核用(2026-08-18:wandb offline 拆除段 traceback 曾误伤
# 两个实际全程健康的彩排)。
if [ "${VERDICT_ONLY:-0}" != 1 ]; then
    rm -f "$SIDEBAND"
    rm -f "/tmp/simopd_h_budget_rehearsal_${ARM}.jsonl"   # h9 反向侧带,同样求新
    # 彩排必须从零开始(2026-08-18 实测):verl 完训即存终局 checkpoint,上一轮
    # 3 步彩排留下的 global_step_3 会让 resume_mode=auto 瞬间"完训"退出——
    # 0 步、exit 0 的假绿灯。清掉彩排自己的 ckpt 命名空间(只动 rehearsal_*,
    # 永不触碰真实 run),再显式关闭 resume,双保险。
    rm -rf "$CKPT_ROOT/simopd/rehearsal_${ARM}"
    echo "rehearsal $ARM on GPUs $GPUS -> $LOG"
    rc=0
    bash scripts/run_opd_baseline.sh \
        data.seed=0 \
        actor_rollout_ref.rollout.seed=0 \
        trainer.resume_mode=disable \
        > "$LOG" 2>&1 || rc=$?
else
    echo "verdict-only re-judge of $LOG"
    rc=0
fi

fail() { echo "REHEARSAL FAIL [$ARM]: $1"; tail -5 "$LOG" | sed 's/^/    /'; exit 1; }

[ "$rc" -eq 0 ] || fail "run exited $rc"
# 主判据:三个协议步真的都跑了。Traceback 只在训练段内(最后一个 step 行
# 之前)出现才致命;之后的属拆除噪音(wandb offline teardown 实测会吐
# traceback 而 run 退出 0),记 note 放行。
steps=$(grep -c "step:" "$LOG" || true)
[ "${steps:-0}" -ge 3 ] || fail "only ${steps:-0}/3 training steps in log"
last_step=$(grep -n "step:" "$LOG" | tail -1 | cut -d: -f1)
first_tb=$(grep -n "Traceback (most recent call last)" "$LOG" | head -1 | cut -d: -f1 || true)
if [ -n "$first_tb" ] && [ "$first_tb" -lt "$last_step" ]; then
    fail "Traceback during training (line $first_tb < last step line $last_step)"
elif [ -n "$first_tb" ]; then
    echo "note [$ARM]: teardown-phase traceback tolerated (line $first_tb > last step $last_step, exit 0)"
fi

case "$ARM" in
  a5_aggrevate)
    grep -q 'a5_aggrevate armed' "$LOG"            || fail "a5 wrapper never armed"
    grep -q 'training-prompt keys loaded' "$LOG"   || fail "membership keys not loaded"
    grep -q 'teacher_registry: published' "$LOG"   || fail "teacher handles never published"
    ! grep -iEq 'request.*already|duplicate request' "$LOG" || fail "duplicate request id (NEW-ISSUE 3)"
    ;;
  h6_gen_sched)
    grep -q 'h_horizon armed' "$LOG"               || fail "h_horizon never armed"
    grep -q 'h_horizon: .* training-prompt keys loaded' "$LOG" || fail "membership keys not loaded"
    ;;
  h9_prune_adapt)
    grep -q 'h_horizon armed' "$LOG"               || fail "h_horizon never armed"
    grep -q 'h_horizon: .* training-prompt keys loaded' "$LOG" || fail "membership keys not loaded"
    grep -q 'h9_controller: first budget' "$LOG"   || fail "controller never produced a budget"
    ;;
  h10_task_subset)
    grep -q 'train_sub50.parquet' "$LOG"           || fail "subset parquet absent from launch config"
    ;;
  h7_gen512)
    # 无横幅可验,但配置必须携带臂的帽——恰好抓"launcher 覆写臂设值"这类雷。
    grep -q "max_response_length=512" "$LOG"       || fail "resp cap 512 absent from launch config"
    ;;
  h8_gen2048)
    grep -q "max_response_length=2048" "$LOG"      || fail "resp cap 2048 absent from launch config"
    ;;
  *)
    grep -q 'gkd_mix armed' "$LOG"                 || fail "gkd_mix wrapper never armed"
    grep -q 'gkd_mix: cache loaded' "$LOG"         || fail "teacher cache not loaded"
    ;;
esac

# 无侧带臂(纯 env / 数据臂)跳过侧带判据。
case "$ARM" in
  h7_gen512|h8_gen2048|h10_task_subset)
    echo "rehearsal sideband n/a [$ARM]"
    echo "REHEARSAL PASS [$ARM]"
    exit 0 ;;
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
    # 30s 完成时 flush 带会在步中途拍累积快照(n_seen 提交时计数、结局完成时
    # 落账,快照天然 sum<n_seen)——按步折叠取末行再判:闭步末行必须精确平衡
    # (步界 flush 发生在同步屏障之后),末步容忍快照;sum>n_seen 任何行恒致命。
    by_step = {}
    for r in rows:
        if "n_seen" in r:
            by_step[r.get("step")] = r
    if not by_step:
        sys.exit(f"REHEARSAL FAIL [{arm}]: no a5 rows in sideband")
    OUT = ("mixed", "pure_student", "full_teacher", "cap_full", "degraded", "aborted")
    last = max(by_step)
    seen = deg = teach = 0
    for s in sorted(by_step):
        r = by_step[s]
        tot = sum(r.get(k, 0) for k in OUT)
        ns = r.get("n_seen", -1)
        if tot > ns:
            sys.exit(f"REHEARSAL FAIL [{arm}]: outcome sum {tot} EXCEEDS n_seen {ns} @step {s}")
        if s != last and tot != ns:
            sys.exit(f"REHEARSAL FAIL [{arm}]: outcome sum {tot} != n_seen {ns} @step {s} (closed step must balance)")
        if s == last and tot != ns:
            print(f"note [{arm}]: final step {s} row is a mid-step snapshot ({tot}/{ns}), tolerated")
        seen += r.get("n_seen", 0)
        deg += r.get("degraded", 0)
        teach += r.get("mixed", 0) + r.get("full_teacher", 0) + r.get("tail_tokens", 0)
    # 教师参与度硬闸:全程零教师交付 == 挂着 a5 名字的香草臂(两香草同名事故类)。
    if teach == 0:
        sys.exit(f"REHEARSAL FAIL [{arm}]: teacher route delivered NOTHING "
                 f"(mixed+full_teacher+tail_tokens all zero) -- a5 would train as vanilla")
    if seen and deg / seen > 0.10:
        sys.exit(f"REHEARSAL FAIL [{arm}]: degraded {deg}/{seen} > 10% of eligible -- teacher route unhealthy")
elif arm in ("h6_gen_sched", "h9_prune_adapt"):
    by_step = {}
    for r in rows:
        if "h_target" in r:
            by_step[r.get("step")] = r
    if not by_step:
        sys.exit(f"REHEARSAL FAIL [{arm}]: no h rows in sideband")
    st = [by_step[k]["h_target"] for k in sorted(by_step)]
    tr = sum(r.get("n_train", 0) for r in by_step.values())
    ms = sum(r.get("n_miss", 0) for r in by_step.values())
    if tr <= 0:
        sys.exit(f"REHEARSAL FAIL [{arm}]: zero training rollouts counted")
    if arm == "h6_gen_sched":
        if len(st) >= 2 and not all(b > a for a, b in zip(st, st[1:])):
            sys.exit(f"REHEARSAL FAIL [{arm}]: horizon not ascending across steps: {st}")
    else:
        if not all(256 <= x <= 16384 for x in st):
            sys.exit(f"REHEARSAL FAIL [{arm}]: budget outside [256, 16384]: {st}")
    print(f"rehearsal sideband OK [{arm}]: {len(rows)} rows, h_targets {st}, "
          f"train {tr}, val-exempt {ms}")
    sys.exit(0)
else:
    lams = [r["lam_target"] for r in rows if "lam_target" in r]
    if not lams or not all(0.0 <= x <= 1.0 for x in lams):
        sys.exit(f"REHEARSAL FAIL [{arm}]: lam_target illegal: {lams}")
    # 120s 快照会让同一步产生多行(同 lam)——按步折叠取末行后再验单调。
    by_step = {}
    for r in rows:
        if "lam_target" in r:
            by_step[r.get("step")] = r["lam_target"]
    slams = [by_step[k] for k in sorted(by_step)]
    if arm == "a4_dagger_anneal" and len(slams) >= 2 and not all(b < a for a, b in zip(slams, slams[1:])):
        sys.exit(f"REHEARSAL FAIL [{arm}]: schedule not descending across steps: {slams}")
    if arm == "a1_gkd_mix0.5" and abs(lams[-1] - 0.5) > 1e-9:
        sys.exit(f"REHEARSAL FAIL [{arm}]: constant lambda drifted: {lams}")
    if arm == "a3_offpolicy" and abs(lams[-1] - 1.0) > 1e-9:
        sys.exit(f"REHEARSAL FAIL [{arm}]: constant lambda drifted: {lams}")
print(f"rehearsal sideband OK [{arm}]: {len(rows)} rows")
PY

echo "REHEARSAL PASS [$ARM]"
