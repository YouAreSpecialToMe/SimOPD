#!/usr/bin/env bash
# 把两个 HF 仓库的下载内容拼成 verl / 评测端认的目录布局。
#
# 为什么需要这一步:一条臂的运行目录 $CKPT_ROOT/simopd/<arm>_s0/ 里同时放着
#   * checkpoint(global_step_*/)          -> 传在 Jerrycool/opd-ckpt
#   * 训练产物与运行元数据(traj/ val_gen/ metrics/ manifest/ run_manifest.json
#     simopd_fingerprint.txt simopd_stop_contract.txt latest_checkpointed_iteration.txt)
#                                          -> 传在 Jerrycool/opd
# 两边必须合回同一个目录,少一边都跑不起来:
#   - 没有 simopd_stop_contract.txt,eval_offline.py 的 resolve_stop_contract() 从
#     --model 路径往上找不到停止契约,评测口径就和产出这批数据时不一致;
#   - 没有 latest_checkpointed_iteration.txt,run_opd_baseline.sh 会当作从 0 开始。
#
# 还有一处形状差异:opd-ckpt 里
#   resume/<arm>/global_step_N/   是 verl 的原样布局(actor/ + actor/huggingface/ + data.pt)
#   eval/<arm>/global_step_N/     是**平铺**的 hf 权重(config.json / model.safetensors / ...)
# 而评测入口写死在 <step>/actor/huggingface(deploy/dsw/eval_worker_exp.sh:93 —— actor/ 是
# FSDP 分片,vLLM 读不了)。所以 eval/ 那一份必须重新嵌套进 actor/huggingface/,直接
# cp 过去评测端是找不到的。
#
# 用法:
#   huggingface-cli download Jerrycool/opd-ckpt              --local-dir dl_ckpt
#   huggingface-cli download Jerrycool/opd --repo-type dataset --local-dir dl_data
#   . ./simopd_env.sh
#   bash scripts/assemble_handoff.sh dl_ckpt dl_data
#
# 幂等:重跑不会重复解包已经解过的臂(除非 FORCE=1)。
set -uo pipefail

CKPT_IN=${1:?用法: assemble_handoff.sh <opd-ckpt 下载目录> <opd 下载目录>}
DATA_IN=${2:?用法: assemble_handoff.sh <opd-ckpt 下载目录> <opd 下载目录>}
DEST=${CKPT_ROOT:?先 source simopd_env.sh}/simopd
FORCE=${FORCE:-0}

command -v zstd >/dev/null || { echo "FATAL: 需要 zstd 才能解 runs/*.tar.zst" >&2; exit 1; }
mkdir -p "$DEST"

# ── 1. 训练产物:解包 runs/<arm>_s0.tar.zst ────────────────────────────────
n_run=0
for t in "$DATA_IN"/runs/*.tar.zst; do
    [ -e "$t" ] || { echo "FATAL: $DATA_IN/runs/ 下没有 tar.zst" >&2; exit 1; }
    arm=$(basename "$t" .tar.zst)
    if [ -d "$DEST/$arm/traj" ] && [ "$FORCE" != 1 ]; then continue; fi
    tar -I zstd -xf "$t" -C "$DEST" || { echo "FATAL: 解包失败 $t" >&2; exit 1; }
    n_run=$((n_run + 1))
done
echo "解包训练产物: $n_run 条臂 -> $DEST"

# ── 2. resume 存档:verl 原样布局,直接搬 ───────────────────────────────────
n_res=0
for d in "$CKPT_IN"/resume/*/global_step_*; do
    [ -d "$d" ] || continue
    arm=$(basename "$(dirname "$d")"); step=$(basename "$d")
    mkdir -p "$DEST/$arm"
    [ -e "$DEST/$arm/$step/actor/model_world_size_1_rank_0.pt" ] && [ "$FORCE" != 1 ] && continue
    rm -rf "$DEST/$arm/$step"
    cp -a "$d" "$DEST/$arm/$step" && n_res=$((n_res + 1))
done
echo "续训存档(带优化器): $n_res 个"

# ── 3. eval 权重:平铺 -> <step>/actor/huggingface/ ────────────────────────
n_ev=0
for d in "$CKPT_IN"/eval/*/global_step_*; do
    [ -d "$d" ] || continue
    arm=$(basename "$(dirname "$d")"); step=$(basename "$d")
    hf="$DEST/$arm/$step/actor/huggingface"
    # resume 那一份已经带了完整的 actor/huggingface,不要用平铺版覆盖它
    [ -e "$hf/model.safetensors" ] && [ "$FORCE" != 1 ] && continue
    mkdir -p "$hf"
    cp -a "$d"/. "$hf"/ && n_ev=$((n_ev + 1))
done
echo "评测权重(重新嵌套进 actor/huggingface): $n_ev 个"

# ── 4. A 轴离策缓存 ───────────────────────────────────────────────────────
if [ -d "$DATA_IN/assets" ]; then
    mkdir -p "${DATA_DIR:?}"
    cp -n "$DATA_IN"/assets/gkd_offpolicy.parquet* "$DATA_DIR"/ 2>/dev/null
    echo "A 轴离策缓存 -> $DATA_DIR/  ($(ls "$DATA_DIR"/gkd_offpolicy.parquet* 2>/dev/null | wc -l) 个文件)"
else
    echo "注意: $DATA_IN/assets/ 不在,A 轴四条臂要先跑 slurm/retrain_gen_offpolicy.sbatch"
fi

# ── 5. 自检 ───────────────────────────────────────────────────────────────
# 分三类,不是"有没有权重"一刀切:
#   * 在 opd-ckpt 里出现过的臂  -> 权重必须齐,缺了就是下载或摆放出了问题
#   * 两边都没出现的臂          -> 是**故意**不传的(训练完 + 评测齐全,权重对完成剩余
#                                  工作没有用,见 opd-ckpt 的 README)。读数在数据集里,
#                                  要复现得重训。不算失败。
echo
echo "== 自检 =="
bad=0; done_noweight=0; okn=0
for d in "$DEST"/*_s0; do
    [ -d "$d" ] || continue
    arm=$(basename "$d")
    miss=""
    [ -f "$d/simopd_stop_contract.txt" ] || miss="$miss stop_contract"
    [ -d "$d/traj" ] || miss="$miss traj/"
    latest=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc 0-9)
    [ -n "$latest" ] || miss="$miss latest_iter"

    want_res=$(ls -d "$CKPT_IN/resume/$arm"/global_step_* 2>/dev/null | wc -l)
    want_ev=$(ls -d "$CKPT_IN/eval/$arm"/global_step_* 2>/dev/null | wc -l)
    got_res=$(ls "$d"/global_step_*/actor/optim_world_size_1_rank_0.pt 2>/dev/null | wc -l)
    got_hf=$(ls -d "$d"/global_step_*/actor/huggingface 2>/dev/null | wc -l)

    [ "$got_res" -lt "$want_res" ] && miss="$miss 续训档($got_res/$want_res)"
    [ "$got_hf" -lt "$want_ev" ]  && miss="$miss 评测权重($got_hf/$want_ev)"

    if [ -n "$miss" ]; then
        printf "  !! %-28s 缺:%s\n" "$arm" "$miss"; bad=$((bad + 1))
    elif [ "$want_res" -eq 0 ] && [ "$want_ev" -eq 0 ]; then
        printf "  -- %-28s 训练完 + 评测齐全,故意未传权重(读数在数据集里)\n" "$arm"
        done_noweight=$((done_noweight + 1))
    else
        printf "  ok %-28s latest=%-4s 可评步数 %2d  可续档 %d\n" "$arm" "$latest" "$got_hf" "$got_res"
    fi
done
echo
echo "  可开工 $(( $(ls -d "$DEST"/*_s0 2>/dev/null | wc -l) - bad - done_noweight )) 条 | 已完结 $done_noweight 条 | 不完整 $bad 条"
[ "$bad" -eq 0 ] || { echo "  先把上面缺的补齐再开跑。"; exit 1; }
echo "  全部通过。下一步见 docs/HANDOFF-20260917.md 的 §7(续训)与 §8(评测)。"
