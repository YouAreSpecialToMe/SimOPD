#!/usr/bin/env bash
# 按 docs/data/rescue_manifest.csv 抢救 ckpt 的 HF 权重(只拷 actor/huggingface,每个约 3.4 GB)。
#
#     bash scripts/rescue_ckpts.sh <目标目录> [tier...]        # 例:bash scripts/rescue_ckpts.sh /rescue R1 R2
#     DRY=1 bash scripts/rescue_ckpts.sh /rescue               # 只打印要拷什么、多大
#
# 为什么只拷这 40 个:全量 ckpt 33 TB、全部欠账曲线 1.02 TB,而"还没跑的评测分别在回答
# 什么问题"决定了性价比 —— R1 判决收口(3 个)、R2 修正波 @200 排名表(25 个)、
# R3 省 token 假说三步(12 个),合计 136 GB 就保住了几乎全部剩余科学价值。
# 目录结构原样保留:下游脚本按 <run>/global_step_<N>/actor/huggingface 找权重。
# 另外别忘了 evals/*.parquet(14 GB)—— 回答原文列只在那里,分析表重建不出来。
set -uo pipefail
DEST=${1:?用法: rescue_ckpts.sh <目标目录> [tier...]}; shift || true
TIERS="${*:-R1 R2 R3}"
D=${SIMOPD_STORE:-/mgfs/shared/Group_GY/changhao/simopd_data}
MAN="$(dirname "$0")/../docs/data/rescue_manifest.csv"
[ -f "$MAN" ] || { echo "!! 找不到清单 $MAN"; exit 1; }
ok=0; miss=0
while IFS=, read -r tier run step why path gb; do
    [ "$tier" = tier ] && continue
    case " $TIERS " in *" $tier "*) ;; *) continue ;; esac
    src="$D/$path"; dst="$DEST/$path"
    if [ ! -d "$src" ]; then echo "MISS  $tier  $src"; miss=$((miss+1)); continue; fi
    if [ -n "${DRY:-}" ]; then echo "would copy $tier  $src  ($(du -sh "$src" 2>/dev/null | cut -f1))"; ok=$((ok+1)); continue; fi
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$(dirname "$dst")/" && { echo "OK    $tier  $run@$step"; ok=$((ok+1)); } || { echo "FAIL  $tier  $src"; miss=$((miss+1)); }
done < "$MAN"
echo "---- $ok 个就位 / $miss 个缺失或失败"
[ -n "${DRY:-}" ] || {
    echo "生成清单校验文件…"
    ( cd "$DEST" && find . -type f -printf '%s %p\n' | sort > MANIFEST.sizes )
    echo "写好 $DEST/MANIFEST.sizes —— 到新集群后用同样命令再生成一份逐行 diff"
}
echo "别忘了:cp -a $D/evals $DEST/   # 14 GB,回答原文列的唯一副本"
