#!/usr/bin/env bash
# 一次巡检:把"现在到底怎么样"压成一屏。给 cron 每轮读,不做任何改动(只读)。
#   bash slurm/campaign_tick.sh
set -u
cd $SIMOPD_ROOT
. ./simopd_env.sh >/dev/null 2>&1
# 不要依赖环境里的 USER:crontab 拉起时环境是最小集,USER 可能根本没有。
# 2026-09-06 就是这么炸的 —— 看门狗查不到守卫 -> 每 5 分钟多起一个守卫 ->
# 每个守卫起一个调度器 -> 每个调度器 `squeue -u ""` 查不到作业、以为 0/4 ->
# 各自投满 4 个,一共超投到 7 个作业。id -un 不依赖环境,永远对。
ME=$(id -un)
CK=${CKPT_ROOT:-$SIMOPD_STORE/ckpt}/simopd
LOGS=$OPD_LOGS
ROSTER=configs/retrain_roster.tsv
LEDGER=$SIMOPD_STORE/scheduler/dispatched.tsv

# 一个 run "还活着吗"要看每步都写的文件,不能看目录 mtime:run 目录顶层只在每 25 步
# 落 global_step_* 时才变,活跃的 run 也会被判成半小时没动。
_age_min() {   # $1=run 目录 -> 最近一次写盘距今几分钟(没有就 99999)
    local newest
    newest=$(ls -t "$1"/metrics/launch_*.jsonl "$1"/traj/light.jsonl 2>/dev/null | head -1)
    [ -z "$newest" ] && { echo 99999; return; }
    echo $(( ( $(date +%s) - $(stat -c %Y "$newest") ) / 60 ))
}

echo "########## $(date -Is)  campaign tick ##########"

echo "== slurm =="
squeue -u "$ME" -o "%.7i %.14j %.9T %.11M %.11l %R" 2>/dev/null
for j in $(squeue -u "$ME" -h -o %i 2>/dev/null); do
    lim=$(scontrol show job "$j" 2>/dev/null | grep -oE "TimeLimit=[^ ]+" | cut -d= -f2)
    # 365-00:00:00 是分区默认值,等于没有时限;只有真会砍到臂的才值得喊
    case "$lim" in UNLIMITED|3[0-9][0-9]-*|[1-9][0-9][0-9][0-9]-*) continue ;; esac
    run=$(scontrol show job "$j" 2>/dev/null | grep -oE "RunTime=[^ ]+" | cut -d= -f2)
    echo "   !! job $j 有时限 $lim(已跑 $run)—— 长臂会被砍,靠 25 步一档的 resume 兜底"
done

echo "== 我们节点上的空卡 =="
free_total=0
for j in $(squeue -u "$ME" -h -t RUNNING -o %i 2>/dev/null); do
    n=$(squeue -j "$j" -h -o %R 2>/dev/null)
    idle=$(srun --jobid="$j" --overlap nvidia-smi --query-gpu=index,memory.used \
             --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '$2<2000{c++} END{print c+0}')
    [ -z "$idle" ] && idle="?"
    echo "   $n (job $j): 空 $idle / 8"
    [ "$idle" != "?" ] && free_total=$((free_total + idle))
done
echo "   -> 合计空卡 $free_total 张 = $((free_total / 2)) 条 lane 的容量"
echo "   集群里还空着的节点: $(sinfo -h -t idle -o %N 2>/dev/null | tr '\n' ' ')"

echo "== 名册 =="
done_n=0; run_n=0; held_n=0; todo_n=0; todo_list=""
while IFS=$'\t' read -r arm steps prio note; do
    case "$arm" in ''|'#'*) continue ;; esac
    d=$CK/${arm}_s0
    # 分类语义必须分清,否则巡检会照着误报动手(2026-09-06 实测踩过两次):
    #   RUN     台账里有人管 + 最近在写盘
    #   STALLED 台账里有人管 + 30 分钟没写盘   <- 只有这个是真出事
    #   WAIT    不在台账、但已有进度          <- MAX_JOBS 满了在等空位,正常
    #   -       不在台账、也没进度            <- 还没轮到
    in_ledger=0; cut -f1 "$LEDGER" 2>/dev/null | grep -qx "$arm" && in_ledger=1
    # 步数读 metrics 里 step 字段的最大值,不能数行数:FileLogger 每次启动开一个新文件、
    # 续跑不覆盖旧的,行数相加会把续跑过的臂重复计数(c5_union_rkl 真实 103 被报成 112)。
    last=$(cat "$d"/metrics/launch_*.jsonl 2>/dev/null | python3 -c '
import sys, json
m = 0
for l in sys.stdin:
    l = l.strip()
    if l:
        try: m = max(m, json.loads(l).get("step", 0))
        except Exception: pass
print(m)' 2>/dev/null)
    last=${last:-0}
    if [ -d "$d/global_step_$steps" ]; then
        st="DONE"; done_n=$((done_n+1))
    elif [ "$in_ledger" = 1 ]; then
        age=$(_age_min "$d")
        if [ "$age" -lt 30 ]; then st="RUN @${last}/$steps"; run_n=$((run_n+1))
        else st="!! STALLED @${last}/$steps (${age}m 没写盘)"; run_n=$((run_n+1)); fi
    elif [ "$last" -gt 0 ]; then
        st="WAIT @${last}/$steps"; held_n=$((held_n+1))
    else
        st="-"; todo_n=$((todo_n+1)); todo_list="$todo_list $arm:$steps:P$prio"
    fi
    [ "$st" = "-" ] || printf "   %-26s %-22s P%s\n" "$arm" "$st" "$prio"
done < "$ROSTER"
echo "   已完成 $done_n / 在跑 $run_n / 有进度待派工 $held_n / 未开工 $todo_n"
echo "== 未开工(按优先级) =="
echo "$todo_list" | tr ' ' '\n' | grep -v '^$' | sort -t: -k3,3 -k2,2r | sed 's/^/   /'

echo "== 归档层(每个在跑的 run) =="
for d in "$CK"/*_s0; do
    [ -d "$d" ] || continue
    [ "$(_age_min "$d")" -gt 60 ] && continue
    out=$(python scripts/check_archive.py "$d" 2>&1)
    if echo "$out" | grep -q FAIL; then
        echo "   !! $(basename "$d"):"; echo "$out" | grep -E "FAIL" | sed 's/^/      /'
    else
        echo "   ok $(basename "$d"): $(echo "$out" | grep -oE 'light.jsonl +[0-9]+ 行[^,]*, *[0-9]+ 步' | head -1)"
    fi
done

echo "== 出事的 lane(排除已经重投、现在正常在跑的臂) =="
# 一条臂早先失败、后来被调度器重投并且正在跑,那不是待处理的事故 —— 按 12 小时窗口
# 无差别列出来的话,今天那两轮已修的失败(未打标守卫 / V1 脱钩)会一直刷屏,真事故反而
# 淹在里面。只报"现在盘上没有活跃 run 目录"的那些。
_bad=0
for f in "$LOGS"/lanes/*.log; do
    [ -f "$f" ] || continue
    hrs=$(( ( $(date +%s) - $(stat -c %Y "$f") ) / 3600 ))
    [ "$hrs" -gt 12 ] && continue
    hit=""
    for a in $(grep -oE '^[a-z0-9_.]+ +FAIL' "$f" 2>/dev/null | awk '{print $1}'); do
        # 已经重投并且在写盘 = 不用管;在台账里(排在 lane 中等着) = 有人管,也不用管
        [ -d "$CK/$a" ] && [ "$(_age_min "$CK/$a")" -lt 30 ] && continue
        cut -f1 "$LEDGER" 2>/dev/null | grep -qx "${a%_s0}" && continue
        hit="$hit $a"
    done
    [ -n "$hit" ] && { echo "   $(basename "$f")  [${hrs}h 前]:$hit"; _bad=1; }
done
[ "$_bad" = 0 ] && echo "   无(早先的失败都已重投并在跑)"

echo "== 盘 / 缓存 =="
df -BG --output=avail "$SIMOPD_STORE" 2>/dev/null | tail -1 | xargs echo "   可用"
C=${DATA_DIR:?}/gkd_offpolicy.parquet
echo "   gkd 缓存: $([ -f "$C" ] && echo "在 ($(du -h "$C" | cut -f1))" || echo "还没有 —— A 轴 a1/a3/a4 被卡住")"
echo "   a5 keys : $([ -f "$C.dry" ] && echo 在 || echo 缺)"
Q=$SIMOPD_STORE/evalq_exp
echo "   评测队列: $(wc -l < "$Q/pending.txt" 2>/dev/null || echo 0) 条待评, $(ls -A "$Q/claims" 2>/dev/null | wc -l) 在跑, $(ls -A "$Q/failed" 2>/dev/null | wc -l) 有失败记录"
echo "   ckpt 上传: ${CKPT_SYNC_REPO:-<未配置,ckpt 只有一份>}"
