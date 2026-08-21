#!/usr/bin/env python3
"""把 evalq_exp 队列按盘上真相重建,并报告 20 卡规模的产能缺口。

    # 跳板机上,先看不动手
    python scripts/eval_refill_exp.py

    # 真写队列(原子替换)
    python scripts/eval_refill_exp.py --write

    # 常驻,每 20 分钟补一次(新 ckpt 每 25 步落一个)
    setsid nohup python scripts/eval_refill_exp.py --write --watch 1200 > $D/evalq_exp/refill.log 2>&1 &

为什么需要:队列是 2026-08-21 01:38 手工排的一份快照,之后每条臂每 25 步落的新 ckpt
一个都没进去 —— 老的 eval_refill.sh 指向的是另一个队列($D/evalq,DSW 时代)且没在跑。
worker 吃完存量就会空转,而卡是刚从跑满的 lane 交接过来的。

产能的现实(2026-08-22):corr 波跑在 DLC 的 7 个 pod 上,跳板机 ssh 不进 pod,所以
worker 只能由 pod 自己的 corr_wave_fleet.sh 在 lane 跑满时通过 _eval_handoff 拉起。
本脚本因此不"起 worker",而是保证队列里始终有活,并把产能缺口报出来:
  * 每个 pod 有几个 worker 在跑 / 在等卡;
  * 哪些 pod 还没有 _eval_handoff 代码(2026-08-19 之前起的 pod)—— 那些 pod 的
    lane 跑满后卡会像 slot4 那样静静空转,要靠 touch fleet_relaunch_slot<k>_s0 让它
    re-exec 拿到新代码。
"""
import argparse
import glob
import os
import re
import sys
import time
from collections import defaultdict

D = os.environ.get("SIMOPD_STORE", "/mgfs/shared/Group_GY/changhao/simopd_data")
Q = f"{D}/evalq_exp"
EVALS = f"{D}/evals"
CKPT = f"{D}/ckpt/simopd"
BENCH = ("aime24", "aime25", "amc23", "minerva", "math500")

# 优先级:队列是从上往下取的,所以顺序就是调度策略。
#  0 判决臂 —— vanilla_corr 是整个 corr 波的裁决对象,它的每个 ckpt 都要
#  1 本轮新格 —— c5 并集、wave20/21 的 C 族
#  2 corr/n0 波其余
#  3 入库老臂的补漏
def prio(run):
    if run.startswith("vanilla_corr"):
        return 0
    if run.startswith(("c5_union", "c4_rep", "c4_carrier", "c4_hq", "c4_state",
                       "c4_pi_tail_budget_corr", "c2_quantile_budget_corr")):
        return 1
    if "_corr_" in run or "_n0_" in run:
        return 2
    return 3


def scan():
    done = defaultdict(set)          # (run, step) -> {bench}
    for p in glob.glob(f"{EVALS}/*.parquet"):
        m = re.match(r"(.+?)__([a-z0-9]+)__step(\d+)__seed", os.path.basename(p))
        if m:
            done[(m.group(1), int(m.group(3)))].add(m.group(2))

    have = []                        # 盘上真实存在、且 actor/huggingface 齐的 ckpt
    for d in sorted(glob.glob(f"{CKPT}/*/global_step_*")):
        if not os.path.isdir(f"{d}/actor/huggingface"):
            continue
        run = os.path.basename(os.path.dirname(d))
        step = int(d.rsplit("_", 1)[1])
        have.append((run, step))

    claimed = {c for c in os.listdir(f"{Q}/claims")} if os.path.isdir(f"{Q}/claims") else set()
    todo = []
    for run, step in have:
        if len(done[(run, step)]) >= len(BENCH):
            continue
        if f"{run}__{step}" in claimed:
            continue                 # 有人正在跑;claim 过期由 worker 侧处理
        todo.append((prio(run), run, step, len(done[(run, step)])))
    todo.sort(key=lambda t: (t[0], t[1], t[2]))
    return have, done, todo


def workers():
    """从 worker 日志推断产能:跑着的 / 在等卡的 / 已退出的。"""
    out = []
    for p in sorted(glob.glob(f"{D}/corr_wave/evalw_slot*.log")):
        tail = ""
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4000))
                tail = f.read().decode("utf-8", "replace").strip().split("\n")[-1]
        except OSError:
            pass
        age = (time.time() - os.path.getmtime(p)) / 60 if os.path.exists(p) else 1e9
        if "exiting" in tail:
            st = "已退出"
        elif "sitting out" in tail or "busy" in tail:
            st = "等卡"
        elif age > 30:
            st = f"静默 {age:.0f}m"
        else:
            st = "在跑"
        out.append((os.path.basename(p).replace("evalw_", "").replace(".log", ""), st, age))
    return out


def pods_without_handoff():
    """哪些 pod 还没有 _eval_handoff 代码 —— 它们的 lane 跑满后卡会静静空转。"""
    bad = []
    for k in range(7):
        logs = sorted(glob.glob(f"{D}/corr_wave/fleet_slot{k}_s0_*.log"),
                      key=os.path.getmtime)
        if not logs:
            continue
        try:
            txt = open(logs[-1], errors="replace").read()
        except OSError:
            continue
        # 交接行只有新代码才会打;没有 lane 跑满过的 pod 也没有,所以同时看它有没有
        # 跑满的 lane —— 有跑满的却没有交接行,才是真的缺代码。
        finished = "跑满 250 步" in txt
        handoff = "交给 eval worker" in txt
        if finished and not handoff:
            bad.append(k)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="真写 pending.txt(原子替换)")
    ap.add_argument("--watch", type=int, default=0, help="常驻,每 N 秒补一次")
    ap.add_argument("--limit", type=int, default=0, help="只排前 N 项(0=全部)")
    a = ap.parse_args()

    while True:
        have, done, todo = scan()
        cur = 0
        if os.path.exists(f"{Q}/pending.txt"):
            cur = sum(1 for _ in open(f"{Q}/pending.txt"))
        print(f"\n===== {time.strftime('%m-%d %H:%M:%S')} =====")
        print(f"盘上 ckpt {len(have)} 个;已出全部 {len(BENCH)} 个基准的 "
              f"{sum(1 for k in done if len(done[k]) >= len(BENCH))} 个;"
              f"待评 {len(todo)} 个(当前队列 {cur} 行)")
        by = defaultdict(int)
        for p, run, step, n in todo:
            by[p] += 1
        names = {0: "判决臂 vanilla_corr", 1: "本轮新格 c5/c4/c2", 2: "corr/n0 波其余", 3: "入库老臂补漏"}
        for p in sorted(by):
            print(f"   P{p} {names[p]:<22} {by[p]:>4} 项")
        print("   队首 8 项:", ", ".join(f"{r}@{s}" for _, r, s, _ in todo[:8]))

        w = workers()
        run_n = sum(1 for _, st, _ in w if st == "在跑")
        wait_n = sum(1 for _, st, _ in w if st == "等卡")
        print(f"\nworker: {len(w)} 个,{run_n} 在跑 / {wait_n} 等卡 / "
              f"{len(w) - run_n - wait_n} 已退出或静默   (20 卡目标缺 {max(0, 20 - run_n)})")
        for nm, st, age in w:
            print(f"   {nm:<16} {st:<10} {age:>5.0f}m 前")

        bad = pods_without_handoff()
        if bad:
            print(f"\n!! 这些 pod 有 lane 跑满却没有交接代码(卡在空转):slot{bad}")
            print("   修法:touch $D/corr_wave/fleet_relaunch_slot<k>_s0 让它 re-exec 取新脚本;"
                  "代价是同槽健康 lane 回退到最近 ckpt(<=25 步)")

        if a.write:
            lines = [f"{run} {step}\n" for _, run, step, _ in (todo[:a.limit] if a.limit else todo)]
            tmp = f"{Q}/.pending.new"
            with open(tmp, "w") as f:
                f.writelines(lines)
            os.replace(tmp, f"{Q}/pending.txt")   # 原子:worker 整行读,不会读到半截
            print(f"\n已写 {len(lines)} 行 -> {Q}/pending.txt")
        else:
            print("\n(未写;加 --write 才真写队列)")

        if not a.watch:
            break
        sys.stdout.flush()
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
