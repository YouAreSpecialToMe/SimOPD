#!/usr/bin/env python3
"""AGENT.md section 2.4 的验收清单,机械化版:看文件,不看横幅。

    python scripts/check_archive.py $CKPT_ROOT/simopd/vanilla_corr_s0
    python scripts/check_archive.py <run-dir> --step 25      # 只查某一步

为什么要有这个:归档层的失败语义是"喊一次、训练继续",而 2026-09-04 那次连喊都没喊 ——
verl 0.10.0.dev 把 trainer.use_v1 默认成 true,traj_dump 打的 RayPPOTrainer 根本没被实例化,
于是横幅照打 "traj_dump armed"、verl 自己那份未打补丁的 dump 照写 _verl_text/,而
light.jsonl / ids_*.parquet 一个都没有、div/rank*.jsonl 的 step 全是 null。人眼看日志
看不出来,因为该有的字样一个不缺。这个脚本把"该在盘上的东西"逐条对出来。

退出码 0 = 全过;1 = 有 FAIL。WARN 不影响退出码(比如还没跑到 25 步)。
"""
import argparse
import glob
import json
import os
import sys

FAIL, WARN, OK = "FAIL", "WARN", "ok  "
_rc = 0


def say(status, item, detail=""):
    global _rc
    if status == FAIL:
        _rc = 1
    print(f"[{status}] {item}" + (f"   {detail}" if detail else ""), flush=True)


def _read_jsonl(p, limit=None):
    rows = []
    with open(p) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                      # 末行可能正在写
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--step", type=int, default=None, help="dump 步(默认取盘上最大的 25 的倍数)")
    a = ap.parse_args()
    R = os.path.abspath(a.run_dir)
    T = os.path.join(R, "traj")
    if not os.path.isdir(R):
        sys.exit(f"no such run dir: {R}")
    print(f"=== {R}")

    # ---- 启动期产物 ---------------------------------------------------------
    mani = os.path.join(R, "run_manifest.json")
    if os.path.isfile(mani):
        m = json.load(open(mani))
        sha = ((m.get("git") or {}).get("simopd") or {}).get("sha")
        k = (m.get("eval_protocol") or {}).get("suite_k")
        say(OK if sha else FAIL, "run_manifest.json",
            f"git={(sha or '?')[:10]} suite_k={k} env_keys={len(m.get('env', {}))}")
    else:
        say(FAIL, "run_manifest.json", "缺失 —— run_manifest 没跑")
    mfiles = glob.glob(os.path.join(R, "metrics", "launch_*.jsonl"))
    done_steps = 0
    for f in mfiles:
        try:
            done_steps += sum(1 for line in open(f) if line.strip())
        except OSError:
            pass
    # 一个 metrics 文件都没有,有两种截然不同的情况:
    #   (a) 训练起来了但 FileLogger 没挂上 -> 真故障
    #   (b) run 还没跑完第一步就被砍了(窗口到点、抢占)-> 完全正常
    # 用 traj/ 有没有东西来区分:(b) 的目录里只有启动期产物。2026-09-06 窗口收尾时
    # 三条臂正好卡在这个状态,被误报成 FAIL。
    _bare = not mfiles and not os.path.isdir(T)
    if _bare:
        say(WARN, "metrics/launch_*.jsonl", "0 个 —— 这个 run 还没跑完第一步就停了(不是故障)")
    else:
        say(OK if mfiles else FAIL, "metrics/launch_*.jsonl", f"{len(mfiles)} 个,{done_steps} 步已记")

    # 第 1 步还没跑完时,每步产物本来就该不在 —— 那不是脱钩,别喊狼来了。归档层的每步
    # 文件是在 _log_rollout_data 里写的,而那是一步的最后一段。
    started = done_steps > 0
    STEP_MISS = FAIL if started else WARN
    if not started:
        print("     (还没有任何一步跑完;每步产物的缺失记 WARN,等第 1 步落盘后再验)")

    # ---- 每步产物:light + ids ------------------------------------------------
    light = os.path.join(T, "light.jsonl")
    steps_seen = set()
    if os.path.isfile(light):
        rows = _read_jsonl(light)
        steps_seen = {r.get("step") for r in rows if r.get("step") is not None}
        per = {}
        for r in rows:
            per[r.get("step")] = per.get(r.get("step"), 0) + 1
        sizes = sorted(set(per.values()))
        say(OK if rows and steps_seen else FAIL, "traj/light.jsonl",
            f"{len(rows)} 行,{len(steps_seen)} 步,每步 {sizes}")
    else:
        say(STEP_MISS, "traj/light.jsonl", "缺失 —— traj_dump 的 driver 侧没在写(V1 trainer?)")

    ids = sorted(glob.glob(os.path.join(T, "ids_*.parquet")))
    if ids:
        try:
            import pandas as pd
            n = len(pd.read_parquet(ids[-1], columns=["seq_key"]))
        except Exception as e:
            n = f"unreadable: {e!r}"
        say(OK, "traj/ids_<n>.parquet", f"{len(ids)} 个,最新 {os.path.basename(ids[-1])} {n} 行")
        if steps_seen and len(ids) < len(steps_seen):
            say(WARN, "  ids 每步一档", f"{len(ids)} 档 < {len(steps_seen)} 步")
    else:
        say(STEP_MISS, "traj/ids_<n>.parquet", "一个都没有 —— 契约是每步一档")

    # ---- worker 侧分歧面板 ---------------------------------------------------
    ranks = sorted(glob.glob(os.path.join(T, "div", "rank*.jsonl")))
    div_keys = set()
    if ranks:
        rows = [r for p in ranks for r in _read_jsonl(p)]
        nonnull = [r for r in rows if r.get("step") is not None]
        div_keys = {r.get("seq_key") for r in nonnull}
        # section 2.4 第 2 条明写 "step 非空":步号由 driver 经 meta_info 注入,注不进来
        # 这些行就没法和训练曲线对齐,整个 (2) 类特征作废。
        say(OK if nonnull else FAIL, "traj/div/rank*.jsonl",
            f"{len(ranks)} rank,{len(rows)} 行,其中 step 非空 {len(nonnull)}")
        if rows and not nonnull:
            say(FAIL, "  step 注入", "全是 null —— _update_actor 的补丁没生效")
        # section 2.4 第 3 条:qS_mean 接近 1 = 教师块覆盖住了学生的采样
        qs = [r["qS_mean"] for r in nonnull if isinstance(r.get("qS_mean"), (int, float))]
        if qs:
            avg = sum(qs) / len(qs)
            say(OK if avg > 0.8 else WARN, "  qS_mean(教师块覆盖)",
                f"{avg:.4f}  (接近 1 才对)")
    else:
        say(STEP_MISS, "traj/div/rank*.jsonl", "缺失 —— div_panel 没在写")

    # ---- light ⋈ div 按 seq_key 对得上吗(section 2.4 第 3 条)-------------------
    if os.path.isfile(light) and div_keys:
        lk = {r.get("seq_key") for r in _read_jsonl(light)}
        both = lk & div_keys
        say(OK if both else FAIL, "light ⋈ div by seq_key",
            f"light {len(lk)} / div {len(div_keys)} / 交 {len(both)}")

    # ---- 每 25 步产物 --------------------------------------------------------
    dumps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                   for p in glob.glob(os.path.join(T, "summary_*.parquet")))
    step = a.step if a.step is not None else (dumps[-1] if dumps else None)
    # 一个洞:靠 summary_*.parquet 存不存在来决定查哪一步的话,driver 侧的 25 步档整个
    # 不写也只会记 WARN —— 正是它该抓的那种"安静地少几个文件"。所以先用已完成步数
    # 独立判断:跑过 25 步了却一个 summary 都没有,那就是 FAIL,不是"还没到"。
    EVERY = int(os.environ.get("SIMOPD_TRAJ_EVERY", "25"))
    expect = (done_steps // EVERY) * EVERY
    if step is None:
        if expect >= EVERY:
            say(FAIL, "traj/summary_<n>.parquet",
                f"已跑 {done_steps} 步,第 {expect} 步的 dump 档一个都没有 —— driver 侧没在写")
        else:
            say(WARN, "traj/summary_<n>.parquet", f"还没到 dump 步(每 {EVERY} 步一次),稍后再验")
    else:
        for name in (f"summary_{step}.parquet", f"step_{step}.parquet"):
            p = os.path.join(T, name)
            say(OK if os.path.isfile(p) else FAIL, f"traj/{name}")
        tok = glob.glob(os.path.join(T, "div", f"tok_step{step}_*.parquet"))
        say(OK if tok else FAIL, f"traj/div/tok_step{step}_*", f"{len(tok)} 个")
        sp = os.path.join(T, f"summary_{step}.parquet")
        if os.path.isfile(sp):
            try:
                import pandas as pd
                df = pd.read_parquet(sp)
                if "tch_lp_nan" in df and "resp_len" in df:
                    # tch_lp_nan 是**计数**(该序列有多少个位置的采样 token 不在教师块里),
                    # 不是比例 —— 直接和 0 比会漏掉 70% 缺失这种事。按响应长度归一。
                    n = float(df["tch_lp_nan"].sum())
                    d = float(df["resp_len"].sum()) or 1.0
                    frac = n / d
                    # 高了说明采样列没对上。2026-09-04 就是这么发现教师块差一位的:
                    # 块的位置 j 预测第 j+1 个 token,而查找用的是第 j 个。
                    say(OK if frac < 0.05 else FAIL, "  summary.tch_lp_nan",
                        f"{frac:.3f} 的采样位置查不到教师值 ({n:.0f}/{d:.0f})  (要接近 0)")
            except Exception as e:
                say(WARN, "  summary 读取", repr(e))
        vg = os.path.join(R, "val_gen", f"{step}.jsonl")
        say(OK if os.path.isfile(vg) else WARN, f"val_gen/{step}.jsonl")

    print("=== 全过" if _rc == 0 else "=== 有 FAIL:先修再铺 lane(AGENT.md 2.4)")
    return _rc


if __name__ == "__main__":
    sys.exit(main())
