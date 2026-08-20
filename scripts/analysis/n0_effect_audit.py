#!/usr/bin/env python3
"""N0 对已有结果到底改了什么 —— 四表收据。

问题(用户 2026-08-20):"N0 到底对之前的实验结果有影响吗"。
corr 波正是这个消融:每个臂 = 老臂 + SIMOPD_TERM_EVENT=1(终止符身份修正),
臂自己的 registry 函数 / mask / transform / 聚合都不动。于是 corr 与 mfleet 的
逐步配对就是 N0 的因果读数。

四表:
  T1  vanilla:legacy(k1_rec)vs +N0(k1_termfix),逐步长度/截断/精度
  T2  19 个家族臂:最大公共步上的 Δlen / Δclip / Δacc,对照三种子零假设带
  T3  爆炸时钟:legacy 三种子的 clip>=0.9 首达步 vs corr 现在跑到哪
  T4  终止面板:停止事件密度(N0 的修正机会)与词级 raw vs 事件级 applied

输入(先跑 export_corr.py 与 export_n0_panels.py):
  $D/tmp_export/inloop_corr_vs_mfleet.csv
  $D/tmp_export/n0_term_panels.csv
  $D/exp_patrol/combined_metrics.csv.gz      # 三种子老队,零假设带 + 爆炸时钟
"""
import collections
import csv
import gzip
import os
import statistics as st

D = os.environ.get("SIMOPD_DATA_ROOT", "/mgfs/shared/Group_GY/changhao/simopd_data")
INLOOP = os.path.join(D, "tmp_export", "inloop_corr_vs_mfleet.csv")
PANELS = os.path.join(D, "tmp_export", "n0_term_panels.csv")
BANKED = os.path.join(D, "exp_patrol", "combined_metrics.csv.gz")
MATH500_SIGMA = (0.6 * 0.4 / 500) ** 0.5          # 精度面板的 1sigma,判"动没动"用


def _load_inloop():
    A = collections.defaultdict(dict)
    for r in csv.DictReader(open(INLOOP)):
        cell = A[(r["wave"], r["arm"])].setdefault(int(r["step"]), {})
        for k in ("val_acc", "resp_len", "clip_ratio"):
            if r[k] != "":
                cell[k] = float(r[k])
    return A


def _load_panels():
    P = collections.defaultdict(dict)
    if not os.path.exists(PANELS):
        return P
    for r in csv.DictReader(open(PANELS)):
        cell = P[r["arm"]].setdefault(int(r["step"]), {})
        for k in ("n_stop", "dl", "dl_raw", "is_stop", "len", "clip"):
            if r.get(k) not in ("", None):
                try:
                    cell[k] = float(r[k])
                except ValueError:
                    pass
    return P


def _load_banked():
    """三种子老队:长度(零假设带)与截断(爆炸时钟)。"""
    L = collections.defaultdict(dict)
    C = collections.defaultdict(dict)
    for r in csv.DictReader(gzip.open(BANKED, "rt")):
        try:
            s = int(float(r["step"]))
        except (TypeError, ValueError):
            continue
        v, c = r.get("response_length/mean"), r.get("response_length/clip_ratio")
        if v not in ("", None):
            L[(r["arm"], r["seed"])][s] = float(v)
        if c not in ("", None):
            C[(r["arm"], r["seed"])][s] = float(c)
    return L, C


def smooth(series, step, field=None, win=4):
    """±win 步的均值 —— 单步噪声比我们要读的效应大,不平滑读不出东西。"""
    vals = []
    for t in range(step - win, step + win + 1):
        if t not in series:
            continue
        v = series[t] if field is None else series[t].get(field)
        if v is not None:
            vals.append(v)
    return st.mean(vals) if vals else None


def nearest(series, step, field, win=6):
    cand = [(abs(t - step), series[t][field]) for t in series
            if abs(t - step) <= win and field in series[t]]
    return min(cand)[1] if cand else None


def onset(clip_by_step, thr):
    """首个越过 thr 并且往后 5 步不掉回 0.85*thr 以下的步 —— 一次尖刺不算爆。"""
    for t in sorted(clip_by_step):
        if clip_by_step[t] < thr:
            continue
        if all(clip_by_step.get(u, 0.0) >= thr * 0.85 for u in range(t, t + 6) if u in clip_by_step):
            return t
    return None


def t1_vanilla(A):
    print("=" * 78)
    print("T1  vanilla:legacy(k1_rec) vs +N0(k1_termfix)  —— 唯一旋钮是停止处的读数")
    print("=" * 78)
    print(f"{'step':>5} | {'len_legacy':>10} {'len_N0':>9} {'Δ%':>7} | "
          f"{'clip_L':>7} {'clip_N0':>8} | {'acc_L':>11} {'acc_N0':>11}")
    for s in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 94, 100, 110, 120, 125, 150, 200, 250):
        L = smooth(A[("mfleet", "vanilla")], s, "resp_len")
        N = smooth(A[("corr", "vanilla")], s, "resp_len")
        cL = smooth(A[("mfleet", "vanilla")], s, "clip_ratio")
        cN = smooth(A[("corr", "vanilla")], s, "clip_ratio")
        aL = nearest(A[("mfleet", "vanilla")], s, "val_acc")
        aN = nearest(A[("corr", "vanilla")], s, "val_acc")
        if L is None and N is None:
            continue
        d = f"{100 * (N - L) / L:+.1f}%" if (L and N) else ""
        f = lambda v, p=0: ("%.*f" % (p, v)) if v is not None else "-"   # noqa: E731
        print(f"{s:>5} | {f(L):>10} {f(N):>9} {d:>7} | {f(cL, 3):>7} {f(cN, 3):>8} | "
              f"{f(aL, 3):>11} {f(aN, 3):>11}")


def t2_family(A, band):
    print()
    print("=" * 78)
    print("T2  家族臂:最大公共步上的配对差  (零假设 = 同臂三种子的自然离散)")
    print("=" * 78)
    print(f"零假设带(222 个 (臂,步) 格):中位 {band['p50']:.1f}%  p75 {band['p75']:.1f}%  p90 {band['p90']:.1f}%")
    print(f"精度带:MATH-500 1σ = {MATH500_SIGMA:.3f},±2σ = ±{2 * MATH500_SIGMA:.3f}")
    print(f"{'arm':<22} {'@step':>6} | {'len_L':>7} {'len_N0':>7} {'Δ%':>8} | "
          f"{'clip_L':>7} {'clip_N0':>7} | {'acc_L':>6} {'acc_N0':>7} | verdict")
    rows = []
    for arm in sorted({a for (_, a) in A}):
        if arm in ("vanilla", "n2"):
            continue
        sl, sc = A.get(("mfleet", arm), {}), A.get(("corr", arm), {})
        if not sl or not sc:
            continue
        s = max(t for t in sc if t <= max(sl))
        L, N = smooth(sl, s, "resp_len"), smooth(sc, s, "resp_len")
        if not (L and N):
            continue
        cL, cN = smooth(sl, s, "clip_ratio"), smooth(sc, s, "clip_ratio")
        aL, aN = nearest(sl, s, "val_acc"), nearest(sc, s, "val_acc")
        d = 100 * (N - L) / L
        rows.append((abs(d), arm, s, L, N, d, cL, cN, aL, aN))
    for _, arm, s, L, N, d, cL, cN, aL, aN in sorted(rows, reverse=True):
        f = lambda v, p=0: ("%.*f" % (p, v)) if v is not None else "-"   # noqa: E731
        verdict = ("超出种子带" if abs(d) > band["p90"] else
                   "带边缘" if abs(d) > band["p75"] else "种子带内")
        print(f"{arm:<22} {s:>6} | {f(L):>7} {f(N):>7} {d:>+7.1f}% | {f(cL, 3):>7} {f(cN, 3):>7} | "
              f"{f(aL, 3):>6} {f(aN, 3):>7} | {verdict}")


def t3_clock(A, banked_clip):
    print()
    print("=" * 78)
    print("T3  爆炸时钟:legacy 三种子 clip>=0.9 首达步  vs  +N0 现在跑到哪")
    print("=" * 78)
    print("时钟是每臂近乎确定的常数(种子间 ±2~6 步),所以'越过悬崖还没爆'不是种子噪声。")
    print(f"{'arm':<22} {'legacy onset (3 seeds)':<26} {'+N0 step':>9} {'+N0 clip':>9}  status")
    arms = sorted({a for (a, _) in banked_clip})
    for arm in arms:
        seeds = sorted(s for (x, s) in banked_clip if x == arm)
        o9 = [onset(banked_clip[(arm, s)], 0.9) for s in seeds]
        if not any(o9):
            continue
        sc = A.get(("corr", arm), {})
        if not sc:
            continue
        s = max(sc)
        cN = smooth(sc, s, "clip_ratio")
        ref = st.mean([o for o in o9 if o])
        if cN is None:
            status = "-"
        elif cN >= 0.9:
            status = f"也爆了,晚 {s - ref:+.0f} 步"
        elif cN >= 0.6:
            # 还没到 0.9 但已经在半程往上爬:是"推迟",不是"避免"。这条别读成好消息。
            status = f"正在爆(clip {cN:.2f} 且在爬),已推迟 ~{s - ref:.0f} 步"
        elif s > ref:
            status = f"越过悬崖 +{s - ref:.0f} 步仍未爆"
        else:
            status = f"还差 {ref - s:.0f} 步到判决点"
        print(f"{arm:<22} {str(o9):<26} {s:>9} {cN if cN is not None else float('nan'):>9.3f}  {status}")


def t4_panels(P):
    if not P:
        return
    print()
    print("=" * 78)
    print("T4  终止面板:N0 的修正机会有多少、每次纠正多少")
    print("=" * 78)
    print("n_stop = 学生自己停下来的事件数/micro-batch;survive = len*is_stop ≈ 会自然终止的比例;")
    print("raw = 词级 Δℓ(老臂实际吃到的),applied = 事件级 Δℓ(N0 换上的),gap = 差多少 nat。")
    head = "n_stop survive   raw   appl"
    steps = (10, 50, 100, 150)
    print(f"{'arm':<20} " + " ".join(f"{'step ' + str(s):>28}" for s in steps))
    print(f"{'':<20} " + " ".join(f"{head:>28}" for _ in steps))
    for arm in sorted(P):
        cells = []
        for s in steps:
            n = smooth(P[arm], s, "n_stop")
            fr = smooth(P[arm], s, "is_stop")
            ln = smooth(P[arm], s, "len")
            dl = smooth(P[arm], s, "dl")
            rw = smooth(P[arm], s, "dl_raw")
            if n is None:
                cells.append(f"{'-':>28}")
                continue
            surv = (fr * ln) if (fr and ln) else float("nan")
            cells.append(f"{n:>6.1f} {surv:>6.2f} {rw if rw is not None else float('nan'):>6.1f} "
                         f"{dl if dl is not None else float('nan'):>6.1f}")
        print(f"{arm:<20} " + " ".join(cells))


def main():
    A = _load_inloop()
    P = _load_panels()
    banked_len, banked_clip = _load_banked()
    spreads = []
    for arm in sorted({a for (a, _) in banked_len}):
        seeds = [s for (x, s) in banked_len if x == arm]
        if len(seeds) < 3:
            continue
        for step in (50, 75, 100, 125, 150, 200, 250):
            vals = [smooth(banked_len[(arm, s)], step) for s in seeds]
            vals = [v for v in vals if v]
            if len(vals) < 3:
                continue
            spreads.append(100 * (max(vals) - min(vals)) / st.mean(vals))
    spreads.sort()
    q = lambda f: spreads[min(len(spreads) - 1, int(f * len(spreads)))]   # noqa: E731
    band = {"p50": q(.5), "p75": q(.75), "p90": q(.9), "n": len(spreads)}
    t1_vanilla(A)
    t2_family(A, band)
    t3_clock(A, banked_clip)
    t4_panels(P)


if __name__ == "__main__":
    main()
