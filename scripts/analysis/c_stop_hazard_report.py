"""Aggregate the six C2/C4 hazard-probe conditions into the four tables of the 2026-08-20 design.

    python scripts/analysis/c_stop_hazard_report.py [docs/data]

Conditions per arm: (Y100, theta100), (Y100, theta250), (Y250, theta250); base is forwarded
on every text set. Tables:
  T1  fixed Y100: theta100 vs theta250 vs base at stop / close / body  (did the stopping
      POLICY change 100->250 on the same states?)
  T2  theta250 on Y250 vs base on Y250                                 (does the final
      stopping come from visiting different completion states?)
  T3  correct / wrong / trunc: h(t) bands and Lambda at 2k/4k/6k        (failure = unwilling
      to stop, or never reaching a stop state?)
  T4  C2 vs C4: A_S, A_out, P(F_stop>0), |S_T|, q_cov by region         (why different
      length operating points?)
"""
import glob
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "docs/data"
files = sorted(glob.glob(os.path.join(root, "c_stop_hazard_*_txt*_ckpt*.npz")))
runs = {}
for f in files:
    z = np.load(f, allow_pickle=True)
    arm = str(z["arm"]); Y = int(z["step"]); th = int(z["ckpt_step"])
    runs[(arm, Y, th)] = list(z["per"])
print("loaded:", sorted(runs.keys()))


def band(p, which):
    R = p["R"]
    sl = {"stop": slice(R - 1, R), "close": slice(max(0, R - 201), R - 1), "body": slice(0, max(1, R - 201)),
          "deep": slice(2000, R) if R > 2000 else slice(0, 0)}[which]
    return sl


def agg(per, grp, which, key):
    vals = []
    for p in per:
        if grp != "all" and p["grp"] != grp:
            continue
        if which == "stop" and not p["stopped"]:
            continue
        vals.append(np.asarray(p[key])[band(p, which)])
    return np.concatenate(vals) if vals else np.array([])


def med(x, fmt="{:.3g}"):
    return fmt.format(float(np.median(x))) if x.size else "-"


def share(per, grp, which):
    AS, AO = agg(per, grp, which, "A_S"), agg(per, grp, which, "A_out")
    return f"{np.abs(AS).sum() / max(np.abs(AS).sum() + np.abs(AO).sum(), 1e-12):.2f}" if AS.size else "-"


def lam_at(p, t, key):
    lam = p[key]
    return float(lam[min(t, len(lam)) - 1])


def first_cross(p, key, thr=1.0):
    lam = p[key]
    return int(np.argmax(lam >= thr)) if lam[-1] >= thr else -1


for arm in sorted({k[0] for k in runs}):
    print(f"\n################ {arm}")
    # ---------------- T1: fixed Y100
    if (arm, 100, 100) in runs and (arm, 100, 250) in runs:
        a, b = runs[(arm, 100, 100)], runs[(arm, 100, 250)]
        print(f"\n--- T1 fixed Y100 (n={len(a)}): same states, policy 100 -> 250 (base = column 3)")
        print(f"{'region':<7}{'h_th100':>9}{'h_th250':>9}{'h_base':>9}{'m_100':>8}{'m_250':>8}{'m_base':>8}{'A_S100':>8}{'A_S250':>8}{'A_out100':>9}{'A_out250':>9}{'F>0_100':>8}{'F>0_250':>8}")
        for grp in ("correct", "wrong"):
            for which in ("stop", "close", "body"):
                row = [f"{grp[:4]}/{which}"]
                row += [med(agg(a, grp, which, "h_theta")), med(agg(b, grp, which, "h_theta")), med(agg(a, grp, which, "h_base"))]
                row += [med(agg(a, grp, which, "m_theta"), "{:.3f}"), med(agg(b, grp, which, "m_theta"), "{:.3f}"), med(agg(a, grp, which, "m_base"), "{:.3f}")]
                row += [med(agg(a, grp, which, "A_S"), "{:+.2f}"), med(agg(b, grp, which, "A_S"), "{:+.2f}"), med(agg(a, grp, which, "A_out"), "{:+.2f}"), med(agg(b, grp, which, "A_out"), "{:+.2f}")]
                fa, fb = agg(a, grp, which, "f_stop"), agg(b, grp, which, "f_stop")
                row += [f"{(fa > 0).mean():.2f}" if fa.size else "-", f"{(fb > 0).mean():.2f}" if fb.size else "-"]
                print(f"{row[0]:<12}" + "".join(f"{x:>9}" for x in row[1:]))
        # first-crossing overlap
        for grp in ("correct", "wrong"):
            A = [p for p in a if p["grp"] == grp]; B = [p for p in b if p["grp"] == grp]
            same = sum(1 for x, y in zip(A, B) if first_cross(x, "lam_theta") == first_cross(y, "lam_theta"))
            same0 = sum(1 for x in A if first_cross(x, "lam_theta") == first_cross(x, "lam_base"))
            print(f"  {grp}: t_(Lambda>1) identical theta100 vs theta250: {same}/{len(A)}; theta100 vs base: {same0}/{len(A)}; "
                  f"Lambda(T_obs) med th100={np.median([p['lam_theta'][-1] for p in A]):.2f} th250={np.median([p['lam_theta'][-1] for p in B]):.2f} base={np.median([p['lam_base'][-1] for p in A]):.2f}")
    # ---------------- T2: theta250 on Y250 vs base
    if (arm, 250, 250) in runs:
        c = runs[(arm, 250, 250)]
        print(f"\n--- T2 theta250 on Y250 (n={len(c)}) vs base on the same texts")
        print(f"{'region':<12}{'h_theta':>9}{'h_base':>9}{'m_theta':>9}{'m_base':>9}{'A_S':>8}{'A_out':>8}{'|A_S|sh':>8}{'F>0':>6}{'q_T(E_T)':>9}{'|S_T|':>6}")
        for grp in ("correct", "wrong", "trunc"):
            for which in (("stop", "close", "body") if grp != "trunc" else ("body", "deep")):
                h = agg(c, grp, which, "h_theta")
                if not h.size:
                    continue
                print(f"{grp[:5]+'/'+which:<12}{med(h):>9}{med(agg(c, grp, which, 'h_base')):>9}{med(agg(c, grp, which, 'm_theta'), '{:.3f}'):>9}"
                      f"{med(agg(c, grp, which, 'm_base'), '{:.3f}'):>9}{med(agg(c, grp, which, 'A_S'), '{:+.2f}'):>8}{med(agg(c, grp, which, 'A_out'), '{:+.2f}'):>8}"
                      f"{share(c, grp, which):>8}{(agg(c, grp, which, 'f_stop') > 0).mean():>6.2f}{med(agg(c, grp, which, 'q_ET'), '{:.2g}'):>9}{med(agg(c, grp, which, 'budget'), '{:.0f}'):>6}")
        for grp in ("correct", "wrong"):
            G = [p for p in c if p["grp"] == grp]
            same0 = sum(1 for x in G if first_cross(x, "lam_theta") == first_cross(x, "lam_base"))
            print(f"  {grp}: t_(Lambda>1) identical theta250 vs base: {same0}/{len(G)}; Lambda(T_obs) med theta={np.median([p['lam_theta'][-1] for p in G]):.2f} base={np.median([p['lam_base'][-1] for p in G]):.2f}")
    # ---------------- T3: curves by outcome (theta on its own texts, both conditions)
    for (Y, th) in ((100, 100), (250, 250)):
        if (arm, Y, th) not in runs:
            continue
        c = runs[(arm, Y, th)]
        print(f"\n--- T3 Y{Y}/theta{th}: cumulative hazard by outcome (median over trajectories)")
        print(f"{'group':<9}{'n':>3}{'R_med':>7}" + "".join(f"{'L_th('+str(t)+')':>10}{'L_0('+str(t)+')':>9}" for t in (1000, 2000, 4000, 6000)) + f"{'q_T(E_T)stop':>13}{'q_T(E_T)deep':>13}")
        for grp in ("correct", "wrong", "trunc"):
            G = [p for p in c if p["grp"] == grp]
            if not G:
                continue
            row = f"{grp:<9}{len(G):>3}{np.median([p['R'] for p in G]):>7.0f}"
            for t in (1000, 2000, 4000, 6000):
                row += f"{np.median([lam_at(p, t, 'lam_theta') for p in G]):>10.3f}{np.median([lam_at(p, t, 'lam_base') for p in G]):>9.3f}"
            qs = [p["q_ET"][-1] for p in G if p["stopped"]]
            qd = agg(c, grp, "deep", "q_ET")
            row += f"{(np.median(qs) if qs else float('nan')):>13.2g}{(np.median(qd) if qd.size else float('nan')):>13.2g}"
            print(row)
# ---------------- T4: C2 vs C4 on matched conditions
print("\n--- T4 C2 vs C4 (theta250 on Y250): mechanism knobs by region")
print(f"{'arm/region':<24}{'A_S':>8}{'A_out':>8}{'|A_S|sh':>8}{'F>0':>6}{'|S_T|':>7}{'q_cov':>7}{'m_theta':>9}{'h_theta':>9}")
for arm in sorted({k[0] for k in runs}):
    if (arm, 250, 250) not in runs:
        continue
    c = runs[(arm, 250, 250)]
    for which in ("stop", "close", "body"):
        h = agg(c, "all", which, "h_theta")
        if not h.size:
            continue
        print(f"{arm[:12]+'/'+which:<24}{med(agg(c, 'all', which, 'A_S'), '{:+.2f}'):>8}{med(agg(c, 'all', which, 'A_out'), '{:+.2f}'):>8}{share(c, 'all', which):>8}"
              f"{(agg(c, 'all', which, 'f_stop') > 0).mean():>6.2f}{med(agg(c, 'all', which, 'budget'), '{:.1f}'):>7}{med(agg(c, 'all', which, 'q_cov'), '{:.3f}'):>7}"
              f"{med(agg(c, 'all', which, 'm_theta'), '{:.3f}'):>9}{med(h):>9}")
