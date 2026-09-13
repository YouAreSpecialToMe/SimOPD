#!/usr/bin/env python3
"""Does the terminator fix TEACH stopping, or merely STOP PUNISHING it?

At the position where the student stopped, compare two readings of the same event:

  token-level (legacy k1_rec):  dl_tok = log p_th(eot) - log q_T(eot)
  event-level (k1_termfix):     dl_evt = log p_th(E_S) - log q_T(E_T)

both in the archive's stu-minus-tch convention, so POSITIVE = the student is more
confident than the teacher = the update pushes that token DOWN (a penalty for
stopping, under the PG form where advantage = -(stu-tch)).

  E_S = {151643}            what actually ends a rollout for the Base student
  E_T = {151643, 151645}    what ends a turn for the Instruct teacher

p_th(E_S) comes from stu_lp_last on sequences whose last token is the stop id;
q_T(E_T) = p_last_151643 + p_last_151645 (teacher probs at that same position).

If the fix merely removes a penalty, dl_evt sits near 0 while dl_tok is +26..+69.
If the fix actively teaches stopping, dl_evt goes clearly NEGATIVE.
"""
import os
import sys
import glob
import re
import math
import statistics as st
import pyarrow.parquet as pq

root = sys.argv[1] if len(sys.argv) > 1 else "/home/zz865/opd_pack/extracted"
arms = sys.argv[2:] or ["vanilla_corr_s0", "vanilla_te_s0"]
EOT_P, IME_P = "p_last_151643", "p_last_151645"


def ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


for arm in arms:
    fs = sorted(glob.glob(os.path.join(root, arm, "traj", "summary_*.parquet")),
                key=lambda f: int(re.search(r"summary_(\d+)", f).group(1)))
    if not fs:
        continue
    print("=" * 96)
    print(arm)
    print("%4s %6s %10s %10s %10s %10s %10s %10s" %
          ("step", "n_stop", "p_th(E_S)", "q_T(E_T)", "dl_tok", "dl_evt", "evt_p10", "evt_p90"))
    for f in fs:
        step = int(re.search(r"summary_(\d+)", f).group(1))
        t = pq.read_table(f)
        cn = t.column_names
        if not all(c in cn for c in ("stu_lp_last", "last_is_stop", "dl_last", EOT_P, IME_P)):
            continue
        slp = t.column("stu_lp_last").to_pylist()
        isstop = t.column("last_is_stop").to_pylist()
        dltok = t.column("dl_last").to_pylist()
        pe = t.column(EOT_P).to_pylist()
        pi = t.column(IME_P).to_pylist()

        pth, qt, evt, tok = [], [], [], []
        for s, stop, d, a, b in zip(slp, isstop, dltok, pe, pi):
            if not stop or not ok(s):
                continue
            qa = (a if ok(a) else 0.0) + (b if ok(b) else 0.0)
            if qa <= 0:
                continue
            pth.append(math.exp(s))
            qt.append(qa)
            evt.append(s - math.log(qa))     # log p_th(E_S) - log q_T(E_T)
            if ok(d):
                tok.append(d)
        if not evt:
            continue
        ev = sorted(evt)
        p10 = ev[max(0, int(0.10 * (len(ev) - 1)))]
        p90 = ev[max(0, int(0.90 * (len(ev) - 1)))]
        print("%4d %6d %10.3f %10.3f %10.2f %10.2f %10.2f %10.2f" %
              (step, len(evt), st.mean(pth), st.mean(qt),
               (st.mean(tok) if tok else float("nan")), st.mean(evt), p10, p90))
    print("  dl>0 = student more confident than teacher = update pushes the stop token DOWN")
