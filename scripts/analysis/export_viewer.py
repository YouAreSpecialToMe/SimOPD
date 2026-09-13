#!/usr/bin/env python3
"""Export a compact JSON for the trajectory viewer.

Per arm: the step-level series (length / acc / truncation / dl_last / rep4).
Per (arm, step): a few full sequences, each as
  - a downsampled whole-sequence trace of r_t, entropy and q_T(im_end)
  - token-level detail for a first window, a last window, and the window around
    the position where the teacher most wanted to end

r_t = stu_lp - tch_lp is the archive convention and is the quantity the loss
averages, so it is what the colouring shows.
"""
import glob
import json
import math
import os
import re
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/zz865/opd_pack/extracted"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/zz865/viewer_data.json"
ARMS = sys.argv[3:] or ["vanilla_corr_s0", "g1_verified_only_corr_s0",
                        "d3_teachability_corr_s0", "h1_first_segment_corr_s0"]

N_SEQ = 3          # sequences kept per (arm, step)
BINS = 400         # whole-sequence trace resolution
WIN = 220          # tokens of detail per window
IMEND = 151645

tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
_cache = {}


def tok_text(i):
    s = _cache.get(i)
    if s is None:
        s = tk.decode([i])
        _cache[i] = s
    return s


def ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def r2(x):
    return None if x is None else round(float(x), 2)


def downsample(vals, n):
    if not vals:
        return []
    if len(vals) <= n:
        return [r2(v) for v in vals]
    step = len(vals) / n
    out = []
    for k in range(n):
        a, b = int(k * step), max(int(k * step) + 1, int((k + 1) * step))
        seg = [v for v in vals[a:b] if v is not None]
        out.append(r2(sum(seg) / len(seg)) if seg else None)
    return out


def window(ids, r, ent, qime, lo, hi):
    toks = []
    for j in range(max(0, lo), min(len(ids), hi)):
        toks.append([tok_text(ids[j]), r2(r[j]), r2(ent[j]), r2(qime[j])])
    return {"o": max(0, lo), "toks": toks}


data = {"meta": {"root": ROOT, "r": "stu_lp - tch_lp", "bins": BINS, "win": WIN},
        "arms": {}}

for arm in ARMS:
    adir = os.path.join(ROOT, arm)
    if not os.path.isdir(adir):
        continue
    name = arm[:-3] if arm.endswith("_s0") else arm
    entry = {"steps": [], "series": {k: [] for k in
                                     ("len", "trunc", "dl_last", "rep4", "stop", "q_ime")},
             "samples": {}}

    sums = sorted(glob.glob(os.path.join(adir, "traj", "summary_*.parquet")),
                  key=lambda f: int(re.search(r"summary_(\d+)", f).group(1)))
    for f in sums:
        step = int(re.search(r"summary_(\d+)", f).group(1))
        t = pq.read_table(f)
        col = {c: t.column(c).to_pylist() for c in t.column_names}
        n = len(col.get("resp_len", []))
        if not n:
            continue

        def mean(k):
            v = [x for x in col.get(k, []) if ok(x)]
            return r2(sum(v) / len(v)) if v else None

        entry["steps"].append(step)
        entry["series"]["len"].append(mean("resp_len"))
        entry["series"]["trunc"].append(r2(sum(1 for x in col.get("truncated", []) if x) / n))
        entry["series"]["stop"].append(r2(sum(1 for x in col.get("last_is_stop", []) if x) / n))
        entry["series"]["dl_last"].append(mean("dl_last"))
        entry["series"]["rep4"].append(mean("rep4"))
        entry["series"]["q_ime"].append(mean("p_last_151645"))

    steps_f = sorted(glob.glob(os.path.join(adir, "traj", "step_*.parquet")),
                     key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    for f in steps_f:
        step = int(re.search(r"step_(\d+)", f).group(1))
        t = pq.read_table(f)
        cn = t.column_names
        if not all(c in cn for c in ("response_ids", "stu_lp", "tch_lp")):
            continue
        rid = t.column("response_ids").to_pylist()
        slp = t.column("stu_lp").to_pylist()
        tlp = t.column("tch_lp").to_pylist()
        ent = t.column("ent").to_pylist() if "ent" in cn else [[]] * len(rid)
        qim = t.column("tch_lp_151645").to_pylist() if "tch_lp_151645" in cn else [[]] * len(rid)
        gt = t.column("gt").to_pylist() if "gt" in cn else [""] * len(rid)
        sc = t.column("score").to_pylist() if "score" in cn else [None] * len(rid)
        tr = t.column("truncated").to_pylist() if "truncated" in cn else [None] * len(rid)

        picks = []
        for k in range(len(rid)):
            if rid[k] and slp[k] and tlp[k]:
                picks.append(k)
            if len(picks) >= N_SEQ:
                break

        out = []
        for k in picks:
            ids = rid[k]
            L = min(len(ids), len(slp[k]), len(tlp[k]))
            ids = ids[:L]
            r = [(slp[k][j] - tlp[k][j]) if (ok(slp[k][j]) and ok(tlp[k][j])) else None
                 for j in range(L)]
            e = [(ent[k][j] if (ent[k] and j < len(ent[k]) and ok(ent[k][j])) else None)
                 for j in range(L)]
            q = [(math.exp(qim[k][j]) if (qim[k] and j < len(qim[k]) and ok(qim[k][j])) else None)
                 for j in range(L)]

            hot = None
            best = 0.0
            for j in range(L - 1):
                if q[j] is not None and q[j] > best:
                    best, hot = q[j], j

            wins = [window(ids, r, e, q, 0, WIN)]
            if L > 2 * WIN:
                wins.append(window(ids, r, e, q, L - WIN, L))
            if hot is not None and best > 0.2 and not (hot < WIN or hot > L - WIN):
                wins.append(window(ids, r, e, q, hot - WIN // 2, hot + WIN // 2))
            wins.sort(key=lambda w: w["o"])

            out.append({
                "seq": k, "len": L, "trunc": bool(tr[k]) if tr[k] is not None else None,
                "score": r2(sc[k]) if sc[k] is not None else None,
                "gt": (gt[k] or "")[:80],
                "hot": hot, "hotq": r2(best),
                "trace": {"r": downsample(r, BINS), "ent": downsample(e, BINS),
                          "q": downsample(q, BINS)},
                "wins": wins,
            })
        if out:
            entry["samples"][str(step)] = out

    data["arms"][name] = entry

with open(OUT, "w") as fh:
    json.dump(data, fh, separators=(",", ":"), ensure_ascii=False)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
for a, v in data["arms"].items():
    print(f"  {a:28} steps={len(v['steps'])} sampled_steps={len(v['samples'])}")
