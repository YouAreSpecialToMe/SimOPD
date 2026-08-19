#!/usr/bin/env python3
"""One-screen status of the corrected-rerun wave (DLC slots): phase per slot, rehearsal
verdicts, per-lane latest step / response length / clip ratio / last log line.

    python scripts/corr_wave_status.py            # snapshot
    python scripts/corr_wave_status.py --watch 600  # re-print every 10 min

Reads only shared disk: $D/corr_wave/*.log (fleet + rehearse + lane logs) and the
checkpoint tree $D/ckpt/simopd/<arm>_s<seed>_16k/. Read-only.
"""
import argparse
import glob
import os
import re
import time

D = "/mgfs/shared/Group_GY/changhao/simopd_data"
LOGD = f"{D}/corr_wave"
SLOTS = {
    0: ["vanilla_corr", "n2_corr", "b1_skew_kl_corr", "b5_k2_corr"],
    1: ["f1_soft_log_corr", "f2_hard_clip_corr", "f3_power_corr", "g1_verified_only_corr"],
    2: ["g4_failure_only_corr", "g6_seqmean_corr", "h2_last_segment_corr", "h3_random_segment_corr"],
    3: ["h4_random_scatter_corr", "b2_forward_kl_corr", "e2_set_coverage_a0_corr", "c3_intersection_corr"],
    4: ["g2_fire_likelihood_corr", "g5_rgopd_gate_corr", "d2_selectkd_corr", "d3_teachability_corr"],
    5: ["n2_termcal", "d1_tip_corr", "f2_clip2.3_corr", "h1_first_segment_corr"],
}
STEP_RE = re.compile(r"step:(\d+)")
KV_RE = {
    "len": re.compile(r"response_length/mean:([0-9.]+)"),
    "clip": re.compile(r"response_length/clip_ratio:([0-9.]+)"),
    "dl_stop": re.compile(r"actor/distillation/eos_dl_at_stop:([-0-9.e]+)"),
    "dl_raw": re.compile(r"actor/distillation/eos_dl_at_stop_raw:([-0-9.e]+)"),
    "n_stop": re.compile(r"actor/distillation/eos_n_stop:([0-9.e]+)"),
    "val": re.compile(r"val-core/[^:]*math500[^:]*mean@1:([0-9.]+)"),
}


def newest(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    return hits[-1] if hits else None


def tail(path, n=1):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            lines = f.read().decode(errors="replace").splitlines()
        return lines[-n:] if lines else []
    except Exception:
        return []


def phase_of_slot(slot, seed):
    log = newest(f"{LOGD}/fleet_slot{slot}_s{seed}_*.log")
    if not log:
        return "not started", None
    last = tail(log, 40)
    text = "\n".join(last)
    age = time.time() - os.path.getmtime(log)
    if "REHEARSAL MARKERS MISSING" in text and "Phase R: rehearsing" not in text:
        ph = "OLD script: idling for markers"
    elif "Phase R: rehearsing" in text and "rehearsals finished" not in text:
        ph = "rehearsing on pod"
    elif "Phase L:" in text or "lanes launched" in text:
        ph = "lanes running"
    elif "CORR_WAVE_FLEET" in text and "DONE" in text:
        ph = "DONE"
    elif "TREE STALE" in text or "ARM_LINT" in text and "FAILED" in text:
        ph = "BLOCKED (see log)"
    elif "CARRIER REHEARSAL FAILED" in text:
        ph = "carrier rehearsal FAILED"
    else:
        ph = "starting"
    return f"{ph} (log {os.path.basename(log)}, {age/60:.0f} min old)", log


def lane_status(arm, seed):
    out = {"arm": arm}
    ok = os.path.exists(f"{LOGD}/rehearsal_{arm}.OK") or os.path.exists(f"{D}/n2/rehearsal_{arm}.OK")
    rlog = f"{LOGD}/rehearse_{arm}_s{seed}.log"
    if ok:
        out["rehearsal"] = "PASS"
    elif os.path.exists(rlog):
        t = "\n".join(tail(rlog, 5))
        out["rehearsal"] = "FAIL" if "REHEARSAL FAIL" in t else "running"
    else:
        out["rehearsal"] = "-"
    ck = sorted(int(os.path.basename(p).split("_")[-1]) for p in glob.glob(f"{D}/ckpt/simopd/{arm}_s{seed}_16k/global_step_*"))
    out["ckpt"] = ck[-1] if ck else None
    llog = f"{LOGD}/lane_{arm}_s{seed}.log"
    if os.path.exists(llog):
        lines = tail(llog, 400)
        steps = [l for l in lines if "step:" in l and "response_length" in l]
        if steps:
            l = steps[-1]
            m = STEP_RE.search(l)
            out["step"] = int(m.group(1)) if m else None
            for k, rx in KV_RE.items():
                mm = rx.search(l)
                if mm:
                    out[k] = float(mm.group(1))
        out["age_min"] = (time.time() - os.path.getmtime(llog)) / 60
        errs = [l for l in lines if "Traceback" in l or "Error" in l or "OOM" in l or "out of memory" in l]
        out["err"] = errs[-1][:120] if errs else ""
        out["last"] = lines[-1][:100] if lines else ""
    return out


def render(seed):
    print(f"corrected wave status  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}   (read-only view of {LOGD})")
    for slot, arms in SLOTS.items():
        ph, log = phase_of_slot(slot, seed)
        print(f"\nSLOT {slot}: {ph}")
        for arm in arms:
            s = lane_status(arm, seed)
            step = s.get("step")
            row = f"   {arm:<26} reh={s['rehearsal']:<7} ckpt={str(s['ckpt']):>4}"
            if step is not None:
                row += (f" step={step:>3} len={s.get('len', float('nan')):7.0f} clip={s.get('clip', float('nan')):.3f}"
                        f" dl_stop={s.get('dl_stop', float('nan')):+.2f} dl_raw={s.get('dl_raw', float('nan')):+.1f}"
                        f" n_stop={s.get('n_stop', float('nan')):.0f}")
                if "val" in s:
                    row += f" val={s['val']:.3f}"
            if "age_min" in s:
                row += f"  ({s['age_min']:.0f}m ago)"
            if s.get("err"):
                row += f"\n      !! {s['err']}"
            print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watch", type=int, default=0, help="seconds between refreshes (0 = once)")
    a = ap.parse_args()
    while True:
        render(a.seed)
        if not a.watch:
            break
        time.sleep(a.watch)
        print("\n" + "=" * 100)
