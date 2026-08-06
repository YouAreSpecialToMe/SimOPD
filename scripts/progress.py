"""The whole campaign on one screen: what state is every column of the paper in?

    python scripts/progress.py                # one shot
    python scripts/progress.py --watch 300    # refresh every 5 min

watch.py answers "is this run healthy" -- flight-recorder altitude. This answers the
campaign question: every row of configs/campaign.tsv, joined with whatever evidence
the shared filesystem holds about it. Run it on ANY DSW box and it sees all of them:
lane logs, checkpoints, pool claims and the machine map all live on the shared mount,
which is also why it needs no SSH and no per-machine agents.

cornell is a separate filesystem. Its rows show as "other island" here, and this same
script run ON cornell shows its slice (and marks the DSW machines as islands in turn).
The rule for telling "not started" from "not visible": a machine registered in THIS
claim directory's MACHINE_MAP shares this filesystem, so absence of its logs is real;
an unregistered machine's absence proves nothing.

Read-only by design. The enforcing watcher is watch.py --enforce; a monitor that can
kill things is two tools glued together, and the glue is where the accidents live.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watch import collect  # noqa: E402  (battle-tested log parsing, reused not rewritten)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def manifest_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            rows.append({"wave": parts[0].strip(), "machine": parts[1].strip(),
                         "arm": parts[2].strip(), "seed": parts[3].strip(),
                         "note": parts[4].strip() if len(parts) > 4 else ""})
    return rows


def read_kv_dir(claim_dir):
    """MACHINE_MAP, claim owners, pin -- everything the campaign wrote about itself."""
    info = {"map": {}, "claims": {}, "pin": None, "probes": []}
    mp = os.path.join(claim_dir, "MACHINE_MAP")
    if os.path.isfile(mp):
        for ln in open(mp):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 2:
                info["map"][p[1]] = p[0]  # machine -> hostname
    cd = os.path.join(claim_dir, "claims")
    if os.path.isdir(cd):
        for name in os.listdir(cd):
            try:
                txt = open(os.path.join(cd, name, "owner")).read()
                info["claims"][name] = next((f.split("=", 1)[1] for f in txt.split()
                                             if f.startswith("machine=")), "?")
            except OSError:
                info["claims"][name] = "?"
    ref = os.path.join(claim_dir, "CAMPAIGN_REF")
    if os.path.isfile(ref):
        info["pin"] = open(ref).read().split()[0][:12]
    pd = os.path.join(claim_dir, "_probe")
    if os.path.isdir(pd):
        info["probes"] = sorted(os.listdir(pd))
    # Daemon heartbeats, by age. Stale is the alarm: a daemon that stopped beating is
    # a machine that will sit idle after its current runs finish -- exactly the state
    # this monitor exists to make visible before it costs a night.
    info["daemons"] = {}
    for f in os.listdir(claim_dir) if os.path.isdir(claim_dir) else []:
        if f.startswith("daemon.alive."):
            info["daemons"][f[len("daemon.alive."):]] = age_min(os.path.getmtime(os.path.join(claim_dir, f)))
    return info


def age_min(mtime):
    return (time.time() - mtime) / 60


def fmt_age(m):
    return f"{m / 60:.1f}h" if m >= 90 else f"{m:.0f}m"


def row_state(row, runs, info):
    """One manifest row -> (state, detail). The joins, in evidence order."""
    name = f"{row['arm']}_s{row['seed']}"
    r = runs.get(name)
    owner = info["claims"].get(name)
    machine = owner if row["machine"] == "any" and owner else row["machine"]

    if r:
        last_step = r["steps"][-1]["step"] if r["steps"] else (r["tqdm"][0] if r.get("tqdm") else 0)
        total = r["tqdm"][1] if r.get("tqdm") else 250
        val = f"{r['val'][-1][1]:.3f}@{r['val'][-1][0]}" if r["val"] else "-"
        ck = f"ckpt@{r['ckpt_step']}" if r.get("ckpt_step") else ""
        fl = " ".join(f for f in r.get("flags", []) if not f.startswith("VAL-0"))
        if r["status"] == "OK":
            return machine, "DONE", f"val {val} {ck}".strip()
        if r["status"] == "FAIL":
            return machine, "FAILED", f"at step {last_step}; campaign.sh retries it {ck}".strip()
        detail = f"step {last_step}/{total}  val {val}  age {fmt_age(age_min(r['mtime']))}"
        if fl:
            detail += f"  [{fl}]"
        return machine, "RUNNING", detail

    # No log evidence anywhere on this filesystem.
    if row["machine"] != "any" and row["machine"] not in info["map"]:
        return machine, "ISLAND", "other filesystem; run progress.py there for its slice"
    if row["note"].startswith("needs="):
        need = row["note"][len("needs="):]
        return machine, "BLOCKED", f"needs {need}" + ("" if os.path.exists(need) else " (absent here)")
    if owner:
        return machine, "CLAIMED", f"by {owner}, no log yet (starting, or its launch failed)"
    if row["machine"] == "any":
        return machine, "POOL", "unclaimed; next free lane takes it"
    return machine, "QUEUED", "assigned, not started"


def render(args):
    rows = manifest_rows(args.manifest)
    runs = collect(args.log_dir, args.ckpt_root, args.save_freq)
    info = read_kv_dir(args.claim_dir)

    now = datetime.now(timezone.utc).strftime("%m-%d %H:%MZ")
    host = os.uname().nodename
    me = next((m for m, h in info["map"].items() if h == host), "?")
    print(f"\n=== SimOPD campaign @ {now}  (viewed from {me}:{host}) ===")
    beats = "  ".join(f"{m}:{fmt_age(a)}" + ("!" if a > 45 else "") for m, a in sorted(info["daemons"].items()))
    print(f"pin {info['pin'] or 'UNPINNED'}   machines {', '.join(f'{m}={h}' for m, h in sorted(info['map'].items())) or '(none registered)'}"
          f"   fs-probes {len(info['probes'])}   daemons {beats or 'NONE RUNNING'}")

    counts = {}
    by_wave = {}
    for row in rows:
        machine, state, detail = row_state(row, runs, info)
        counts[state] = counts.get(state, 0) + 1
        by_wave.setdefault(row["wave"], []).append((machine, row, state, detail))

    order = {"RUNNING": 0, "FAILED": 1, "DONE": 2, "CLAIMED": 3, "QUEUED": 4,
             "POOL": 5, "BLOCKED": 6, "ISLAND": 7}
    for wave in sorted(by_wave):
        print(f"\n-- wave {wave} " + "-" * 66)
        for machine, row, state, detail in sorted(by_wave[wave], key=lambda x: (order.get(x[2], 9), x[0])):
            name = f"{row['arm']}_s{row['seed']}"
            print(f"  {machine:8s} {name:28s} {state:8s} {detail}")

    done, total = counts.get("DONE", 0), sum(counts.values())
    visible = total - counts.get("ISLAND", 0)
    bar = "#" * done + "." * max(visible - done, 0)
    print(f"\n{done}/{visible} done here ({total} rows total)   [{bar}]")
    print("   " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: order.get(kv[0], 9))))

    # A machine whose daemon is beating but refusing work: fresh heartbeat + QUEUED
    # rows + nothing moving looks healthy from every other line on this screen.
    for m in sorted(info["daemons"]):
        st = os.path.join(args.claim_dir, f"daemon.status.{m}")
        try:
            txt = open(st).read()
        except OSError:
            continue
        if not txt.startswith("ok"):
            print(f"\nATTENTION -- {m}'s daemon is alive but its launches are REFUSED:")
            for ln in txt.splitlines()[:5]:
                print(f"  {ln}")
            print(f"  full context: logs/<{m}-host>_daemon.log")

    # Anything running-but-silent is the line to read first; everything else can wait.
    quiet = [(n, r) for n, r in runs.items()
             if r["status"] == "running" and any(f.startswith(("STALLED", "ABANDONED")) for f in r.get("flags", []))]
    if quiet:
        print("\nATTENTION -- running but silent past the healthy window:")
        for n, r in sorted(quiet, key=lambda x: -age_min(x[1]["mtime"])):
            print(f"  {n:28s} quiet {fmt_age(age_min(r['mtime']))}   {r['log']}")
        print("  py-spy dump the actor pid before sweeping anything; the stack names the cause.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join(ROOT, "configs", "campaign.tsv"))
    p.add_argument("--log-dir", default=os.environ.get("LOG_DIR", os.path.join(ROOT, "logs")))
    p.add_argument("--claim-dir", default=os.environ.get("CLAIM_DIR", os.path.join(ROOT, ".campaign")))
    p.add_argument("--ckpt-root", default=os.environ.get("CKPT_ROOT", "/scratch/zz865/simopd/ckpt"))
    p.add_argument("--save-freq", type=int, default=int(os.environ.get("SAVE_FREQ", "50")))
    p.add_argument("--watch", type=int, metavar="SEC")
    args = p.parse_args()
    while True:
        render(args)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
