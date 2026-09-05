"""Read configs/arms.yaml — the arm registry that drives every run in the audit.

  python scripts/arm.py list [--status stock|needs]
  python scripts/arm.py env <run_id>     # eval-able env block for run_opd_baseline.sh
  python scripts/arm.py check            # registry sanity + implementation gate report
"""

import argparse
import os
import re
import sys

import yaml

REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "arms.yaml")


def load():
    with open(REGISTRY) as f:
        arms = yaml.safe_load(f)["arms"]
    ids = [a["run_id"] for a in arms]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        sys.exit(f"duplicate run_id in registry: {sorted(dup)} — run_id is the ledger key, it must be unique")
    return arms


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list")
    pl.add_argument("--status", choices=["stock", "needs", "shelved"])
    pe = sub.add_parser("env")
    pe.add_argument("run_id")
    sub.add_parser("check")
    args = p.parse_args()

    arms = load()

    if args.cmd == "list":
        for a in arms:
            if args.status and a["status"] != args.status:
                continue
            print(a["run_id"])

    elif args.cmd == "env":
        arm = next((a for a in arms if a["run_id"] == args.run_id), None)
        if arm is None:
            sys.exit(f"unknown arm '{args.run_id}'; see: python scripts/arm.py list")
        if arm["status"] == "shelved":
            sys.exit(f"arm '{args.run_id}' is SHELVED (out of the roster by dated decision — "
                     f"see its note in configs/arms.yaml); flip status to re-enlist")
        if arm["status"] != "stock":
            sys.exit(f"arm '{args.run_id}' is not runnable yet — blocked on: {arm.get('seam', '?')}")
        print(f"export EXPERIMENT_NAME={arm['run_id']}")
        for k, v in (arm.get("env") or {}).items():
            # 2026-09-04:登记表里的资产路径写成 $SIMOPD_STORE/...(从前是写死的旧集群绝对路径,
            # 换集群后六条名册臂全指向一块已失联的盘)。这里展开,而不是留给 shell 的 eval ——
            # python 侧消费者(run_manifest、lint)也要看到真实路径。展不开就拒绝,不给静默默认。
            v = os.path.expandvars(str(v))
            left = re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", v)
            if left:
                sys.exit(f"arm '{args.run_id}': {k}={v!r} 引用了未设置的环境变量 {sorted(set(left))} "
                         f"—— 先 export SIMOPD_STORE=<数据根>(见 AGENT.md §6 第 0 步)")
            print(f"export {k}={v}")

    elif args.cmd == "check":
        stock = [a for a in arms if a["status"] == "stock"]
        needs = [a for a in arms if a["status"] == "needs"]
        shelved = [a for a in arms if a["status"] == "shelved"]
        line = f"registry: {len(arms)} arms — {len(stock)} runnable on stock verl, {len(needs)} blocked"
        if shelved:
            line += f", {len(shelved)} shelved ({', '.join(a['run_id'] for a in shelved)})"
        print(line + "\n")
        print("runnable now:")
        for a in stock:
            print(f"  {a['run_id']:<26} [{a['axis']}] {a['desc']}")
        print("\nblocked on implementation:")
        for a in needs:
            print(f"  {a['run_id']:<26} [{a['axis']}] {a.get('seam', '?')}")


if __name__ == "__main__":
    main()
