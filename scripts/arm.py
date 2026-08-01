"""Read configs/arms.yaml — the arm registry that drives every run in the audit.

  python scripts/arm.py list [--status stock|needs]
  python scripts/arm.py env <run_id>     # eval-able env block for run_opd_baseline.sh
  python scripts/arm.py check            # registry sanity + implementation gate report
"""

import argparse
import os
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
    pl.add_argument("--status", choices=["stock", "needs"])
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
        if arm["status"] != "stock":
            sys.exit(f"arm '{args.run_id}' is not runnable yet — blocked on: {arm.get('seam', '?')}")
        print(f"export EXPERIMENT_NAME={arm['run_id']}")
        for k, v in (arm.get("env") or {}).items():
            print(f"export {k}={v}")

    elif args.cmd == "check":
        stock = [a for a in arms if a["status"] == "stock"]
        needs = [a for a in arms if a["status"] == "needs"]
        print(f"registry: {len(arms)} arms — {len(stock)} runnable on stock verl, {len(needs)} blocked\n")
        print("runnable now:")
        for a in stock:
            print(f"  {a['run_id']:<26} [{a['axis']}] {a['desc']}")
        print("\nblocked on implementation:")
        for a in needs:
            print(f"  {a['run_id']:<26} [{a['axis']}] {a.get('seam', '?')}")


if __name__ == "__main__":
    main()
