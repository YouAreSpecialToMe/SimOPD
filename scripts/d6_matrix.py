"""D6: separate the teacher's SIZE from its CAPABILITY, per problem.

    python scripts/d6_matrix.py
    python scripts/d6_matrix.py --bench amc23 --report docs/d6.md

plan §3's Phase 0 diagnostic, restated 2026-08-04 after the ceilings came in. It was
designed as a ladder on the assumption that size order is capability order. Measured
non-thinking, it is not: 4B-Instruct-2507 beats 8B on MATH500 by 10.4 points and on
AMC23 by 23.1, because enable_thinking=False costs a hybrid model more than its extra
parameters buy.

That inversion is what makes the diagnostic possible. Prior work cannot separate the
two -- their teachers are always both bigger and better -- so "large teacher-student
gap" is a single confounded axis in the literature. Ours has a teacher that is bigger
and WEAKER, which splits it.

The quantity that matters is not the ceiling but the per-problem one: how often the
teacher is right where the student is wrong. That is the supervision actually
available to distil, and two teachers with the same ceiling can differ in it.
"""

import argparse
import glob
import os
import sys

# Size relative to the 1.7B student, for the axis the ceilings do not order.
SIZE = {"Qwen3-1.7B": 1.0, "Qwen3-4B-Instruct-2507": 2.4, "Qwen3-8B": 4.7}


def load(pattern):
    import pandas as pd

    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        return None
    df = pd.read_parquet(hits[-1])
    # avg@32 has 32 rows per problem; collapse to a per-problem rate so teacher and
    # student are compared on the same shape whatever k each was run at.
    return df.groupby("problem_id")["correct"].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals", default="/home/zz865/data/simopd_evals")
    p.add_argument("--bench", default="math500")
    p.add_argument("--student", default="student_base_1p7b")
    p.add_argument("--report", default=None)
    a = p.parse_args()

    stu = load(os.path.join(a.evals, f"{a.student}__{a.bench}__*.parquet"))
    if stu is None:
        raise SystemExit(
            f"no student artifact for {a.bench}. The ceilings alone cannot give D6 its\n"
            f"per-problem column -- run:\n"
            f"  python scripts/eval_offline.py --model Qwen/Qwen3-1.7B-Base "
            f"--benchmarks {a.bench} --run-id {a.student} --step 0")

    print(f"D6 on {a.bench}: student = Qwen3-1.7B-Base, {len(stu)} problems, "
          f"acc {stu.mean():.4f}\n")
    print(f"{'teacher':26s} {'size':>5} {'acc':>7} {'T>S':>7} {'S>T':>7} {'headroom':>9}")
    rows = []
    for name in SIZE:
        tea = load(os.path.join(a.evals, f"teacher_ceiling_{name}__{a.bench}__*.parquet"))
        if tea is None:
            print(f"{name:26s}   (no artifact)")
            continue
        common = stu.index.intersection(tea.index)
        s, t = stu.loc[common], tea.loc[common]
        # The supervision that exists: problems the teacher gets right and the student
        # does not. Its complement -- student right, teacher wrong -- is where
        # distilling actively teaches the wrong answer, which no ceiling reveals.
        t_over_s = float(((t > s)).mean())
        s_over_t = float(((s > t)).mean())
        print(f"{name:26s} {SIZE[name]:5.1f} {t.mean():7.4f} {t_over_s:7.2%} "
              f"{s_over_t:7.2%} {t.mean() - s.mean():9.4f}")
        rows.append((name, SIZE[name], t.mean(), t_over_s, s_over_t, t.mean() - s.mean()))

    if len(rows) >= 2:
        by_size = sorted(rows, key=lambda r: r[1])
        by_cap = sorted(rows, key=lambda r: r[2])
        print(f"\n  size order       : {' < '.join(r[0] for r in by_size)}")
        print(f"  capability order : {' < '.join(r[0] for r in by_cap)}")
        print("  -> " + ("SAME order; this benchmark does not separate the two axes"
                         if [r[0] for r in by_size] == [r[0] for r in by_cap] else
                         "DIFFERENT order -- size and capability are separable here, which is\n"
                         "     the comparison the literature cannot make: their teachers are\n"
                         "     always both bigger and better."))

    if a.report and rows:
        with open(a.report, "w") as f:
            f.write(f"# D6: size vs capability ({a.bench})\n\n")
            f.write(f"Student Qwen3-1.7B-Base, acc {stu.mean():.4f}, {len(stu)} problems.\n")
            f.write("`T>S` is the fraction of problems the teacher gets right and the student\n"
                    "does not -- the supervision that actually exists. `S>T` is the reverse,\n"
                    "where distilling teaches the wrong answer; no ceiling shows it.\n\n")
            f.write("| teacher | size x | acc | T>S | S>T | headroom |\n|---|---|---|---|---|---|\n")
            for n, sz, acc, ts, st, hr in rows:
                f.write(f"| {n} | {sz} | {acc:.4f} | {ts:.2%} | {st:.2%} | {hr:.4f} |\n")
        print(f"\nwrote {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
