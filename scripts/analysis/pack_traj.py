#!/usr/bin/env python3
"""把分析要用的轨迹从各 run 目录里挑出来,打成可下载的包(GitHub Release 资产 / HF 数据集都行)。

    # 集群上、在仓库树里(SIMOPD_STORE 指向 simopd_data):
    python scripts/analysis/pack_traj.py --dry                                # 只算大小、列文件、列缺失
    python scripts/analysis/pack_traj.py --out $SIMOPD_STORE/traj_pack_20260910
    gh release create traj-20260910 $SIMOPD_STORE/traj_pack_20260910/*.tar \
        --title "traj pack 2026-09-10" --notes-file $SIMOPD_STORE/traj_pack_20260910/README.md

为什么要挑:一个 200 步的 run 的 traj/ 有 2–3 GB(ids_ 每步整批),29 条臂 ≈ 60 GB,不该整搬;
而分析要回答的问题(c2_fixed8 的熵爆、c5 学不会终止符、d3/h2/g1/e2 各自的坏法、载体对照)只要
**全部臂的每序列摘要 + 特定臂在特定步的整序列**。所以分两层:

  tier1  全部臂的小文件:traj/light.jsonl(每步每序列)、traj/div/rank*.jsonl(每步每序列 FKL/RKL/JSD)、
         traj/summary_*.parquet(每 25 步每序列的终止事件账)、traj/meta.json、run_manifest.json、
         manifest/*.json、simopd_*.txt。约 30 MB/臂,29 臂 ≈ 0.8 GB。
  tier2  指定臂 × 指定步:traj/step_<n>.parquet(32 条整序列逐 token)、traj/ids_<n>.parquet(整批无损 id)、
         val_gen/<n>.jsonl(在环 500 题生成文本);更小一撮点位再带 traj/div/tok_step<n>_*.parquet
         (整批逐 token 分歧,~60 MB/步)。默认挑法在 PICKS / TOK_PICKS(2026-09-10 按 results/20260909 的
         读数定),--picks / --tok-picks 可覆盖,别改默认值。

不解析任何文件,只 stat / hash / 写 tar;纯标准库,任何 python3 能跑。缺文件**不致命**:列进 MISSING.tsv
继续 —— 上一次 archive_inventory 的 n_tok_parquet 全 0 是 extract.py 的 glob 少了一层 div/,这里按
traj_dump / div_panel 的真实路径找。产物:<out>/MANIFEST.tsv(包内路径、字节、sha256、层)、<out>/MISSING.tsv、
<out>/README.md(挑法 + 校验法)、<out>/traj_pack.part<k>.tar(每个 ≤ --part-gb;GitHub Release 单文件上限 2 GiB)。
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.environ.get("SIMOPD_STORE") or os.path.join(os.path.dirname(ROOT), "simopd_data")
CK = os.path.join(STORE, "ckpt", "simopd")

TIER1_FILES = ["traj/light.jsonl", "traj/meta.json", "run_manifest.json",
               "simopd_fingerprint.txt", "simopd_stop_contract.txt", "latest_checkpointed_iteration.txt"]
TIER1_GLOBS = ["traj/summary_*.parquet", "traj/div/rank*.jsonl", "manifest/*.json"]

# 臂 -> 要整序列的步(2026-09-10 定;理由见 README 段)
PICKS = {
    "vanilla_corr":            [25, 50, 75, 100, 125, 150, 175, 200, 225, 250],   # 载体全程 = 一切行为分析的对照
    "vanilla_te":              [50, 150, 250],                                    # 回归对照,三点够
    "c2_qb_fixed8_corr":       [75, 100, 125, 150, 175],                          # 熵爆:75 起爆前 / 100–125 爆 / 之后
    "c2_quantile_budget_corr": [100, 125, 150],                                   # 同族同步对照
    "c5_union_rkl":            [50, 75, 100, 125],                                # 91 步锁死前后
    "c5_union_fkl":            [100, 150, 175],                                   # 100 步截断 .01 → 175 步 .88
    "d3_teachability_corr":    [100, 150, 175, 200],                              # 变长、停止概率 .12
    "e2_set_coverage_a0_corr": [75, 100, 125],                                    # 低熵锁死
    "g1_verified_only_corr":   [50, 75, 100, 125],                                # 深 U 后往锁死漂
    "h2_last_segment_corr":    [25, 50],                                          # 复读锁死起点
    "h1_first_segment_corr":   [100, 200],                                        # 4k 短答、停止最果断
    "c3_intersection_corr":    [100, 200],                                        # 熵 2.4
    "a3_offpolicy_n0":         [75, 125],                                         # 纯离策,停止意愿没了
    "a1_gkd_mix0.5_n0":        [75, 125],
}
TOK_PICKS = {                                                                     # 逐 token 分歧,~60 MB/步,省着取
    "vanilla_corr": [100, 250], "c2_qb_fixed8_corr": [100, 125],
    "c5_union_rkl": [75, 100], "d3_teachability_corr": [175],
}


def _parse_picks(s, base):
    """'arm:25,50;arm2:100' -> dict;空串 = 用默认。"""
    if not s:
        return dict(base)
    out = {}
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        arm, steps = part.split(":")
        out[arm.strip()] = [int(x) for x in steps.split(",") if x.strip()]
    return out


def _sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(ck, arms, seed, picks, tok_picks, tier1_only, no_val_gen):
    """-> (entries [(arcname, abspath, bytes, tier)], missing [(arm, relpath, why)])"""
    entries, missing, seen = [], [], set()

    def add(arm, run_dir, rel, tier):
        p = os.path.join(run_dir, rel)
        arc = f"{os.path.basename(run_dir)}/{rel}"
        if arc in seen:
            return
        if os.path.isfile(p):
            seen.add(arc)
            entries.append((arc, p, os.path.getsize(p), tier))
        else:
            missing.append((arm, rel, "no such file"))

    for arm in arms:
        run_dir = os.path.join(ck, f"{arm}_s{seed}")
        if not os.path.isdir(run_dir):
            missing.append((arm, "", "run dir missing"))
            continue
        for rel in TIER1_FILES:
            add(arm, run_dir, rel, "tier1")
        for pat in TIER1_GLOBS:
            hits = sorted(glob.glob(os.path.join(run_dir, pat)))
            if not hits:
                missing.append((arm, pat, "glob empty"))
            for p in hits:
                add(arm, run_dir, os.path.relpath(p, run_dir), "tier1")
        if tier1_only:
            continue
        for step in picks.get(arm, []):
            add(arm, run_dir, f"traj/step_{step}.parquet", "tier2")
            add(arm, run_dir, f"traj/ids_{step}.parquet", "tier2")
            if not no_val_gen:
                add(arm, run_dir, f"val_gen/{step}.jsonl", "tier2")
            if step in tok_picks.get(arm, []):
                pat = f"traj/div/tok_step{step}_*.parquet"
                hits = sorted(glob.glob(os.path.join(run_dir, pat)))
                if not hits:
                    missing.append((arm, pat, "glob empty (逐 token 分歧没写?)"))
                for p in hits:
                    add(arm, run_dir, os.path.relpath(p, run_dir), "tier2")
    return entries, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", default=CK, help="run 目录的父目录(默认 $SIMOPD_STORE/ckpt/simopd)")
    ap.add_argument("--arms-from", default=os.path.join(ROOT, "results/20260909/progress.tsv"),
                    help="臂名清单(TSV 第一列 arm;不存在就按 <ckpt-root>/*_s<seed> 扫)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--picks", default="", help="覆盖 PICKS:'arm:25,50;arm2:100'")
    ap.add_argument("--tok-picks", default="", help="覆盖 TOK_PICKS,同格式")
    ap.add_argument("--tier1-only", action="store_true")
    ap.add_argument("--no-val-gen", action="store_true")
    ap.add_argument("--part-gb", type=float, default=1.5, help="每个 tar 分卷上限(GitHub Release 单文件 2 GiB)")
    ap.add_argument("--out", default=None, help="输出目录;不给 = --dry")
    ap.add_argument("--dry", action="store_true", help="只统计,不写")
    a = ap.parse_args()
    dry = a.dry or not a.out

    if os.path.isfile(a.arms_from):
        with open(a.arms_from) as f:
            arms = [ln.split("\t")[0].strip() for ln in f.read().splitlines()[1:] if ln.strip()]
    else:
        arms = sorted(os.path.basename(d)[: -len(f"_s{a.seed}")] for d in glob.glob(os.path.join(a.ckpt_root, f"*_s{a.seed}")))
    picks, tok_picks = _parse_picks(a.picks, PICKS), _parse_picks(a.tok_picks, TOK_PICKS)
    entries, missing = collect(a.ckpt_root, arms, a.seed, picks, tok_picks, a.tier1_only, a.no_val_gen)

    tot = {t: sum(b for _, _, b, tt in entries if tt == t) for t in ("tier1", "tier2")}
    by_arm = {}
    for arc, _, b, _ in entries:
        by_arm[arc.split("/")[0]] = by_arm.get(arc.split("/")[0], 0) + b
    print(f"root {a.ckpt_root}: {len(arms)} arms listed, {len(entries)} files, "
          f"tier1 {tot['tier1']/2**30:.2f} GiB + tier2 {tot['tier2']/2**30:.2f} GiB = {sum(tot.values())/2**30:.2f} GiB; "
          f"{len(missing)} missing")
    for run, b in sorted(by_arm.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {run:32s} {b/2**20:8.1f} MiB")
    if missing:
        print("  missing (first 12):")
        for arm, rel, why in missing[:12]:
            print(f"    {arm:26s} {rel:40s} {why}")
    if dry:
        return

    os.makedirs(a.out, exist_ok=True)
    limit = int(a.part_gb * 2**30)
    parts, cur, cur_bytes, k = [], None, 0, 0
    with open(os.path.join(a.out, "MANIFEST.tsv"), "w") as mf:
        mf.write("arcname\tbytes\tsha256\ttier\tpart\n")
        for arc, p, b, tier in sorted(entries):
            if cur is None or (cur_bytes + b > limit and cur_bytes > 0):
                if cur is not None:
                    cur.close()
                k += 1
                name = os.path.join(a.out, f"traj_pack.part{k}.tar")
                parts.append(name)
                cur, cur_bytes = tarfile.open(name, "w"), 0
                print(f"  writing {name}", flush=True)
            cur.add(p, arcname=arc, recursive=False)
            cur_bytes += b
            mf.write(f"{arc}\t{b}\t{_sha256(p)}\t{tier}\tpart{k}\n")
        if cur is not None:
            cur.close()
    with open(os.path.join(a.out, "MISSING.tsv"), "w") as f:
        f.write("arm\trelpath\twhy\n")
        for arm, rel, why in missing:
            f.write(f"{arm}\t{rel}\t{why}\n")
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(f"# traj pack — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n\n"
                f"源:`{a.ckpt_root}`,{len(arms)} 条臂,seed {a.seed}。{len(entries)} 个文件,"
                f"tier1 {tot['tier1']/2**30:.2f} GiB,tier2 {tot['tier2']/2**30:.2f} GiB,{len(parts)} 个分卷。\n\n"
                "## 挑法\n\n- tier1(全部臂):`" + "`, `".join(TIER1_FILES + TIER1_GLOBS) + "`\n"
                "- tier2(臂 × 步,每步 `traj/step_<n>.parquet` + `traj/ids_<n>.parquet`"
                + ("" if a.no_val_gen else " + `val_gen/<n>.jsonl`") + "):\n\n```\n"
                + json.dumps(picks, ensure_ascii=False, indent=1) + "\n```\n\n- 逐 token 分歧 `traj/div/tok_step<n>_*.parquet`:\n\n```\n"
                + json.dumps(tok_picks, ensure_ascii=False, indent=1) + "\n```\n\n"
                "## 校验\n\n```\nfor t in traj_pack.part*.tar; do tar -xf $t; done\n"
                "awk -F'\\t' 'NR>1{print $3\"  \"$1}' MANIFEST.tsv | sha256sum -c --quiet && echo OK\n```\n\n"
                "缺的文件见 `MISSING.tsv`。列说明见仓库 `docs/RUN-ARCHIVE.md`;各表按 `seq_key` 对接。\n")
    print(f"wrote {len(parts)} part(s) + MANIFEST.tsv/MISSING.tsv/README.md under {a.out}")
    print("next:\n  gh release create traj-" + time.strftime("%Y%m%d") + " " + os.path.join(a.out, "*.tar")
          + f" --title 'traj pack' --notes-file {os.path.join(a.out, 'README.md')}")


if __name__ == "__main__":
    main()
