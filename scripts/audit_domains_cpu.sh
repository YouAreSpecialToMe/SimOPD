#!/usr/bin/env bash
# CPU correctness audit for the domain campaigns: everything provable without GPUs.
# Each section is independent -- one failure must not hide the others.
set -uo pipefail
cd /mgfs/shared/Group_GY/changhao/SimOPD-exp
source simopd_env.sh
DATA=/mgfs/shared/Group_GY/changhao/simopd_data
export HF_HUB_OFFLINE=1

echo "===== 1. reward 单测(当前 HEAD)====="
python scripts/test_domain_reward.py 2>&1 | tail -3 || echo "SECTION1 FAILED"

echo
echo "===== 2. verl 调用契约(reward manager 怎么调 compute_score)====="
python - <<'PY' || echo "SECTION2 FAILED"
import os, re, glob, verl, inspect
vroot = os.path.dirname(verl.__file__)
# call sites inside reward managers
for f in sorted(glob.glob(f"{vroot}/workers/reward_manager/*.py")):
    for i, ln in enumerate(open(f), 1):
        if "self.compute_score(" in ln or "compute_score_fn(" in ln:
            print(f"  {os.path.basename(f)}:{i}: {ln.strip()[:110]}")
# how the custom fn gets wrapped
from verl.trainer.ppo import reward as R
src = inspect.getsource(R.get_custom_reward_fn)
keep = [l for l in src.splitlines() if ("config" in l and "custom" in l) or "partial" in l or "def " in l]
print("  --- get_custom_reward_fn 关键行 ---")
[print("  ", l.strip()[:110]) for l in keep[:8]]
PY

echo
echo "===== 3. data_source 路由(codecontests -> ? 在这份 checkout 里)====="
python - <<'PY' || echo "SECTION3 FAILED"
import inspect
from verl.utils import reward_score
src = inspect.getsource(reward_score)
hits = [l.strip() for l in src.splitlines() if "codecontests" in l or "prime_code" in l]
[print("  ", h[:110]) for h in hits[:8]]
# empirical: a trivially-correct program through the real path
from verl.utils.reward_score import default_compute_score
import json
tests = json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]})
good = "a,b=map(int,input().split());print(a+b)"
print("  实测 codecontests 判分(正确程序):", default_compute_score("codecontests", f"```python\n{good}\n```", tests))
PY

echo
echo "===== 4. parquet schema 对表(code vs math)====="
python - <<'PY' || echo "SECTION4 FAILED"
import pandas as pd
m = pd.read_parquet("/mgfs/shared/Group_GY/changhao/simopd_data/simopd_math/train.parquet")
c = pd.read_parquet("/mgfs/shared/Group_GY/changhao/simopd_data/simopd_code/train.parquet")
print("  math 列:", sorted(m.columns))
print("  code 列:", sorted(c.columns))
print("  列集合相同:", sorted(m.columns) == sorted(c.columns))
for col in sorted(set(m.columns) | set(c.columns)):
    tm = type(m[col].iloc[0]).__name__ if col in m else "缺失"
    tc = type(c[col].iloc[0]).__name__ if col in c else "缺失"
    flag = "" if tm == tc else "   <-- 不一致"
    print(f"    {col:<16} math={tm:<12} code={tc}{flag}")
print("  code data_source:", dict(c["data_source"].value_counts()))
print("  math data_source:", dict(m["data_source"].value_counts()))
p = c["prompt"].iloc[0]
print("  code prompt 结构:", type(p).__name__, "->", str(p)[:160].replace(chr(10)," "))
r = c["reward_model"].iloc[0] if "reward_model" in c else None
print("  code reward_model keys:", list(r.keys()) if hasattr(r, "keys") else type(r).__name__)
PY

echo
echo "===== 5. prompt 长度审计(1.7B tokenizer,cap=1024)====="
python - <<'PY' || echo "SECTION5 FAILED"
import numpy as np, pandas as pd
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
def plen(p):
    msgs = list(p) if isinstance(p, (list, np.ndarray)) else [{"role": "user", "content": str(p)}]
    try:
        txt = tok.apply_chat_template([dict(x) for x in msgs], tokenize=False, add_generation_prompt=True)
    except Exception:
        txt = " ".join(str(x.get("content", x)) for x in msgs)
    return len(tok(txt, add_special_tokens=False).input_ids)
for name in ["simopd_code/train.parquet", "simopd_code/val_holdout.parquet", "simopd_math/train.parquet"]:
    df = pd.read_parquet(f"/mgfs/shared/Group_GY/changhao/simopd_data/{name}")
    n = min(len(df), 3000)
    sample = df["prompt"].sample(n, random_state=0) if len(df) > n else df["prompt"]
    ls = np.sort(np.array([plen(p) for p in sample]))
    over = int((ls > 1024).sum())
    print(f"  {name}: n={len(ls)} p50={ls[len(ls)//2]} p90={ls[int(len(ls)*.9)]} "
          f"p99={ls[int(len(ls)*.99)]} max={ls[-1]}  超1024: {over} ({100*over/len(ls):.1f}%)")
PY

echo
echo "===== 6. 官方 preflight 对 code parquet(域 env 同款参数)====="
python scripts/preflight.py --student Qwen/Qwen3-1.7B-Base --teacher Qwen/Qwen3-4B-Instruct-2507 \
    --data "$DATA/simopd_code/train.parquet" --val "$DATA/simopd_code/val_holdout.parquet" \
    --loss k1_rec --max-prompt-length 1024 2>&1 | tail -12 || echo "SECTION6 FAILED (preflight 非零退出)"

echo
echo "===== 7. 三 domain 臂覆盖对表(新臂必须三处都在)====="
python - <<'PY' || echo "SECTION7 FAILED"
import yaml
arms = yaml.safe_load(open("configs/arms.yaml"))["arms"]
stock = {a["run_id"] for a in arms if a.get("status") == "stock"}
def manifest_arms(p):
    s = set()
    for ln in open(p):
        if ln.strip().startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 4:
            s.add(f[2])
    return s
mmath = manifest_arms("configs/campaign.tsv")
mif   = manifest_arms("configs/campaign_if.tsv")
mcode = manifest_arms("configs/campaign_code.tsv")
print(f"  stock 臂总数: {len(stock)}")
for name, s in [("math", mmath), ("if", mif), ("code", mcode)]:
    missing = stock - s
    print(f"  {name:<5} manifest 臂数={len(s)}  缺 stock: {sorted(missing) if missing else '无'}")
new = ["f5_tanh", "c1_direct", "c1_tailbucket", "h5_gen100", "c2_qb_fixed8",
       "c2_qb_perseq", "f2_clip2.3", "e1_pl_rank_a0", "e2_set_coverage_a0", "b4_jsd_b0.1", "b4_jsd_b0.9"]
print("  新补臂逐个对表 (math/if/code):")
for a in new:
    print(f"    {a:<20} {'✓' if a in mmath else '✗'} / {'✓' if a in mif else '✗'} / {'✓' if a in mcode else '✗'}")
PY
echo
echo "AUDIT_DONE"
