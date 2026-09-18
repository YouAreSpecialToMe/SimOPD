# 评测结果入账(2026-09-18)

`Jerrycool/opd` 这个数据集长出了一个 `evals/` 树 —— 155 个 parquet,全部戳着
2026-09-17。9/12 那版交接包(`simopd_retrain_20260912`,集群上 `/home/zz865/opd_pack`)
里**一个评测结果都没有**,只有 traj / metrics / val_gen / manifest。所以这是评测
结果第一次随数据发出来。

两个包现在都在 Cornell:

| 位置 | 内容 | 体积 | 校验 |
|---|---|---|---|
| `/share/rush/zz865/opd-ckpt` | 10 条待续训存档(带优化器态)+ 93 个纯推理权重 | 866 G | MANIFEST 668/668 ✓ |
| `/share/rush/zz865/opd` | 29 条臂轨迹 + 155 个评测结果 | 32 G | MANIFEST 23,977/23,977 ✓ |

`/share/rush` 用掉 4.4 T,余 17 T。

## 1 这 155 格覆盖了什么

| arm | 25 | 50 | 100 | 150 | 200 | 250 | 单元 |
|---|---|---|---|---|---|---|---|
| `vanilla_corr` | 全 | 全 | 全 | 全 | 全 | 全 | **24** |
| `c2_quantile_budget_corr` | 全 | 全 | 全 | 全 | 全 | — | 19 |
| `c4_hq` | 全 | 全 | 全 | 全 | 全 | — | 19 |
| `c4_pi_tail_budget_corr` | 全 | 全 | 全 | 全 | 全 | — | 19 |
| `c4_state` | 全 | 全 | 全 | 全 | 全 | — | 19 |
| `a1_gkd_mix0.5_n0` | 全 | 全 | 3/5 | 2/3 | 3/5 | — | 14 |
| `c5_union_fkl` | 全 | 全 | 全 | 1/3 | 2/5 | — | 14 |
| `a3_offpolicy_n0` | 全 | 全 | 3/5 | 1/3 | 2/5 | — | 12 |
| `c5_union_rkl` | 全 | 全 | 3/5 | 1/3 | — | — | 10 |
| `a4_dagger_anneal_n0` | 2/3 | 全 | — | — | — | — | 5 |

前五条正是交接 README §C 那五条"已完成、五基准齐全"的臂,曲线完整一格不缺。

**这不是重跑。** 第一次读的时候我把它读成了重跑 —— 对账脚本比的是"README 表 B
列为**缺**的格子"对"`evals/` 里**有**的格子",零重叠。正确的读法是反过来:`evals/`
装的就是**已做完**的,而它的空洞**逐格等于表 B**:

| 臂 | 表 B 说缺 | `evals/` 实际缺 |
|---|---|---|
| `a1_gkd_mix0.5_n0` | 100(math500,minerva) 150(math500) 200(math500,minerva) | 一致 |
| `a3_offpolicy_n0` | 100(math500,minerva) 150(math500,minerva) 200(amc23,math500,minerva) | 一致 |
| `c5_union_fkl` | 150(math500,minerva) 200(amc23,math500,minerva) | 一致 |
| `c5_union_rkl` | 100(math500,minerva) 150(math500,minerva) | 一致 |
| `a4_dagger_anneal_n0` | 25(math500) … | step25 缺 math500,一致 |

**另外 19 条臂在这个数据集里零评测结果**(b1 / b2 / c2_qb_fixed8 / c2_qb_perseq /
c3 / d2 / d3 / f2 / f3 / h1 / n2 / vanilla_te / a5 / e2 / g1 / g6 / h2 / h3 / h4)。
补评测的主体在那边,对应 `opd-ckpt` 里 597 GB 纯推理权重。

每个文件是逐题逐样本的原始记录,不是汇总分:1500 行 = 500 题 × avg@3,带 `response`
全文、`token_ids`、`finish_reason`、`truncated`、`resp_len`、`correct`。固定量:
`git_sha=461486d`、`max_tokens=32768`、`stop_contract=pin`。

## 2 `\boxed{}` 那条风险解除

09-12 的轨迹读数里 `\boxed{}` 在晚期 rollout 只剩 ~3%,而判分器就是靠解析它
拿答案 —— 当时把这条标成了最高优先级,因为它能把下游所有准确率数字掏空。
评测有全文,可以直接量:

- **判分器确实完全依赖 `\boxed{}`**:54,000 条已判响应里,不含 `\boxed{}` 的
  正确率是 **0.0%**,一格例外都没有。
- **但学生在评测里 85–98% 都写了**,而且随训练上升(`vanilla_corr`
  92.3% @25 → 98.2% @250)。

3% 出自**训练 rollout**(16k 上限、temperature 1.0);评测是 **32768 预算、t=0.7**。
截断率随长度走,截断了就来不及收尾写 box。**3% 是 16k 上限的截断产物,不是策略
退化。**下游准确率数字没有被掏空。

## 3 在环 val 与评测口径不可比 —— §6 的更正

同一条 `vanilla_corr`,两种协议给的曲线不是一个量:

| step | 25 | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|
| 在环 val(verl grader,greedy mean@1) | 60.8 | 60.4 | 62.2 | 63.6 | 62.4 | 66.0 |
| 新评测(本项目 harness,avg@3 t=0.7,32k) | 59.6 | 60.9 | 65.9 | 68.8 | 68.8 | **71.7** |

差距在晚期拉到 5.7 分。**查过一个假说并证伪了**:不是 16k 截断。把评测里长度
超过 16384 的响应一律记 0(截断后没有 `\boxed{}`,判 0 已实测),
`vanilla_corr` 各步最多只掉 **0.5 分**(step250 71.7 → 71.1)。剩下的差别是
判分器(verl 内置 vs 本项目 harness)和采样(greedy mean@1 vs avg@3 t=0.7),
**哪个占主导尚未定**。

### 对 `TRAJ-ANALYSIS-20260912.md` §6 的影响

§6 的表格用的是在环 val,应当标明口径。逐条看:

- **"长度与能力解耦"这个结论两种协议都支持,不动。** 长度暴涨那一段
  (step 25→50,训练里 7.6×、评测里 2602→9036 token)在环给 −0.4,新评测给
  **+1.3**;长度持平那一段(50→250,稳在 ~5k)在环给 +5.6,新评测给 **+10.8**。
  新协议下解耦**更明显**:收益几乎全部落在长度不动的阶段。
- **"准确率的收益几乎全部发生在前 25 步"这句要撤。** 新评测显示 step 25 之后
  还有 **+12.1**。在环 val 低估了晚期 checkpoint,原因见上(不是截断)。
- `corr(Δlen, Δacc) = −0.116` 是在 130 个 (臂, 步) 差分点上用在环 val 算的,
  同样带在环口径。用 155 格评测在**题级**重算是待办 —— 那次 `resp_len` 和
  `correct` 同表,能配对算,比跨臂回归干净。

## 4 复现

```
python3 scripts/analysis/eval_new.py    /share/rush/zz865/opd/evals   # 覆盖了哪些格
python3 scripts/analysis/eval_ledger.py                              # 对账表 B
python3 scripts/analysis/boxed_check.py /share/rush/zz865/opd/evals math500
python3 scripts/analysis/cap16k.py      /share/rush/zz865/opd/evals vanilla_corr_s0
python3 scripts/analysis/verify_opd.py  /share/rush/zz865/opd
```

拉包:`scripts/cluster/fetch_opd.py`(数据集)、`scripts/cluster/fetch_ckpt.py`(存档),
两者都走 sbatch —— 登录节点每用户 4 GiB cgroup 会把 `snapshot_download` OOM 杀掉。
