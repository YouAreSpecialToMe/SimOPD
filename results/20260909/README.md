# 2026-09 名册重训 — 结果数据集（导出于 2026-09-09）

本目录是**数据搬运**，不是分析。每张表的每一列要么直接来自磁盘上的运行产物，
要么由它们按下文写明的公式算出。没有阈值判定，没有"正常 / 异常"的分类，
没有对任何一条臂的结论。

- 导出脚本：`extract.py`（可重跑，幂等）
- 导出时刻：2026-09-09 15:4x UTC（= 08:4x PDT），紧接第二个训练窗口结束之后
- 数据来源：各臂运行目录 `<arm>_s0/` 下的 `metrics/launch_*.jsonl`、
  `val_gen/*.jsonl`、`global_step_*/`、`traj/`，以及 `configs/retrain_roster.tsv`
  和 `configs/arms.yaml`

---

## 文件清单

| 文件 | 行数 | 内容 |
|---|---|---|
| `arms.tsv` | 29 | 每条臂的轴、目标步数、名册优先级、**相对载体的 env 增量**、原始描述与名册备注 |
| `progress.tsv` | 29 | 每条臂的存档最大步、存档个数、日志最大步、记录步数、是否达标、launch 文件数 |
| `checkpoints.tsv` | 173 | 逐个存档点（arm, global_step） |
| `val_accuracy_wide.tsv` | 29 | 臂 × 步 的在环评测准确率宽表（步为 25 的倍数） |
| `metrics_summary.tsv` | 4579 | 臂 × 步 的 24 个常用指标长表 |
| `metrics_full/<arm>.tsv` | 29 个文件 | 每条臂**全部**指标的逐步时间序列（该臂出现过的所有键，最多 200+ 列） |
| `val_gen_summary.tsv` | 160 | 每次在环评测的逐题汇总（题数、acc 均值、reward 均值、输出字符长度） |
| `archive_inventory.tsv` | 29 | §2.4 归档层各类文件的计数 |
| `teacher_columns.tsv` | 173 | 每个 `summary_<step>.parquet` 里教师列的行数与均值 |
| `non_roster_dirs.tsv` | 5 | 运行目录下**不属于名册 29 条臂**的目录 |

逐题生成文本（每次评测 500 条，含 `input/output/gts/acc/reward/score`）**没有**复制到本目录，
体量太大；它们留在各臂的 `val_gen/<step>.jsonl`，`val_gen_summary.tsv` 是对它们的汇总。

---

## 口径定义

### 在环评测 `val_acc`

键名 `val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1`。
MATH-lighteval 验证集，**每次 500 题**，`test_freq=25` 即每 25 步评一次，
贪心单样本（`mean@1`）。`val_gen_summary.tsv` 里的 `acc_mean` 是对同一次评测的
500 行 `acc` 字段**独立**求均值，两者可互相校验。

> `critic/score/mean`（表中列名 `train_score_mean`）是**训练批**的打分，
> 不是验证集准确率，量级不同（前者 0.0x 量级，后者 0.6 量级）。两列都在表里，
> 不要混用。

### 步数的三个口径

| 口径 | 列 | 含义 |
|---|---|---|
| 存档 | `ckpt_max_step` | 磁盘上 `global_step_<n>/` 的最大 n。`save_freq=25`，所以永远是 25 的倍数。**这是续跑的起点。** |
| 日志 | `launch_max_step` | `launch_*.jsonl` 里 `step` 字段的最大值，跨该臂所有 launch 文件取最大 |
| 达标 | `reached_target` | `global_step_<目标>/` 目录是否存在（1/0） |

`launch_max_step` 可以大于 `ckpt_max_step`：进程被停时最后不足 25 步的进度不落盘。
两者之差就是那一段。

### `env_delta_vs_carrier`

`arms.tsv` 的这一列 = 该臂在 `configs/arms.yaml` 里的 `env`，减去**载体**
`vanilla_corr` 的 `env` 之后剩下的键值对。载体本身是：

```
DISTILLATION_LOSS_MODE=k1_termfix   DISTILLATION_TOPK=66
SIMOPD_KEEP_SAMPLED=1               SIMOPD_GATHER_EOS=1
SIMOPD_EOS_IDS=151643               SIMOPD_EOS_TEACHER_IDS=151643,151645
SIMOPD_STOP_IDS=off                 SIMOPD_SHADOW=0
```

因此 `vanilla_corr` 这一行的增量为空。名册臂名带载体后缀（`_n0` / `_corr`），
查 `arms.yaml` 时按剥掉后缀的裸旋钮名匹配。

### 共同训练配置（全部 29 条臂一致）

学生 `Qwen3-1.7B-Base`，教师 `Qwen3-4B-Instruct-2507`；
响应上限 16384 token、prompt 上限 1024；`train_batch_size=128`、
`ppo_mini_batch_size=128`、`rollout_n=1`、temperature 1.0、top_p 1.0；
学习率 1e-6；`save_freq=25`、`test_freq=25`；全部 seed 0。

### 同一步被写多次时的取舍

续跑会与上一次运行在步号上重叠。`extract.py` 按文件 mtime 升序遍历，
**后写的覆盖先写的**，即同一 step 取最近一次运行的值。
`n_launch_files` 列给出该臂被启动过多少次，可据此判断重叠程度。

---

## 汇总数字

| 项 | 值 |
|---|---|
| 名册臂数 | 29 |
| 达标（存在 `global_step_<目标>`） | **10** |
| 未达标 | 19 |
| 存档步数合计 / 目标合计 | **4325 / 6050** |
| 名册臂的存档点总数 | 173 |
| 运行目录总体积 | 4.7 T |
| 有在环评测记录的臂 | 29 / 29 |
| 在环评测总次数 | 160（`val_gen_summary.tsv` 行数） |

达标的 10 条：
`c2_qb_fixed8_corr`、`c2_qb_perseq_corr`、`c2_quantile_budget_corr`、
`c3_intersection_corr`、`c4_hq`、`c4_pi_tail_budget_corr`、
`d3_teachability_corr`、`h1_first_segment_corr`、`vanilla_corr`、`vanilla_te`。

### 在环评测准确率（`val_accuracy_wide.tsv` 的内容，空格表示该步无记录）

```
arm                          25     50     75    100    125    150    175    200    225    250
vanilla_corr              0.608  0.604  0.632  0.622  0.638  0.636  0.662  0.624  0.664  0.660
vanilla_te                0.630  0.610  0.610  0.628  0.604  0.630  0.654  0.666  0.592  0.654
a1_gkd_mix0.5_n0          0.514  0.584  0.566  0.468  0.432
a3_offpolicy_n0           0.506  0.572  0.540         0.486
a4_dagger_anneal_n0       0.518  0.610  0.516
a5_aggrevate_n0           0.520  0.586
b1_skew_kl_corr           0.608  0.622         0.608  0.606  0.618
b2_forward_kl_corr        0.548  0.570  0.506
c2_qb_fixed8_corr         0.600  0.630  0.570  0.510  0.472  0.496  0.522
c2_qb_perseq_corr         0.612  0.600  0.618  0.602  0.642  0.612  0.636  0.662
c2_quantile_budget_corr   0.620  0.618  0.592  0.626  0.614  0.538  0.668  0.658
c3_intersection_corr      0.528  0.568  0.598  0.578  0.562  0.554         0.570
c4_hq                     0.610  0.624  0.584  0.590  0.610  0.606  0.628  0.636
c4_pi_tail_budget_corr    0.616  0.622         0.588  0.480  0.612  0.610  0.640
c4_state                  0.612  0.620         0.562  0.564  0.600
c5_union_fkl              0.550  0.570  0.514  0.480  0.462  0.472  0.394
c5_union_rkl              0.638  0.620  0.598  0.452  0.474
d2_selectkd_corr          0.614  0.634  0.598  0.626                0.666
d3_teachability_corr      0.588  0.634  0.564  0.548  0.540         0.484
e2_set_coverage_a0_corr   0.596  0.584  0.572  0.568  0.556
f2_hard_clip_corr         0.610  0.610  0.630  0.632
f3_power_corr             0.548  0.564  0.554  0.616         0.544  0.558
g1_verified_only_corr     0.350  0.566  0.660  0.632  0.612
g6_seqmean_corr           0.596  0.636
h1_first_segment_corr     0.592  0.648  0.626  0.592  0.630  0.638  0.604  0.616
h2_last_segment_corr      0.512  0.454
h3_random_segment_corr    0.568  0.636  0.620  0.606
h4_random_scatter_corr    0.604  0.640  0.618  0.618
n2_corr                   0.538  0.618  0.592         0.592
```

n = 500，故单点的二项标准误约 0.021–0.022。

### 判决基线的逐步序列（`vanilla_corr`，取自 `metrics_summary.tsv`）

| step | val_acc | resp_len_mean | resp_clip_ratio | entropy_student | entropy_teacher_topk |
|---|---|---|---|---|---|
| 25 | 0.608 | 1556 | 0.000 | 0.432 | 0.416 |
| 50 | 0.604 | 11905 | 0.609 | 0.152 | 0.216 |
| 75 | 0.632 | 9241 | 0.188 | 0.280 | 0.360 |
| 100 | 0.622 | 9745 | 0.172 | 0.321 | 0.389 |
| 125 | 0.638 | 8946 | 0.070 | 0.358 | 0.429 |
| 150 | 0.636 | 8846 | 0.039 | 0.382 | 0.442 |
| 175 | 0.662 | 9060 | 0.039 | 0.367 | 0.431 |
| 200 | 0.624 | 9230 | 0.086 | 0.347 | 0.405 |
| 225 | 0.664 | 9394 | 0.070 | 0.349 | 0.414 |
| 250 | 0.660 | 9809 | 0.117 | 0.348 | 0.404 |

---

## 自洽性核对

导出后做的三项对账，结果如实记录：

1. **准确率的两个独立来源逐点一致。** `metrics_summary.tsv` 的 `val_acc`
   （来自 `launch_*.jsonl` 的 `val-core/...acc/mean@1`）与 `val_gen_summary.tsv`
   的 `acc_mean`（对同次评测 500 行 `acc` 字段自行求均值）在 `vanilla_corr`
   的全部 10 个评测点上完全相同（0.608 / 0.604 / 0.632 / 0.622 / 0.638 /
   0.636 / 0.662 / 0.624 / 0.664 / 0.660）。

2. **归档层 29/29 齐全。** `archive_inventory.tsv` 中 29 条臂的
   `light_jsonl_lines`、`n_ids_parquet`、`n_summary_parquet`、`n_step_parquet`、
   `n_div_rank_jsonl` 均 > 0，且 `has_run_manifest` 均为 1。

3. **存档计数对得上。** `progress.tsv` 的 `ckpt_max_step` 求和 = 4325；
   `checkpoints.tsv` 行数 = 173 = `progress.tsv` 的 `n_checkpoints` 求和。
   磁盘上 `global_step_*` 目录共 177 个，差额 4 个见 `non_roster_dirs.tsv`。

---

## 已知的数据缺口与不确定性

以下都是**观察到的事实**，不含推断：

1. **部分步在日志中缺失。** `val_accuracy_wide.tsv` 里的空格表示该步在
   `launch_*.jsonl` 中没有对应记录。`progress.tsv` 的
   `n_steps_logged` 与 `launch_max_step` 可比对缺失量。缺失出现在作业被
   取消或到时限的时刻附近。

2. **续跑造成的步号重叠。** 19 条未达标臂全部经历过至少一次从存档续跑
   （`n_launch_files` ≥ 2，最多 13）。同一步号可能被写过多次，本导出取最近一次。

3. **部分指标只在部分臂上出现。** `entropy_student` 等
   `actor/distillation/*` 键只在开了相应通道的臂上有值，
   `metrics_summary.tsv` 中留空。各臂真实拥有的键见 `metrics_full/<arm>.tsv` 的表头。

4. **`c2_qb_fixed8_corr` 的 `entropy_student` 末值为 8.06**，
   而其余有该键的臂在 0.03–2.2 区间。已核对为该臂 `metrics_full` 中的原样数值，
   未做任何处理，未核查成因。

5. **`teacher_columns.tsv` 的 `tch_lp_nan_mean` 在本次导出的全部行上为 0**。
   该列在 2026-09-04 之前的运行中曾为非零（traj_dump 教师块索引修复前）；
   本目录导出的是修复后重跑的数据。

6. **wandb 处于离线模式**，无在线曲线；本目录的时间序列即为可得的全部指标记录。

7. **尚无离线评测。** 173 个存档点没有任何一个跑过训练环外的评测套件，
   `evals/` 目录为空。本目录里所有准确率均来自训练过程中的在环评测。

8. **`non_roster_dirs.tsv`** 记录了 4 个 A 轴彩排目录（各 1 个 3 步存档、27 G）
   和 1 个环境自检目录。它们不属于名册 29 条臂，未计入上面的汇总数字。

---

## 复现

```bash
python3 results/20260909/extract.py
```

脚本只读不写运行目录，输出覆盖本目录下的 `*.tsv` 与 `metrics_full/`。
需要 `pyyaml`；`teacher_columns.tsv` 额外需要 `pyarrow`，缺失时跳过该表并打印提示。
