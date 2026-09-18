# 还没跑完的 —— 2026-09 名册(2026-09-18)

**给接手收尾的合作者。**怎么装环境、怎么摆 checkpoint、怎么起作业,全在
[`HANDOFF-20260917.md`](HANDOFF-20260917.md);这份只回答三件事:**还剩什么、先做哪些、
哪里会踩空**。

## 0 一句话

29 条臂 / 6050 步,**落档 5400 步(89.3%)**。剩两块:

- **训练 650 步**,10 条臂 —— 只能在 **80 G 卡**上跑
- **评测 114 格** —— 99 格权重已在盘上、**现在就能评**;15 格要等上面的训练补完

下面的表**不是抄的**,是从实物现算的:目标步数取 `configs/retrain_roster.tsv`,
落档取每条臂自己的 `latest_checkpointed_iteration.txt`,已完成的评测取 `evals/` 里的
parquet(`scripts/analysis/remaining_table.py`)。算出来和交接手册的数字逐项一致:
650 步;150 格应有、36 已齐、114 缺,其中 11 格是部分完成。

## 1 数据在哪

| 位置 | 内容 | 校验 |
|---|---|---|
| [`Jerrycool/opd-ckpt`](https://huggingface.co/Jerrycool/opd-ckpt) | `resume/` 10 条臂带优化器态的存档 + `eval/` 93 个推理权重,865 GiB | MANIFEST 668/668 |
| [`Jerrycool/opd`](https://huggingface.co/datasets/Jerrycool/opd) | `runs/` 29 条臂轨迹 + `evals/` 155 份评测 + `assets/` A 轴缓存 | MANIFEST 23,977/23,977 |
| Cornell `/share/rush/zz865/opd-ckpt` | 上面第一个的完整镜像 | 同上,已核 |
| Cornell `/share/rush/zz865/opd` | 上面第二个的完整镜像,`runs/` 已解到 `extracted/` | 同上,已核;`assets/` 09-18 补齐 |

**两个仓库要合回同一个目录才能续训**,用 `scripts/assemble_handoff.sh`,别手工 `cp`
(手册 §5.4:缺 `latest_checkpointed_iteration.txt` 会从 0 开始;`eval/` 的形状也要改)。

## 2 训练:10 条臂,650 步

按名册优先级排(0 判决基线 / 1 自研方法 / 2 代表面板 / 3 A 轴)。步时是这一轮在
**H100** 上实测的末窗中位数(`scripts/analysis/step_times.py`),一条 lane = 2 张卡。

| 优先级 | 臂 | 落档 / 目标 | 还差 | 步时 (H100) | GPU·h (H100) | 注意 | 认领 |
|---|---|---|---:|---:|---:|---|---|
| 1 | `c5_union_rkl` | 150 / 250 | 100 | 1071 s | 59.5 | 攒一个存档要 7.4 h,见 §5 | |
| 1 | `c5_union_fkl` | 200 / 250 | 50 | 1005 s | 27.9 | | |
| 2 | `g6_seqmean_corr` | 125 / 200 | 75 | 970 s | 40.4 | | |
| 2 | `e2_set_coverage_a0_corr` | 125 / 200 | 75 | 927 s | 38.6 | | |
| 2 | `h2_last_segment_corr` | 100 / 200 | 100 | 692 s | 38.5 | | |
| 2 | `h4_random_scatter_corr` | 150 / 200 | 50 | 720 s | 20.0 | | |
| 2 | `h3_random_segment_corr` | 150 / 200 | 50 | 596 s | 16.6 | | |
| 2 | `g1_verified_only_corr` | 175 / 200 | 25 | 831 s | 11.5 | | |
| 3 | `a5_aggrevate_n0` | 100 / 200 | 100 | 391 s | 21.7 | 要 `assets/gkd_offpolicy.parquet.dry` | |
| 3 | `a4_dagger_anneal_n0` | 175 / 200 | 25 | 238 s | 3.3 | 要 `assets/gkd_offpolicy.parquet` | |
| | **合计** | | **650** | | **278** | | |

- **278 是 H100 上的下界**(步时随长度还在涨)。换 A100 估计 1.5–2 倍,约 420–560 GPU·h
  —— 估计,没实测,开跑前先量几步。
- **墙钟下界由 `c5_union_rkl` 一条定**:100 步 ≈ 30 h(H100),卡再多也快不过它。
- 一次 `sbatch` 只能有一个 `STEPS`:250 步组(两条 c5)和 200 步组分开投,或用
  `RUNS_B` / `STEPS_B` 在一个节点上并跑两组(手册 §7.2)。
- **为什么非 80 G 不可**:存档是 FSDP `world_size_1`,学生 25.3 G 静态态不能拆到多卡;
  48 G 的卡在 vLLM 按 0.45 预留后只剩 1.1 G 给全部激活,而单个全词表 logits 就 4.93 GiB。
  能修它的旋钮全在续训指纹里。算术见 [`CORNELL-FEASIBILITY-20260918.md`](CORNELL-FEASIBILITY-20260918.md)。

**认领**:要跑哪条,先在上表"认领"列填名字推上来,再开跑。**同一条臂的 ckpt 目录不能
有两个写者** —— 两处同时续一条臂,两边的存档会互相覆盖,而且事后看不出来。

## 3 评测:114 格

一"格" = 一个 (臂, 步)。`light` 网格:25 / 50 / 150 步跑 amc23 + minerva + math500,
100 / 200 / 250 步五个基准全跑。`*` = 那一步还没训练到,要等 §2。

| 优先级 | 臂 | 已齐 / 应有 | 现在能评 | 等训练 | 待评的步 |
|---|---|---|---:|---:|---|
| **0** | **`vanilla_te`** | **0 / 6** | **6** | — | 25 50 100 150 200 250 |
| 1 | `n2_corr` | 0 / 6 | 6 | — | 25 50 100 150 200 250 |
| 1 | `c2_qb_fixed8_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 1 | `c2_qb_perseq_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 1 | `c5_union_rkl` | 2 / 6 | 2 | 2 | 100(缺 minerva,math500) 150(缺 minerva,math500) 200* 250* |
| 1 | `c5_union_fkl` | 3 / 6 | 2 | 1 | 150(缺 minerva,math500) 200(缺 amc23,minerva,math500) 250* |
| 2 | `b1_skew_kl_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 2 | `b2_forward_kl_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 2 | `c3_intersection_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 2 | `d2_selectkd_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 2 | `d3_teachability_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 2 | `f2_hard_clip_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 2 | `f3_power_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 2 | `h1_first_segment_corr` | 0 / 5 | 5 | — | 25 50 100 150 200 |
| 2 | `g1_verified_only_corr` | 0 / 5 | 4 | 1 | 25 50 100 150 200* |
| 2 | `h3_random_segment_corr` | 0 / 5 | 4 | 1 | 25 50 100 150 200* |
| 2 | `h4_random_scatter_corr` | 0 / 5 | 4 | 1 | 25 50 100 150 200* |
| 2 | `e2_set_coverage_a0_corr` | 0 / 5 | 3 | 2 | 25 50 100 150* 200* |
| 2 | `g6_seqmean_corr` | 0 / 5 | 3 | 2 | 25 50 100 150* 200* |
| 2 | `h2_last_segment_corr` | 0 / 5 | 3 | 2 | 25 50 100 150* 200* |
| 3 | `a1_gkd_mix0.5_n0` | 2 / 5 | 3 | — | 100(缺 minerva,math500) 150(缺 math500) 200(缺 minerva,math500) |
| 3 | `a3_offpolicy_n0` | 2 / 5 | 3 | — | 100(缺 minerva,math500) 150(缺 minerva,math500) 200(缺 amc23,minerva,math500) |
| 3 | `a4_dagger_anneal_n0` | 1 / 5 | 3 | 1 | 25(缺 math500) 100 150 200* |
| 3 | `a5_aggrevate_n0` | 0 / 5 | 3 | 2 | 25 50 100 150* 200* |
| | **合计** | **36 / 150** | **99** | **15** | |

已完成、不用再碰的 5 条:`vanilla_corr`、`c2_quantile_budget_corr`、`c4_hq`、
`c4_pi_tail_budget_corr`、`c4_state`(权重也故意没传)。

**现在能评的 99 格,按优先级的机时**(H100 单格实测:非锚点 1.6 h、锚点 2.5 h;
部分完成的格按整格计,所以是上界):

| 优先级 | 格 | GPU·h (H100) |
|---|---:|---:|
| 0 判决基线/回归对照 | 6 | ≤ 12.3 |
| 1 自研方法 | 20 | ≤ 40.1 |
| 2 代表面板 | 61 | ≤ 117.4 |
| 3 A 轴 | 12 | ≤ 24.6 |
| 合计 | 99 | ≤ 194.4 |

评测不碰指纹,**任何支持 bf16 的卡都能跑**(1.7B 权重 ~3.4 G),但 A6000 这类
GDDR6 卡带宽只有 H100 的约四分之一,vLLM decode 吃带宽,会慢好几倍 —— 先跑一格计时。
**别用 Turing(TITAN RTX / T4 等)**:没有 bf16,vLLM 静默降 fp16,不报错,但数值路径
不同,进不了同一张表。

## 4 建议的顺序

只能做一部分时:

1. **评 `vanilla_te`**(优先级 0,6 格,≤ 12 GPU·h)。名册注释:它检验 TE Path1 是否
   ≡ termfix,"licenses 每条 `_corr` 行" —— 它没评,其余 `_corr` 行的解读都悬着。
   现在是 0/6。
2. **评优先级 1 的 20 格**(≤ 40 GPU·h):`n2_corr`、两条 `c2_qb`、两条 c5 的残格。
3. **训两条 c5**(优先级 1,预注册臂):150 步,要 80 G 卡,`c5_union_rkl` 要长窗口。
4. 其余 73 格评测(≤ 142 GPU·h)—— 可以铺开在任何 bf16 卡上并行。
5. 其余 8 条臂的训练。
6. 训练补完后的 15 格评测。

评测和训练互不依赖(除了那 15 格),可以两边同时推。

## 5 开跑前必须知道的

每一条都是这一轮真踩过的。

1. **从 `ch-dev`(09-18 已合并 `0917`)或 `0917` 起,别用合并前的 `ch-dev`。**
   合并前有三处会静默踩空:`setup.sh` 钉的 verl 是 `aebd1f8`,而这批权重全部产自
   **`3d36367e`**(`deploy/dsw/verl.pin`);`run_opd_baseline.sh` 的指纹没排除
   `SIMOPD_SHORT_RUN_OK`,200 步的臂必须设它,旧脚本续训会与盘上指纹不符而被拒;
   `verdict.py` 没有 `--base`。
2. **`SIMOPD_SUITE_K=8`。**脚本默认 32,已有 155 份全是 avg@8。`eval_suite.py` 按每题
   采样数过滤,k=8 和 k=32 不会混 —— 用默认值跑出来的是另一套读数,跟已有 36 格不可比。
3. **`eval_refill_exp.py` 必须带 `--grid light`。**不带的话 25/50/150 步永远判不出完成,
   worker 反复认领同一批空活。
4. **`verdict.py --base vanilla_corr`。**不给的话基线默认 `vanilla`,名册里没有这条 run,
   每一行都**安静地** MISSING。
5. **指纹不符不要 `RESUME=force`。**那是把两段配置拼进一条曲线。另外**指纹只哈希配置、
   不哈希库版本** —— 换了 verl/vLLM/torch,指纹照样放行,差异静默骑在曲线上。装完对照
   手册 §3.4 的实测版本逐项核。
6. **`trainer.use_v1=False` 不能动。**V1 下横幅照打、归档一个文件不写。
7. **A 轴要 `assets/`**:a4 读 `gkd_offpolicy.parquet`(`SIMOPD_GKD_CACHE`),a5 读
   `gkd_offpolicy.parquet.dry`(`SIMOPD_A5_KEYS`)。没有它们就得先跑一个 12 小时的生成作业。
8. **在环 val 和离线评测不是一个量,别混用。**同一条 `vanilla_corr`,在环(verl grader,
   greedy mean@1)step 25→250 是 +5.2,离线评测(avg@3 t=0.7 32k)是 +12.1;
   差距不是 16k 截断(模拟一刀切最多掉 0.5 分)。见 [`EVAL-LEDGER-20260918.md`](EVAL-LEDGER-20260918.md) §3。
9. **判分器完全依赖 `\boxed{}`**(没写 box 的 54,000 条响应正确率 0.0%)。学生在评测里
   85–98% 都写了,所以现有读数没问题;但改 prompt 模板或解码设置时要盯住这个比例。

## 6 等拍板的

这几件不是执行问题,需要项目负责人定:

1. **`c5_union_rkl` 怎么存档。**25 步要 7.4 h,短于这个的调度窗口里它**永远落不了档**
   (上游为此卡了九天)。三个选项:给长窗口 / `SAVE_FREQ=10` / 单独连续跑。
   已核:`save_freq` **不在指纹里**、不改训练数值,改 10 的代价只是磁盘(每个存档 26.8 GB)。
2. **avg@8 还是 avg@32。**协议原文是 avg@32,实际产出全是 avg@8。要么接受 avg@8 并在
   论文里写明;要么全部重评(AIME/AMC 的生成量 ×4,先把已有 155 份挪走,别混放)。
3. **训练在哪跑。**Cornell 的 80 G 卡只有 `nlplarge-compute-01` 那 8 张 A100,
   现被 verification 项目占到 9/25 截稿之后。要么等,要么找别处的 80 G 卡。

## 7 怎么复查进度

```bash
# 本文 §2–§3 的表,从实物现算
python3 scripts/analysis/remaining_table.py configs/retrain_roster.tsv <runs 解包目录> <evals 目录>
# 训练还差多少机时(读各臂 perf/time_per_step)
python3 scripts/analysis/step_times.py <runs 解包目录>
# 评测缺口的逐格清单(手册的工具)
python3 scripts/eval_refill_exp.py --grid light
```

表会随进度变。跑完一批就重算一次,把本文更新掉 —— 过期的"还剩什么"比没有还糟。
