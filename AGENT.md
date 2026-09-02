# AGENT.md — 换集群 / 接手先读这一页

**写于 2026-08-24,起因:08-23 14:19 两张 DLC 单同时掉线,64 卡停机,可能要换集群。**
这页只回答三件事:哪些结论已经定了(别重跑)、现在要跑什么(按优先级和预算)、
换集群要带走什么。方法与机制读 `docs/MECHANISMS.md`,派工真源是 `configs/campaign.tsv`,
臂的唯一真源是 `configs/arms.yaml`。

---

## 0 一句话现状

修正终止符身份错位(内核 `k1_rec` → `k1_termfix`)之后,**vanilla 不再晚期塌缩**;
坏载体上量到的"药效"在修正载体上**归零**。所以现在缺的不是新方法,是**把修正载体上的
曲线补齐到能下判决**——训练还差约 610 GPU·小时,评测欠着 1636 个格。

---

## 1 已经定了的(别重跑)

| 结论 | 证据 | 位置 |
|---|---|---|
| 晚期掉分是"不会停",不是能力退化 | legacy 三种子截断 .92–1.00、长度撞 32k 帽 | MECHANISMS §M-I |
| 修正内核消除塌缩 | 单旋钮对照(同为 `STOP_IDS=off`):.247 → **.348@200**,截断 .97 → .10 | MECHANISMS §M-I cure |
| 独立复现 | `c4_carrier` / `c4_rep` 跑满 250,composite **.353**、截断 .07/.08 | 同上 |
| 坏载体上的"药效"是抗塌缩 | c2/c4 在坏载体 +0.107@250,修正载体 ±0.01;同批极差 .086 → ≤.012 | 判读台 §3 |
| 大多数臂动态上分不出来 | 25/34 条健康臂晚期均长挤在 8.5k–9.7k | 判读台 §4 |
| 没有早期预警阈值 | 健康的 vanilla_corr 在 35–48 步把三个阈值全触发过 | 判读台 §8 |

**限制**(每次引用结论都要一起说):corr 侧全是 **n=1 种子**;`vanilla_corr@250` 的评测
**没跑**;`_n0` 那组同时改了载体和停止契约,不是单旋钮,且其 `stop_set=151645`
与 `off` 的臂**不可做离线相减**。

---

## 2 重训计划(2026-08-24 定案:旧机器不再用,ckpt 拿不回来)

旧集群不再使用,而**代码里从来没有任何上传路径**(`scripts/`/`deploy/`/`src/` 无
`push_to_hub`/`create_repo`/`upload_folder`;verl 的 FSDP checkpoint manager 的 `hdfs_path`
明写 `Unused`;训练命令只有 `default_local_dir`;wandb 也不传 artifact)——
**ckpt 只在共享盘存过一份,308 个还没评测的 ckpt 连同 1636 个评测格一起没了。**

### 2.1 不用重训的(已在 git 里,结论已经成立)

08-24 那次归档保住了大部分东西。**这些不要重跑**:

| 已成立的结论 | 靠什么数据 | 状态 |
|---|---|---|
| 治愈对照(headline) | legacy vanilla 三种子全评完 + `vanilla_corr` 25–225 九个完整格 | ✅ 曲线到 225,判据用晚期斜率,**不需要 250** |
| 药效归零 | b2 / c2 / c4 的 legacy 与 corr 版在 ≤175 步的同步差 | ✅ 全部已评 |
| 独立复现 | `c4_carrier` / `c4_rep` 评到 250(.353) | ✅ 完整 |
| 吸引子 / 早期阈值负结果 / 长度分层 | 51 run 的逐步动态(git 里有,wandb 云端还有一份) | ✅ 双备份 |
| 整个 legacy 波(29 臂 × 3 种子) | `post_eval_cells.csv` 4700 格 | ✅ 完整 |

### 2.2 重训名册(2026-09-01 定稿:**全部重跑 = 论文臂表**,34 条 + 2 条零成本)

单种子、评测 k=8、每 25 步。全文与合并依据见 `docs/ARM-REVIEW-20260901.md` §5。

| 轴 | 臂 | 步 | GPU·h | 它回答什么 |
|---|---|---|---|---|
| 对照 | `vanilla_corr` | 250 | 157 | 判决基线 |
| 对照 | **`vanilla_te`(新登记)** | 250 | 157 | TE Path 1 是否 ≡ termfix |
| B | b1_skew_kl_corr · b2_forward_kl_corr | 200 | 235 | null 代表 · 分数≈交付坏 |
| C | c3_intersection_corr · c5_union_rkl · c5_union_fkl | 200/250 | 319 | 省 token 候选 · 两条预注册(c5 = OURS) |
| C·OURS | **c2_quantile_budget_corr · c4_hq · c4_state** | 200 | 289 | 自研方法必须在表里有自己的行:c2(分位预算)、c4 的状态自适应控制器两档。归档 null / 对照裁决仍然成立,这三条是把它们**在同一张表里评出来**,不是重新提问 |
| C·OURS 梯子 | **c2_qb_fixed8_corr · c2_qb_perseq_corr(新登记)** | 200 | 236 | c2 的 QB pinning ladder:固定 top-8(零自适应)→ 每轨迹一个 τ → 全量分位。讲透「预算的自适应哪一级在起作用」。登记为 rung + 与 c2_quantile_budget_corr **相同的修正**(TE=1),梯子内部才是单变量 |
| C | *c4_carrier / c4_rep* | — | **0** | 已评满 250 |
| D | d2_selectkd_corr · d3_teachability_corr | 200 | 194 | D 代表(d1 并入 d2:+0.1%)· d3 修正后 **+32% 长度**,动态不同,必须自己一条 |
| E | e2_set_coverage_a0_corr | 200 | 140 | 第二条塌缩通路 |
| F | f2_hard_clip_corr · f3_power_corr | 200 | 236 | 文献标准做法(硬截 ±10)· legacy 效应最大处(+.090)的 null;合并 f1/f2_clip2.3/f4/f5 |
| G | g6_seqmean_corr · g1_verified_only_corr | 200 | 288 | G null 代表(合并 g2/g4/g5,±2% 内)· g1 修正后 13.2k/.53 有副作用,G 轴最经典的一条,要评分数 |
| H | h1 · h2 · h3 · h4 | 200 | 538 | 最短 · 照塌 · null 代表 · h4 有自己的预注册问题(窗口 vs 散点)且动态不同(+23%,.46) |
| N | **`n2_gather`(新登记)** | 250 | 167 | N2 自己的效果,TE off |
| A | a1_gkd_mix0.5_n0 · a3_offpolicy_n0 · a4_dagger_anneal_n0 · a5_aggrevate_n0 | 200 | 410 | 四种轨迹来源各一条:固定混合 / 纯离策 / DAgger 退火 / AggreVaTe。a4、a5 与 a1、a3 在修正载体上动态**不同**(9.9k/.22、9.7k/.21 vs 8.3k/.12、6.8k/.08),且 A-AXIS-REGISTRATION 对 a4 有 R1/R2 预注册裁决,不能合并。对照仍是 vanilla_corr,契约差异属数据来源本身,论文里声明 |
| | **34 条** | | **≈3830** | **32 卡 5.0 天 / 64 卡 2.5 天** |

**不跑的**:legacy 载体(eos 错位版)一条都不跑——塌缩已由三种子 + 16 条老臂 + w 对钉死,用户 2026-09-02 定 (归档 5 点已给 null)· b5/d1/d3/
g2/g4/g5/f1/f2_clip2.3/f4/f5/b5/d1(修正后与代表 ±3% 内且无独立预注册问题,合并)· n2_termcal/n2_corr(已答/被 TE 混淆)·
b3/b4/c1/e1/e3/j1/vanilla_n8/a2(入库数据或 exempt)。

**四条新登记**(需用户在 `configs/arms.yaml` 登记):`vanilla_te` = `k1_rec` + gather +
`TERM_EVENT=1`;`n2_gather` = `k1_termcal` + gather、TE off;`c2_qb_fixed8_corr` / `c2_qb_perseq_corr`
= 对应 rung + 与 `c2_quantile_budget_corr` 相同的修正旋钮(TOPK 66、gather、TE=1)。
**搁置**(用户 2026-09-02 定):d1 的 entropy_only / divergence_only 分解、g2 的 filter_only / reweight_only
分解、c1_tailbucket ——不进本轮,登记保留。**登记纪律**:top-k 族里原生读
终止符列的内核(c4/c5)走 gather-only,TE 只给采样式 k1 族。

### 2.2b 评测减重:**每 25 步照测,k 从 32 降到 8**

步数网格不动(10 个点全测),减重靠采样数。成本结构 —— **一格约 68M token**:

| benchmark | 题 × k | 生成条数 | 中位长度 | token 占比 |
|---|---|---|---|---|
| aime24 / aime25 | 30 × **32** ×2 | 1,920 | ~19k | **54%** |
| amc23 | 40 × **32** | 1,280 | 12,173 | **23%** |
| minerva / math500 | 272×3 + 500×3 | 2,316 | ~7k | 23% |

**k=32 的三项占 77%**,所以杠杆只有 k。**定为 k=8**(用户 2026-08-24 定):

| k | 省 token | 评测 GPU·h | 单格总噪声 | 分辨 ±0.01 |
|---|---|---|---|---|
| 32(旧) | — | 1350 | 0.0056 | 1.8σ |
| 16 | 38% | 830 | 0.0068 | 1.5σ |
| **8** | **58%** | **570** | **0.0086** | **1.2σ** |

**文献位置**:k=8 = FiRe 的 Avg@8;文献里 k∈{1,4,8,16,32} 全都有(`docs/METRICS.md` §0.1),
我们原来的 32 来自 Demystifying、采样参数却来自 Rethinking —— 本就是混搭。8 在分布中游,
高于 ESR 的 4 与一票 pass@1。

**代价与用法(必须照做)**:composite 用 `avg@k`(逐题正确率均值),**k 变了估计量不变、只是
方差变大**;但 k=8 时单格总噪声 0.0086,**"药效归零"那档 ±0.01 的差只有 1.2σ —— 逐格读不出来**。
所以那条结论**必须在曲线层面读**:6 个点取均值后噪声降到 0.0035,±0.01 = **2.9σ**,仍站得住。
逐格的小差异一律不得单独下结论。

**pass@k 不可跨 k 比**:已入库 4700 格是 k=32,新跑是 k=8;只有 avg / composite 可比。
产物不会串味 —— `newest()` 按每题采样数过滤(2026-08-07 F1 的 era 门)。

```
SIMOPD_SUITE_K=8      # 写进 simopd_env.sh;只作用于原 k=32 的两组,minerva/math500 的 k=3 不动
```

**不动的**:`MAX_TOKENS=32768` —— 健康臂只写 ~9k 用不到帽,只有塌缩臂跑满,
而**截断率是头号仪表**,降帽等于改测量对象。

**逐批预算见 §2.2 的表:训练 1380 + 评测 509 = 1888 GPU·小时。**

**不再重训**:全部 legacy 载体臂、`n2_termcal`(每步 1444 秒最慢且已塌透)、
以及没进 T2 面板的其余 f/g/d/h 臂(动态已归档、分类已完成)。

### 2.3 新集群第一天就要做的一件事

**起 ckpt 同步守护**,别再让"只存一份"发生:

```
export HF_TOKEN=...            # 只从环境读
unset HF_HUB_OFFLINE
python scripts/ckpt_sync.py --repo <org>/simopd-ckpts --watch 600 \
    --with-optimizer vanilla_corr_s0_16k,vanilla_corr_s1_16k,vanilla_corr_s2_16k
```

只传 `actor/huggingface/`(3.4 GB/个,够评测);`--with-optimizer` 点名的 run 连
FSDP 分片一起传(28.4 GB/个),保住"以后想真 resume"的选项。幂等、断点续传、单个失败
不拖垮整轮;**写完再传**(目录静置 120 秒才认,不传半截)、**落地校验**(比对远端文件名与
字节数,不一致就不记账、下轮重传)。

**更省事的做法:不用手动起。** 舰队引擎已接入(opt-in),在 `simopd_env.sh` 里设好
`CKPT_SYNC_REPO`(可选 `CKPT_SYNC_FULL` 点名要连 optimizer 的 run)和 `HF_TOKEN`,
每个训练 pod 起 lane 时会自己拉起同步器、按 pod 去重;没设 `CKPT_SYNC_REPO` 就是空操作。
设了却没有 `HF_TOKEN` 会**大声报错**而不是静默跳过 —— 静默正是这次丢数据的方式。

## 3 现在要跑什么

### P0 · 判决收口(3 个格,约 27 GPU·小时)

`vanilla_corr@250`、`n2_corr@250`、`b1_skew_kl_corr@250` 的 ckpt 都在盘上,五基准评测没跑。
**这是唯一挡着判决表出不来的东西**,一上电先跑它。

```
# 队列会自动带上(refill 给 vanilla_corr 优先级 0);要手工插队就直接写队首
printf '%s\n' "vanilla_corr_s0_16k 250" "n2_corr_s0_16k 250" "b1_skew_kl_corr_s0_16k 250" \
  | cat - $D/evalq_exp/pending.txt > /tmp/p && mv /tmp/p $D/evalq_exp/pending.txt
```

### P1 · 省 token 假说(约 450 GPU·小时,**纯评测,不占训练卡**)

同契约、同窗口下,这几条**比 vanilla_corr 短**,但修正版一个格都没评:

| 臂 | 比 vanilla_corr 短 | 老版 @250 分数 |
|---|---|---|
| `h1_first_segment_corr` | −48.8% | 0.303(明显偏低) |
| `c3_intersection_corr` | −15.5% | 0.345 |
| `d2_selectkd_corr` | −6.3% | 0.227 |
| `g2_fire_likelihood_corr` | −3.3% | 0.269 |

已证实"同分更省 token"的只有 c4 家族(−9.4%、.353)。上面四条评出来才知道是"省"还是"废"。
**别评满 10 个 ckpt**:先评 100 / 175 / 250 三步,不合格就停。

### P2 · 训练补满(约 610 GPU·小时 ≈ 32 卡跑 19 小时)

16 条 lane 停在半路,从最近 ckpt 续跑即可(引擎自己会续):

| 剩余 | 臂 |
|---|---|
| 25 步 | e2_set_coverage_a0_corr · g4_failure_only_corr · g6_seqmean_corr · a3_offpolicy_n0 |
| 50 步 | f1_soft_log_corr · f2_hard_clip_corr · h6_gen_sched_n0 · d1_tip_corr |
| 75–125 步 | f5_tanh_corr · h4_random_scatter_corr · f2_clip2.3_corr · f4_posclip_corr |
| 150–175 步 | h10_task_subset_n0 · c5_union_fkl · n2_termcal · c5_union_rkl |

**c5_union_fkl 必须跑过 163 步**——它的预注册 onset 在那之后,现在停在 114 步,
既没证实也没证伪。`n2_termcal` 每步 1444 秒是全场最慢,单它就占 60 卡对·小时。

### P3 · 缺的种子与覆盖

- **`vanilla_corr` 的 s1 / s2**:整条治愈结论现在压在一个种子上,这是最脆的一环(每种子约 60 卡对·小时)。
- 评测欠 **1636 格**(全评 ≈ 14.7k GPU·小时 / 32 卡 19 天)——**不要全评**,按论文主表需要的臂和步挑。
- `c4_hq` / `c4_state` 只跑到 100 步,它们的预注册判据("greedy-32k 截断 20% → 个位数")要跑深才看得见。
- `vanilla_s0_w` 的评测当年失败,w 对至今**没有自己的基线**;补上它,跨对复现才完整。

### P4 · 一条读数(不要钱)

集群一通就重导一次全键动态,把 c5 的预注册判据读出来,并让尾巴补丁退休:

```
python scripts/export_wave_metrics.py --since 2026-08-19 --out docs/data/training_metrics_corr_allkeys.csv.gz
python scripts/analysis/export_wave_dynamics.py --out docs/data/inloop_wave_dynamics.csv   # 已含 un_p_imend 三列
python scripts/analysis/collapse_status.py --write && python scripts/analysis/make_cure_page.py
python scripts/make_dynamics_page.py && python scripts/make_campaign_tables.py
```

判据:**`un_p_imend` 上升而 `un_p_eot` 下降** = 学生把教师的终止符当 token 学会了
(`c5_union_fkl` 本地 dump 只到第 4 步,读不出来)。重导后把
`training_metrics_corr_tail_20260823.csv.gz` 从 `make_dynamics_page.py` 的默认输入里删掉。

---

## 4 换集群:带什么 / 重建什么

**必须带走**(`$D = .../simopd_data`):

| 东西 | 体量 | 说明 |
|---|---|---|
| `ckpt/simopd/<arm>_s0_16k/global_step_*` | 33 TB 全量 | **续训只需每臂最新那个**(19 臂 × ~83 GB ≈ **1.6 TB**);**评测只需 `actor/huggingface`**(每个约 3.4 GB,395 个 ≈ 1.3 TB) |
| `evals/*.parquet` | 14 GB / 4833 份 | 已入库的分析表能重建,但原文列(`response`)只在这里 |
| `simopd_math/` | 数据集 | 训练/评测都要 |
| `evalq_exp/` | 小 | 队列与 claims;`pending.txt` 可由 `eval_refill_exp.py` 从盘上重建 |
| `eval_worker_exp.sh` | 小 | **不在 git 里**,只在盘上;含 2026-08-23 的读卡修补 |
| `simopd_env.sh` | 小 | 含 wandb key,**别进 git、别回显** |

**可以重建**:`docs/data/*.csv`(从 evals 重跑抽取)、两张 HTML、报告表、队列。

**必须重搭**:venv(`simopd/`,pandas+pyarrow+vLLM+verl)、共享盘挂载、以及"能跑长作业"的载体。

### 4.1 ckpt 走 HuggingFace(2026-08-24 定稿:**只传权重**)

前提已经变了:换机器后**不能 resume**(优化器状态与数据顺序留在旧盘),而 §2.5 的方案是
"降锚点 + 只从 0 重跑 4 条",**所以全量 `global_step_N/` 不用传**。传两样就够:

| 传什么 | 体量 | 换来什么 |
|---|---|---|
| 每个 ckpt 的 `actor/huggingface/`(约 3.4 GB) | 395 × 3.4 GB ≈ **1.3 TB** | **全部 1636 格评测欠账**在任何有卡的地方都能跑 —— 这是剩余工作量的绝大头 |
| `evals/*.parquet` → 一个 **dataset** repo | **14 GB** | 回答原文列(`response`)只在这里;分析表能重建,它不能 |
| ~~非最新步的 FSDP 分片 + optimizer~~ | ~~约 31 TB~~ | ~~从旧步精确续~~ —— 已放弃 resume,不传 |

**合计约 1.3 TB**(原 33 TB 的 4%)。上传顺序:先 @200 主表要用的 31 条臂,再补其余。

**传的时候**:仓库设 **private**(未发表产物);模型卡注明基座 `Qwen3-1.7B-Base` ←
教师 `Qwen3-4B-Instruct-2507` 及其许可;上传前 **unset `HF_HUB_OFFLINE`**
(`simopd_env.sh` 里是开的,不 unset 会静默失败);大目录用 `hf upload-large-folder`;
`HF_HUB_DISABLE_XET=1` 是为在线拉取设的,**批量上传建议放开 Xet**(块级去重,重复的
tokenizer/config 不必重传);**目录结构原样保留** `<arm>_s0_16k/global_step_N/actor/huggingface/`
—— 下游脚本全按这个 glob 找 ckpt,拍平了要改一堆代码。

**删源盘前必须做**:传前在源盘生成清单(文件大小 + 关键文件 sha256),拉下来逐条比对;
再抽一个 ckpt 真跑一次 5 基准评测,确认权重能载入、数字对得上。
**注意这一步不可逆**:全量 ckpt 一删,"以后想真 resume"这个选项就永远没有了 ——
如果对某条臂还存有"将来要接着跑"的念头,那条臂的最新 `global_step_N/` 全量(28.4 GB)
要单独留一份。

**载体的可移植性**:`deploy/dlc/forever.sh` 只依赖三件事——共享文件系统、一个能长跑的作业、
容器里有 `python3`。它**不依赖 DLC 特有的东西**,pod 识别门认的是
`MLP_ROLE_INDEX / RANK / MASTER_ADDR / POD_NAME` 里任意一个存在(见 `_podsig`)。
换到 Slurm/K8s 时:确认新平台会注入其中之一(否则载体会以为自己是提交端、打卡片就退出),
其余照旧。仓库里 `slurm/` 有一套旧脚本可参考。

---

## 5 环境与凭据(只列名字,值不写进任何文件)

`WANDB_API_KEY`(在 `$ROOT/simopd_env.sh`,source 它,别 echo)、`HF_HOME` / `HF_ENDPOINT` /
`HF_HUB_OFFLINE`、`VLLM_USE_MODELSCOPE` 与 `VERL_USE_MODELSCOPE`(**新集群务必显式设成
False**,除非真装了 modelscope——2026-08-23 就是容器注入了 `True` 把整条评测队列烧成 FAILED)、
`CKPT_ROOT` / `DATA_DIR` / `SIMOPD_STORE`。

---

## 6 上电顺序

1. 挂盘、建 venv、`git clone` 到 `$ROOT`,`source simopd_env.sh` 自检 wandb 可达。
2. 起载体(DLC 形态见 `bash deploy/dlc/forever.sh` 打印的控制台卡片;**表单里的自动重启一定要开**)。
3. 装 payload:评测节点 `task.sh set <槽> $ROOT/deploy/dlc/eval_farm.sh`;
   训练节点 `task.sh set <槽> $ROOT/deploy/dlc/slot_resume.sh` + 写 lane 图
   `$D/corr_wave/slot<k>_s0_lanes`(格式 `arm:gpu,gpu`,一行)。
4. `task.sh status` / `task.sh alive` 看心跳;`task.sh sh <槽> '<命令>'` 当 ssh 用;
   `task.sh tty <槽>` 是真交互 shell。
5. P0 的三个格插队,然后 P2 的 lane 图铺开。

---

## 7 会咬人的坑

- **过滤器按命名习惯写死**:这一轮连中三次(`run` 名含 `16k`、臂名正则要 `_corr_s0_16k$`、
  按 `run` 而不是 `arm` 关联)。症状都是"数据在、脚本不报错、页面安静地少几条"。
  加过滤器时**排除不可比的**,别**包含"看起来对的"**。
- **契约不可混池**:`stop_set=off` 与 `151645` 是两套协议,composite 不能相减;
  训练侧 `STOP_IDS` 同理会机械地改变长度与截断。比之前先摊平契约。
- **舰队引擎的单槽守卫**:`SLOT` 显式给定且 `rank != 0` 时会永久空转 —— 载体分槽的场景要
  `CORR_SLOT_OWNED=1` 豁免(`slot_resume.sh` 已经带了)。24 卡曾因此白转 19.5 小时。
- **交卡竞态**:lane 跑满后 `_eval_handoff` 会把卡交给 eval worker;重发时镰刀要连
  `eval_worker_exp.sh` **外壳**一起杀(已修),否则新 lane 会 OOM 三次然后放弃。
- **nvidia-smi 不认数字 `CUDA_VISIBLE_DEVICES`**:worker 判"我的卡空不空"必须按 index 精确取,
  否则同 pod 所有 worker 会误判卡忙、集体坐等(已修,但换集群后值得复验一次)。
- **别在跑着的 bash 脚本上原地改**:换 inode(写新文件再 mv)才安全。
- **同一个 ckpt 目录不能有两个写者**:迁移时先确认旧 lane 真死透(日志 >15 分钟没动静)。
