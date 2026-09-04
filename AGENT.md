# AGENT.md — 换集群 / 接手先读这一页

**写于 2026-08-24,起因:08-23 14:19 两张 DLC 单同时掉线,64 卡停机,可能要换集群。**
这页只回答三件事:哪些结论已经定了(别重跑)、现在要跑什么(按优先级和预算)、
换集群要带走什么。方法与机制读 `docs/MECHANISMS.md`,派工真源是 `configs/campaign.tsv`,
臂的唯一真源是 `configs/arms.yaml`。

---

## 0 一句话现状

修正终止符身份错位(内核 `k1_rec` → `k1_termfix`)之后,**vanilla 不再晚期塌缩**;
坏载体上量到的"药效"在修正载体上**归零**。所以现在缺的不是新方法,是**把修正载体上的
曲线补齐到能下判决**——旧机器已弃、ckpt 未上云,所以是 §2 的全部重跑(35 条 ≈ 3950 GPU·h)。
2026-09-02 起每个 run **自带分析归档**(§2.4):轨迹、逐 token 学生/教师量、每序列 FKL/RKL/JSD、
熵、判分、run 定义,随权重一起上 HF —— 旧集群失联时"只剩 eval parquet"的局面不会再有。

> **2026-09-02 复核发现的硬阻塞已解除**:`eval_worker_exp.sh` 从未进过 git、随旧盘丢失,
> 而评测农场整条链子指着它。**已按契约重写并放进仓库**(`deploy/dlc/eval_worker_exp.sh`,
> 30 条断言的无 GPU 测试台 `deploy/dlc/test_eval_worker.sh` 全绿),三个调用点改为优先用
> 仓库里那份、盘上副本兜底。详见 §2.6。

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

### 2.2 重训名册(2026-09-01 定稿:**全部重跑 = 论文臂表**,35 条 + 2 条零成本)

单种子、评测 k=8、每 25 步。合并依据见 `docs/ARM-REVIEW-20260901.md` **§6–§6.5**
(§5.3 那张「20 条」表已被 §6 的复查整体推翻 —— 它还写着 `n2_gather`、c4 行 = `c4_carrier`、
「删 h5/h7/h8/h9 预算臂」,**照它铺 lane 会跑错**;终稿是 §6.5 的 35 条 ≈3950)。

| 轴 | 臂 | 步 | GPU·h | 它回答什么 |
|---|---|---|---|---|
| 对照 | `vanilla_corr` | 250 | 157 | 判决基线。**它已跑完(旧集群到 250,评测 25–225 九格已入库),但仍要重跑**:新集群所有臂是新栈 + k=8,入库的是旧栈 + k=32,基线不同栈则全表相减都是跨栈跨 k;`vanilla_te` 也只能对同栈的它比。入库那条改任「治愈的跨集群复现」,与新的并排 |
| 对照 | **`vanilla_te`(新登记)** | 250 | 157 | TE Path 1 是否 ≡ termfix |
| B | b1_skew_kl_corr · b2_forward_kl_corr | 200 | 235 | null 代表 · 分数≈交付坏 |
| C | c3_intersection_corr · c5_union_rkl · c5_union_fkl | 200/250 | 319 | 省 token 候选 · 两条预注册(c5 = OURS) |
| C·OURS | **c2_quantile_budget_corr · c4_pi_tail_budget_corr · c4_hq · c4_state** | 200 | 408 | 自研方法必须在表里有自己的行。**c4 的表行 = `c4_pi_tail_budget_corr`(TE=1,与全表同旗标)**;c4_hq / c4_state 的闸门读 q_et,代码拒绝折叠(`if EG.enabled() and not TERM_EVENT`),只能 gather-only,声明为例外 |
| C·OURS 梯子 | **c2_qb_fixed8_corr · c2_qb_perseq_corr(新登记)** | 200 | 236 | c2 的 QB pinning ladder:固定 top-8(零自适应)→ 每轨迹一个 τ → 全量分位。讲透「预算的自适应哪一级在起作用」。登记为 rung + 与 c2_quantile_budget_corr **相同的修正**(TE=1),梯子内部才是单变量 |
| C | *c4_carrier / c4_rep* | — | **0** | 已评满 250(.353);作 c4 行的**消融**「修而不折叠」,与 c4_pi_tail_budget_corr 成单变量对 |
| D | d2_selectkd_corr · d3_teachability_corr | 200 | 194 | D 代表(d1 并入 d2:+0.1%)· d3 修正后 **+32% 长度**,动态不同,必须自己一条 |
| E | e2_set_coverage_a0_corr | 200 | 140 | 第二条塌缩通路 |
| F | f2_hard_clip_corr · f3_power_corr | 200 | 236 | 文献标准做法(硬截 ±10)· legacy 效应最大处(+.090)的 null;合并 f1/f2_clip2.3/f4/f5 |
| G | g6_seqmean_corr · g1_verified_only_corr | 200 | 288 | G null 代表(合并 g2/g4/g5,±2% 内)· g1 修正后 13.2k/.53 有副作用,G 轴最经典的一条,要评分数 |
| H | h1 · h2 · h3 · h4 | 200 | 538 | 最短 · 照塌 · null 代表 · h4 有自己的预注册问题(窗口 vs 散点)且动态不同(+23%,.46) |
| H 预算线 | **h5_gen100_n0 · h7_gen512_n0 · h8_gen2048_n0 · h6_gen_sched_n0 · h9_prune_adapt_n0 · h10_task_subset_n0** | 200 | 465 | 深度剂量 R-H1–R-H4 四条预注册裁决。曾被我在压预算时整条删掉、被用户抓回(ARM-REVIEW §6 「不能删 → 整条预算线恢复」)。训练极便宜(h5 约 8 秒/步),成本几乎全在评测;**它们自带小帽,截断率天然高,别按主表阈值误判成塌缩** |
| N | n2_corr | 250 | 167 | N2 校准损失在修正基座上的效果(对照 vanilla_te,同为 TE 路线)。`n2_termcal` 已是「termcal + gather、TE off」且已塌,不再另登记 n2_gather |
| A | a1_gkd_mix0.5_n0 · a3_offpolicy_n0 · a4_dagger_anneal_n0 · a5_aggrevate_n0 | 200 | 410 | 四种轨迹来源各一条:固定混合 / 纯离策 / DAgger 退火 / AggreVaTe。a4、a5 与 a1、a3 在修正载体上动态**不同**(9.9k/.22、9.7k/.21 vs 8.3k/.12、6.8k/.08),且 A-AXIS-REGISTRATION 对 a4 有 R1/R2 预注册裁决,不能合并。对照仍是 vanilla_corr,契约差异属数据来源本身,论文里声明 |
| | **35 条** | | **≈3950** | **32 卡 5.1 天 / 64 卡 2.6 天** |

**表里是简写,lane 图要精确名。** 下面 37 行是 `arm.py env` 全部解析通过的真名
(2026-09-02 逐条验过),直接抄:

```
vanilla_corr            vanilla_te
b1_skew_kl_corr         b2_forward_kl_corr
c3_intersection_corr    c5_union_rkl              c5_union_fkl
c2_quantile_budget_corr c4_pi_tail_budget_corr    c4_hq          c4_state
c2_qb_fixed8_corr       c2_qb_perseq_corr
d2_selectkd_corr        d3_teachability_corr
e2_set_coverage_a0_corr
f2_hard_clip_corr       f3_power_corr
g6_seqmean_corr         g1_verified_only_corr
h1_first_segment_corr   h2_last_segment_corr      h3_random_segment_corr   h4_random_scatter_corr
h5_gen100_n0            h7_gen512_n0              h8_gen2048_n0
h6_gen_sched_n0         h9_prune_adapt_n0         h10_task_subset_n0
n2_corr
a1_gkd_mix0.5_n0        a3_offpolicy_n0           a4_dagger_anneal_n0      a5_aggrevate_n0
# 零成本(已评满 250,不重跑,只作 c4 行的消融):c4_carrier  c4_rep
```

`h1`/`h2`/`h3`/`h4` **不是** `arms.yaml` 里的 run_id,写进 lane 图会被 `arm.py env` 当场拒掉
(`unknown arm`)——那是好事,但别以为简写能用。

**不跑的**:legacy 载体(eos 错位版)一条都不跑——塌缩已由三种子 + 16 条老臂 + w 对钉死,用户 2026-09-02 定 (归档 5 点已给 null)· b5/d1/d3/
g2/g4/g5/f1/f2_clip2.3/f4/f5/b5/d1(修正后与代表 ±3% 内且无独立预注册问题,合并)· n2_termcal/n2_corr(已答/被 TE 混淆)·
b3/b4/c1/e1/e3/j1/vanilla_n8/a2(入库数据或 exempt)。

**四条新登记(2026-09-02 已入 `configs/arms.yaml`)**:`vanilla_te`、`c2_qb_fixed8_corr`、`c2_qb_perseq_corr`、`h5_gen100_n0`。
**基座规则(用户定)**:`k1_termfix` / N0 载体是**所有臂的 base**,臂 = base + 恰好一个旋钮,名册里没有
任何未修 eos 的臂。k1 族的旋钮经 `TERM_EVENT=1` 骑在 base 上——代码保证先跑 termfix 再套臂的数学,
所以每条 `_corr` 就是 `k1_termfix` + 旋钮,`vanilla_te` 是这条规则的**回归对照**(应与 vanilla_corr 重合)。
top-k 族**同样统一走 `TERM_EVENT=1`**(用户定,2026-09-02),c4 的表行是 `c4_pi_tail_budget_corr`,与全表同旗标;
唯一例外是代码拒绝折叠的内核——c5 union(`topk_losses:1527` 直接 raise,该格本就是测「无事件语义的几何」)和
c4 的 HQ/REP 闸门(q_et 只在 `not TERM_EVENT` 时读得到)——它们 gather-only 并在表里声明。c4_carrier/c4_rep
保留为 c4 行旁边的消融「修而不折叠」(单变量对:折叠让 pi_tail 长 25%)。
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

**预算以 §2.2 的表为准:35 条 ≈ 3950 GPU·h**(训练 + 评测,32 卡 5.1 天 / 64 卡 2.6 天)。
> 这里原先写着「训练 1380 + 评测 509 = 1888」——那是 2026-08-24 那版 16 条名册的数,
> 名册两次恢复(§6 的 9 条、§6.1–6.5 的 6 条)之后早就不成立了,已作废。

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

### 2.4 每个 run 自带分析归档(2026-09-02 起默认开,`SIMOPD_ARCHIVE=0` 关)

复盘一条曲线所需的原始量在训练时就写在 run 自己的 ckpt 目录里,`ckpt_sync.py` 随权重一起上 HF
(`<run>@aux`),不再依赖 wandb 登录或一个还活着的集群。全文与列说明见 `docs/RUN-ARCHIVE.md`。

| 文件 | 内容 | 频率 |
|---|---|---|
| `metrics/launch_<ts>.jsonl` | verl `file` logger:wandb 拿到的每个标量 | 每步 |
| `traj/light.jsonl` | 每序列:长度 / 熵(均值·末位·末 256)/ 末 id / 是否自然停止 / 截断 / 正文含终止符次数 / 重复 4-gram 率 / Σ学生 logprob / 训练题判分 | 每步 |
| `traj/div/rank<r>.jsonl` | **worker 侧**每序列:学生轨迹上的 forward KL / reverse KL / JSD / TV(教师块 + 尾桶,所有臂同一定义)+ 覆盖 qS/pS + top1 一致率,各取 mean/last/tail256 | 每步 |
| `traj/summary_<n>.parquet` | 整批每序列:上面 + 教师采样列 Σ / Δℓ 均值·末位·最大 / 末位教师 top1 / 末位各终止符概率 | 每 25 步 |
| `traj/ids_<n>.parquet` | 整批每序列的无损 prompt/response id(行为分析;解码即带终止符的原文) | **每步** |
| `traj/step_<n>.parquet` | 采样子集(seq_key % 8 == 0,≤32 条):逐 token 学生 logprob/熵、教师采样列、top1、各终止符列 | 每 25 步 |
| `traj/div/tok_*` | **整批**逐 token FKL/RKL/JSD/TV/qS/pS/agree(`SIMOPD_DIV_MOD`,默认 1) | 每 25 步 |
| `val_gen/<step>.jsonl` | 在环 math500 生成(verl 文本 dump) | 每 25 步 |
| `run_manifest.json` + `manifest/launch_*.json` | 解析后的臂 env、hydra 覆盖、契约、指纹、git sha、评测 k、机器 | 每次启动 |

三处记录靠 `seq_key`(响应 id 的 63 位哈希,driver/worker 各自算)对接。体量一个 run ~2–3 GB(`ids_` 每步 + 整批逐 token 分歧;见 §7 对 aux 同步的影响)。教师列按 id 在教师块里查找(不依赖 top-K / gather 的列布局),找不到记 NaN;
事件级修正的量可由各终止符列离线 logsumexp 重构。**都不进 resume 指纹**(改的是"记什么",不是 loss)。
verl 自己那份剥特殊符的文本 dump(`traj/_verl_text/`,每步整批 ~1 GB/run)默认不写(`SIMOPD_TRAJ_TEXT=1` 放行)。

**要保存的三类特征 → 文件**(2026-09-02 定):

| 特征 | 在哪 |
|---|---|
| ① rollout 本身:长度、熵、自然停止/截断、结尾 id | `traj/light.jsonl`(`resp_len ent_mean ent_last ent_tail256 last_is_stop truncated`);batch 级在 `metrics/` |
| ② 学生模仿教师:学生轨迹上的 forward KL / reverse KL / JSD,采样 token 的 Δℓ | `traj/div/rank*.jsonl`(`fkl rkl jsd tv qS pS agree` × mean/last/tail256);`summary_<n>` 的 `dl_*`;逐 token 在 `div/tok_*` 与 `step_<n>` |
| ③ 学生能力:训练题判分、在环 math500、坏模式频率 | `light.jsonl` 的 `score`;`metrics/` 的 `val/*` 与 `val_gen/`;坏模式 = `rep4 truncated n_stop_body ent_tail256` |
| 轨迹本身(行为分析) | `traj/ids_<n>.parquet`(整批无损 id,**每步**)+ `step_<n>.parquet`(子集逐 token 量,每 25 步)+ 离线评测 parquet 的 `token_ids/response/finish_reason` |

② 只能在 actor worker 上算(只有它同时握着学生完整分布与教师 top-K 块):`simopd/div_panel.py`
包住 verl 的 `compute_topk_loss`(逐 token)与注册表查表(按序列聚合落盘),**所有 top-k 臂同一定义**
(教师块 ∪ gather 终止列 ∪ 采样列按 id 去重,两侧各加尾桶;是真 KL 的下界),与臂自己的
`SUPPORT_MODE` / `TERM_EVENT` 折叠无关;分块 logsumexp,不物化 [N, V]。步号由 driver 经
`meta_info["simopd_step"]` 注入;熵由 driver 在 `_compute_old_log_prob` 处暂存。

旋钮(默认值都对;只列名字):`SIMOPD_ARCHIVE`、`SIMOPD_TRAJ_DIR / EVERY / IDS_EVERY / N / MOD / LIGHT / TEXT`、`SIMOPD_DIV_MOD`、
`SIMOPD_DIV_PANEL / DIV_CHUNK`、`LOGGER`(显式给了就不再加 `file`)。

**首个 run 的验收**(本地只用合成 batch 与 stub 过的 verl 接缝测过,真 batch 没跑过):
1. stderr 里有 `traj_dump armed` 和 `div_panel armed` 两行 —— 但**横幅不是证据,文件才是**。
2. 第 1 步后:`traj/light.jsonl` 每步 +256 行、`traj/ids_1.parquet` 出现且 256 行、`traj/div/rank*.jsonl` 有行且 `step` 非空;
   第 25 步后:`summary_25 / step_25.parquet` 与 `div/tok_step25_*.parquet` 出现(后者各 rank 文件合起来 256 行,整批),
   `metrics/launch_*.jsonl` 在长,`val_gen/` 有 25.jsonl。
3. 数值:`div` 里 `qS_mean` 接近 1(教师块覆盖),`summary` 里 `tch_lp_nan` 接近 0(采样列找对了),
   `light` 与 `div` 按 `seq_key` merge 后行数不掉。
4. `ckpt_sync` 日志有 `OK <run>@25` 与 `OK aux <run>`,HF 仓库里 `SYNCED.json` 在长。
任何一条不成立都是"看着武装了其实没有"的形状(h9 中继烧 66 步的那种),先修再铺 lane。

### 2.6 `eval_worker_exp.sh`:已丢失,已重写(2026-09-02)

**原状态**:从未提交进 git,本机无副本,旧盘不可达 = **没了**。§4 的"必须带走"清单写于
08-24(当时盘还在),现在那一行只是历史记录。

**现状态**:按下面的契约重写,**放在 `deploy/dlc/eval_worker_exp.sh`(在 git 里)**。
三个调用点已改成 `EVALW=${EVALW:-$ROOT/deploy/dlc/eval_worker_exp.sh}`、盘上副本兜底,
所以这次它不会再随任何一块盘消失。配套测试台 `deploy/dlc/test_eval_worker.sh` 不需要 GPU、
任何机器都能跑(mac 也行),30 条断言覆盖:抢单原子性(两 worker 只跑一次)、心跳打在
**目录**上、陈 claim 收尸、带 `owner` 的 claim 删得掉、**按 index 判卡**(index 0 忙时
不误判)、`--bench` 子集、失败退避、`ckpt` 缺失不热重试、队列行序即优先级、日志里
refill 要 grep 的状态词。**第一天先跑一遍它**:

```
bash deploy/dlc/test_eval_worker.sh      # 期望 RESULT: ALL PASS
```

**谁在等它**:

| 调用点 | 行为 |
|---|---|
| `deploy/dlc/eval_farm.sh` `_start_worker()` | 每卡 `nohup bash "$EVALW" <gpu> <队列目录>` |
| `deploy/dlc/eval_farm.sh` 值守循环 | 每 5 分钟按 `pgrep -f "eval_worker_exp.sh $g $Q"` 补起(basename 不变,所以换路径不影响去重) |
| `deploy/dlc/corr_wave_fleet.sh` `_eval_handoff()` | 交卡前检查存在;**不在就只打一行日志、卡闲着**(静默不评) |
| `deploy/dlc/worker.sh` backfill 段 | 同样的起法 |

**契约**(全部从现存代码反推,已实现;改这个脚本前先读):

1. 入参 `$1 = GPU index`(数字)、`$2 = 队列目录 `$D/evalq_exp``。
2. 队列:`$2/pending.txt` 每行一格,由 `scripts/eval_refill_exp.py --write --watch 1200`
   原子替换写出(整行读,读不到半截)。行格式见 `eval_refill_exp.py` 的 `_RUNPAT`。
3. 认领:在 `$2/claims/<run>__<step>/` 建目录 + 写 `owner` 文件,**每 5 分钟 touch 一次心跳**。
   refill 侧按 mtime 收尸:`age >= 7200` 视为陈旧、`rm -rf` 回收(`eval_refill_exp.py:71-95`)。
   **删 claim 必须 `rm -rf` 不能 `rmdir`** —— 目录里有 `owner` 文件,`rmdir` 删不动,
   2026-08-23 就是这么攒出 722 个僵尸 claim 把 651 行队列锁死的。
4. 干活:`python scripts/eval_offline.py`(在 git 里,没丢),`SIMOPD_SUITE_K=8`。
5. 选卡:`nvidia-smi` **不认数字 `CUDA_VISIBLE_DEVICES`**,要按 index 精确取行
   (2026-08-23 的读卡修补就在丢掉的那份里,重写时别再踩)。
6. 环境:顶上 `. ./simopd_env.sh` 并 `export VLLM_USE_MODELSCOPE=False VERL_USE_MODELSCOPE=False`
   (`eval_farm.sh` 已经做了双保险,worker 里再确认一次)。

**真集群验收**(测试台证明不了 vLLM 真起得来):队列放 1 格 → worker 认领(claims 里出现
目录)→ 出 parquet → claim 消失 → `pending.txt` 下一轮少一行。**在第一条臂跑到 25 步之前做。**

**还没被证明的**:真 vLLM 起得来、`eval_suite.py` 在新栈上跑得通、单卡显存够 32768 token。
这三件只能在有卡的地方验。

## 3 旧波数据的最后一次收口(不占卡)

> 这里原来是 P0–P4 五批"续跑"计划(2026-08-24 写,当时以为 ckpt 还在)。**P0–P3 已整体作废**:
> P0 的三格 @250 评测和 P2 的 16 条续跑都依赖盘上 ckpt(随旧集群失联);P1 的四条省 token 候选
> 现在要重训(在 §2.2 名册里);P3 的 s1/s2 种子按用户决定不跑、w 对基线不进本轮。
> **唯一还能做、且不占卡的只剩下面这一件。**

### 旧波 wandb 重导一次(c5 的预注册读数还能读到 114 步)

wandb 在云端,不随集群走。集群一通(或任何能连 wandb 的机器)重导一次全键动态,把 c5 的
预注册判据从旧波里读出来,并让尾巴补丁退休:

```
python scripts/export_wave_metrics.py --since 2026-08-19 --out docs/data/training_metrics_corr_allkeys.csv.gz
python scripts/analysis/export_wave_dynamics.py --out docs/data/inloop_wave_dynamics.csv   # 已含 un_p_imend 三列
python scripts/analysis/collapse_status.py --write && python scripts/analysis/make_cure_page.py
python scripts/make_dynamics_page.py && python scripts/make_campaign_tables.py
```

判据:**`un_p_imend` 上升而 `un_p_eot` 下降** = 学生把教师的终止符当 token 学会了。旧波里
`c5_union_fkl` 跑到 114 步、`c5_union_rkl` 到 91 步,趋势读得到,但预注册的 onset(163 步后)
读不到——那由 §2.2 里重跑的 c5 两条回答。重导后把 `training_metrics_corr_tail_20260823.csv.gz`
从 `make_dynamics_page.py` 的默认输入里删掉。

---

## 4 换集群:带什么 / 重建什么

> **这一节写于 2026-08-24,当时旧盘还连得上,所以它讲的是"搬运"。现在旧盘已不可达,
> 下面这张"必须带走"表已经是历史记录 —— 表里的东西没有一样搬得出来了。**留着是因为它
> 说清了每样东西的作用,而这正是 §2(全部重跑)与 §2.4(归档随权重上云)存在的理由。
> 唯一还要照做的是最后那行 `eval_worker_exp.sh`:它没了,必须重写,见 §2.6。

**当时的"必须带走"**(`$D = .../simopd_data`):

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

前提已经变了:换机器后**不能 resume**(优化器状态与数据顺序留在旧盘),而现在的方案是
**§2.2 的全部重跑(35 条,全从 0 起)**,**所以全量 `global_step_N/` 不用传**。传两样就够:

> (原文这里引的 "§2.5「降锚点 + 只从 0 重跑 4 条」" 是 08-24 那版计划,小节与方案都已作废。
> **本节整体也只对"旧盘还在"的世界成立**;对新集群真正生效的是 §2.3 的 `ckpt_sync.py`
> —— 边训边传,而不是训完再搬。)

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
`CKPT_ROOT` / `DATA_DIR` / `SIMOPD_STORE`;ckpt 与归档上云:`HF_TOKEN`(只从环境读)、
`CKPT_SYNC_REPO` / `CKPT_SYNC_FULL` / `CKPT_SYNC_EVERY`;评测协议:`SIMOPD_SUITE_K`(重训 = 8);
归档层:`SIMOPD_ARCHIVE`、`SIMOPD_TRAJ_*`、`SIMOPD_DIV_PANEL`、`VERL_FILE_LOGGER_PATH`(启动器自己算)。

---

## 6 上电顺序

0. **先造 `simopd_env.sh`** —— 它**不在 git 里**(`.gitignore:20`,因为含 wandb key),
   而 `run_opd_baseline.sh` / `eval_offline.py` / `preflight.py` / `fetch_assets.py` /
   `eval_farm.sh` / `coldstart.sh` 全都 source 它。新集群上它不存在,得先写。必须有:
   `WANDB_API_KEY`、`CKPT_ROOT`、`DATA_DIR`、`SIMOPD_STORE`、`HF_HOME`、
   `VLLM_USE_MODELSCOPE=False`、`VERL_USE_MODELSCOPE=False`、`SIMOPD_SUITE_K=8`、
   `CKPT_SYNC_REPO`、`HF_TOKEN`。**别进 git、别回显。**
1. 挂盘、建 venv、`git clone` 到 `$ROOT`,`source simopd_env.sh` 自检 wandb 可达。
1b. **拉模型与数据**(`HF_HUB_OFFLINE` 默认是 1,不先拉就是离线找不到权重):
   `python scripts/fetch_assets.py --data-dir $DATA_DIR`(学生 Qwen3-1.7B-Base + 教师
   Qwen3-4B-Instruct-2507 + 评测集),训练集 `scripts/prep_nemotron_math.py`。
   拉之前 `unset HF_HUB_OFFLINE`,拉完再设回去。
1c. **改路径**:`deploy/dlc/*.sh` 里有 20+ 处写死 `/mgfs/shared/Group_GY/changhao/...`。
   两处从前**硬写死没有覆盖口**的已经改成认环境变量(默认不变):舰队的数据根
   `corr_wave_fleet.sh` 的 `D` 现在认 **`SIMOPD_STORE`**(ckpt / 评测队列 / 日志 / lane 图
   全挂在它下面,是换集群第一个要改的),`eval_farm.sh` 的 venv 路径认 **`SIMOPD_PY`**。
   其余带 `${VAR:-默认}` 的用环境覆盖即可;`rehearse_n2.sh` 里的 `D` 仍是写死的,要彩排先改它。
2. 起载体(DLC 形态见 `bash deploy/dlc/forever.sh` 打印的控制台卡片;**表单里的自动重启一定要开**)。
3. 装 payload:评测节点 `task.sh set <槽> $ROOT/deploy/dlc/eval_farm.sh`
   (worker 已在仓库里,§2.6;装之前先 `bash deploy/dlc/test_eval_worker.sh` 看全绿);
   训练节点 `task.sh set <槽> $ROOT/deploy/dlc/slot_resume.sh` + 写 lane 图。
3b. **lane 图别手写** —— 35 条要带对步数(30 条 200、5 条 250),手抄错一条不会报错,
   只会安静地多跑或少跑 50 步。用生成器,它逐条过 `arm.py env`,解析不了就当场失败:
   ```
   python scripts/make_lane_map.py --print                      # 先看
   python scripts/make_lane_map.py --out $D/corr_wave --seed 0   # 写 slot<k>_s0_lanes
   ```
   格式是 `arm:gpu,gpu[:steps]`,第三段不写 = 250。8 卡 pod / 2 卡 lane 时 35 条 = 9 个 slot、
   70 张卡;64 卡就先铺 8 个 slot,余下的等前面跑完再链上去(`.next` 机制,见舰队脚本)。
   **没有覆盖文件时舰队现在会停下来喊**,不再回落到内置表 —— 内置的是**旧波**的 lane 图,
   24 条里有 8 条(b5 / d1_tip / f1 / f2_clip2.3 / g2 / g4 / g5 / n2_termcal)是本轮明确不跑的,
   少写一个文件就静默开跑另一个实验。真要用旧表:`FLEET_ALLOW_BUILTIN_LANES=1`。
3c. **彩排(Phase R)会先跑,别以为卡住了。** 舰队在起 lane 之前对每条没有 `.OK` 标记的臂
   跑一次三步彩排(`deploy/dsw/rehearse_n2.sh`,证明 sitecustomize 钩子在教师的 vLLM
   engine-core 里真的生效、终止符块真的到了内核),通过就自动 `touch
   $D/corr_wave/rehearsal_<arm>.OK`。**载体 `vanilla_corr` 的彩排必须先过**,否则所有 lane
   都不会起,日志每 2 分钟打一行 `CARRIER REHEARSAL FAILED/MISSING`。单条彩排失败只会
   丢掉那一条 lane(`rehearsal not PASS -- skipped`),别的照跑。新集群第一次会把 35 条
   全彩排一遍,要留出时间。
4. `task.sh status` / `task.sh alive` 看心跳;`task.sh sh <槽> '<命令>'` 当 ssh 用;
   `task.sh tty <槽>` 是真交互 shell。
5. `simopd_env.sh` 里给 `CKPT_SYNC_REPO` 与 `HF_TOKEN`(§2.3),`SIMOPD_SUITE_K=8`(§2.2b)。
6. **先只铺一条 lane**(建议 `vanilla_corr:0,1:250`,它也是载体彩排要过的那条),按 §2.4 的
   验收清单看归档层与 ckpt_sync 真在写、真在传;过了再用 3b 的生成器铺满。

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
- **步数从前写死在舰队里**:`_launch_lane` 里曾是 `export TOTAL_TRAINING_STEPS=250`,而名册
  35 条有 30 条是 200 步 —— 「200 步」那一列根本没有执行路径,照跑就是每条多 25% 训练加一个
  评测点。现在步数从 lane 图第三段来(`arm:gpu,gpu:steps`),<250 时舰队会显式打开
  `SIMOPD_SHORT_RUN_OK=1` 并打进日志:`run_opd_baseline.sh` 拒绝「未打标的 <250 步」,
  那个守卫防的是从 shell 漏出来的 `STEPS=3`,而 lane 图就是「on the record」的那份记录。
- **只活在盘上的脚本等于没有**:`eval_worker_exp.sh` 从未进 git,旧盘一失联就永久丢了,
  而三处调用只会打一行"找不到,卡先闲着"——静默不评。已重写并入库(§2.6)。
  **凡是被 payload/fleet 调用的东西都必须在仓库里**,盘上只放数据。
- **`ckpt_sync.py` 曾经根本跑不起来**:校验段有个 `(dp := dp)` 的 SyntaxError,2026-09-02 才发现 ——
  它此前在任何机器上都没执行过。"自动上传"只认日志里的 `OK` 行和仓库里的 `SYNCED.json`,别信脚本存在。
- **verl 的 `FileLogger` 以 `wb` 打开会截断**:所以一次启动一个 `metrics/launch_<ts>.jsonl`,续跑/重试
  不会覆盖上一段;读的时候按 step 拼、后写覆盖先写。
- **`seq_key` 必须 ≤ 63 位**:`pandas.read_json` 读不了 > 2^63 的整数(本地测试当场炸),
  `simopd/seqkey.py` 已截到 63 位;别"顺手"改回 64 位。
- **归档层的失败语义是"喊一次、训练继续"**:`traj_dump` / `div_panel` / `run_manifest` 任何一环坏了
  只在 stderr 出一行,不会停训练。所以验收看文件、不看横幅(§2.4 清单);发现 stderr 有
  `写盘失败` / `面板失败` 就当这个 run 没有归档。
- **`ids_` 每步落盘后,run 目录不再是"几十 MB 的小文件"**:一个 200 步的 run 的 `traj/` 长到 GB 级,而 `ckpt_sync`
  的 aux 同步是"目录里有新文件就整目录交给 `upload_folder`",HF 端按内容哈希跳过没变的,但**每轮都要把整个目录
  重新哈希一遍**,多 pod 各自扫同一块盘时读放大成倍。先看 `ckpt_sync_slot*.log` 里一轮 aux 的耗时;接近扫描间隔
  就把 `CKPT_SYNC_EVERY` 调大,或把 aux 同步改成只提交上次之后新增的文件(`SYNCED.json` 里已记 `newest`)。
