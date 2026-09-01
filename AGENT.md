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

## 2 先分清:重跑 / 续跑 / 新跑

**没有任何东西需要从 0 重跑。** 盘上每条臂都有 ckpt,舰队引擎起来就自己从最近的续,
最多丢 25 步。三类要分开看:

| 类 | 条数 | 说明 |
|---|---|---|
| **只欠评测**(训练已满 250) | 18 | 一张训练卡都不用。含三条判决臂,**也含"省 token"四条候选(c3 / d2 / h1 / g2)全部已满 250** |
| **续跑**(从 ckpt,不是重跑) | 33 | 其中 16 条是掉线时正在跑的 lane;另 17 条停在更早的波次 |
| **新跑**(从 0) | 2 | `vanilla_corr` 的 s1 / s2 种子 —— 唯一真正要从头起的东西 |

**续跑里有 5 条建议直接放弃**:`a1_gkd_mix0.5` / `a4_dagger_anneal` / `a5_aggrevate` /
`h6_gen_sched` / `h10_task_subset` 的**老载体版**。它们跑满只是再证明一次"坏载体会塌"
(a4/h6/h10 已经塌透了),而结论早已由 vanilla 的单旋钮对照定死。省下的卡直接给种子。

**不用管的**:`c5_union_fkl_s0_16k.renorm-defect-20260822` 是缺陷版存档(干净版在跑,
别合并);modelscope 事故烧掉的那批评测没产出 parquet,`eval_refill_exp.py` 会按
"盘上 ckpt 减去盘上 parquet"自动重排,不需要人工补。

## 2.5 如果换机器后**不能 resume**(2026-08-24 的现实假设)

只有 HF 权重能跟着走、优化器状态和数据顺序留在旧盘 → **33 条半途的 run 都续不了**。
拿权重"暖启"接上去不是续跑:优化器矩清零、数据顺序重排,曲线中段留一道缝,
而我们的判据恰恰读晚期斜率 —— 那道缝正好落在要读的地方。所以**不暖启**,改成:

**主表锚点从 @250 降到 @200。** 盘上已有 ckpt 的臂数:

| 锚点 | 可进表的臂 |
|---|---|
| @250 | 18 / 51 |
| **@200** | **31 / 51** ← 推荐 |
| @175 | 36 / 51 |

@200 一步不用重跑就能拿到 31 条臂的同步对比,而 corr 波的核心结论(治愈对照、药效归零、
吸引子)全部在 ≤200 步内成立 —— 判决臂三条本来就已经满 250,c4_carrier / c4_rep 也满了。
停在 200–225 的这 13 条(e2 / g4 / g6 / a3_n0 / f1 / f2_hard / d1 / h6_n0 / g5 / b5 / a1_n0 /
a4_n0 / h6_legacy)**直接进表,不重跑**。

**只有 4 条值得从 0 重跑**(共 ~360 GPU·小时,32 卡约 11 小时):

| 臂 | 为什么非跑不可 | 成本 |
|---|---|---|
| `vanilla_corr` **s1** | 整条治愈结论现在压在一个种子上 | 119 GPU·h |
| `vanilla_corr` **s2** | 同上 | 119 GPU·h |
| `c5_union_fkl` | 预注册 onset 在 163 步之后,现停 100 步,**既没证实也没证伪** | 76 GPU·h |
| `c5_union_rkl` | 预注册"比 vanilla 更早爆",现停 75 步,同样未决 | 86 GPU·h |

其余低于 200 的 16 条(f3 / f4 / f5 / f2_clip2.3 / g1 / h2 / h4 / h10_n0 / c2 / c4_hq /
c4_state / c4_pi_tail_corr / n2_termcal / 以及 a1/a4/a5/h10 的老载体版)**冻结在原地**,
按已有步数报,不重跑:它们的曲线已经够长到能分类(健康/危险/塌缩),而"药效归零"用的是
≤175 步的同步对比,不依赖它们跑满。`n2_termcal` 单条重跑就要 200 GPU·h(全场最慢),
而它已经塌透了,再跑一遍只是把塌缩看得更清楚。

**结论:失去 resume 的代价 ≈ 360 GPU·小时**,而不是原计划的 610 —— 因为真正必须跑满的
只有 4 条,其余靠降锚点就能进表。

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

### 4.1 ckpt 走 HuggingFace(2026-08-24 决定的迁移路径)

**先记住一件事**:`actor/huggingface/`(约 **3.4 GB**)只够**评测**;**续训要整个
`global_step_N/` 目录**——FSDP 分片 + optimizer + extra,约 **28.4 GB**(HF 权重只占其中
14%)。只传 HF 权重就不是"续跑",而是拿权重重开一条新 run:优化器状态和
StatefulDataLoader 的数据顺序都换了,一条 250 步曲线的中段被这样接上,协议就断了。
`resume_mode=auto` 认的是目录里最新的 `global_step_*`,而拒绝 resume 的是**旋钮指纹**
(从 env 读),**不是路径**——所以换集群、换挂载点都不影响,只要臂的 env 一模一样。

**分层传,别全传**(33 TB → 约 2.1 TB):

| 层 | 传什么 | 体量 | 换来什么 |
|---|---|---|---|
| 1 | 本波每个 ckpt 的 `actor/huggingface/` | 395 × 3.4 GB ≈ **1.3 TB** | 整个评测欠账(1636 格)在任何有卡的地方都能跑 |
| 2 | 要续训那些臂的**最新那个** `global_step_N/` 全量 | ~28 × 28.4 GB ≈ **0.8 TB** | 真续跑,不破协议 |
| 3 | `evals/*.parquet` → 一个 **dataset** repo | 14 GB | 原文列(`response`)只在这里,分析表能重建、它不能 |
| — | **不传**:非最新步的 25 GB 分片/优化器 | 省下 ~30 TB | 我们从不从旧步精确续 |

**上传注意**:仓库设 **private**(未发表的研究产物);模型卡注明基座
`Qwen3-1.7B-Base` ← 教师 `Qwen3-4B-Instruct-2507` 与其许可;上传前
**unset `HF_HUB_OFFLINE`**(`simopd_env.sh` 里是开的),大目录用
`hf upload-large-folder`;`HF_HUB_DISABLE_XET=1` 是为在线拉取设的,**大批量上传建议放开
Xet**(块级去重,重复的 tokenizer/config 不必重传)。目录结构**原样保留**
(`<arm>_s0_16k/global_step_N/...`),下游脚本按这个 glob 找 ckpt。

**落地后必须验**:传之前在源盘生成清单(`find … -type f -printf '%s %p\n'` + 关键文件
sha256),拉下来逐条比对;抽一条臂真跑一次 25 步续训,确认指纹没被拒。**验完再删源盘。**

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
