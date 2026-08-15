# SimOPD 基线与各臂详解

> 面向新加入这个 campaign 的人:**vanilla 到底在算什么**,**每个臂改了哪一处**,
> **怎么把它跑起来**。
>
> 这份文档不做判决也不下结论 —— 判决规则在 `SimOPD-plan.md` §5,协议在
> `PROTOCOL-unified.md`,指标在 `METRICS.md`。这里只解释机制。
>
> 符号:`π` = student policy,`q` = teacher,`x` = prompt,`y` = rollout,
> `S` = teacher top-k 支撑集,`S̄` = 其补集,`T` = 一条 rollout 的 response token 数。

---

## Part 1 — 基础 OPD setting

### 1.1 OPD 是什么,以及它不是什么

三种"让小模型学大模型"的做法,区别在**数据从谁的分布来**、**监督信号是什么**:

| | 数据来源 | 监督信号 | 问题 |
|---|---|---|---|
| SFT / 离线 KD | teacher 生成(固定语料) | teacher 的 token 或分布 | 学生推理时走到自己会去、语料里没有的状态,没人教过它怎么办(exposure bias) |
| RL(GRPO/PPO) | **student 自己生成** | 稀疏的 answer-level 奖励 | 一条 8k token 的 rollout 只换来一个 0/1,信号极稀 |
| **OPD(本项目)** | **student 自己生成** | teacher 对这条 rollout 的**逐 token** 打分 | —— |

OPD 拿的是两边的好处:轨迹来自学生自己的分布(所以教的正是它真会犯的错),
而监督是逐 token 稠密的(所以一条 rollout 给 T 个信号而不是 1 个)。

### 1.2 一步训练里发生了什么

```
  ┌─ 1. student π 对 prompt x 采样一条 rollout y   (vLLM, τ=1.0, n=1)
  │
  ├─ 2. teacher q 对 (x, y) 做一次 forward,吐出逐位置的 logprob
  │      · sampled-token 路径:只要 log q(y_t | x, y_<t) 这一个数
  │      · top-k 路径:要 q 在每个位置的 top-k 分布(k=32/64)
  │
  ├─ 3. student 对同一个 (x, y) 做一次 forward,拿 log π(y_t | ·)
  │
  ├─ 4. 算逐 token 的 advantage,喂给 PPO-clip
  │
  └─ 5. 一个 optimizer epoch(batch = mini-batch = 128),然后回到 1
```

关键:teacher **只做推理,永不更新**,所以它常驻一张卡当 vLLM server;student 要
训练,占另一张卡。verl 把 teacher 池注册成**独立的 Ray resource pool**,不与 actor
共卡 —— 所以 **2 卡是一个 run 的地板,不是一个选择**。

### 1.3 vanilla 的目标函数

目标是最小化 **reverse KL**:

```
    L(θ) = KL( π_θ ‖ q )  =  E_{y ~ π_θ} [ Σ_t ( log π_θ(y_t|·) − log q(y_t|·) ) ]
```

期望是对 `π_θ` 自己取的(这就是 "on-policy" 的含义),所以求导要用 policy gradient。
把括号里那一项取负记作**逐 token advantage**:

```
    Δℓ_t  ≜  log q(y_t | x, y_<t)  −  log π(y_t | x, y_<t)
```

于是

```
    ∇L  =  − E [ Σ_t  Δℓ_t · ∇ log π_θ(y_t | ·) ]
```

**直观读法**:`Δℓ_t > 0` 表示"老师在这个位置比你更愿意说这个词" → 把 `π` 往上推;
`Δℓ_t < 0` 表示"你说得比老师还起劲" → 往下压。它不是奖励,是**两个 logprob 的差**,
所以每个 token 都有,而且自带方向和大小。

为什么是 **reverse** KL 而不是 forward:reverse KL 是 mode-seeking —— 它惩罚
"学生在老师不去的地方放质量",逼学生收敛到老师的一个模式上;forward KL 是
mass-covering,会逼学生去覆盖老师的全部支撑,在容量差大的时候摊薄成一团糊。
`b2_forward_kl` 就是这个"公认更差"的对照臂。

**代码里的符号约定**(`src/simopd/losses.py` 头部有完整推导):

```python
kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")  # = log π − log q = −Δℓ_t
# verl 的 distillation_loss 随后用 -losses 当 advantage  →  advantage = Δℓ_t  ✓
```

所以**在 loss 空间做的变换作用在 −Δℓ 上**;我们所有的变换都是奇函数
(`sign(d)·log(1+|d|)`、`clip`),对 −Δℓ 变换等价于对 Δℓ 变换再取负 —— 这一点在
`f1`/`f2` 上必须成立,否则符号就反了。

### 1.4 严格 on-policy 的两个后果

协议钉死 `batch = mini_batch = 128`,即**每个 rollout batch 恰好一个 optimizer epoch**。
于是:

1. **PPO ratio ≡ 1**,clip(0.2) 结构上永不触发。它留在配方里只是因为 verl 的 PG
   路径走这条公式,不是一个起作用的旋钮。
2. **异步训练是协议禁区,不是工程取舍**。verl 的 `fully_async` / `one_step_off`
   会引入 staleness ≥ 1,而 `Δℓ-as-advantage` 的推导假设 rollout 就是当前策略;
   异步会给每个臂叠加一层 IS/clip 混淆。飞行记录仪里 `trajectory_staleness` 恒为 0
   就是凭证。墙钟压力走 **run 级并行**(多泳道),不走步内异步。

### 1.5 锁定协议(所有臂共用,只有该臂那一处例外)

| 项 | 值 | 依据 |
|---|---|---|
| student | **Qwen3-1.7B-Base** | 0.6B 收敛到 0.468,而 1.7B **未训练就是 0.468** —— 0.6B 能测的全部范围在真实学生的零点之下 |
| teacher | **Qwen3-4B-Instruct-2507** | 非思考 MATH500 天花板 **0.896** > 8B 0.792 > 1.7B 0.702。`enable_thinking=False` 下 8B 的主武器被关掉,**更大的老师不等于更强的老师** |
| 训练集 | nvidia/Nemotron-Cascade-RL-Math,14,476 题(过 1024 prompt 帽后有效 **~12,478**) | 锚点对齐 |
| val | MATH500 pass@1,**greedy**,每 25 步 | Demystifying/LSM 同款 |
| 模板 | chat template + `enable_thinking=False` | 学生 rollout 与老师打分**同模板** |
| rollout | n=1/prompt,τ=1.0,top-p=1.0 | Demystifying 对齐 |
| 长度帽 | prompt 1024 / response **8192**(筛选)· 16384(锚点与终审) | 截断率必报 |
| 优化 | AdamW,**lr 1e-6 恒定无 warmup**,grad-clip 1.0,bf16 FSDP,loss 聚合 **token-mean** | 文献众数 |
| horizon | **250 步 + 预注册早停** | 2026-08-06 依 DSW 锚点 run 修订(见 plan §4)⚠ 见下 |
| teacher 精度 | **bf16,禁量化** | logprob 是被审对象,不能先把它压坏 |
| 显存份额 | `ROLLOUT_GPU_MEM_UTIL` 全场钉死一个值 | **非数值中性**:cache 尺寸 → batch 组成 → RNG 流,入指纹 |

**为什么 teacher 全现货、零自训**:受审池里 4 篇训了 GRPO 老师,但每一篇**也都报了
现货老师**,自训不是任何一篇的必要条件;而一个只有我们有的老师,会成为这份审计里
唯一别人无法复现的部件。

> ⚠ **horizon 注(2026-08-07 更新)**:上游 merge(`ee6200d`)已把
> `run_parallel.sh` / `campaign.sh` / daemon 的默认值统一为 **250**,且早停规则改为
> **只记账不杀进程**("the stop rule measures, it no longer kills")。
> 仍留一处地雷:**裸调 `scripts/run_opd_baseline.sh` 的 fallback 还是 150** ——
> 手动调试单臂时记得 `TOTAL_TRAINING_STEPS=250`;走 campaign/泳道路径则无需操心。

### 1.6 一个 run 的成本

2 卡 · 250 步 · 8k 响应帽。A100 上实测外推约 18–23 GPU·时/臂,H100 会更快。
checkpoint 每 `SAVE_FREQ` 步落一次,1.7B 一个 ckpt ≈ 25GB(bf16 权重 + fp32 master
+ AdamW 矩 + 梯度),`MAX_CKPT_KEEP=2` → **每 run ≈ 50GB**。

---

## Part 2 — 八轴十六臂

### 2.0 轴的组织逻辑

一个 OPD 算法可以拆成互相正交的几个问题,每个轴回答一个:

```
  A  轨迹从哪来?        on-policy / 混合 / 先冷启动
  B  用哪个散度?        reverse KL / skew-KL / forward KL
  C  在多宽的词表上比?   只看采样到的那一个 token / teacher top-k / 自适应
  D  哪些位置要教?      全都教 / 高熵的 / 老师认可的 / 可教的
  E  在支撑内比什么?     概率值 / 排序
  F  信号要不要压?      不压 / 软压 / 硬截
  G  哪些轨迹配被教?     全都要 / 验证器过的 / 老师似然高的
  H  一条轨迹教多长?     全长 / 只教前 512 token
```

**纪律:一个臂 = vanilla + 恰好一个旋钮。** `configs/arms.yaml` 里每个臂只列它自己那
一处偏离,其余全部继承锁定协议。同轴内总监督量严格相等(H 轴故意例外 —— 削预算正是
它的主张)。

---

### 轴 A — 轨迹来源与日程

#### `a2_coldstart` — 离策略冷启动 → OPD

**直观**:训练刚开始时学生和老师的分布差得太远,学生采样出来的轨迹落在老师概率很低
的区域,老师给的信号又大又噪。先用老师的生成做一轮 SFT 把两个分布拉近,再开 OPD。

**做法**(两阶段):

```
  阶段 1  从训练集里划出一块保留 prompt → teacher 采样 → 验证器拒绝采样 → verl SFT
  阶段 2  从这个 SFT checkpoint 出发跑标准 OPD
```

**关键纪律**:冷启动用掉的 prompt **从 OPD 阶段剔除**,否则"见过同一道题两遍"会被
错记成冷启动的收益。代价是这个臂见到的 distinct prompt 更少,记在
`coldstart_meta.json` 里。

**执行**:

```bash
sbatch slurm/coldstart.sbatch        # 阶段 1(或改写成本地脚本)
# 阶段 2 走标准路径,arms.yaml 里 env 已把 STUDENT_MODEL 指向 SFT ckpt
```

> ⚠ 这个臂的 `env` 硬编码了 Cornell 路径(`/scratch/zz865/...`),**在我们的集群上必须改**。
> 它在 `campaign.tsv` 里被显式排除在共享池之外,正是因为它曾经抢到一条泳道、失败、
> 然后被无限重试。

#### `a1_gkd_mix0.5` — GKD 混合(门控:等预计算缓存)

**直观**:每个 batch 里混入一部分老师自己生成的响应(off-policy),λ=0.5。

**历史**:最初以为要动 TransferQueue 的 agent loop(真 trainer 手术)所以推迟;
2026-08-06 借 i1 的 rollout-server 注入路径落地(`src/simopd/gkd_mix.py`:
per-(prompt,step) 确定性掷币,命中改为对缓存的 teacher 响应打分而非生成)。
现门控在 `scripts/gen_offpolicy.py` 的一次性预计算上,与 a3/a4 共享解锁。

#### `a4_dagger_anneal` — DAgger 退火(arXiv 2605.12913)

**直观**:和 a1 同一台机器,但 off-policy 比例不是常数而是 **λ(step) 从 1.0
退火到 0.0** —— 训练开始时全部抄老师范文,结束时全部自己写,渐进过渡。
每条 response 仍整条归属单一作者(论文 turn 级规则的单轮镜像;token 级穿插
在 multi-turn 工作中被判不稳定,这里同样不做)。

**做法**:`SIMOPD_GKD_SCHEDULE="mode=linear,start=1.0,end=0.0,warmup=0,decay=250"`
(`src/simopd/gkd_schedule.py`,STACX DAggerSchedule 的逐式移植:linear/cosine/
exponential/step 四种模式 + warmup,快降慢降全可调)。主形状全窗线性的平均剂量
0.502 ≈ a1 的常数 0.5,{a1,a4} 构成匹配平均剂量对;decay=125 半窗变体(= 发表
recipe 的归一化形状)已注册为裁定臂。实测遥测(λ 目标/实现、师生 token 比例)
经 `gkd_stats` 接力落到 wandb 的 `distillation/gkd_*` 面板。

**执行**:解锁与 a1 完全相同(同一缓存、同一 3 步彩排),彩排额外要求看到
λ_realized 贴合调度曲线、wandb gkd 面板出数。

#### `a5_aggrevate` — AggreVaTe 单切换(已落地,待 keys 预计算 + 彩排)

**直观**:每条 response 只有一个切换点 κ~U{0, T_max(step)}:学生写 0..κ−1,
teacher **在线**从 κ 续写到结束(前缀依赖当前学生权重,缓存不可能)。T_max
从 0 线性升到 16384,step 0 即论文的 iter-0 纯 teacher 冷启动;与 a4 共享
归一化窗口,{a4,a5} 把"混合结构"隔离成唯一变量。κ~U 天然前载 teacher 剂量
(实测 teacher-token 占比衰减快于名义曲线)—— 以遥测实测为准,记录在案。

**做法**:`src/simopd/a5_aggrevate.py` 同 seam 包装(与 gkd_mix 互斥,install
互检拒绝):学生按 κ 截断生成 → 经具名 `simopd_teacher_registry`(sitecustomize
在 trainer 建完 teacher 池后发布 handle)拿 teacher server actor,token-ids
在线续写 → 打分调用取全响应学生 logprobs → 缝合返回。κ=0 的裸 prompt 尾调
用靠哨兵剥离防递归。遥测:`gkd_tmax`/`gkd_kappa_mean`/`gkd_tail_token_frac`/
结局计数走通用 gkd 接力。解锁:`gen_offpolicy.py --dry` 出 keys(纯 CPU,
不等 GPU 预计算)+ 3 步彩排(κ 直方图、实测尾占比、哨兵剥离)。

---

### 轴 B — 散度

#### `b1_skew_kl` — Skew-KL,α=0.1

**直观**:老师在某些位置给的概率接近零,`log q → −∞`,`Δℓ` 爆炸,**单个 token 就能
主导整个 batch 的梯度**。Skew-KL 的做法是先把老师和学生**混合**一下再比 —— 混进去的
那一点学生自己的质量给了对数一个下限。

**数学**:

```
    SKL_α(π ‖ q)  =  KL( π ‖ α·π + (1−α)·q )

    per-token:  log π(y_t) − log( α·π(y_t) + (1−α)·q(y_t) )
              = log π − logaddexp( log π + log α ,  log q + log(1−α) )
```

- **有界**:估计量上界为 `−log α = log 10 ≈ 2.303`,不管老师多么不同意
- **α → 0 退化为 k1**:混合分布 → q,式子回到 `log π − log q`

**实现要点**(`src/simopd/losses.py`):混合分布在**采样到的那个 token** 处的对数密度
是**精确可算**的 —— 两个已有的 sampled-token logprob 就够,**不需要 top-k 也不需要
full-vocab**。所以这个臂和 vanilla 停在同一条 C 轴上,B 轴的比较不被支撑大小污染。
在 fp32 里算,因为 bf16 下光 `log(0.1)` 就吃掉约两位十进制精度。

**执行**:
```bash
DISTILLATION_LOSS_MODE=skew_kl_a0.1
```

#### `b2_forward_kl` — Forward KL(完整性对照)

**直观**:方向反过来。reverse KL 说"别去老师不去的地方"(mode-seeking);forward KL
说"老师去的地方你都得覆盖"(mass-covering)。在容量差大的蒸馏里公认更差 —— 审计应该
包含一个"已知更差"的对照,否则读者无法校准其他臂的效应量。

**数学**(verl 原生 `forward_kl_topk`):

```
    Σ_{v ∈ S}  q(v) · ( log q(v) − log π(v) )        S = teacher top-32
```

> ⚠ **verl 的这个实现是 top-k 且不重归一化** —— 这是一处 C 轴混淆。本臂只能报成
> **forward-KL-topk**,不能报成 plain forward KL。已记在 `arms.yaml`。

**执行**:
```bash
DISTILLATION_LOSS_MODE=forward_kl_topk  DISTILLATION_TOPK=32
```

---

### 轴 C — 词表支撑(比多宽)

这是 headline 假设所在的轴。定理:

```
    截断 reverse KL 的误差  =  π(S̄) · KL( π(·|S̄) ‖ q(·|S̄) )
```

**由 student 尾质量 π(S̄) 控制**,不是由 teacher 的质量控制。而现行做法(按 teacher
top-k 截断)是**按 teacher 的质量切的 —— 切错了边**。文献只报交集(overlap ratio),
从没人报这条尾巴,所以每个 top-k 臂都免费带 `pi_tail_k{8,16,32}` 面板。

#### `c1_lsm_topk32_renorm` — LSM 截断 reverse KL

**直观**:vanilla 只看采样到的那**一个** token,老师 top-k 分布里的其余信息全丢了。
LSM 说:整个 top-k 都是信号,用起来。代价是 top-k 之外的全扔掉。

**数学**(`SIMOPD_SUPPORT_MODE=renorm`,默认):

```
    S = teacher top-k                      (k=32)
    π̃(v) = π(v) / π(S) ,  q̃(v) = q(v) / q(S)        两侧各自重归一化成真分布
    loss  =  KL( π̃ ‖ q̃ )  =  Σ_{v∈S} π̃(v)·( log π̃(v) − log q̃(v) )
```

**内部消融** `SIMOPD_SUPPORT_MODE=tailbucket`:不丢尾巴,额外拼一个桶装剩余质量
`π(S̄)` 和 `q(S̄)`,让尾巴对散度有贡献而不是凭空消失(RSKD 的修正,精神版)。
这一对就是预注册的"重归一化 vs 尾桶"对照 —— **同一个臂的内部消融,不是两个臂**。

**执行**:
```bash
DISTILLATION_LOSS_MODE=lsm_topk_renorm  DISTILLATION_TOPK=32  SIMOPD_SUPPORT_MODE=renorm
# 消融:                                                        SIMOPD_SUPPORT_MODE=tailbucket
```

#### `c2_quantile_budget` — 分位预算分配 [自研]

**直观**:固定 top-32 对每个位置一视同仁 —— 学生已经笃定的 token(比如公式里的
`\frac` 后面必然跟 `{`)和真正的推理分叉点,花掉的词表预算一样多。改成**按需分配**:
分歧大的地方支撑开宽,笃定的地方收窄,**平均预算不变**。

**数学**:

```
    候选 margin:   m(v) = max( q(v), π(v) )              (SIMOPD_QB_MARGIN=max|q|pi)
    batch 级阈值:  τ = Quantile( {m(v)}_全 batch ,  1 − B/k )      B=8, k=64
    支撑:         S_t = { v : m(v) ≥ τ }  ∪  { teacher 的 top-1 }
    loss  =  KL( π̃ ‖ q̃ )  在 S_t 上重归一化后计算
```

**关键性质**:`|S_t|` **逐 token 变化**,而 `E[|S_t|] = B` —— 与固定 top-B 在**期望上
预算配平**,所以两者可比。实测 target=8 时 budget 的 p5/p50/p95 = 6/9/12,确实自适应,
不是伪装的定 k。`keep[..., 0] = True` 保证 teacher 自己的 top-1 永不被丢,支撑不会空。

**为什么 margin 用 `max(q, π)`**:只看 `q` 会漏掉"学生给了高概率但老师没给"的位置 ——
而那正是 reverse KL 要惩罚的地方(mode-seeking 的着力点);只看 `π` 则漏掉老师想教
但学生还没看见的词。取 max 两边都收。`q` / `pi` 两个单独取值是替补拆解臂。

**执行**:
```bash
DISTILLATION_LOSS_MODE=qb_quantile_budget  DISTILLATION_TOPK=64 \
SIMOPD_QB_TARGET_BUDGET=8  SIMOPD_QB_MARGIN=max
```

---

### 轴 D — token 选择(哪些位置要教)

**三个臂共享同一个 kernel 工厂**(`_d_axis_kernel`),base loss 全部保持 **vanilla 的
sampled-token reverse KL 不变**:

```
    loss_t  =  1[ t ∈ Keep ] · Δℓ_t · ( T / |Keep| )
                                        └─ 缩放回来
```

**那个缩放因子是必须的**:`agg_loss` 的分母是全部 token,丢掉一半 token 就等于把有效
学习率砍半 —— 不缩放的话,这个臂掉点会被归咎于"更小的学习率"而不是"后面的 token
本来有信号",而后者才是它要测的唯一一件事。

三个选择器共享 `SIMOPD_D_RETENTION=0.5`,所以**同轴预算按构造配平**(SelecTKD 例外)。

> **前置依赖**:D 轴同时需要 teacher top-k(算判据)和采样 token 的 teacher logprob
> (保持 vanilla 目标),而 verl 在 `vllm_rollout/utils.py:483` 把后者丢了。
> `src/simopd/teacher_patch.py` 把它作为一列尾巴接回来 —— 所以三个臂都必须
> `SIMOPD_KEEP_SAMPLED=1`。

#### `d1_tip` — TIP:高熵 + 自信但错

**直观**:两种 token 值得教 —— 学生**犹豫**的(高熵,真正的分叉点),和学生**自信但
和老师分歧大**的(低熵但错得笃定)。soft-OR 把两者并起来:满足任意一个就够。

**数学**:

```
    h_t  =  H( π(·| x, y_<t) )                    全词表熵,batch 内 98 分位截顶
    d_t  =  Σ_{v∈S} q(v)·( log q(v) − log π(v) )  该位置的 forward 散度
    ĥ, d̂ =  min-max 归一化(batch 内)
    s_t  =  ĥ + d̂ − ĥ·d̂                          soft-OR
    Keep =  s_t 最高的 50%
```

**实现要点**:熵**在 kernel 里从 `student_logits` 现算**,不取 `model_output['entropy']` ——
后者要靠 `entropy_coeff != 0` 打开,而那会往 loss 里加一项熵奖励,把臂污染掉。

**执行**:
```bash
DISTILLATION_LOSS_MODE=tip_select  DISTILLATION_TOPK=32 \
SIMOPD_KEEP_SAMPLED=1  SIMOPD_D_RETENTION=0.5
```

#### `d2_selectkd` — SelecTKD:propose-verify

**直观**:借用投机解码的结构。学生先"提议"它最想说的词(top-1),老师"验证"这个词
在不在自己的 top-k 里。**在** → 学生这一步走在老师认可的路上,信号可信,教;
**不在** → 跳过。

**数学**:

```
    Keep_t  =  1[ argmax_v π(v | x, y_<t)  ∈  teacher top-k ]
    TAR     =  mean(Keep)     ← SelecTKD 的 acceptance rate
```

**与另外两个的区别**:保留率**由数据决定,不可调** —— 所以它**不是**按构造和
`d1`/`d3` 预算配平的。TAR 照实记录,台账报实际预算而不是假装配平。

**执行**:
```bash
DISTILLATION_LOSS_MODE=selectkd_verify  DISTILLATION_TOPK=32  SIMOPD_KEEP_SAMPLED=1
```

#### `d3_teachability` — Teachability:分歧 × 可达

**直观**:一个 token 值得教要**同时**满足两件事 —— (a) 老师和学生确实不一样
(disagreement,不然没什么可教);(b) 学生**够得着**老师想要的东西
(compatibility,够不着的就是白教,梯度全浪费在一个学生根本到不了的方向上)。

**数学**:

```
    dis_t   =  Σ_{v∈S} q(v)·( log q(v) − log π(v) )
    comp_t  =  Σ_{v ∈ S ∩ StudentTopK} q(v)              teacher 落在学生 top-K 上的质量
    s_t     =  minmax(dis_t) · minmax(comp_t)            乘法 = 两个都要
    Keep    =  s_t 最高的 50%
```

> ⚠ **已登记偏离**:teacher 只返回自己的 top-k,所以 `comp_t` 实际算的是**交集上的
> 质量 —— 一个下界**。当学生 top-K 落在 teacher top-K 内时取等,Rethinking 报告这是
> 常见情形。

**执行**:
```bash
DISTILLATION_LOSS_MODE=teachability_select  DISTILLATION_TOPK=32 \
SIMOPD_KEEP_SAMPLED=1  SIMOPD_D_RETENTION=0.5
```

---

### 轴 E — 支撑内目标(比什么)

#### `e1_pl_rank` — Plackett-Luce 排序 + value 锚 [自研]

**直观**:容量差摆在那里,1.7B 学不出 4B 的精确概率**值**。那就退一步:**只要求学生把
老师的排序复现出来**。而且 greedy 解码本来就只关心谁排第一 —— 值对不对无所谓,序对了
就行。这是个更弱、也可能更容易迁移的目标。

**数学**(verl 返回的 teacher top-k **已按秩排序**,所以目标排列就是恒等):

```
    s_i  =  学生在老师第 i 名 token 上的 log 概率

    PL 对数似然(老师这个排序在学生分布下的概率):
        log P_PL  =  Σ_{i=1..k} [  s_i  −  logsumexp( s_i, s_{i+1}, ..., s_k )  ]

    rank_loss  =  − log P_PL / k

    value 锚(防止排序对了但 margin 全塌):
        anchor   =  KL( π̃ ‖ q̃ )        两侧在 top-k 上重归一化

    loss  =  rank_loss  +  0.1 · anchor          (系数是本臂的内部消融轴)
```

后缀 logsumexp 用 `flip → logcumsumexp → flip` 一次算完,O(k)。

**为什么必须配锚**:纯排序损失对概率值不敏感,但终评有 **AMC23 avg@32** 这种采样评测,
采样是要看 margin 的 —— 序对了而 margin 塌成一团,采样质量会掉。锚强度本身就是要测的
东西。

**验证过的性质**:loss 在无序度上单调 —— 学生与老师一致时 0.34 < 均匀分布 1.10 <
完全反序 2.84。

**执行**:
```bash
DISTILLATION_LOSS_MODE=pl_rank_anchor  DISTILLATION_TOPK=32  SIMOPD_PL_ANCHOR_COEF=0.1
```

---

### 轴 F — 信号调节(尾巴要不要压)

`Δℓ` 的分布尾巴很重:老师给某个 token 的概率比学生高两个数量级,`Δℓ` 就有 4.6,
而中位数可能在 0.1 量级。**少数 token 主导梯度**,是 Mode A(长度爆炸)的怀疑对象。

#### `f1_soft_log` — 软 log 压缩(Demystifying 的赢家)

**数学**:

```
    Δℓ'_t  =  sign(Δℓ_t) · log( 1 + |Δℓ_t| )
```

**直观**:大信号被压成对数增长,小信号几乎不动(`log(1+x) ≈ x` 当 x 小),而且
**处处连续可导** —— 不像硬截断那样在阈值处制造一个断点。

面板同时记 `softlog_raw_absmean` 和 `softlog_shrink_ratio`,后者 ≈1.0 就说明这个变换
在这个 batch 上是惰性的 —— 效应可见而非推测。

**执行**:
```bash
DISTILLATION_LOSS_MODE=k1_softlog
```

#### `f2_hard_clip` — 硬截断

**数学**:

```
    Δℓ'_t  =  clip( Δℓ_t, −c, +c )        c = 10.0
```

用 verl 自带的 `loss_max_clamp`,**loss_mode 仍是 `k1_rec`** —— 保证 clamp 是这个臂
相对 vanilla 的**唯一**偏离。

> 因为 clamp 在 loss 函数下游生效,这个臂自己的 Δℓ 面板是**裁剪前**的分布,和 vanilla
> 一模一样 —— 机制会隐形。所以 `_clip_metrics` 专门记 `clip_hit_rate`
> (`|signal| > c` 的 token 占比)。

**预注册**:此臂**只在 D5 诊断测到 Mode B 才进场**;但它是 stock 的,所以跟着一起冒烟。

**执行**:
```bash
LOSS_MAX_CLAMP=10.0        # 不设 DISTILLATION_LOSS_MODE,继承 k1_rec
```

---

### 轴 G — 轨迹门控(哪些 rollout 配被教)

两个臂共用 `_reweight_kept`:丢掉的轨迹置零,再按 `T/T_keep` 缩放 —— 理由同 D 轴,
**不缩放的话门控就退化成学习率实验**。做成 loss mask 而不是 batch filter,是为了它能在
贪心轮里和别的轴组合;缩放让两者在期望上等价。

#### `g1_verified_only` — 只蒸验证器通过的轨迹

**直观**:学生写出一个错误答案,老师逐 token 打分照样会给出信号 —— 但那是在**把一条
错路教得更顺**。规则验证器能判对错,那就只在判对的 rollout 上蒸。

**数学**:

```
    keep(y)  =  1[ Σ_t A_t  >  0 ]
    loss_t   =  1[keep] · Δℓ_t · ( T / T_keep )
```

**纪律**:**验证只做过滤,答案永不进训练输入。**

**实现要点**:门控读的是 `advantages` 而不是原始验证器分数 —— verl 的 trainer 只把
`advantages`/`returns` 写回给 actor,`token_level_scores` 根本到不了 loss 函数。
在本协议下(`adv_estimator=grpo`, `use_task_rewards=False`)advantage 只来自验证器分数,
所以"总 advantage > 0" ⟺ "验证器接受"。**这个假设是 assert 出来的**,估计器一改就
大声失败,而不是悄悄门控在别的东西上。

`gate_keep_frac` 必记:一个"什么都没通过"的 batch 会产生零更新 —— 对这个臂是正确行为,
但必须可见,否则看起来就像一步静默死掉了。

**执行**:
```bash
DISTILLATION_LOSS_MODE=k1_verified_only
```

#### `g2_fire_likelihood` — FiRe:按老师似然丢底部 20%

**直观**:没有验证器的域怎么办?用**老师的似然当代理** —— 老师觉得离谱的轨迹,大概率
也是错的,先扔掉。

**数学**:

```
    s(y)   =  (1/T) Σ_t log q(y_t | x, y_<t)         长度归一化的 teacher logprob
    阈值   =  Quantile( {s(y)}_micro-batch , 0.2 )
    keep   =  1[ s(y) ≥ 阈值 ]
```

> ⚠ 阈值取在 **micro-batch** 内,比 FiRe 原文的 full batch 是更嘈杂的排序总体。
> 实际保留比例 `gate_keep_frac` 照实记录,让这层噪声可见。

**前置检验(重要)**:这个臂的**前提**是"老师似然能区分对错"。
`scripts/informativeness.py` 直接量这件事(Demystifying Eq.4 的 informativeness ℐ,
以及把它当判别器的 AUROC,Rethinking 报 0.73–0.75)。**AUROC 低就说明这个臂在过滤
噪声 —— 不训练它就得到了关于它的判决**。`campaign.tsv` 里这一行的备注就是这个意思。

**执行**:
```bash
python scripts/informativeness.py --model Qwen/Qwen3-1.7B-Base --run-id student_base --step 0   # 先跑这个
DISTILLATION_LOSS_MODE=k1_fire_gate  SIMOPD_FIRE_DROP_FRAC=0.2
```

---

### 轴 H — 序列视野(一条轨迹教多长)

#### `h1_first_segment` — 只监督前 K 个 token

**直观**:ESR 的主张是**老师的信号集中在响应前段** —— 推理的"骨架"(用哪个方法、
设哪个变量)在开头几百个 token 就定了,后面是执行。如果成立,监督预算可以砍掉一大半。
这是个**可证伪的科学主张**,不是效率件,所以从效率区捞回来单独成轴。

**数学**:

```
    loss_t  =  1[ t < K ] · Δℓ_t · ( T / T_keep )        K = 512
```

**纪律:生成不截断,只动监督窗口。** ESR 原文是 rollout 截断形式,那会**改变采样分布
本身**,和 A 轴混淆 —— 这里只把 loss mask 前移。

**这个臂故意削监督预算**(那正是它的主张),所以台账**报预算而不是配平**,
`firstseg_covered_frac` 记实际覆盖比例。

**执行**:
```bash
DISTILLATION_LOSS_MODE=k1_firstseg  SIMOPD_FIRST_SEGMENT_K=512
```

---

### 2.9 一览表

| run_id | 轴 | 一句话 | loss_mode | 状态 |
|---|---|---|---|---|
| `vanilla` | — | sampled-token reverse KL,PG 形式 | `k1_rec` | stock |
| `a1_gkd_mix0.5` | A | GKD λ=0.5 混合 | `k1_rec` | needs(等预计算) |
| `a2_coldstart` | A | 离策略 SFT 冷启动 → OPD | `k1_rec` | stock*(需前置) |
| `a4_dagger_anneal` | A | DAgger:λ(step) 1→0 退火,整条 response 掷币 | `k1_rec` | needs(等预计算) |
| `a5_aggrevate` | A | AggreVaTe:学生前缀 κ + teacher 在线续写 | `k1_rec` | needs(等彩排) |
| `b1_skew_kl` | B | `KL(π ‖ 0.1π+0.9q)`,有界 | `skew_kl_a0.1` | stock |
| `b2_forward_kl` | B | forward KL(已知更差的对照) | `forward_kl_topk` | stock |
| `c1_lsm_topk32_renorm` | C | teacher top-32 上的截断 reverse KL | `lsm_topk_renorm` | stock |
| `c2_quantile_budget` | C | **[自研]** 按 margin 分位自适应支撑 | `qb_quantile_budget` | stock |
| `d1_tip` | D | 高熵 ∨ 自信但错 | `tip_select` | stock |
| `d2_selectkd` | D | 学生提议,老师验证 | `selectkd_verify` | stock |
| `d3_teachability` | D | 分歧 × 可达 | `teachability_select` | stock |
| `e1_pl_rank` | E | **[自研]** 保序不保值 + value 锚 | `pl_rank_anchor` | stock |
| `f1_soft_log` | F | `sign(d)·log(1+\|d\|)` | `k1_softlog` | stock |
| `f2_hard_clip` | F | `clip(d, ±10)` | `k1_rec` + clamp | stock |
| `g1_verified_only` | G | 只蒸验证器通过的轨迹 | `k1_verified_only` | stock |
| `g2_fire_likelihood` | G | 丢老师似然最低的 20% | `k1_fire_gate` | stock |
| `h1_first_segment` | H | 只监督前 512 token | `k1_firstseg` | stock |

---

## Part 3 — 每个 top-k 臂免费带的三组面板

全部是 kernel 已持有张量的纯函数,**零额外前向**。

### `pi_tail_k{8,16,32}` — student 尾质量 π(S̄)

headline 定理的**直接量**:截断 reverse KL 误差 = `π(S̄)·KL(π‖q | S̄)`。文献只报交集
(overlap ratio),**从没人报这条尾巴**。

因为 teacher 支撑是**按秩排序**的,更窄的支撑就是它的前缀 —— **一次前向拿到整条 K 扫描**。
它顺带回答"两个 K 值到底是不是实质不同",在花掉一个 run 之前。

### `shadow_*` — D 轴影子掩码

**每个 run 都同时评估 TIP / Teachability / SelecTKD 三个选择器会选哪些 token**,
输出两两的交、并两个逐 token 指示量。

Jaccard 不直接发,因为指标管道报的是掩码内均值,而 `mean(A∧B)/mean(A∨B)` **恰等于**
`|A∩B|/|A∪B|`(token 数约掉)—— 台账做一次除法即可。

用途有二:(a) 预注册冗余预测 #4("三个选择器高度重叠")本来要三个训练 run 才能答一次,
现在每个 run 都顺带答一遍;(b) **组合筛选** —— 影子几乎一致的两个设定,是同一个臂的
两个名字,不值得各花一个 run。

`SIMOPD_SHADOW=0` 可关。

### Rethinking Eq.6/7/8 — overlap 与熵差

| key | 含义 |
|---|---|
| `overlap_ratio` | Eq.6:两个 top-k **集合**的交集比例 |
| `overlap_teacher_mass` / `overlap_student_mass` | App B.1:交集 token 携带**多少概率质量** |
| `overlap_token_advantage` | Eq.7 原式 |
| `entropy_student` / `entropy_teacher_topk` / `entropy_gap_abs` | Eq.8,`\|H(q)−H(p)\|` |

**为什么质量版是关键**:他们那句"交集持 97–99% 质量"正是"支撑大小无所谓"的**承重步**,
而只有质量版能证实或推翻它 —— 一大堆可忽略 token 的大交集,**在计数上一模一样**。

**熵那项两侧分开发**而不是只发差值:student 侧精确(全词表),teacher 侧只能在返回的
top-k 上算,**系统性低估**它看不见的尾巴。分开发,近似程度就不会被埋进减法里 ——
`teacher_mass` 低于 1.0 的部分正好是缺失的量。

### 命名纪律(重要)

**只有 k1 族的 loss 等于 `−Δℓ`,才可以用 `delta_ell_*` 这组 key。**
C/E 轴优化的是散度(≥0)或排序损失,一律报 `loss_*`。两组 key 不相交 —— 否则三种不同
的量会被画到同一张跨臂对比图上,曲线看起来可比而其实在量不同的东西。

---

## Part 4 — 怎么跑

### 4.1 装环境(一台机器一次)

```bash
cd /mgfs/shared/Group_GY/changhao/SimOPD
bash deploy/dsw/setup.sh          # clone verl + 建 ./simopd venv + torch/vLLM/flash-attn + 模型数据
source simopd_env.sh              # 当前 shell 立刻生效(setup 已挂进 ~/.bashrc)
bash deploy/dsw/doctor.sh         # 只读体检,一屏看清哪里坏了
```

### 4.2 冒烟(换机器先跑这个)

```bash
bash deploy/dsw/envtest.sh                              # 单臂 3 步,失败自动 triage
LANES=1 bash deploy/dsw/run_parallel.sh --rehearsal "vanilla:0"    # 加泳道包装
bash deploy/dsw/run_parallel.sh --rehearsal             # 加 4 路并发
```

### 4.3 跑一个臂(手动,调试用)

```bash
source simopd_env.sh
eval "$(python scripts/arm.py env c1_lsm_topk32_renorm)"   # 展开该臂的 env 块
export EXPERIMENT_NAME=c1_lsm_topk32_renorm_s0
export TOTAL_TRAINING_STEPS=250 TEST_FREQ=25 SAVE_FREQ=50
bash scripts/run_opd_baseline.sh \
    data.seed=0 \
    actor_rollout_ref.rollout.seed=0
```

`arm.py env <run_id>` 打印的就是那个臂的全部偏离,**别手抄** —— 注册表是唯一真相源。
臂如果是 `needs` 状态,这条命令会直接拒绝并告诉你卡在哪条接缝上。

### 4.4 跑多个臂(正常路径)

```bash
python scripts/arm.py check                    # 看清哪些能跑
python scripts/arm.py list --status stock

# 8 卡 = 4 泳道,round-robin 分派。STEPS=250 是必须的 —— 默认值 150 还没跟上预注册
STEPS=250 bash deploy/dsw/run_parallel.sh                          # 默认工作表
STEPS=250 bash deploy/dsw/run_parallel.sh "vanilla:0 vanilla:1 vanilla:2 f1_soft_log:0"
STEPS=250 LANES=2 GPU_LIST="2,3 6,7" bash deploy/dsw/run_parallel.sh "d1_tip:0 d3_teachability:0"
```

默认工作表 = `vanilla` × 3 seeds(**这三个 seed 就是噪声底**,`|Δ| < 噪声底 → 判平`
的操作定义靠它),然后每个 stock 臂一个 seed。

**一个 run = 2 卡**(actor + teacher 池)。泳道之间只共享文件系统:各自的
`CUDA_VISIBLE_DEVICES`、各自的 Ray 临时目录、各自的日志。

> ⚠ 泳道清理**不能**用 `ray stop --force` —— 它是全机范围的,会连带杀掉其它泳道。
> 用 `bash deploy/dsw/sweep_lane.sh <n>` 按泳道精确回收。

### 4.5 看

```bash
python scripts/watch.py                  # 全部 run 一屏
python scripts/watch.py --watch 60       # 每 60 秒刷新
python scripts/watch.py --run vanilla_s0 # 单 run 的 val 轨迹 + 长度/步时
python scripts/progress.py               # campaign 高度:每一列纸在什么状态
python scripts/triage.py                 # 出错时:logs/ 里最新那个
python scripts/triage.py /tmp/x.log --ray   # 顺带挖 Ray worker 日志
```

### 4.6 评

```bash
python scripts/transfer_eval.py --selfcheck      # 换机器先跑:验代码沙箱(不需要 GPU)
bash scripts/eval_transfer.sh vanilla_s0         # 单臂 final ckpt 的迁移列
bash scripts/eval_transfer.sh --all              # 所有已出 checkpoint 的臂
python scripts/d6_matrix.py --bench amc23 --report docs/d6_amc23.md
python scripts/informativeness.py --model <ckpt>/actor --run-id vanilla_s0 --step 250
```

> ⚠ **代码题会因机器负载假失败**:同一批 canonical solution,load≈20 时 160/164、
> load≈6 时 163/164。默认已把单测时限地板抬到 4.0s,但**别在泳道满载时跑代码评测**。

---

## Part 5 — 我们这套集群(Group_GY)

### 5.1 机器

`~/.ssh/config` 里有 `dsw251`–`dsw265`。**当前分配给我们跑实验的 7 台**:

| host | GPU | 状态 |
|---|---|---|
| dsw251 / dsw252 / dsw253 / dsw254 | 8 × **H100 80GB HBM3** | 空 |
| dsw258 / dsw261 / dsw262 | 8 × **H100 80GB HBM3** | 空 |

**共 56 卡 = 28 条泳道**(2 卡/run)。其余机器暂时有人用。

- driver **550.127.08**,全机一致;Python **3.11.11**(无 3.12,无 uv,无 conda,无 nvcc)
- `gpuall` 一条命令看全集群 GPU 状态(**不要另写脚本**)
- 这些 pod 里 `nvidia-smi` **看不到 PID**,判断占用要靠显存/利用率

**ssh 拓扑是星形,不是网状**(2026-08-07 实测):只有 **dsw243**(当前这台)的
`~/.ssh/config` 里有 251–265 的 IP,worker 之间既没有这份 config 也没有 DNS ——
在 dsw251 上 `ssh dsw252` 直接 `Name or service not known`。

操作后果:**dsw243 是唯一的编排点**。启动 7 台的 campaign 只能从这里逐台 ssh 过去
`nohup` 拉起,worker 之间不能互相调度。好在**文件系统是共享的**(见下),所以
"谁认领了哪个臂"这件事走文件系统而不走网络,星形拓扑不构成障碍。

### 5.2 文件系统:**是真共享的**

实测(2026-08-07):在 dsw243 写一个文件,7 台全部立刻看到;
`/mgfs/shared/Group_GY/changhao/SimOPD` 在每台机器上是同一份。

**这比原设计好。** 原来 m1–m3 共享而 cornell 是孤岛,`campaign.tsv` 为此专门区分了
静态分派行和 `machine=any` 的池子行。我们这里 **7 台全在一个池子里** —— `mkdir` 原子
认领、共享 HF cache、共享 ckpt、共享日志、一次 `git pull` 全体生效,全都成立。

> `df` 对 `/mgfs/shared` 报的是容器 overlay(7.0T,可用 3.6T),不是这个卷的真实配额。
> 按 3.6T 规划:**每 run ≈ 50GB**(2 个 ckpt × 25GB),15 臂 ≈ 750GB,余量充足。

### 5.3 上机前必须改的四处

1. ~~`CKPT_ROOT` 要改~~ —— **不用管,setup.sh 已经处理**。`run_opd_baseline.sh:171`
   的 fallback 确实是 Cornell 路径 `/scratch/zz865/simopd/ckpt`,但 `setup.sh` 生成的
   `simopd_env.sh` 会 export `CKPT_ROOT=$DATA_ROOT/ckpt`,而 `DATA_ROOT` 默认就是
   `<repo>/../simopd_data` —— 在我们这里正好落在共享盘上。`DATA_DIR` / `HF_HOME` /
   `WANDB_DIR` 同理。**只要 `source simopd_env.sh`,那个 Cornell 默认值永远轮不到生效。**

2. **flash-attn wheel 版本对不上**。仓库里 vendored 的是
   `flash_attn-2.8.3.post1-**cp312**-cp312-linux_x86_64.whl`,而这 7 台上是
   **Python 3.11.11,没有 3.12,也没有 uv/conda**。三条路:
   - 换一个 cp311 的官方 release wheel(最省事)
   - 用 `slurm/build_flash_attn.sbatch` 从源码编 —— 但**这些机器上没有 nvcc**,要先装 CUDA toolkit
   - 先 `USE_REMOVE_PADDING=False` 绕过(有性能代价,且**入指纹** —— 会让这批 run 与
     开了 padding 的不可比,所以要么全场关要么全场开)

3. **`a2_coldstart` 的 env 硬编码了 `zz865` 路径**,`slurm/` 下 10 个 sbatch 同样。
   要跑这个臂必须先改。

4. **H100 不是 A100**。`run_opd_baseline.sh` 里那套显存账
   (0.45 → vLLM 36GB / 61.3 of 80 used)是按 80GB 算的,H100 同为 80GB 所以数值仍成立;
   但**步时会更快**,`plan §7.5` 里 "150 步 ≈ 23h" 的外推要在这里重新校准。
   另外 flash-attn 要有 **sm90** 支持。

### 5.4 分派与启动(2026-08-07 定稿;上游 daemon 机制 merge 后重写)

`configs/campaign.tsv` 已重写为本机队的分派表:**m1=dsw251 / m2=dsw252 /
m3=dsw253 / m4=dsw254**,wave 1 = 全部 16 个 Phase-1 run(15 臂 + vanilla×3,
一次铺满 4 台);**dsw258/261/262 刻意留白** —— 评测、informativeness、D6 探针
这类临时活不跟 campaign 抢泳道。

每台机器的启动是一行(从 dsw243 ssh 过去执行):

```bash
MACHINE=m1 bash deploy/up.sh     # 首次:注册身份 + campaign.sh --dry + 起 daemon
bash deploy/up.sh                # 之后:幂等,随时可重跑
```

daemon 每 15 分钟重新调用一次 `campaign.sh` 补空泳道;它**显式 unset 一切
run-defining 环境变量**(STEPS/TEST_FREQ/…),继承性配置进不来。停一台:
`touch .campaign/daemon.stop.<machine>`(共享盘,任何机器上都能发)。

`deploy/campaign.sh --plan` 审计 manifest 对 `arms.yaml` 的覆盖:每个可跑臂必须
恰好出现一次。改完 manifest 先跑这个。

---

## 附:环境变量速查

| 变量 | 作用 | 默认 |
|---|---|---|
| `DISTILLATION_LOSS_MODE` | 选哪个 loss(臂的主开关) | `k1_rec` |
| `DISTILLATION_TOPK` | teacher top-k 宽度 | 32 |
| `SIMOPD_KEEP_SAMPLED` | 接回采样 token 的 teacher logprob(**D 轴必需**) | 0 |
| `SIMOPD_D_RETENTION` | D 轴保留比例 | 0.5 |
| `SIMOPD_SUPPORT_MODE` | C1 内部消融:`renorm` / `tailbucket` | `renorm` |
| `SIMOPD_QB_TARGET_BUDGET` / `SIMOPD_QB_MARGIN` | C2 目标预算 / margin 定义 | 8 / `max` |
| `SIMOPD_PL_ANCHOR_COEF` | E1 value 锚系数 | 0.1 |
| `SIMOPD_FIRST_SEGMENT_K` | H1 监督窗口 | 512 |
| `SIMOPD_FIRE_DROP_FRAC` | G2 丢弃比例 | 0.2 |
| `LOSS_MAX_CLAMP` | F2 硬截断阈值 | 不设 |
| `SIMOPD_SHADOW` / `SIMOPD_PI_TAIL_WIDTHS` | 诊断面板开关 / 宽度(**不入指纹**) | 1 / `8,16,32` |
| `ROLLOUT_GPU_MEM_UTIL` | vLLM 显存份额(**非数值中性,入指纹**) | 0.55 |
| `TOTAL_TRAINING_STEPS` / `TEST_FREQ` / `SAVE_FREQ` | horizon 与节奏 | **150**(⚠ 预注册是 250)/ 25 / -1 |
| `MAX_CKPT_KEEP` | 保留几个 checkpoint | 2 |
| `CKPT_ROOT` / `DATA_DIR` | **必须按本集群改** | Cornell 路径 |
