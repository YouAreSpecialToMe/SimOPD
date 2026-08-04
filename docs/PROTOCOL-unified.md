# SimOPD 统一实验协议 v1(预注册;2026-07-31)

> 依据:10 篇受审/竞品论文 setup 实测调研(下表)+ Demystifying 协议实录
> (PROTOCOL-demystifying.md)。**所有臂、所有 phase 共用本协议;每臂只动自己的
> 那一个旋钮。** 本文档冻结后改动需在台账记录理由。

## 0. 调研结论:文献 setup 现状(= 审计动机的直接证据)

| 论文 | student | teacher | 训练集 | n | τ/top-p | resp len | batch | lr | eval 协议 |
|---|---|---|---|---|---|---|---|---|---|
| Demystifying 2607.13399 | Qwen3-1.7B-Base | 1.7B/4B-GRPO(自训)、**4B-Instruct-2507**、8B | **Nemotron-Cascade-RL-Math** | **1** | 1.0/1.0 | 16384 | 128 | 未报 | MATH500/Minerva pass@1;AMC23+AIME24-26+HMMT25 avg@32 |
| Rethinking 2604.13016 | Qwen3-1.7B-Base;R1-Dist-1.5B | Qwen3-4B-NT、4B-Base-GRPO、R1-7B、Skywork-OR1、JustRL | DAPO-17K(+OpenThoughts3 冷启动、DeepMath-57K) | 4 | 1.0 | 7168 | 64 | 1e-6 | AIME24/25+AMC23 **avg@16 @ τ0.7/p0.95** |
| LSM 2603.25562 | Qwen2.5-7B/1.5B-**Inst** | OpenThinker3-7B、GiGPO | DAPO-17K en + agentic | ? | 1.0/**0.9** | 16384 | 128/mini64 | **2e-6** | **pass@1** MATH500/AIME/Minerva/Olympiad;400 步;K=32 双侧重归一化;verl |
| TIP 2604.14084 | Qwen3-4B;Llama-8B;Q2.5-1.5B | Qwen3-8B-GRPO;Llama-70B;Q2.5-14B-think | DAPO | **16** | 1.0/1.0 | 8192 | 8 | 1e-6 cosine | MATH500/AIME mean@16;OPSD 库 fork |
| SelecTKD 2510.24021 | (摘要未名) | — | IF/math/code/VLM | — | — | — | — | — | propose-verify,TAR 曲线 |
| Teachability 2605.26844 | Qwen3-1.7B/4B;Q2.5-3B | Qwen3-4B/8B-GRPO/14B;R1-14B | DAPO | 未报 | 未报 | 未报 | 未报 | 未报 | AIME/GPQA-D/HumanEval/IFEval/MATH500,5 eval seeds ±std;64×H800 |
| RG-OPD 2607.04037 | Qwen2.5-1.5B-Inst | Q2.5-14B-Inst | UltraInteract 子集 | N | eval 0.6 | 1024/8096 | ? | **5e-6**+warmup50 cosine | GSM8K/MATH/MBPP/IFEval acc,3 seeds;topK=50 RKL+尾桶修正 |
| FiRe 2606.02684 | Qwen3-4B-NT | 30B-A3B-Inst;4B-RL-Math | DeepMath-103K lvl6 | ? | 1.0/1.0 | 16384 | **1024** | 1e-6 | 数学 Avg@8 + 代码 pass@1;165 步;PPO-clip 加权 adv |
| ESR 2605.27028("Less is More") | Q2.5-Math/Qwen3/Gemma 1.5-32B | 至 72B | NuminaMath 等 | 1 | **0.7** | **N=100(只生成前段!)** | 16 | **5e-5 LoRA** | MATH500 avg@4;24× 提速主张 |
| EasyOPD 2607.11012 | Phi-4-mini;Qwen3-8B/1.7B | Q2.5-7B-I;Qwen3-8B/4B | 4k 长度混合 | ? | **0.6** | 4096 | 64 | **5e-7** wu0.05 | 框架论文;verl0.5+vllm0.8.5 钉死 |

**读法**:11 个 setup 里,student 从 1.5B-Inst 到 7B、n 从 1 到 16、lr 横跨 5e-7~5e-5、
eval 从 greedy pass@1 到 avg@16@τ0.7 —— **没有任何两篇可逐行比**。这就是本项目存在的理由,
也是统一协议必须一次钉死的原因。

### 0.1 域维度(2026-08-03 补;上表只有超参,而"域"是我们要主张的那一维)

| 论文 | 训练域 | 评测域 | 跨域形态 |
|---|---|---|---|
| **Demystifying**(锚点) | math | math | — |
| **Rethinking**(锚点) | math | math | — |
| TIP | math | math | — |
| ESR | math | math(+HumanEval) | 迁移评测 |
| LSM | math + agentic | math | — |
| FiRe | math | math + **code** | **迁移评测** |
| Teachability | math | math + **code/IFEval/GPQA** | **迁移评测** |
| RG-OPD | UltraInteract(混合) | math + code + IFEval | 混合域训练 |
| SelecTKD | IF/math/code/VLM(多域) | 多域 TAR | 混合域训练 |
| EasyOPD | 4k 混合 | math + code | 混合域训练 |

**结论(直接支撑我们的设计与主张)**:

1. **训练只在 math:6/10;评测只在 math:5/10。两篇锚点(Demystifying/Rethinking)
   训练与评测全在 math。** → 我们"筛选只在 math"不是妥协,是被审对象的主场。
2. 跨域的四篇分两类,**都不是逐域对照**:迁移评测型(FiRe/Teachability)= math 训、
   跨域评;混合域训练型(RG-OPD/SelecTKD/EasyOPD/LSM)= 多域搅在一个训练集里报总分,
   **拆不出"同一 trick 在 math 与在 code 分别如何"**。
3. 故:**逐臂迁移列**(统一协议)与 **code 域蒸馏内部量**(π(S̄))在本池内零先例。
   前者已落地(METRICS §2,每臂 final ckpt);后者是 headline 预言的检验入口,未启动。

⚠ 两处待复核(HTML 抽取所得,写进 paper 前必须核):Teachability 训练超参整行"未报";
ESR 评测集在 BENCHMARKS §1(列了 HumanEval)与本表(MATH500 avg@4)口径不一致。

## 1. 锁定协议(所有臂共用)

### 模型
- student:**Qwen3-1.7B-Base**(筛选 = 终验档 = 复现锚点,2026-08-04 起统一)
- teacher:**Qwen3-4B-Instruct-2507**(主档,2.4×)。全现货,零自训。

**师生档位重定(2026-08-04,依据实测;取代 v3.1 的 0.6B 提速版)**

*学生*:0.6B-Base 收敛到 MATH500 **0.468**,而 **1.7B-Base 未训练就是 0.468** ——
0.6B 整个 campaign 能测到的范围,全部位于真实学生的零点之下。且 0.6B 第 25 步即饱和、
第 90 步进 Mode A,而 1.7B 到第 50 步仍在涨(0.468 → 0.604 → 0.636)。
另:**Base 学生在受审池里只有两家,恰好就是两篇锚点**(Demystifying / Rethinking)。

*老师*:非思考 MATH500 天花板实测(greedy,协议同款模板,2026-08-04):

| teacher | pass@1 | 平均长度 | 备注 |
|---|---|---|---|
| **Qwen3-4B-Instruct-2507** | **0.896** | 1548 | 2507-Instruct 是 Qwen 的**非思考原生**线 |
| Qwen3-8B | 0.792 | 1082 | hybrid,强在 thinking;`enable_thinking=False` 砍掉其主武器 |
| Qwen3-1.7B | 0.702 | 982 | |

**在非思考约束下,更大的老师不等于更强的老师。** 4B-Instruct-2507 以一半尺寸高 10.4 点,
故取为主档;它同时是 Demystifying 的现货格,**筛选档与复现锚点合为同一个 run**。

*为什么不自训*:4 篇用了自训 GRPO 老师(Demystifying/Rethinking/TIP/Teachability),
但**每一篇都同时报了现货老师**,自训不是任何一篇的必要条件;而一个只有我们有的老师,
会成为这份审计里**唯一别人无法复现**的部件。真需要更强老师时,走公开 RL 权重
(Skywork-OR1 / JustRL / R1-7B,均在 Rethinking 的老师列表内),仍是零训练。
- teacher bf16,禁量化(logprob 是被审对象)。

### 数据
- 训练:**nvidia/Nemotron-Cascade-RL-Math**(14,476 题)—— 锚点对齐决定。
  文献众数是 DAPO-17K;如 W4 有余力,vanilla 在 DAPO-17K 上加一个 spot-check run
  证明配方不依赖训练集选择(可选,不预注册为义务)。
- prompt 模板:chat template + `enable_thinking=False`(空 think 前缀,
  对齐 Demystifying 非思考约束);所有臂、student rollout 与 teacher 打分同模板。

### Rollout
- **n=1**/prompt,τ=1.0,top-p=1.0(Demystifying 对齐;注意 LSM 用 0.9 —— 那是
  他们 C 轴主张的一部分,作为 C 轴消融存在,不进默认协议)
- max prompt 1024;max response **8192(筛选)/ 16384(锚点与 Phase 3 终审)**;
  截断率必报(飞行记录仪)。

**输出上限文献对照**(2026-07-31 实测;释义:训练帽 7k-16k 是 math OPD 的文献包络):

| 方法 | 训练 response 帽 | val/eval 帽 | 备注 |
|---|---|---|---|
| Demystifying | 16,384 | 未报 | 非思考约束 |
| Rethinking/thunlp | 7,168(库内选项 3072–31,744) | **31,744** | R1-distill 思考型 student → val 帽反而更大 |
| LSM | 16,384(math)/512(agentic) | 未报 | |
| TIP | 8,192(prompt 2,048) | 未报 | 与我们筛选帽相同 |
| FiRe | 16,384 | — | |
| EasyOPD | 4,096 | — | |
| RG-OPD | 1,024(主)/8,096(长程) | — | 短表单任务 |
| ESR | **100(只生成前段)** | — | 24× 提速主张来源;H 轴语境 |
| Teachability / SelecTKD | 未报 | 未报 | SelecTKD 是 Qwen2 时代短输出设定 |

**val 帽规则**:val 帽 = 训练帽(8k 筛/16k 终)。文献无 "val 帽 < 训练帽" 先例
(thunlp 反而放大),不做 val 缩帽提速;step-0 val 慢是未训练 base 跑满帽的
暂态,非思考训练后长度自然坍缩。若筛选 val 截断率持续 >5%,升格为协议问题再议。

### 目标与优化
- vanilla 基线:sampled-token reverse-KL(k1)作逐 token advantage(Δℓ_t),
  PPO-clip(0.2),**PG 公式化**(verl `use_policy_gradient=True`),
  `use_task_rewards=False`;loss 聚合 **token-mean**。
- AdamW,**lr 1e-6 恒定,无 warmup**(文献众数:thunlp/TIP/FiRe 同值;
  Demystifying 未报 —— 台账记录此推断),grad-clip 1.0,
  batch = mini-batch = **128**(每 rollout batch 恰一个 optimizer epoch,严格 on-policy)。
- **异步训练裁定**:verl 的 fully_async / one_step_off 管线**不用**(协议禁区,非工程
  取舍)—— 异步引入 staleness≥1,正是 Demystifying "eliminate off-policy bias" 排除的
  偏差;且 k1 Δℓ-as-advantage 假设 rollout≡当前策略(ratio≡1),异步会给每个臂叠加
  IS/clip 混淆。步内 serving 异步(rollout.mode=async,continuous batching)保留。
  凭证:飞行记录仪 `trajectory_staleness` 恒为 0。墙钟压力走 run 级并行(rush 池),
  不走步内异步。
- student FSDP bf16 混合精度(偏离 thunlp 的 fp32 actor —— 记录;1.7B 下
  bf16 是 2026 事实标准)。gradient checkpointing 开。

### 训练长度与种子
- 筛选(贪心 R1-R4):**150 步上限 + 预注册早停**(plan §4;2026-08-04 从 300 步改),
  单 seed;判据 = MATH500 逐题配对 McNemar p<0.05,|Δ|<噪声底判平。
  跨臂比较取**最小公共步**,每臂的停步记入 `logs/early_stops.tsv`。
  ⚠ 150 这个上限是在**已废弃的 0.6B 档**上标定的;1.7B 到第 50 步仍在涨、
  clip 仅 0.27,**可能偏短而非偏长** —— 待第一个完整 1.7B vanilla run 后重定。
- 噪声底:**vanilla 1.7B-Base ← 4B-Instruct-2507** 以 3 个不同 seed 重跑,
  MATH500 pass@1 的极差即噪声底(W1 交付)。同一批 checkpoint 顺带产出**逐域**噪声底
  (math/code/IFEval),决定迁移列里哪些域留得下(plan §4)。
- 终审:配方/vanilla/全家桶 ×2 seeds 起步(最终配方行 3 seeds)× 16k × 三域。

### 评测(全部 run 统一)
- 训练内 val:**MATH500 pass@1,greedy(τ=0)**,每 25 步(筛选)/ 每 5 步(锚点前期)。
- checkpoint 终评:**AMC23 avg@32,τ=0.7/top-p=0.95**(thunlp 惯例;
  若 Demystifying 精读给出确切参数则改从其,改动记台账)。
- AIME24/25 avg@32:全档(学生统一为 1.7B 后不再有地板问题;这正是换档的收益之一)。
- code(HumanEval+/MBPP+ pass@1)与 IFEval:**每臂 final ckpt 的迁移列**(METRICS §2)。
- eval 用训练同款非思考模板。

### 飞行记录仪(每 run 必录)
overlap_ratio(verl 内建打点)、student/teacher 逐 token mass、Δℓ 分布分位数、
响应长度曲线 + 截断率、熵曲线、val 子集 pass@8(多样性面板)。

## 2. 各臂实现来源(不重复造轮子;能移植就移植)

**代码复用政策(2026-08-01 定,写进 paper 的 reproducibility 节)**

| 复用什么 | 做不做 | 理由 |
|---|---|---|
| 他们的**方法实现**(作为规格) | **必须** | 论文文字有歧义,代码没有。"你把我的方法实现错了"是审计论文最致命的审稿意见 |
| 他们的 **harness**(训练框架/数据管线/评测) | **不用** | 各家框架互不相同(TA-OPD 自建 PyTorch 64×H800、TIP fork OPSD、LSM 用 verl-agent、FiRe 自建)。直接跑各自仓库 = 原样复制"没有两篇可比"这个病,而那正是本文要治的东西 |

**执行方式**:每个臂实现前先读对应仓库,把差异逐条记进本表;实现后与其代码做
数值/结构对照,对照结论写进臂的 `note`。已完成的对照:

- **LSM(C 轴)vs EasyOPD `kl_renorm_topk`**(`methods/opcd/core.py`):双侧 logsumexp
  重归一化 + KL=Σp(log p−log q) 与我们一致 ✓;**但他们多一层 `> -1e15` 填充位掩码,
  我们原先没有 —— 已补**(padded slot 若进入归一化会静默压低所有概率)。

**尚未对照(实现前必做)**:TIP(HJSang/OPSD fork)、Teachability(wyy-code/TA-OPD)、
FiRe、RG-OPD。SelecTKD / LSM 仓库地址待查。


| 臂 | 参考实现 | 移植方式 |
|---|---|---|
| C: top-k 截断 RKL(±重归一化) | thunlp/OPD fork(`LOG_PROB_TOP_K`,`TOP_K_STRATEGY={only_stu,only_tch,intersection,union}`)+ LSM 论文(K=32 双侧重归一化) | 语义照抄进 mainline 损失注册表(~80 行);尾桶修正参考 RG-OPD(K=50+tail correction) |
| C: 分位预算(自研) | 无(QB 移植) | 新写,走注册表(~120 行) |
| D: TIP | HJSang/OPSD_OnPolicyDistillation fork | soft-OR score fn 移植(熵+自信错,批内 top2% clip) |
| D: Teachability | wyy-code/TA-OPD | s=D̃·C̃ score fn 移植 |
| D: SelecTKD | 库待查(W2 第一件事) | propose-verify 掩码;TAR 打点进飞行记录仪 |
| **D 轴共同前置** | **thunlp/OPD 官方库(已本地 clone)** | 其 `ray_trainer.py` 用**两次前向**:先算 student top-k ids/logprobs,再让 teacher 按 `TOP_K_STRATEGY∈{only_stu,only_tch,intersection,union}` 在选定支撑上打分 —— 这正是 verl 主线缺的那块(主线 teacher 只返回 top-k **或**采样 token,二者不可兼得),也是 D 轴三个臂的解锁钥匙。移植其**设计**,不移植其代码(fork 已与主线分叉) |
| E: PL-rank / set-coverage(自研) | PLD 2506.12542 公式 | 新写(~100 行) |
| F: 软 log 压缩 | Demystifying 公式 sign(Δℓ)·log(1+|Δℓ|) | 估计器路径逐 token 变换(~20 行);硬 clip 已有(`loss_max_clamp`) |
| G: verified-only | verl reward 管线现成 | trainer 侧 batch filter(~50 行);发表代表 RG-OPD 的符号对齐门控作替补语义参考 |
| G: teacher 似然过滤 | FiRe github | 轨迹 bottom-20% 按 (1/T)Σlog π* 过滤,移植 |
| H: 首段监督 | ESR 2605.27028 | **只取 loss 掩码形式**(前 K token 计损,生成不截断)~10 行;ESR 原版的"只生成前 N"是 rollout 分布改变,维持工程落选(与计划 §A 落选条目同一件事,案卷需注明) |
| A: 冷启动 | thunlp recipe(OpenThoughts3 math 子集 SFT→OPD) | 复用其 SFT 配置思路;verl sft 或 LlamaFactory |
| B: skew-KL / JSD / FKL | DistiLLM / GKD 公式 | 估计器族加 loss_mode(各 ~40 行;数学三行,不需要移植代码) |

## 2.5 环境偏离(跨集群必须记录)

DSW 上 `setup.sh` 会**按驱动版本自动选 CUDA 家族**:驱动 ≥580 走 cu130
(torch/vLLM 全部走阿里 PyPI 镜像,不碰 GitHub);<580 走 cu129(torch 走阿里
pytorch-wheels,vLLM 轮子只有 GitHub release 一处,需 GITHUB_PROXY)。

**torch 与 vLLM 的版本号两条路完全相同(2.11.0 / 0.26.0),只是 CUDA runtime 不同。**
数值差异应落在 run-to-run 噪声内(噪声底本来就在量这个),但**这是一条跨集群偏离,
必须记进台账**:vanilla 双跑校验的 pass@1 差值若超噪声底,CUDA 家族是第一嫌疑。

## 3. 与文献 setup 的显式偏离(防审稿人问)

1. lr 1e-6 无 warmup:Demystifying 未报,取文献众数;RG-OPD(5e-6+warmup)与
   EasyOPD(5e-7+warmup)的选择作为已知变体记录,不扫描。
2. bf16 student(thunlp 用 fp32):bf16 为主流;若锚点对不上 Demystifying 曲线,
   fp32 是第一嫌疑switch。
3. 训练集用 Nemotron-Cascade 而非文献众数 DAPO-17K:锚点对齐优先;可选 spot-check。
4. n=1(Demystifying/ESR)而非 4/8/16(thunlp/TIP):Demystifying 已消融 n,
   n=1 是其协议;我们不再扫描。
5. AMC23 avg@32 采样参数暂用 thunlp 值(0.7/0.95),待 Demystifying 精读校正。

## 4. 开口(精读时逐项关闭)

- [ ] Demystifying:lr/步数/评测采样参数/空 think 模板确切句法(HTML 已抽两轮,均 ABSENT
  → 需查其 repo 或邮件作者)
- [ ] SelecTKD 代码库定位 + 其 on-policy 变体的确切掩码规则
- [ ] LSM 是否公开代码(论文 verl 系;fetch 未见 repo 链接)
- [ ] 各臂论文 arXiv 编号最终核验(案卷 v1 的已知风险)
