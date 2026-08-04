# SimOPD 实验计划 v2(独立项目版)

> **v1→v2 变更**:与 VeTRA/ToolQA 完全脱钩,全新独立项目;场景从 agentic tool-use 改为
> **general OPD**;模型从 Llama 线换成 **Qwen3 家族**(与两篇竞品同设定,逐行可比);
> agentic 专属内容(oracle 决策 span、H 轴、ToolQA 协议)全部移除。

> 一句话:把现有 OPD 变体按统一协议审判成三类(必需 / 无用 / 有副作用),
> 组合出最小有效配方 **SimOPD**。模板血统:LitePPO (Tricks or Traps, 2508.08221)。

---

## 0. 定位:文献审计(audit),与机理研究不同线

**血缘声明**:Rethinking/Demystifying 之于 SimOPD == GRPO/DAPO 之于 LitePPO。
它们是一次科学(研究 OPD 本身 + 自提组件);本文是**文献审计**(研究对象 = OPD
变体文献本身,动物园 462 条目、核心受审池 ~55 篇,统一协议判决 + 最小配方合成;
普查与参赛名单详见 SimOPD-casefile.md)。
**它们不是竞品,是案卷** —— 其组件(冷启动/prompt 对齐/信号调节)全部作为参赛臂入庭。

| 线 | 代表 | 做什么 |
|---|---|---|
| 机理研究 | Rethinking 2604.13016 · Demystifying 2607.13399 | OPD 何时有效 / 如何失败 |
| 变体提案 | LSM、TIP、SelecTKD、EOPD…(核心 ~55 篇,zoo 462 条) | 各提一个改法,各自设定自证 |
| **审计与合成(本文)** | SimOPD | 统一协议判决全场 + 最小配方 |

三条纪律:
1. **轴分两类,论文明标**:fresh 仲裁(token 判据、支撑设计、rank、门控、跨域)vs
   replication+boundary(粒度轴 —— "sampled-token 够用"只在 overlap 97–99% 甜区
   验证过,我们测低 overlap 区间是否翻转;调节轴 —— 软压缩的适用域);
2. **自研臂防火墙**:分位支撑 / PL-rank / set-coverage 标注为"变体空间的三个自然
   空格,接受与所有被告完全相同的预注册协议" —— 防"既当裁判又当选手"之讥;
3. **genre 差不豁免赛跑**:护城河 = 覆盖广度 + 方法论器物(台账/副作用面板/
   双向贪心/预注册)+ 速度。
另:跨域鲁棒性列(math+code+IFEval)保留为审计的固有维度。**措辞更正(2026-08-03,
调研复核后)**:"没人有的一列"说得太满 —— 迁移评测(math 训→code/IF 评)已有先例,
FiRe 报 code、Teachability 报 code/IFEval/GPQA。精确的空位是两处:
(a)**统一协议下的逐臂**迁移列(他们各自只测自己那一两个 trick,互不可比);
(b)**code 域蒸馏的内部量**(π(S̄) 等)—— 混合域训练的四篇(RG-OPD/SelecTKD/
EasyOPD/LSM)把域搅在一个训练集里,拿不出"同一 trick 在 math 与在 code 分别如何"。

**Headline 假设**(对话 Rethinking):
"支撑大小无所谓 / overlap 够了" 只在 **student 尾质量小**时成立。
定理:尾桶截断 reverse KL 误差 = π(S̄)·KL(π(·|S̄)‖q(·|S̄)),由 student 尾质量控制
(forward KL 对称地由 teacher 尾质量控制);现行按 teacher 质量截断 = 切错边。
预言支撑/token 设计复活的三个区间:**训练早期、大师生差距、跨域迁移(code 的候选
标识符空间大)** —— 可测、可证伪、直接对话竞品结论的适用域。

---

## 1. 域、模型与协议

**域三元组**(全部带规则验证器,支撑 G 轴门控臂):
| 域 | bench | 验证器 | 结构标签(oracle 替代)|
|---|---|---|---|
| math | MATH500 (pass@1), AMC23 + AIME24/25 (avg@32) | 答案精确校验 | boxed 答案 span(弱标签)+ 高熵 fork 位 |
| code | HumanEval+ / MBPP+ | 单元测试 | **AST 类别**:标识符/关键字/运算符/字面量(半 oracle)|
| general | IFEval | 约束检查器 | 约束相关 token(可解析)|

**模型(v3 提速版)**:主力 student = **Qwen3-0.6B-Base**;teachers 全现货 =
Qwen3-1.7B(2.8× gap,甜/中档)+ Qwen3-4B(6.7×,失配档)—— 零自训 GRPO。
终验档 = 1.7B-Base ← 4B(该格子同时是 Demystifying 阶梯现成格 = **复现锚点**,
一个 run 校验全栈)。0.6B 地板效应 → 筛选主指标 MATH500 pass@1 + AMC23 avg@32,
AIME 只在 1.7B 档报;0.6B 噪声底重测。

**训练配置基线**(对齐 Demystifying 便于复现):batch 128、n=1 rollout/prompt、
max 16,384 token、τ=1.0、top-p=1.0、每 batch 一个 epoch。

**统计协议**:每域先实测 run-to-run 噪声底(新项目,没有现成的 0.0054 了);
筛选阶段单 seed + 逐题配对检验(math/code 逐题二值 → McNemar);终审 3 seeds。

**基础设施**:verl OPD 或 EasyOPD (2607.11012),二选一;三域 rollout+验证器环境。

---

## 2. 参赛臂(7 轴;预算配平:同轴内总监督量严格相等)

### 轴 A — rollout 来源与日程
纯 on-policy (λ=1) / GKD λ∈{0, 0.5} / 离策略冷启动→OPD(Rethinking recipe 复核)。
落选:Early-Stopping Rollout、异步生成 —— 工程默认,不进比赛。

### 轴 B — 散度
Reverse KL(默认)/ JSD-β / Forward KL。
落选:EOPD 熵切换 (2603.07079) —— D2 诊断若显示熵剖面支持可复活。

### 轴 C — 支撑(词表轴)★含自研臂
sampled-token(基线)/ teacher top-k+重归一化 (LSM 2603.25562) / teacher top-p /
**分位预算分配 × margin∈{q, π, max(q,π)}(自研,QB 移植,无人占)** /
消融对:重归一化 vs 尾桶。
full-vocab 只跑一次当上界;RSKD (2503.16870) 作尾桶修正可选件并入。

### 轴 D — token 处理(位置轴)
uniform(基线)/ TIP 高熵+自信错 (2604.14084) / Teachability (2605.26844) /
SelecTKD propose-verify (2510.24021) / **code 域加:AST 结构臂(半 oracle)**。
落选:TRACE(需标注)、SafeSteer(安全域)、SEAD/SCOPE(TIP 同族)、
Evidence(TIP+结构夹住)、Rock tokens(Phase 2 余力再加)。
⚠ v2 弱化声明:general 域没有真 oracle,token 判据对打退为"判据 vs 判据"+
code AST 半仲裁 —— 这是放弃 agentic 的主要代价,写进 limitations。

### 轴 E — 支撑内目标 ★含自研臂
value 对齐(KL,默认)/ set-coverage / **PL-rank(保序不保值;loss 形式源自
PLD 2506.12542 视觉离线版,OPD 版无人占)+ 小系数 value 锚(锚强度是消融轴)**。
动机:容量差 + greedy 指标对齐。风险:采样评测(avg@32)下 margin 有用 → 锚必配。

### 轴 F — 信号调节
无(基线)/ 软 log 压缩(Demystifying 赢家复核)。
硬 clip 仅当 D5 测到 Mode B 才进场。

### 轴 G — 轨迹门控
全 rollout(基线)/ **verified-only(规则验证器通过的 rollout 才蒸)** /
teacher 似然过滤 (FiRe-OPD 2606.02684)。
纪律:验证只做过滤,答案不进训练输入。

### 固定约定(不进比赛)
特殊 token 掩码、n=1 题目多样性(Demystifying 已判)、top-p rollout 采样。

---

## 3. Phase 0 — 病理诊断(infra 就绪后一周,零训练)

**顺序:D6 先行**(teacher 选型),再 D1'–D5(相对选定 teacher 计算)。

| # | 测什么 | 决定 |
|---|---|---|
| D6 | teacher 梯子:逐题 "teacher对&student错" 率,4B/8B/32B × 三域 | teacher 定档;复核"强 teacher 停滞" |
| D1' | student 尾质量 π(S̄) 按位置类别 × 域 × 训练阶段分解 | 轴 C 生死;headline 第一证据 |
| D2' | overlap ratio 按位置类别分解(math fork 位 / code AST 类 / IFEval 约束位)| 轴 D 生死;对话 Rethinking |
| D3 | 初始 overlap / 熵差 | 轴 A 冷启动臂 |
| D5 | 长度/截断病理(Mode A/B)| 轴 F;硬 clip 是否复活 |

(v1 的 D4 轮次深度诊断随 agentic 场景移除。)

---

## 4. Phase 1–3 — 仲裁流程

**Phase 1 单轴筛(选择只在 math 域)**:每臂 vs vanilla,单 seed + 配对检验;
p<0.05 晋级,|Δ|<噪声底判平局。约 15–18 runs。

**筛选步长 300 → 150 + 早停(2026-08-04 预注册修订,依据 vanilla_s0 实测)**

vanilla_s0(job 719188)在我们自己的栈上量到:

| step | val pass@1 | resp_len | clip_ratio | s/it | entropy |
|---|---|---|---|---|---|
| 25 | **0.468** | 1.6k | 0.05 | 82 | 1.40 |
| 40 | — | — | — | — | **0.15**(熵塌) |
| 100 | 0.392 | 7.8k | 0.95 | 457 | 0.13(**Mode A**) |
| 200 | 0.456 | 8.2k | **1.00** | 482 | 0.10 |

**val 从第 25 步起再没动过;第 100–300 步是 26 小时全部撞长度帽的 rollout。**
300 步 = 30.3h/臂,其中 85% 买不到 pass@1 上的任何东西。

- **硬上限 H = 150 步**(所有臂一致)。装得下:平台(25)、熵塌(40)、Mode A 起点(90)
  以及 60 步成型的 Mode A —— 最后这段是 **F 轴必须的**,它被判的就是"能否阻止 Mode A"。
- **早停规则(同一条,所有臂适用)**:step ≥ 50 **且** 连续 10 步 clip_ratio ≥ 0.90
  **且** 最近 3 次 val 无超噪声底增益 → 停。两条**同时**满足才停:只退化可能还在涨,
  只平台可能是后续增益前的台阶。
  - clip 阈取 **0.90 不取 0.95**:Mode A 成型后该比值在 0.90–0.97 间抖动,0.95 是被
    噪声打断而非被恢复打断,白等 39 步(5 小时)。
- **纪律:停在哪一步必须记账**(`logs/early_stops.tsv`)。不记的早停会让两个臂悄悄
  不可比 —— 那正是本文要诊断的病。记了之后,**"撑到第几步才进 Mode A"本身就是逐臂
  可报的量**,跨臂比较取**最小公共步**。

实测效果(同一条 vanilla 日志回放):300 步 30.3h → 150 步 10.9h → **+早停 4.6h,6.6×**。

**迁移列改为逐臂(2026-08-03 预注册修订)**:每臂 final checkpoint 加评
HumanEval+/MBPP+/IFEval(math 训、跨域评,greedy),记入副作用面板。
**只评测不训练,不参与晋级** —— 选择仍然纯在 math。
理由:三分类判决里的"**有副作用**"此前只有 Phase 3 的配方级证据,对单个臂无据可依,
只能从 math 域内的长度/熵/多样性去推。成本 ≈0.25 GPU·时/臂(训练是 18 GPU·时/臂),
约 1.5% 开销。

**门槛(先量后用)**:逐域噪声底由 W1 的 vanilla×3 seeds checkpoint 顺带评出
(零额外训练 run)。**某域极差过宽 / 贴地板则预注册剔除该域**,并把实测数字写进 paper
当理由 —— 0.6B-Base 在 IFEval 上尤其可疑(base 模型本就不擅遵循指令,再 math-only
训 300 步)。宁可少一列,不要一列噪声。

**Phase 2 双向贪心(只在 math 域)**:
前向:vanilla 起,逐个加当前最优 trick,增益<噪声底即停;
后向:全家桶起,逐个删,删而不掉点者移除;
两方向收敛同一集合 = 交叉验证;"全家桶 ≤ 小配方"若现即标题级结论。
约 15–20 runs。

**预注册冗余预测**:
1. token 选择 × 支撑分配:同聚焦高不确定位 → 预测冗余;
2. PL-rank × 信号压缩:rank loss 天然有界 → 预测压缩冗余;
3. verified 门控 × 冷启动:同治分布错配 → 预测二选一;
4. TIP ≈ Teachability ≈ SelecTKD 的选中集高度重叠(code 域用 AST 仲裁重叠结构)。

**Phase 3 终审**:胜出配方 + vanilla + 全家桶 × 3 seeds × **三域** × teacher 档;
code/IFEval **只验证不选择**(防配方过拟合 math)。约 25–30 runs。

总预算:**60–80 runs**(1.7B student、16k 上下文;单 run 成本以集群实测为准)。

---

## 5. 判决规则、副作用面板、成功标准(预注册)

| 判决 | 操作定义 |
|---|---|
| 必需 | 配对检验 p<0.05 增益,≥2 设定(域×teacher 档)复现,后向删之必掉点 |
| 无用 | 所有设定 |Δ|<噪声底 → 平局判简,剔除 |
| 副作用 | 主指标涨但面板任一显著恶化,或换设定显著负增益 → 台账+适用域 |

**副作用面板(每 run 必测)**:
- pass@k 多样性(k=8 on math)—— OPD 有在案的多样性坍缩;蒸馏产物常要继续 RL/采样,
  EM 涨 pass@k 塌的配方是隐性毒药;综述已列 open problem;
- 长度漂移(Mode A/B 在线监测);
- 熵曲线;
- 域间迁移损益(math 上选的配方在 code/IFEval 的 Δ)。

**飞行记录仪(每 run)**:overlap 轨迹按位置类别、支撑大小分布、熵、长度。

**SimOPD 成功标准**:≤3 组件;显著胜 vanilla;打平或胜全家桶;每个落选 trick 有判决书。
达不到 ≤3 照实报 —— "general OPD 天生需要 N 件"同样是结论。

---

## 6. 交付物

1. 判决台账(trick × 域 × teacher 档 × 判决 × 副作用);
2. SimOPD 配方 + 机理解释(接 Phase 0 病理 + 定理);
3. 参考实现以 verl / EasyOPD config+PR 开源;
4. 飞行记录仪随代码放出;
5. 图表骨架:Fig1 病理诊断(π(S̄)/overlap 按位置类别×域)→ Tab1 主效应 →
   Fig2 双向贪心曲线 → Fig3 效果×简单性 Pareto → Tab2 判决台账(跨域列)。

命名:SimOPD 未被占(2026-07 查;邻居 SimpleOPD 是 SimCT 内部基线标签、EasyOPD 是框架)。

---

## 7. 风险

1. **正面赛跑是第一风险**:general 场景与 THUNLP 级团队同场;护栏 = 互补轴 + 跨域 +
   台账方法论 + 速度。窗口以周计(4月 Rethinking → 7月 Demystifying/EasyOPD);
2. compute:16k CoT rollout 贵;靠 1.7B student + math-only 筛选压预算;
3. token 判据仲裁失去真 oracle(agentic 的代价):code AST 半补偿,写进 limitations;
4. 各 trick 在 math 域打平的可能:机理指标让"打平"可解释;跨域列保底。

## 7.5 硬件预算(v3.1:8–16 × A100-80G,0.6B 主力,3–4 周冲刺)

**模型(v3 定稿)**:主力 0.6B-Base ← {1.7B 现货(2.8×,甜/中档)、4B 现货
(6.7×,失配档)};**零 GRPO 自训**(GRPO 每步成组采样 4–8M token,是 OPD 的
~8×,砍掉全现货);终验档 1.7B-Base ← 4B = Demystifying 阶梯现成格 = 复现锚点。

**槽位**:0.6B run = 2–3 卡(student+teacher 权重 <5GB,可同卡 vLLM 共驻)→
16 卡 6–8 路并行;终验档 4 卡/run。

**v3.1 三刀**:(1) Phase 1 并入贪心 R1(数学上同一批 run);(2) 筛选一律
300 步早筛(4–8h/run),终审才满步;(3) 后向拆解降级为配方消融
(全家桶 1 run + 逐件删 ≤3 runs);终审 2 seeds 起步,最终配方行 3 seeds。

**纪律**:(1) teacher bf16 禁量化 —— logprob 是被审对象;(2) 贪心轮间串行,
墙钟=轮数×每轮批次;(3) 长度帽 16k→8k(报截断率);(4) step 时间以 W1
复现 run 实测校准;(5) W1 复现锚点是进度闸门:对不上 Demystifying 的数,
停下修 infra,不带病进 W2。

## 8. 里程碑(v3.1 冲刺:实验 3–4 周 + 1 周 buffer;目标 ICLR 2027)

| 周 | 事 |
|---|---|
| **W1** | infra 跑通(verl/EasyOPD 选型)+ **复现锚点**(1.7B←4B vanilla,对 Demystifying 现成格;进度闸门)+ Phase 0 诊断 + 各域噪声底 + 三域验证器 |
| **W2** | 贪心 R1(= 单轴筛,~12 臂 × 300 步早筛,6–8 路并行)+ R2 |
| **W3** | 贪心 R3–R4;终审满步 run 启动(配方 / vanilla / 全家桶);写作开始 |
| **W4** | Phase 3:三域验证 + 1.7B 终验档 + 配方消融;结果冻结 |
| W5 | buffer:返工 + 写作收尾 |

## 附:关键文献索引

方法论模板:LitePPO 2508.08221
竞品:Rethinking 2604.13016(thunlp/OPD)· Demystifying 2607.13399
支撑:LSM 2603.25562 · RSKD 2503.16870 · top-p-k 2410.16215 · tail-aware 2602.20816 ·
  tool-call drift 2607.07050(支撑缺失现象学,引作动机)
token 判据:TIP 2604.14084 · Teachability 2605.26844 · SelecTKD 2510.24021 ·
  Evidence 2606.22830 · Rock 2605.09253 · SEAD 2606.28562 · FiRe 2606.02684
排序目标:PLD 2506.12542
散度:GKD 2306.13649 · EOPD 2603.07079 · DistiLLM 2402.03898
跨词表(备用):SimCT 2605.07711 · Breaking Tokenizer Barrier 2606.09456
框架:verl OPD docs · EasyOPD 2607.11012 · TRL GKD
综述:2606.22793(adaptive support 列为 open)· 2604.00626 · awesome 清单(chrisliu298)
