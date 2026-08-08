# SimOPD 案卷:变体动物园普查与参赛名单 v1

> 来源:awesome-on-policy-distillation(chrisliu298),**462 条目**(badge 实数)。
> 本文档 = 普查(§1)→ 管辖权裁定(§2)→ 8 轴参赛名单(§3)→ 代表选择准则(§4)。
> ⚠ arXiv 编号取自清单摘要,实现前每篇代表作需全文核验(第 1–2 周功课)。

---

## 1. 普查:动物园结构

| 区 | 规模 | 处置 |
|---|---|---|
| Start Here / 综述随笔 | 8 + ~35 | 背景,不受审 |
| Foundations / Gap-Bridging | 2 + ~25 | 部分入 A 轴 |
| **Stability & Objective Design** | **~65** | **主审区,8 轴大部分来源** |
| Self-Distillation (OPSD 族) | ~80 | **管辖权外**(范围决策:不做 OPSD)|
| Context/Experience Internalization | ~13 | 管辖权外(经验内化线)|
| Efficiency/Systems/Privacy | ~23 | 工程默认,不审(例外:见 H 轴)|
| Agents & Tool-Use | ~40 | 管辖权外(范围决策:general)|
| Multimodal/VLM | ~8 | 管辖权外 |
| Cross-Tokenizer | ~13 | 管辖权外(协议用同族师生)|
| Frameworks | — | 基础设施选型用(verl/EasyOPD)|
| 偏好/奖励混合 RL+KD、自博弈、Taxonomy 区 | ~130 | 邻线,引用不受审 |

**核心受审池 ≈ 55 篇**(Objective/Stability/Token/Support/Curriculum/Gating/Reward 七小区)。

---

## 2. 管辖权外裁定(写进 paper 的 scope 声明)

1. OPSD 族(~80):自蒸馏动力学与外部 teacher 不同,独立成域;
2. 跨词表(~13):协议固定 Qwen3 同族;引 SimCT/2606.09456 划界;
3. 多 teacher 路由(DOPD、MOPD、tool-call drift):teacher 数=1 的审计;
4. 隐状态对齐(OPRD 2606.06021):监督信号类型不同(非 logit-KL),邻线;
5. 弱到强合成 teacher(2607.26246):无强 teacher 的不同问题;
6. 纯效率件(Lightning OPD 预计算、f-OPD 异步、EffOPD 外推):不改精度主张,
   工程默认 —— **例外**:序列视界族有精度主张,入 H 轴。

---

## 3. 十轴(A–J;I 搁置)参赛名单(Phase 1 = 18 runs:17 臂 + vanilla 基线)

图例:【审】= 上庭代表;【替】= 替补(代表结果含糊时加赛);【落】= 落选+理由。

### A — rollout 来源与日程(族 ~25,Gap-Bridging)
- 【审】纯 on-policy λ=1(基线约定)
- 【审】GKD λ=0.5 混合(2306.13649)
- 【审】GKD λ=0 端点(a3_offpolicy,2026-08-07 补齐——计划 §2 原注册 λ∈{0,0.5};**后补批**,随 a1 缓存解锁)
- 【审】离策略冷启动→OPD(Rethinking 2604.13016 recipe 复核)
- 【落】PACED 通过率课程(2603.11178)→ 替补;Escaping KL Agreement Trap
  (2606.09471)rollout 终止 → 替补

### B — 散度/目标函数(族 ~14)
- 【审】Reverse KL(基线)
- 【审】Skew-KL α=0.1(DistiLLM 2402.03898 —— 被采纳最广的替代散度)
- 【审】Forward KL(完整性臂:审计应含"公认差"的对照)
- 【审·后补】JSD-β=0.5(GKD)—— **2026-08-07 由替转审**(b4_jsd,后补批;
  计划 §2 的 B 轴承诺兑现;β 消融预注册)
- 【审·后补】k2 估计器阶梯【自研】(b5_k2,后补批)—— 同目标异估计器,
  r3/r5 估计器教训的实证腿;k3(GRPO 默认)留替补
- 【替】CADENCE 前反 KL 逐 token 日程(2607.16955);EOPD 熵切换已转 b3;
  DistiLLM-2 对比式(2503.07067);Skew-FKL top-k 镜像【自研候选】;k3 阶梯位
- 【落】ExOPD(2602.12125,RL 化重构,越界到 RL 线);OPD+/vOPD 估计器修正
  (2606.01039/2605.07865)→ 并入工程默认核对项,不单独成臂;
  Veto/TRPD 近端 teacher(2601.07155/2607.04751)→ 替补(与 F 轴软压缩同治一病)

### C — 词表支撑(族 ~8)★含自研
- 【审】sampled-token(基线;Rethinking 已判"够用"—— 本轴任务是画适用域边界)
- 【审】teacher top-k + 重归一化(LSM/Revisiting 2603.25562)
- 【审】分位预算分配 margin=max(q,π)(**自研**;重归一化 vs 尾桶作内部消融)
- 【审·后补】intersection 支撑(thunlp/OPD 官方策略;c3,2026-08-07)——
  Rethinking overlap 甜区主张的机制探针;其化简直接形逐行码锚
- 【审·后补】π-tail 预算【自研,头条构造臂】(c4,2026-08-07)—— 定理误差项当旋钮,
  ε=0.05;按学生的边切
- 【替】top-p;margin=π / margin=q 拆解;only_stu/union(接缝外:需 thunlp 双前向
  架构,老师需给学生指定 ids 打分——vLLM 打分服务不可为;声明出局)
- 【落】RSKD 无偏采样(2503.16870,离线设定)→ 尾桶修正可选件;全词表(载荷不可行,
  π(S̄) 面板代偿;PowerOPD 曾以其为基线)

### D — token 选择/加权(族 ~15)
- 【审】uniform(基线)
- 【审】TIP 高熵+自信错(2604.14084)—— 熵判据代表;分解旋钮 SIMOPD_TIP_MODE
  (08-07 预注册:entropy_only@ρ=0.5 / divergence_only@ρ=0.1 各对其原文主张)
- 【审】SelecTKD propose-verify(2510.24021)—— 秩验证判据代表
- 【审】Teachability(2605.26844)—— 支撑落点判据代表
- 【替】SEAD(2606.28562)、FiRe-OPD(2606.02684)、Evidence(2606.22830)、
  Rock Tokens(2605.09253)、Position-Bias IW(2606.22600)、
  Blockwise gating(2606.24084)
- 【落】SafeSteer(安全域)、TRACE(需标注)、Prefix-fade(并入 H 轴族)
- 【挂账 08-07】五【替】判据快读 → 可算者入影子面板(只记不训,V-wave Jaccard
  出冗余判决;正交=晋升候选)。任务单 #7,provenance r6 增补有全案;等用户开闸

### E — 支撑内对齐目标 ★自研,防火墙臂(08-07 阶梯补全)
- 【审】value 对齐 KL(基线)
- 【审】e1 PL-rank + 小 value 锚(**自研**;PL 机械源 PLD 2506.12542,r5 口径)
- 【审】e2 set-coverage(**08-07 由替转审**,自研):−log(1−π_tail) 作损失 + e1
  同系数锚(锚=轴公共加项);阶梯关底档,c4 的定理量面板在此当损失
- 【审】e3 z-value(**08-07 增设**,自研;z 形式源 Logit-Std KD 2403.01427 视觉域,
  形式源非审计对象):支撑内 z-分后 KL,仿射不变档,无锚(纯档,不对称已登记)
- 信息阶梯:全值(c1)> z-值(e3)> 全序(e1)> 集合(e2),同支撑 k=32;免费面板
  rank_kendall_tau + student_mass 让"序先收敛还是质量先收敛"在每个 top-k 臂上可测
- 【落】成对边际族(RankNet/BiLD 类:夹档,与两邻分离度低)、top-1 argmax CE
  (锐化端点,熵塌/长度混淆,无主张持有者)、PLD 置信加权(r5 已登记有意不引入)
- 防火墙声明:自研臂接受与所有被告完全相同的预注册协议

### I — 老师条件化(2026-08-07 全轴搁置;设计与预生成机械保留,复位=status 翻回)
- 【审→搁】i0 素打分(挪用 scratchpad 控制)/ i1 私有 CoT 注入(手术 V1–V4 待做)

### J — 奖励耦合(2026-08-07 增设,survey 对表产物)
- 【审】KDRL 统一目标(2506.02208)—— 奖励作共目标代表;n=8 迷你 cell(自带 vanilla_n8 基线)
- 与 G 的分界:G = 奖励**过滤**蒸馏信号;J = 奖励作**独立共目标**。g1↔j1 跨轴构成"滤 vs 加"对照
- 【落】MiMo/rubric 系(工业报告,无孤立可审主张)、HDPO(自蒸馏域,裁定 #1)

### F — 信号调节(族 ~8)
- 【审】无(基线)
- 【审】软 log 压缩(Demystifying 2607.13399 赢家复核)
- 【审】PowerOPD 有界幂奖励(2606.17199)—— **2026-08-07 由替转审**(f3_power,α=1 正典成员;论文按指标挑最优 α 记为 caveat);硬 clip(仅当 D5 出 Mode B → 已因 stock 顺带入列);
  Stable-OPD 散度约束(**注意:2604.08527 是另一篇同名 "Demystifying"**,治长度通胀)
- 【落】CaOPD 过自信修正(2604.16830)→ 替补池

### G — 轨迹/奖励门控(族 ~9)
- 【审】无门控(基线)
- 【审】verified-only(规则验证器过滤;发表代表 = Reward-Gated OPD 2607.04037)
- 【审】teacher 似然过滤(FiRe-OPD 2606.02684 的轨迹级半件)
- 【替】SG-OPD 符号路由(2606.09304);ReNIO 轨迹重权(2606.23104)
- 【落】ATESD/EGRSD/TRAD(自蒸馏设定)、Beyond GRPO and OPD(流程编排非 trick)

### H — 序列视界(族 ~7,新增轴;从效率区捞回,因其有精度主张)
- 【审】全 rollout 监督(基线)
- 【审】首段 token 监督(Less is More 2605.27028 —— "teacher 信号集中在响应前段"
  是可证伪的科学主张;若成立,预算全盘 ×2–47 杠杆)
- 【替】ADWIN 自适应窗(2605.28396);Prefix-Guided 预算分配(2606.21994);
  Prune-OPD 漂移截断(2605.07804);Are Full Rollouts Necessary(2605.31490)

**Phase 1 合计**:A2 + B2 + C2 + D3 + E1 + F1 + G2 + H1 = 14 个非基线臂
(+vanilla,+3 个机动位给替补加赛)≈ 18 runs,与预算吻合。

---

## 4. 代表选择准则(写进 paper,防"为什么是这几个"之问)

1. **机制独异性**:每个机制族一名代表(不重复审五个熵变体);
2. **可复现性**:公式级描述,优先有码;
3. **协议适配**:同族师生、logit 可达、general 推理域内有效性主张;
4. **采纳度/新近度平衡**:被采纳最广的老件(DistiLLM)+ 各族最新明确表述件。
替补规则:代表显著胜/败 → 同族替补不加赛;代表与基线打平但族内主张分歧 → 加赛一名。

---

## 5. 与审计三纪律的对接

- fresh 仲裁轴:C(边界)、D、E、G、H;replication+boundary 轴:A(冷启动)、
  B(部分)、F(软压缩);
- 自研臂防火墙:C 分位、E rank —— 同协议同预算;
- 跨域列:Phase 3 三域(math/code/IFEval)验证 shortlist。

---

## 附录:冻结后清单增量与勘误(2026-08-06 复扫 chrisliu298 列表)

**估计器核对工单已执行** → `docs/estimator-note.md`。结论:保留领域正典底座
(sg-优势 + PPO 代理),k1 族梯度期望无偏,非 k1 臂为领域标准启发式代理
(OPD+ 形式化其偏差);臂间同底同偏可比,limitations 措辞已备。

**冻结(462 条)后的尾巴,分诊:**
- 【必读】2607.23731 Outcome-Confounded Local Supervision —— outcome×局部信号混淆
  批判;g1 判决措辞前必读,可能加 caveat
- 【替→G】SPOT(2608.04419,verifier 稀疏探针)
- 【替→D】Not All Tokens Deserve Equal Credit(2607.27888,反事实敏感度重分配)
- 【落/discussion】2608.04408 可恢复错误回滚(KAT 同族,改采样分布);
  Lightning 2.0 / SAF-OPD / β-OPSD(系统/混合 RL/自蒸馏门类)
- 【落/门类】weak-to-strong 对(2607.26246/27770):teacher 构造门类,协议外

**勘误(写对表段前必须吸收)**:此前对 2606.22793 survey 的"信用分配格全是假设"
论断**不成立** —— KETCHUP(2504.19024,K 步 Bellman 序列级回报;原清单 adjacent 段,
zoo 漏收,今补)与 Bridging(2606.00305,近未来窗口摊布)是该格的实证成员。
对表段应写:"实证工作**集中**于即时信用(含本文全部臂),序列级信用有零星探索
(KETCHUP、Bridging),GAE-OPD/CR-OPD 仍为假设。"
