# SimOPD Metrics 规范 v2(预注册;2026-07-31;v2 = 基于文献 metrics 实测统一)

> 方法:先实测受审/竞品论文各自报什么(§0),再锁定我们的四层体系(§1-§4),
> **每个指标标注出处** —— 借鉴的写论文名,新增的写理由。判决原则不变:
> **判决算在逐题工件上,wandb 曲线只看趋势。**

---

## 0. 文献 metrics 现状(实测,2026-07-31)

### 0.1 判决类(accuracy 家族)

| 论文 | 主指标 | 采样协议 | seeds/误差棒 |
|---|---|---|---|
| Demystifying | MATH500/Minerva pass@1;AMC23/AIME24-26/HMMT25 avg@32 | avg@32 的 Eq.9 提到 M=10 trials(精读核实) | **训练 run 无 seeds/误差棒** |
| Rethinking | AIME24/25+AMC23 avg@16 | τ0.7/p0.95 | 无 |
| LSM | 5 bench pass@1 | greedy | 无 |
| TIP | MATH500/AIME mean@16 | — | 无 |
| Teachability | 6 bench(含 GPQA/HumanEval/IFEval) | — | **5 eval seeds ±std**(仅评测侧) |
| RG-OPD | GSM8K/MATH/MBPP/IFEval acc | — | 3 seeds ±std |
| FiRe | 数学 6 bench Avg@8 + 代码 3 bench pass@1 | — | 无 |
| ESR | MATH500 avg@4/pass@4/maj@4 | τ0.7 | 无 |

**读法**:k∈{1,4,8,16,32}、温度 0/0.6/0.7/1.0 全都有;训练 run 几乎无人报 seeds。
→ 我们的噪声底 + 逐题配对检验是**超出文献惯例的增项**(审计 genre 的方法论卖点,保留)。

### 0.2 机理/诊断类

| 论文 | 定义了什么 | 我们的继承 |
|---|---|---|
| Rethinking | **overlap ratio(Eq.6)= E_t[\|S_t^(p)∩S_t^(q)\|/k]**(top-k 交集比例);**overlap-token advantage(Eq.7)**;**熵差 \|H(q)−H(p)\|(Eq.8)**;交集 token 携带 97-99% 质量(App B.1);逐位置熵(Fig13);**Gap Recovery Rate** =(Acc_后−Acc_前)/(Acc_师−Acc_前);序列均值 reward 的 correct/incorrect AUROC(0.73-0.75) | 全套进飞行记录仪(Eq.6/7 verl 已实现);GRR 进判决层归一化;AUROC 进 G 轴面板 |
| Demystifying | 长度曲线 + 截断率(Fig6/9);逐 token advantage 均值曲线;**Mode A "Endless Exploration" / Mode B "Abrupt Degeneration"**(正式命名);**Informativeness ℐ(Eq.4)**:teacher 对 correct rollout 的 token 给更高概率;**pass@k 到 k=1024(Fig3)**;主曲线轴 = acc/长度/advantage vs steps | 长度/advantage 面板同轴对齐(锚点对表用);ℐ 进 G 轴与信号质量面板;pass@k 面板与其对齐 |
| TIP / Teachability / SelecTKD / FiRe | 熵与保留率分布 / D̃·C̃ 分解 / **TAR 接受率** / teacher 似然 s(y) 与 bottom-20% 阈线 | 各自作为其臂的臂内指标 + 影子掩码统一记录 |

---

## 1. 判决层

| 指标 | 定义与参数 | 出处 | 节奏 |
|---|---|---|---|
| **MATH500 pass@1** | greedy(τ=0),逐题 0/1 | Demystifying/LSM 同款 | val 每 25 步(筛)/每 5 步(锚点前期) |
| **AMC23 avg@32** | τ=0.7/top-p=0.95,40 题×32 样本逐格 0/1 | k=32 从 Demystifying;采样参数暂从 Rethinking(Demystifying Eq.9 精读后校正,含 M=10 trials 语义) | checkpoint 终评(离线) |
| AIME24/25 avg@32(全档) | 同上 | Demystifying(其含 AIME26/HMMT25,我们不扩,预注册可选) | 终审 |
| Minerva pass@1(仅锚点) | greedy | **锚点专用**:Demystifying 报表含 Minerva,对表面加宽 | 锚点 run |
| HumanEval+/MBPP+ pass@1、IFEval strict | 官方 harness(evalplus / instruction_following_eval) | FiRe(+ 变体)/Teachability | **每臂 final ckpt**(迁移列,§2);Phase 3 另有跨域**重训** |
| **Gap Recovery Rate** | (Acc_配方−Acc_前)/(Acc_师−Acc_前)。**两个分母都已实测**:MATH500(greedy)4B-2507 **0.896**/8B 0.792/1.7B 0.702;**AMC23(avg@32)4B-2507 0.9133**/8B 0.683/1.7B 0.434。学生 1.7B-Base 起点 MATH500 **0.468** → 主档可用差距 **0.428** | **Rethinking**;跨 teacher 档可比性 | 台账列(verdict.py 算) |
| **噪声底** | vanilla×3 seeds 的 MATH500 pass@1 极差 | **本文新增**(文献无人做;判"平局"必需) | W1 一次 |

统计装置与逐题工件 schema 不变(v1;`scripts/verdict.py`、parquet 落盘
`/scratch/zz865/simopd/evals/`)。预算配平核查不变(supervised_token_count 同轴相等)。

## 2. 副作用面板

| 指标 | 定义与参数 | 出处 | 节奏 |
|---|---|---|---|
| **pass@k 面板** | 筛选:MATH500 固定 100 题子集 pass@8,τ=1.0/p=1.0(策略自身分布);终审:k 扫到 64(可选 1.7B 档扫更高) | **Demystifying Fig3**(其扫到 1024;多样性坍缩证据同源) | checkpoint 终评 |
| pass@8−pass@1 gap | 同上导出 | 同上 | 同上 |
| **长度漂移 + 截断率** | mean/median/p95 + 截断率 + 滑动斜率;报警语义用官方命名:**Mode A = Endless Exploration,Mode B = Abrupt Degeneration** | **Demystifying Fig6/9**(锚点对表同轴) | 每步 |
| **逐 token advantage 均值曲线** | batch 内 Δℓ 的 token-mean | **Demystifying**(其 Mode 诊断的第三轴;锚点对表需要) | 每步 |
| 熵曲线 | actor 熵(verl `actor/entropy`) | Rethinking Eq.8 族 | 每步 |
| 重复率 | val 生成 3-gram 重复率 | 本文新增(退化循环 pass@1 不可见;成本≈0) | 每 25 步 |
| **域间迁移损益** | **每臂**在 HumanEval+/MBPP+/IFEval 的 Δ vs vanilla(math 训、跨域评,greedy) | 迁移评测本身有先例(**FiRe**:math 训→code 评;**Teachability**:→code/IFEval/GPQA);**逐臂 + 统一协议**是本文新增 | **每臂 final ckpt**(2026-08-03 从 Phase 3 提前) |

## 3. 机理层 / 飞行记录仪

前置补丁不变(估计器路径 top-64 常开,~40 行;带宽 ~0.6GB/batch)。

| 指标 | 定义 | 出处 | 来源 |
|---|---|---|---|
| **overlap ratio** | Rethinking **Eq.6 原式**:top-k 交集比例(k=64 主报,k=16 副报) | Rethinking | verl 现成(移到估计器路径) |
| **overlap 质量版** | 交集 token 携带的双侧概率质量(97-99% 甜区对话用) | Rethinking App B.1 | 要写(~15 行) |
| **overlap-token advantage** | Rethinking Eq.7 原式 | Rethinking | verl 现成 |
| **π_stu(S̄) 尾质量** | student 在 teacher top-k 外的质量,**在 K∈{8,16,32} 三个宽度同时报**(teacher 支撑是秩序的,窄支撑是它的前缀 → 一次前向拿到整条 K 扫描) | **本文 headline**(定理直接量;文献只看交集,没人看尾) | ✅ `pi_tail_k*`,所有 top-k 臂 |
| **Δℓ 分布分位数** | p5/25/50/75/95 + mean\|Δℓ\| | Demystifying(其只报均值;分位数是我们加密度) | ✅ `_signal_quantiles` |
| **命名纪律(重要)** | **只有 k1 族的 loss 等于 −Δℓ**,才可用 `delta_ell_*`;C/E 轴优化的是散度或排序损失,一律报 `loss_*` | 本文新增 | ✅ 两个面板 key 不相交,防止三种不同量被画到同一张跨臂对比图上 |
| **clip 命中率** | \|signal\| > clamp 的 token 占比(F 轴开启时) | METRICS 预注册项 | ✅ `_clip_metrics`;否则 f2 的机制不可见 —— 它的 Δℓ 面板是**裁剪前**分布,和 vanilla 一模一样 |
| **熵差** | \|H(q)−H(p)\|(Rethinking Eq.8 原式;teacher 熵用 top-64 截断近似,近似误差随 recorder 报) | Rethinking | 要写 |
| **逐位置类别分解** | 所有上述 × {boxed span, 高熵 fork, 其他}(math);AST 类(code,Phase 3);约束 token(IFEval) | Rethinking Fig13 只按绝对位置;**按语义类别是我们的加强** | 要写(~80 行) |
| **Informativeness ℐ** | Demystifying **Eq.4 原式**:teacher 概率在 correct vs incorrect rollout 上的差 | Demystifying | 要写(~15 行;reward 标签现成) |
| **teacher 似然 AUROC** | 序列均值 teacher logprob 判别 correct/incorrect 的 AUROC | Rethinking(0.73-0.75 基准值可直接对表)+ FiRe s(y) 同族 | 要写(~10 行) |
| **D 轴影子掩码** | TIP/Teachability/SelecTKD 三掩码影子计算:选中率 + 两两 Jaccard + **TAR**。Jaccard 不直接发,而发交/并两个逐 token 指示量 —— 指标管道报的是掩码内均值,而 **mean(A∧B)/mean(A∨B) 恰等于 \|A∩B\|/\|A∪B\|**(token 数约掉) | 冗余预测 #4 检验(本文新增);TAR 从 SelecTKD;**兼作组合筛选** —— 影子几乎一致的两个设置是同一个臂的两个名字,不值得各花一个 run | ✅ `shadow_*`,`SIMOPD_SHADOW=0` 可关 |
| G 轴通过率 | verifier 通过率 / 似然过滤阈线位置 | FiRe(bottom-20% 阈线同款) | verl reward + ~10 行 |
| **Gap Recovery 曲线** | GRR vs steps(终审档) | Rethinking | verdict.py |

## 4. 效率层(不变,v1)

timing 分解(verl 现成)、相对 vanilla step 开销比、GPU·h、rollout/prefill token 量、组件数。

## 5. 节奏总表(v1 基础上改两行)

| 时点 | 测什么 |
|---|---|
| 每步 | Δℓ 分布+均值曲线、熵、长度+截断率、overlap(Eq.6/质量版)、π(S̄)、ℐ、timing |
| 每 25 步(筛选 val) | MATH500 pass@1 + 重复率;逐题工件落盘 |
| checkpoint 终评 | AMC23 avg@32、pass@8 面板(终审 k→64)、AIME avg@32、(锚点)Minerva;**+ 迁移列 HumanEval+/MBPP+/IFEval(greedy,1083 题 ≈0.25 GPU·时)** |
| W1 一次性 | 噪声底×3 seeds(逐域);**teacher 上限** —— MATH500 ✅2026-08-04(0.896/0.792/0.702),AMC23 仍欠 |
| Phase 0 | D6 逐题矩阵、D1'/D2'/D3 静态版、D5 |
| Phase 3 | code/IFEval **重训**(shortlist 配方,非迁移评测)+ 三域主表 |

## 6. 实施清单(v1 六项 + 三个新件)

1-6 同 v1(eval_offline.py / recorder 补丁 / 影子掩码 / 位置类别 / verdict.py / wandb 约定);
7. ℐ 与 AUROC 打点(reward 标签 × teacher logprob,~25 行,recorder 内);
8. overlap 质量版(~15 行);
9. verdict.py 增加 GRR 列(需 W1 的 teacher 上限数据)。

## 7. 明确不测(v1 不变)

困惑度、与 base 的 KL、MMLU 类通用回归、GPQA、逐层表征。
(HMMT25/AIME26:Demystifying 报了,我们预注册为终审可选扩展,不进筛选。)
