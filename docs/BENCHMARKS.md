# SimOPD Benchmark 选型 v1(预注册;2026-07-31)

> 方法与 METRICS.md 相同:文献频次实测(§1)→ 按档位锁定(§2)→ 数据源/评分器
> 钉死(§3)→ 明确不选(§5)。采样参数与节奏归 METRICS.md 管,此处只定"测哪些"。

## 1. 文献频次(10 篇受审/竞品,2026-07-31 实测)

| bench | 用它的论文数 | 谁在用 |
|---|---|---|
| **MATH500** | 7 | Demystifying/LSM/TIP/Teachability/FiRe/ESR/EasyOPD |
| **AIME24/25** | 7 | Demystifying(含26)/Rethinking/LSM/TIP/Teachability/FiRe/EasyOPD |
| **AMC23** | 3 | Demystifying/Rethinking/FiRe |
| Minerva | 3 | Demystifying/LSM/FiRe |
| HumanEval(+) | 3 | Teachability/FiRe(+)/ESR |
| MBPP(+) | 3 | RG-OPD/FiRe(+)/EasyOPD |
| IFEval | ~3 | Teachability/RG-OPD(/SelecTKD 泛指 IF) |
| OlympiadBench | 2 | LSM/FiRe |
| HMMT | 2 | Demystifying(25)/FiRe |
| LiveCodeBench | 2 | FiRe/EasyOPD |
| GSM8K | 2 | RG-OPD/EasyOPD(其余论文已弃用) |
| GPQA-D / BFCL / MMLU-STEM 族 | 各 1 | Teachability / ESR / RG-OPD |

结论:**MATH500 + AIME 是行业共识,AMC23 是 avg@k 论文的惯用中难度补充**;
code 共识 = HumanEval/MBPP(+ 变体);general 共识 = IFEval。与计划 §1 预设一致,锁定无冲突。

## 2. 分档锁定

| 档位 | bench 套件 | 用途 |
|---|---|---|
| **筛选(1.7B-Base ← 4B-2507,每 run)** | **MATH500 pass@1**(训练内 val,判决主指标)+ **AMC23 avg@32**(checkpoint 终评,判决副指标)+ **AIME24/25 avg@32** | 500 题给 McNemar 检验力。**AIME 现在测得了** —— 2026-08-04 换档后地板效应消失,这是换档的收益之一 |
| **锚点(= 同一格,不再是独立档)** | 上述 + **Minerva pass@1** | 对齐 Demystifying 该格子的报表面(其 AIME26/HMMT25 列为可选,见 §5) |
| **终验 1.7B 档** | MATH500 + AMC23 + AIME24/25 | 主表 |
| **迁移列(每臂,final ckpt)** | **HumanEval+ / MBPP+**(pass@1,greedy)+ **IFEval**(strict,prompt-level 主报、instruction-level 附报) | **只验证不选择**;副作用面板的迁移损益。2026-08-03 从 Phase 3 提前到逐臂:1083 题 greedy ≈0.25 GPU·时,对 18 GPU·时的训练是 1.5% 开销,换来"有副作用"这条判决第一次有逐臂证据 |
| **Phase 3 跨域重训** | 同上 bench,但在 code/IF 域**重训** shortlist 配方(见 §4.5) | 迁移评测测不了"trick 在该域是否有效",只测"是否损坏";重训才测前者 |
| 多样性面板 | MATH500 固定 100 题子集(pass@k) | 子集索引冻结进仓库,全项目同一份 |

## 3. 数据源与 harness(HF ID 已全部验证存在)

| bench | 数据源 | 评分 |
|---|---|---|
| MATH500 | `HuggingFaceH4/MATH-500`(500) | 统一 math 评分器(见下) |
| AMC23 | `math-ai/amc23`(40) | 同上 |
| AIME24/25 | **AIME24 用 `HuggingFaceH4/aime_2024`**(直接 answer 列;`math-ai/aime24` 无 answer 列、只有 \boxed 在 solution 里,弃用)/ AIME25 用 `math-ai/aime25`(各 30) | 同上 |
| Minerva | `math-ai/minervamath`(272) | 同上 |
| HumanEval+ / MBPP+ | evalplus 官方 release(`~/.cache/evalplus`,非 HF),**evalplus 0.3.1 官方 harness** | 单测通过率(plus 集主报,base 附报) |
| IFEval | `google/IFEval`,**官方 instruction_following_eval checker**(仓库内 `third_party/`,无 PyPI 发行版;PyPI 上的 `ifeval` 是无关第三方上传,不可用) | strict acc |

**代码题 prompt** 用 evalplus 官方 chat 模板(instruction prefix + ```` ```python ```` 预填),
但经我们的 `enable_thinking=False` 走一遍 —— 官方那支不接受该参数,而空 think 前缀是
全项目纪律(§1)。预填不是可选:没有它,base 档 student 的失败大头是"没吐出可解析的
代码块",那这一列量的是 markdown 合规而不是代码能力。

⚠ **代码题会因机器负载产生假失败**(2026-08-03 实测):同一批 canonical solution,
load≈20 时 160/164、load≈6 时稳定 163/164。evalplus 的单测上限是
`max(min_time_limit, 4×参考耗时)`,平凡函数落到 1.0s 地板,CPU 争抢就能击穿 ——
而我们自己的四条训练泳道就是争抢源。故默认 `min_time_limit=4.0`
(`SIMOPD_EVALPLUS_MIN_TIME_LIMIT` 可调),ground truth 在 `fetch_assets.py` 里预热。
HumanEval/32(`find_zero`,牛顿法 tol 1e-5)在任何时限下都挂 —— **163/164 是 harness
自身的参考天花板**,不是我们的缺陷。

**统一 math 评分器(单点决定,全项目同一路径)**:boxed 抽取 + `math_verify` 等价判定
—— 训练内 val(verl reward)与 `eval_offline.py` 用同一实现,版本号记入逐题工件;
AMC/AIME 整数答案与 MATH 表达式等价都由 math_verify 兜底。**禁止**训练 val 与
离线 eval 用不同评分器(pass@1 会漂移)。

eval 生成长度:筛选 8192 / 终验 16384(与训练帽一致),截断率必报。
模板:与训练同款非思考 chat template。

## 4. 卫生检查(W2 前完成)

- [ ] 去污染:Nemotron-Cascade-RL-Math 对 {MATH500, AMC23, AIME24/25, Minerva}
  做 13-gram 重叠扫描(NVIDIA 声称已去污,自查一遍写进 paper 卫生表);
- [x] MATH500 100 题子集已冻结(2026-07-31):`data/math500_subset100.json`,seed=42,
  按 unique_id 记录(防重排),难度分布 L1-L5 = 8/16/19/22/35(与全集比例相称);
- [x] **teacher 上限(2026-08-04 完成)**:非思考 MATH500 pass@1,greedy,协议同款模板 ——
  **4B-Instruct-2507 = 0.896**(len 1548)/ **8B = 0.792**(1082)/ **1.7B = 0.702**(982)。
  这是 GRR 的分母,也是 D6 的输入。**阶梯不单调**:尺寸序 1.7B<4B<8B,能力序
  1.7B<8B<4B —— 在 `enable_thinking=False` 下 8B 被砍掉主武器。
  由此改了两件事:主档 teacher 换成 4B-Instruct-2507(PROTOCOL §1),
  D6 从"单调阶梯"改为"尺寸/能力解耦"(plan §3)。
  ⚠ 仍欠:AMC23 上再各测一次(GRR 的副指标分母)。
- [x] AIME/AMC 答案字段抽查(2026-07-31 完成):amc23/aime25 answer 列整数 ✓;
  **math-ai/aime24 无 answer 列 → 已换 HuggingFaceH4/aime_2024**(30 题,answer 列 ✓)。

## 4.5 Phase 3 跨域**训练**集(候选,W3 前锁定 —— 计划原文只定了跨域 eval,训练集是开口)

Phase 3 的"三域验证"是**在 code/IFEval 域重训 shortlist 配方**(配方/vanilla/全家桶),
不是只拿 math 模型跑跨域 bench;因此需要两个带训练时验证器的跨域训练集:

| 域 | 候选(HF ID 已验证存在) | 验证器 | 倾向 |
|---|---|---|---|
| code | `KodCode/KodCode-Light-RL-10K`(10K,自带单测,RL-ready)/ `agentica-org/DeepCoder-Preview-Dataset` / `likaixin/TACO-verified` | 单元测试执行 | **KodCode**(规模与 Nemotron 同量级,单测干净)|
| IF | `allenai/RLVR-IFeval`(Tülu3 系,约束可程序判定) | 约束 checker | **RLVR-IFeval**(与 IFEval bench 同构不同题,无泄漏)|

注意:code 域训练需要沙箱执行单测(verl 有 code reward 工具链;W3 落地时核验其
沙箱在集群上可用 —— 无 docker 权限时用进程级隔离)。

**eval 侧的沙箱已可自证**:`python scripts/transfer_eval.py --selfcheck` 拿两个数据集
自己的 canonical solution 跑一遍,对上参考数字(HumanEval+ 163/164)才算通过。
换机器(DSW / 新节点)先跑它 —— 不需要 GPU,不需要 checkpoint。训练侧沙箱是另一件事,
仍未验。

## 5. 明确不选(判决书式理由,防审稿人问)

| bench | 不选理由 |
|---|---|
| GSM8K | 已饱和,主流 OPD 文献(7/10)弃用;RG-OPD/EasyOPD 沿用是历史惯性 |
| OlympiadBench | 难度带与 AMC/AIME 重叠,eval 预算不换信息量 |
| GPQA-Diamond | 通用知识,三域管辖权外 |
| HMMT25 / AIME26 | **预注册可选扩展**:仅当终审 1.7B 冲高分需要加表时启用(Demystifying 报了,加做零改动) |
| LiveCodeBench | 时间窗管理重;HumanEval+/MBPP+ 已代表 code 域;审稿要求时再加 |
| BFCL / tool 类 | agentic 已裁出范围(v2 范围决策) |
| MMLU-STEM/BBH/SciQ(likelihood 类) | RG-OPD 特有;与生成式判决无关,METRICS §7 已列不测 |
| GSM-Plus / MPMath | 单一论文使用,无社区采纳 |
