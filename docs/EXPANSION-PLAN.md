# SimOPD 扩容总规划(100+ 卡)—— 2026-08-06 预注册

> 一份文档定全部:波次、候选项全集、每个选择的理由与**触发条件**。到卡即照单执行;
> 改动本文件走 dated amendment,与 plan 同纪律。
> 执行机械:`configs/campaign.tsv` + `deploy/up.sh`(新机器一行入列)。
> 相关:`SimOPD-plan.md`(主协议)、`design-thinking-cells.md`(thinking 三格细案)、
> `estimator-note.md`(底座核对)。

## 0. 现状基线(2026-08-06 晚)

- 主表 19 臂(17 可跑;a1 需 trainer 手术、i1 需注入手术)
- 在跑:DSW 12 泳道(vanilla×2、d2/d3、m2×4、m3×4);cornell:P1 测量链
- 完成:vanilla_s0、f1@150(待+100)、c1(cornell);AUROC 对表命中(0.741 ∈ Rethinking 0.73–0.75)
- 关键路径:**vanilla×3 噪声底** → verdict.py 开闸

## 1. 波次总表(按发车顺序)

| 波 | 内容 | 卡 | 时长 | 前置 | 状态 |
|---|---|---|---|---|---|
| **S 种子** | 17 臂 × 补 2 seeds = 34 runs | 68 | ~35h | 无 | **到卡即发,第一优先** |
| **T 二档** | 8B 老师 × 17 臂 + vanilla_t8b×3 = 20 runs | 40 | ~35h | 无(D6 已量) | 与 S 同发 |
| **U 条件化** | i0 三连(vanilla/c1/f1 × 4B-Thinking 素打分)| 6 | ~30h | P1a | P1a 完即发 |
| **锚点 16k** | vanilla + c1 + f1 @16384 | 6 | ~70h | 无 | S 波首批让卡后 |
| **Annex-Think** | 配置 1:7 runs @16k(§4) | 14 | ~4d | P1-AIME + 两处 run-defining 改动 | S 波让卡后 |
| **i1 手术格** | 配置 2b(§5) | 6 | ~30h | 手术 V1–V4 + 预生成 | 手术完 |
| **V 决赛扩展** | 依判决结果(§6 触发表) | ~30 | — | 噪声底 + Phase 1 判决 | 判决后 |

预算合计 ≈ 170 卡·天量级;100 卡下 S+T 同发满负荷,其余接力。

## 2. 教师阶梯(全部候选与取舍,一次说完)

| teacher | 尺寸比 | 非思考天花板 | 决定 |
|---|---|---|---|
| 4B-Instruct-2507 | 2.4× | 0.896 | 主表现役 |
| **8B** | 4.7× | 0.792 | **T 波**:大而弱 → 尺寸/能力解耦(独有对照,D6 在手) |
| 14B | 8.2× | P1b 量取中 | 若 >0.896 → "更大且更强"点,**V 波决赛臂用** |
| 32B | 19× | P1c(兼 tp2 冒烟) | 上限档;老师占双卡 → 泳道 3 卡,**只配最终配方** |
| 1.7B-2507(同尺寸) | 1.0× | 0.702(教错率 6.2%)| 文献对齐端,只配 vanilla+决赛臂 |
| 4B-Thinking-2507 | 2.4× | P1a 量取中 | **I 轴 + Annex 共用**(见 §4/§5) |
| Qwen3.5 系 | — | — | **已查(2026-08-07,config.json 定案):协议外,弃,如预注册**。词表 248,320 ≠ Qwen3 151k(此一条即红线);9B 为 dense(intermediate_size 12288,无 MoE 字段,系列里 MoE 是大杯),但混合注意力(24 linear + 8 full)+ 多模态底座 —— 连自洽 cell 都是新岛(模板/显存/引擎全部重推) |
| Gemma3-12B/27B(族内) | — | 需自量 | **Phase 3 跨家族**,只配最终配方 |

**不做**:pair 网格(稀释统计力)、跨词表对(协议红线,intro 声明)。

## 3. 学生侧候选(V 波与 Phase 3)

| student | 用途 | 触发 |
|---|---|---|
| 1.7B-Base | 全部主表/附表基准 | 现役 |
| 0.6B-Base ← 4B-2507 | 判决对学生容量的依赖;run 0.5× | 决赛臂,V 波 |
| 4B-Base ← 14B | **配方向上迁移**(工业问题);run 2× | 最终配方,V 波 |
| 4B-Base ← 8B(2×) | 与 1.7←4B(2.4×)构成等比对:判决随比例还是档位 | 决赛臂,V 波 |
| 1.7B hybrid(think ON) | Annex 专用(§4) | — |

## 4. Annex-Think(配置 1)定案摘要(细节见 design-thinking-cells.md §3/§6/§7)

- 对:**1.7B hybrid ← 4B-Thinking-2507**(主表尺寸镜像;差异压至 {制度}∪{学生后训练})
- val:**AIME24/25 + AMC23,Wilcoxon on avg@32**(MATH500 饱和预判,降跟踪)
- 臂:vanilla×2 + c1 + f1 + b3 + 主表最佳 D 臂(规则先于数据)
- 备选 A 触发:P1-AIME 显示 4B-Thinking 净空 < 学生 avg@32 波动 → 老师升 8B(think)
- 发车前 run-defining 改动(一次 repin):`ENABLE_THINKING` 旋钮入指纹;preflight 分 regime

## 5. I 轴(配置 2a/2b)摘要(细节见 design-thinking-cells.md §1/§2)

- 阶梯:L0 vanilla → L1 i0(素打分,预注册预测**更差**)→ L2 i1(带私有 CoT)
- i1 手术:teacher 请求注入 `[prompt|cot|response]`;尾长抽取假设 + V1–V4 验证单;
  预生成 12,478 题 ×1 CoT(缓存);`SIMOPD_PRIV_COT` 入指纹
- 若 i1 > L0:头条级(思考监督穿制度注入);若 i1 ≈ i0:私有 CoT 无增益,同样入账

## 6. V 波触发表(判决出来后照查)

| 若 | 则 |
|---|---|
| 某臂 PROMOTE(主表) | 进 3-seed 终审 + 8B 档复现(预注册 ≥2 设定标准) |
| c1 无-Mode-A 在 s1/s2/8B 档复现 | c1 进 16k 锚点 + Annex + 4B-Base↑迁移,作头条线 |
| 全臂平局 | 机理面板(π(S̄)/shadow/Δℓ)解释"为何平" = 结论本身;V 波转投 Phase 2 贪心 |
| shadow Jaccard(TIP/SEAD/teach/selectkd 任两者)> 0.8 | 冗余判决直接写,不再跑重复臂 |
| 14B 天花板 > 0.896 | 决赛臂加 14B 档一列 |
| P1c 32B 冒烟失败(显存) | 32B 出局,阶梯止于 14B,照实报 |
| g3 晋级或含糊 | KDRL-faithful 微格:g3@n=8 + vanilla_n8 对照(2 runs;恢复真 GRPO 组对比) |

## 6.5 W 波:TM 模型对 × 我们的臂(2026-08-06 用户定案)

**Cell**:`Qwen3-8B-Base ← Qwen3-32B`(TM 博客**发表实验**所用模型对)——
溯源注(2026-08-07):TM 博客 2026-06 编者注将 32B/8B-Base 从 Tinker 平台线退役,
cookbook 配方改用 Qwen3.5-9B←9B-Base;**博客的实验与数字仍是 32B→8B 对**,HF 开源
权重不受平台退役影响,W 锚定发表实验、引用注明该编者注。其新配方为同尺寸
后训练→Base 蒸馏,与我们阶梯中 1.7B-2507 同尺寸臂同构,该臂文献对齐价值加强。——
**协议用我们的**:无 SFT init、8k 上限、250 步定点、一臂一旋钮、cell 内自带 vanilla。
定位 = 学生侧规模格(晋级标准的"≥2 设定"从此含学生规模维度),
放弃 TM-faithful 复现(SFT init + 16k + AIME 轨迹对表)—— 若后续想要,单跑一个
TM-faithful vanilla 即可,记为可选项不占波次。

**臂**:首批 = vanilla ×2 seeds + {c1, f1, b3, 主表最佳 D 臂}(规则同附表 §7.4)。
**自动加宽触发**:首批判决与主表一致率 < 3/4(规模翻转多)→ 全 17 臂入 cell。

**工程前置(此格的真实成本)**:
- 泳道形状:8B actor 静态 ~128GB → actor FSDP 2 卡 + 32B teacher tp2 双卡 ≈ **4–6 卡/泳道**;
  执行走 `GPUS_PER_RUN` 参数 + 独立 mini-manifest(现 launcher 已支持该 env)
- 显存算术全部重推(0.45 是 1.7B 专属;此格独立指纹批,天然隔离)
- 单 run ~160–240 GPU·h;首批 6 runs ≈ 1200 GPU·h
- 测量前置:32B 天花板+tp2 冒烟(P1c 已排)、**8B-Base 零点**(入 cornell 链尾)

**变体定案(2026-08-06 追问后)**:学生 = 8B-**Base**(协议锁死:Base+非思考);
老师 = Qwen3-32B **后训练混合体,非思考打分**。Base 老师非选项(无可蒸能力);
双 thinking = Annex² 未来格(一步两变量,不属 W)。
**预注册意外情形**:32B 为混合体,非思考打分付草稿纸税 —— P1c 若量出其非思考
天花板 ≤ 4B-2507 的 0.896,则"名义 19× 的老师在本协议下不强于主表小老师",
这本身入 D6 作规模端发现;**老师仍留 32B 不换**(对的身份即价值),不临时改 14B。

**排期**:V 波同期或其后(噪声底落地为门),不挤 S/T。

## 6.6 a1 复活计划(2026-08-06 翻案)

原延后理由(TransferQueue 手术不值)被 i1 的注入门派推翻:**rollout 注入**方案
(学生 server 按 prompt-hash 掷币 λ=0.5,命中返回缓存 teacher 响应 + 学生引擎打分)
不碰 trainer。GKD 是奠基双璧之一,缺席是审稿人一眼可见的洞。
排序:i1 手术 → a1 手术(共享机械)→ 预生成(cornell)→ 随 S/T 波入列,A 轴齐装。
PG-形式偏离照 estimator-note 论证入 note。

## 7. Phase 2/3 定型(依 plan §4,不变,列此备查)

- 贪心 R1=单轴筛(即主表);R2+:vanilla 起逐加最优 trick,增益<噪声底停;后向删
- Phase 3:三域验证(code/IFEval 训练域扩展**只配最终配方**)+ Gemma3 族内对 + 终审 3 seeds

## 6.7 优化器路径修正(r3 2026-08-07 初判;r5 同日逐臂对官方码后修订)

r3 曾将散度值臂(b2/c1/c2/e1/b3)整体改直接分支;**r5 终态**:直接分支名单 =
**b2/c2/e1**(verl 自家引文与我们自研);**b3** = PG 底座 + 叠加 FKL(官方码,
`b3_additive` 送项);**c1 撤销改判回 PG** —— LSM 官方库以原始 −KL 作优势走 PPO
代理,论文行文与其代码相左,以码为准(estimator-note §8 记方法论教训)。
**处置**:b2/c2/e1 的 PG 旧 run 迁 `*_pgab` 消融、直接版正名重发(已执行,
migrate_stale.py);**cornell c1 的 PG run 恢复为忠实结果**(同集群对照告诫不变),
其直接版从必做重跑降为可选 paper-form 消融;**g2 的 stage-1 旧版**跑完后同走
migrate_stale 迁移、全式版正名重发。d1/d2/d3 底损失为带符号采样 k1,PG 正确,
结果有效。

## 8. 治理提醒(执行者须知)

- 每次动 `src/ configs/arms.yaml run_opd_baseline.sh arm.py` → **repin**(REASON 入 PIN_HISTORY)
- 新机器:`MACHINE=mX bash deploy/up.sh` 一次,此后免名;纯池子机器可直接入列
- 一切失败:先 `logs/campaign_last_*.txt` / daemon 日志 / `triage.py`,**py-spy 后再清扫**
- 熔断 3 败隔离;放行 `MAX_RUN_RETRIES=99`;阈值原则:**先看健康长什么样**
