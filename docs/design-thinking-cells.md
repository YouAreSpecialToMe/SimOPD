# 设计:thinking 三格(配置 1 / 2a / 2b)—— 2026-08-06 预注册

> 在任何一格的数据存在之前登记。三格共享一个骨架问题:**主表(非思考)的判决,
> 有多少能穿过制度边界。** 主表 17 臂协议不动;这里全部是附加格。

## 0. 三点阶梯(同一尺寸、同一家族,只动条件化状态)

```
L0  vanilla: 1.7B-Base ← 4B-2507(非思考原生)       主表基准,在跑
L1  = 2a:    1.7B-Base ← 4B-Thinking-2507 素打分     隔离「调法税」
L2  = 2b:    1.7B-Base ← 4B-Thinking + 私有CoT        隔离「思考内容的监督价值」
A   = 配置1: 1.7B(hybrid, think ON) ← 4B-Thinking     制度整体切换(附表)
```

相邻两级之差各回答一个单变量问题;L0→L1→L2 学生侧字节不变。

## 1. 配置 2a(`i0_think_scorer`)—— 零手术

- **定义**:主表 vanilla 协议,唯一改动 `TEACHER_MODEL=Qwen/Qwen3-4B-Thinking-2507`。
  老师打分时面对无 think 块文本,离开主场分布(机理同 8B 倒置,但同尺寸同代,干净)。
- **预注册预测(可证伪)**:i0 劣于 L0 —— 天花板数据的延伸。若反而优,单独成一节。
- **规模**:vanilla+c1+f1 × 1 seed(它是 L2 的对照,不需要全 17 臂)。
- **登记**:arms.yaml `i0_think_scorer`(I 轴:teacher conditioning),status: stock。
- **前置**:P1 测量(见 §4)。

## 2. 配置 2b(`i1_priv_cot`)—— 中等手术,本文档即手术设计稿

**定义**:i0 + 每题一段离线预生成的老师私有 think 块,打分时拼进老师自己的上下文。
学生看不见任何字;唯一新旋钮(相对 i0)= 私有 CoT 的有无。

### 2.1 预生成(`scripts/gen_priv_cot.py`,待写,~60 行)

- 老师 4B-Thinking、think ON、每题 1 条,think 段截到 4096 token;
  输出 parquet:`prompt_hash → cot_token_ids`(以学生 tokenizer 的 id 存,见 2.2)。
- 跑一次缓存复用(12,478 题 × ~2k think token,2 卡数小时,cornell)。
- **纪律辨析(要写进 paper)**:老师的 CoT 常含最终答案 —— 这是设计意图
  (判卷人心里有数),且与 g1 的红线不冲突:红线是"verifier 答案不进**训练输入**",
  此处答案只进**老师的条件化**,即监督信号的构造,学生输入零污染。

### 2.1.5 三个设计参数定案(2026-08-06,动刀前)

1. **CoT 生成:greedy、单条、不验证。** 用 ground truth 筛选 CoT 会让 verifier 的
   答案进入监督构造 —— g1 红线的近旁,不越。老师 CoT 的对错率(cot_correct)照测
   照存,但**只用于预注册子分析**:错草稿题上的监督质量是否塌(逐题,D6 式)——
   privileged-bias 文献(2607.05184 等)预言的失败模式,我们直接量。
2. **think 上限 4096,截断则强补 `</think>`**,截断率报。
3. **同形合同(取代原 §2.2 的"尾长抽取假设")**:server 侧前缀替换 + 抽取后裁行,
   返回数组与无注入版逐位同形 —— trainer 侧无论头切尾切都不受影响。

### 2.2 注入点(手术)—— 已由候选升级为确认(2026-08-06 读码)

- **确认位置**:`vllm_async_server` 的 server 端 generate 处理器(~L595–650:
  `prompt_ids` 装配进 `TokensPrompt` 之前替换;`extract_prompt_logprobs` 调用之后裁行)。
  与 `teacher_patch` 同模块同 sitecustomize 钩子,复用 loud-fail 纪律。
- **判别器**:`sampling_params.prompt_logprobs is not None` —— 只有打分请求带它,
  学生生成路径从不带;注入被精确限定在打分调用,零误伤。
- **替换语义**:缓存存 `{前16token哈希 → (原前缀ids, 替换前缀ids, cot_correct, truncated)}`;
  命中则换前缀(空 think 块 → 真 CoT),响应尾原样;**未命中即 raise**(臂运行中每个
  训练题都必须命中,miss = 缓存错位,响亮好过带病)。
- **对齐方案(升级)**:抽取后在 server 侧删除注入段对应的行(含 teacher_patch 的
  K+1 采样列,按行走所以结构自保),返回同形数组 —— 原"尾长假设"不再需要成立,
  V2/V3 改为验证裁行本身。
- **开关**:`SIMOPD_PRIV_COT=<parquet路径>`(空 = 完全不挂钩)。经 SIMOPD_* 通道
  自动入指纹。
- **验证清单(动刀后、入列前)**:
  - V1 同一 rollout 带/不带前缀打分,logprob 必须不同(注入生效);
  - V2 尾部对齐:构造已知 token,带前缀打分的 response 段逐位对上无前缀版的位置;
  - V3 K+1 采样列(teacher_patch)在偏移下仍抓对 token id;
  - V4 一次 3-step 彩排,面板数值合理。
- **登记**:arms.yaml `i1_priv_cot`,status: **needs**(seam = 本节手术 + 预生成)。

## 3. 配置 1(Annex-Think)—— 附表,S 波释放后

- **cell**:`Qwen3-1.7B(hybrid)+ enable_thinking=True ← 4B-Thinking-2507`,16k 上限。
- **偏离台账(附表自带)**:学生是后训练混合模型(非 Base)—— 起点语义不同,
  这是"学生必须会开 think 块"的必然代价,声明而非隐藏。
- **run-defining 改动(届时一次 repin)**:`run_opd_baseline.sh` 的
  `enable_thinking=False` 硬编码改为 `${ENABLE_THINKING:-False}` 并入指纹;
  preflight 的 think 块检查按 regime 分叉。
- **规模**:vanilla_think ×2 seeds + c1 + f1 + b3 + 最佳D臂 = 7 runs @16k(~14 卡,3–4 天)。
- **读法**:逐臂"判决是否翻转"表 —— 尤其 c1(长度机制在 think 制度下更烈,
  无-Mode-A 若复现即跨制度规律)。

## 4. 前置测量 P1(cornell,依赖链接在 coldstart 之后)

| 任务 | 量什么 | 用途 |
|---|---|---|
| P1a(1卡) | 4B-Thinking:think@16k 与 非思考@8k 两条天花板;hybrid-1.7B think@16k 零点 | i0/i1 的 D6 行;附表 step-0 锚 |
| P1b(1卡) | 14B 非思考天花板 + D6 | 教师阶梯第 4 点 |
| P1c(2卡, tp2) | 32B 非思考天花板 | 阶梯上端;顺带显存冒烟 |

## 5. 预算与排期(并入 100 卡计划)

```
现在        P1a/b/c 入 cornell 队列(几 GPU 时,零训练卡)
Wave U      i0 三连(6 卡)+ 8B 档(Wave T)同批
i1 手术     设计稿(本文档)→ 动刀 ~1 天 → V1–V4 → 预生成 → 入列(6 卡)
Annex       S 波让卡后(~14 卡,3–4 天)
```

## 6. 配置 1 选型定案(2026-08-06 补)

**对:`Qwen3-1.7B(hybrid, think ON) ← Qwen3-4B-Thinking-2507`。** 对称性论证:与主表
尺寸完全镜像、同代(2507)、老师为主表老师的调法姊妹 —— 附表-主表差异压到
{制度} ∪ {学生后训练}。老师与 I 轴同权重,P1 测量一次喂三格。

**val 集(预注册)**:主 = AIME24/25 avg@32(MATH500 在 thinking 域预计饱和,降为
跟踪);P1-AIME 补充任务量零点与净空。

**备选 A 触发条件**:若 P1-AIME 显示 4B-Thinking→1.7B-think 的 AIME 差距 < 学生
avg@32 的自身波动,老师升 8B(think)(其 thinking 主场天花板由同任务顺带量出)。

## 7. 过审加固(2026-08-06,应"放 paper 行不行"之问)

**7.1 统计功率(n=60 问题,现在补)**:AIME24/25 仅 60 题,二值化 McNemar 的不一致对
将个位数,功率不足。**预注册**:附表主检验 = 逐题 avg@32 通过率上的 Wilcoxon 符号秩
(配对连续量,非二值化);val 加宽为 AIME24/25 + AMC23(~100 题);MATH500 仅跟踪。
(verdict.py 现有 McNemar,Wilcoxon 变体属评测工具,附表发车前补,pin 豁免。)

**7.2 判分器 × think 块(已验,有据)**:verl MATH 判分器取**最后一个** boxed
(`math_dapo.last_boxed_only_string`)—— 终答在 </think> 后且居末,think 内的中间
boxed 不会被误抓;截断中段无终答按错计,截断率在报。P1a 数字可信。

**7.3 血统披露(写进 limitations,三句)**:
(i) 附表学生为后训练混合模型,其思考能力本身由 Qwen3 官方管线蒸馏而来 ——
附表**审计的就是这个生产制度**,原样声明;"纯血"替代路线(Base + thinking 冷启动,
即 TM 配方 = 我们 a2 机械 + thinking 老师)列为替代方案,非本轮;
(ii) 全 Qwen 家族的血统混同由 Phase 3 的 Gemma 族内对覆盖;
(iii) AIME25 早于 Qwen3 发布,预训练暴露为全领域共有风险 —— 同条件内比较 +
avg@32 部分缓解,照实声明。

**7.4 臂选择规则显式化**:附表臂 = vanilla ×2 + {c1, f1, b3} + **主表判决最佳的
D 轴臂**(以附表发车时刻的台账为准)—— 规则先于数据登记,免"挑臂"之嫌。

## 8. val 差异的方法论立场(2026-08-06,预注册)

**立场:附表 val(AIME24/25+AMC23)≠ 主表主判据(MATH500)是接受的,论证三层:**
(i) 附表判决全部在制度内部(arm vs 本制度 vanilla,同尺);跨制度迁移发生在判决层;
(ii) 仪器效度 > 仪器同一性 —— thinking 域沿用 MATH500 = 饱和仪器,判别力为零;
(iii) AIME/AMC 本在主表指标族(plan §1),两制度共测 → 跨制度数值比较的**唯一合法轨道**。

**两条硬规矩**:①禁止跨制度比较不同主判据的量值(只比判决方向、机理量、共享指标);
②MATH500 附表照测:饱和是可证伪预测,不饱和则白得同尺,饱和则照实报。

**训练中曲线 val**:16k thinking 下全量 MATH500 单次 val 达小时级 → 附表 in-training
val = MATH500-sub100(曲线形状用),判决量一律出自 final ckpt 离线 AIME/AMC avg@32。
与主表"verl 曲线 / eval_offline 台账"两轨制同构。

## 9. thinking 域资格测量与"倒置消失"预测(2026-08-06)

**立场**:固定 4B-Thinking 为附表老师;测量不为重开选型,只服务备选 A 触发器与机理图。

**预注册预测(可证伪)**:非思考倒置(4B-2507 0.896 > 8B 0.792)的机理是混合模型的
"草稿纸税" → thinking 模式下税退还:8B 的 think−nothink 差值显著为正,8B(think)
大概率回到 0.896 之上,**倒置随制度消失**。消失 = D6 机理解释被独立证实;
不消失 = 单独成节。(4B-2507 无 think 模式,2×2 恰好只落在 8B 上。)

**测量清单**:P1a(4B-T 双模式 + 学生零点)、P1-AIME(三模型 AIME,含 8B-think)、
P1-8BT(8B-think MATH500,补 2×2)—— 全部入 cornell 队列;artifacts 落地后
`d6_matrix.py` 出 thinking 版逐题矩阵(CPU)。
