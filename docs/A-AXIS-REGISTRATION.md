# A 轴(DAgger/AggreVaTe 增补)裁决预注册 —— 2026-08-18

范围:a1_gkd_mix0.5 / a3_offpolicy / a4_dagger_anneal / a5_aggrevate × 3 seed
× 250 步,协议钉值与战役完全一致(k1_rec + PG,16k,SAVE/TEST=25;见
configs/arms.yaml 各臂 note 与 docs/EXPANSION-PLAN.md 的纪律总纲)。本文件由
git commit 时间戳公证;修订一律追加条目,不改写既有条目(append-only)。

## 注册时已见数据的坦白声明

注册时刻(2026-08-18 ~09:30 集群时,三 seed 训练进行中)已见且仅见:

- s0 step-25 in-house val(math500 acc@1):a1 **0.532**,a5 **0.528**,
  a4 **0.488**,a3 **0.460**;彩排期 a1 step-3 val 0.462(近基线锚点)
- 遥测侧带零星读数:a1 累计 λ_realized≈0.49、teacher_token_frac≈0.75;
  a4 λ(24)=0.904;a5 彩排期 step0-2 结局分布(ft→mixed 迁移,κ̄ 贴 tmax/2)
- **未见**:一切 step>25 结果、一切 s1/s2 指标、一切 post-eval

污染自查:上述 val 排序(a1≈a5>a4>a3)已被注册者看到并作过口头初步解读,
因此 R3 的"在线优"侧在注册时已有一个有利数据点——该条目证据力按此打折。
相对终局裁决面(250 步 × 3 seed × 5 基准 post-eval),已见部分 < 5%。

## R1 a4-vs-a2:渐进过渡 vs 突变切换

- **问题**:教师数据→学生数据的过渡,线性退火(a4)是否优于两阶段突变
  (a2 冷启动 SFT→OPD)?本轴设立的原始动机。
- **预测 G(渐进有益)**:a4 终局决策指标 ≥ a2,且训练曲线无 a2 切换点邻域
  的暂时跌落。论据:DAgger 分布匹配论证 + 本组 agentic 配方(10 iter ×
  0.2/轮)的先例经验。
- **预测 A(突变不劣)**:a2 ≥ a4。论据:单轮设定下切换冲击可在 ~25 步内被
  吸收;两阶段的"纯净"训练信号可能优于全程混合梯度。
- **裁决**:终局(step 250)post-eval 决策指标(math500 greedy pass@1 +
  AMC23/AIME24/AIME25 avg@32 + minerva),3 seed,verdict.py 的
  McNemar/Wilcoxon 标准;辅助证据:in-house val 曲线在 a2 切换步邻域的形态。

## R2 a1-vs-a4:等剂量下时间表是否重要

- **问题**:序列级教师剂量配对(a4 全程均值 0.502 vs a1 恒定 0.500),恒定
  与退火谁优?本对照是配对设计的核心。
- **预测 S(时间表有益)**:a4 > a1——前期教师密度助冷启动,后期纯学生巩固。
- **预测 C(剂量即充分统计量)**:|a4 − a1| 落于噪声内。
- **预先措辞约束**:序列剂量配对 ≠ token 剂量配对(a1 实测教师 token 占比
  ~0.75,因教师答案系统性更长)。裁决必须在两条剂量轴上分别陈述;token 轴
  一律采用侧带实测积分,不得用名义 λ。
- **裁决**:同 R1 决策指标;3-seed 显著性不过线即判 C 侧成立。

## R3 a3-vs-a5:缓存离线教师 vs 在线前缀条件教师

- **问题**:教师 token 剂量相近时,数据来源——整条离线缓存(a3)vs 从学生
  前缀在线续写(a5)——是否影响效果?
- **预测 On(在线优)**:a5 > a3。论据:AggreVaTe 的核心主张,教师尾巴条件
  于学生实际到达的状态,数据分布贴合学生的真实需要。
- **预测 Eq(来源无关)**:a3 ≈ a5,教师 token 就是教师 token。
- **已见披露**:s0@25 的 0.528 vs 0.460 利于 On 侧(见坦白声明);本条目
  的验证性效力以终局裁决为准,该早期点仅计探索性。
- **裁决**:终局 post-eval 3-seed 为主;辅助:整条 val 曲线 AUC;比较前以
  侧带实测剂量差异作协变量校正。

## R4 a5 实际剂量前置(测量性声明)

- **声明**:a5 的实际教师 token 剂量显著前置于名义 T_max ramp——
  κ~U{0,T_max} 的均匀抽取叠加教师尾巴远长于学生前缀,预测 tail_token_frac
  前期贴近 1、其下降显著慢于名义期望;终局累计教师 token 剂量 > 名义中点。
- **约束效力**:一切涉及 a5 的"等剂量"比较必须以侧带 tail_token_frac 实测
  积分为剂量轴;禁用名义 ramp(这正是遥测管道按臂律修通的原因)。
- **裁决**:三 seed 侧带遥测积分曲线 vs 名义曲线,直接测量,无对立面。

## R5 F3 EOS/长度透镜(接既有 F3 注册问题)

- **问题**:教师数据经由长度/EOS 通道(教师答案系统性更长)影响学生的部分
  有多大?
- **预测 L(长度通道真实)**:终局平均回答长度按教师剂量排序
  (a3 ≥ a4 ≈ a1 > vanilla),且以长度分层/控制后,val 增益可观缩水。
- **预测 N(长度效应可忽略)**:增益基本全部来自内容而非长度。
- **裁决**:response_length 训练曲线 + post-eval 输出长度分布;测量工具接
  F3 已注册的 EOS-mass / τ=0 复评方法。

## R5 附录:终止符漂移机制与双停契约(2026-08-19 追加,append-only)

### 已见数据坦白(登记时点)

- 配置事实:学生 Qwen3-1.7B-Base eos=151643(单值);教师 Qwen3-4B-Instruct-2507
  eos=[151645,151643](tokenizer eos=`<|im_end|>`);两模型词表共享,两 id 两侧同号;
  训练存档原样复制学生 generation_config(eos 仍 151643);rollout 与 eval 采样
  均未显式传 stop,故只认被服务模型自身的 eos。
- 行为事实:学生基座训练前 textdump 12 题 100% 正常停止(均长 257);vanilla@250
  同题 8.3% 停止、均长 15053,截断样本在 1108 字符处已给出正确答案后将
  "Final Answer/We are done" 重复 980 次至 32k 帽。
- 剂量形态:aime24@step25 截断率 vanilla .22-.26 < a5 .27-.29 < a1 .33 < a4 .39-.45
  < a3 .48 < a2_coldstart(纯教师文本 SFT).985;step50 vanilla 爆至 .854-.864 而
  a1_s2 仅 .383。终档 fr_stop 与"尾部被监督量"整齐对应(h2 尾段监督 0.000,
  h1/h5 前段 .76-.79)。
- **未见**:任何双停止集下的重评数据;学生 `<|im_end|>` 发射率的任何 token 级
  直接测量。

### 机制假说与对立面

- **M-drift(登记方)**:反向 KL 在学生采样的终端 token 上压制其唯一句号
  (教师在该处质量 ~0),教师授权文本同时教会教师的句号;采样器只认学生
  单一 eos,于是"想停而停不下"。两条病理:P1 句号被压(vanilla 型,慢),
  P2 学了采样器不认的句号(a2 型,快)。
- **M-null(对立)**:截断是与终止符约定无关的普通长度失控/退化;换停止集
  重评不会实质改变截断率。

### 三方判定(实验未跑,预测先行)

用停止集 {151643,151645} 对既有 checkpoint 重评(aime24,余参数与原格一致):

- **P1** a2_coldstart@25:截断率大幅回落(自 .985 至少减半);
- **P2** vanilla@250:基本不动(回落 <10 个百分点)——它是句号全无而非句号换牌;
- **P3** a1_gkd_mix0.5@50:落在两者之间。

M-drift 要求 P1∧P2∧P3 同时成立;P1 不成立即 M-drift 的 P2 病理被推翻,
P2(vanilla)大幅回落则 P1 病理被推翻(vanilla 其实学会了 im_end)。

### 契约法令(实现已入库,部署另行决定)

自本条起,双终止符 {151643,151645}(即教师自带停止集,双约定之并)为**修正
协议 v2**:训练 rollout 经 `SIMOPD_STOP_IDS`(进指纹;run 级 pin 文件
`simopd_stop_contract.txt` 保证任何 run 终身单一契约,先于契约的 run 一律
grandfather 为 off 且环境不设、指纹与历史逐字节一致);评测 `eval_offline.py`
默认双停,每行产物落 `stop_token_ids` 列。**跨契约比较必须显式声明**。在跑
A/H wave 的处置(按旧契约跑完 vs 推倒按 v2 重训)与 vanilla/a2 是否补训 v2
版,待判定实验结果后决定;该选择不影响本条预测的效力。

### R5 附录补记(2026-08-19 追加,append-only;测量方 = ch-dev 终止章工作)

**"未见"一栏现在有数据了**——token 级 `<|im_end|>` / `<|endoftext|>` 直接测量,
CPU 上对真实 checkpoint 与真实 rollout 做的(脚本 `scripts/analysis/eos_stop_probe.py`
/ `eos_stop_audit.py` / `a2_coldstart_probe.py`,回执 `docs/data/eos_stop_*.txt`、
`docs/data/a2_coldstart_probe.txt`;机制正文 `docs/MECHANISMS.md` §M-I):

- 学生停止位(30 条真实已停止 rollout,vanilla@125/225、c4@100):教师 q(`<|im_end|>`)
  中位 0.95、q(`<|endoftext|>`) 中位 1.4e-11(秩 ~2×10⁴);base 学生 p(`<|endoftext|>`)
  中位 0.98、p(`<|im_end|>`) ~1e-11。vanilla 采样列 k1 在每次停止事件给停止 token
  Δℓ = −25 nats(中位),教师终止 token 永远采不到——**M-drift 的 P1 病理("句号被压")
  逐 token 证实**:vanilla@250 自然停止位 p(`<|endoftext|>`) 中位 6e-5,p(`<|im_end|>`)
  ~1e-13。
- **M-drift 的 P2 病理("学了采样器不认的句号",a2 型)在 token 级不成立**:冷启动 SFT
  目标确以 `<|im_end|>\n` 收尾且 0/6358 行含 `<|endoftext|>`,但 98 步 SFT 只把
  p(`<|im_end|>`) 从 1e-11 抬到 ~1e-3(每 ~2.3k token 一个目标),同时把 base 的
  p(`<|endoftext|>`) 从 0.98 压到 ~1e-3(从不是目标、正是那些位置的 CE 竞争者),质量落
  在续写(`\n\n` 0.52 → 教师收尾模板"---\n\n**Answer:** \boxed{}",即终末循环);
  a2@25 p(`<|im_end|>`) 4e-6~8e-4、@50 起两个终止符都 <1e-7。它不是"想用 `<|im_end|>`
  停而停不下",是"两个句号都没有"。因此**预测 P1(a2@25 双停重评截断率至少减半)
  预计不成立**,且这一"不成立"不推翻 M-drift 的训练时机制——它只说明错位的伤害发生
  在训练时(SFT 学不到采样器认的句号;OPD 惩罚学生的句号),事后换停止集重评救不回
  v1 checkpoint。**P2(vanilla 基本不动)预计成立**(同上,`<|im_end|>` ~1e-13);P3
  未测。建议把三方判定的读法从"重评救回多少"改为"token 级两个终止符各在何处、
  何时被压/被教"。
- 原配方差异:LlamaFactory `qwen3` 模板 `replace_eos=True`、`stop_words=[<|im_end|>]`
  → Rethinking 的 SFT 学生 eos = `<|im_end|>`,其 OPD rollout 停在那儿;我们的
  verl-sft_trainer 移植保留 Base tokenizer,丢了这一步。a2 修正格见 RESULTS-GAPS A1'。

**契约 v2 的落地口径(与本条实现同日,`stop_set` 之上的三处防护)**:
(1) `eval_offline.py --stop-token-ids` 默认改为 `auto` = 该 checkpoint **训练时**的
契约(读 run 目录 `simopd_stop_contract.txt`;无 pin 的 pre-contract run 与 hub id
= `off`)——正在跑的 post-hoc drain 从共享盘读默认值,全局翻到双停会把 819 格 v1 与
余下格子 v2 混进同一张表;每行产物仍落 `stop_token_ids` + `stop_contract_source`,
`extract_post_eval.py` 把 `stop_set` 带进 cells/bystep 并拒绝跨契约合成 composite。
(2) 数学 campaign 的 `_lane.sh`:未在臂 env 里指定时用 batch 契约
`.campaign/STOP_CONTRACT`(默认 off = 该 batch 登记时的契约;把 batch 翻到 v2 =
写文件,而不是拉一次 launcher 的副作用)。DLC A/H fleet 启动器**不覆盖**——A 轴
v2 重启波(`stop2` 标记,legacy 同名 run 已移侧)按 launcher 的新 run 默认走 v2,
是登记方的明确决定。(3) 终止族臂(n0_termfix / n2_termcal)env 显式钉
`SIMOPD_STOP_IDS: off`(与 v1 vanilla 单旋钮可比;v2 配对需先有 vanilla_v2),
`eos_gather` 在 import 时校验 E_S == 当前契约下真正终止 rollout 的集合,arm_lint 同查。

