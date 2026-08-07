# 估计器核对:OPD+ (2606.01039) 的偏差指控 vs 我们的底座

> casefile 将 OPD+/vOPD 判为【落】时写下"并入工程默认核对项"。本页即该核对,
> 执行于 2026-08-06(campaign 12 臂在跑期间;结论:**不改底座**,理由见末节)。

## 1. 我们的底座到底算什么(verl 实况,非文档转述)

`verl/trainer/distillation/losses.py`,`use_policy_gradient=True` 路径(我们全部臂):

```python
advantages = -distillation_losses.detach()      # L277:stop-gradient 优势
distillation_loss, _ = policy_loss_fn(          # 标准 PPO 代理:ratio + clip
    old_log_prob=..., log_prob=..., advantages=advantages, ...)
```

即:**A_t = sg(−ℓ_t) 作逐 token 优势,走 PPO 裁剪代理**。代码注释自证血统:
"as done by thinkingmachines.ai/blog/on-policy-distillation" —— **这是领域正典估计器**,
也是 Demystifying(我们的协议锚)所采用的 PG 形式。

## 2. OPD+ 的指控,逐条对表

其主张:对**一般 f-散度**,把依赖学生似然的散度奖励做 stop-gradient,
得到的目标值与梯度**双双有偏**;OPD+ 给出修正并支持任意 f-散度。

对我们的分臂适用性:

| 臂族 | 损失 | sg-优势估计器的性质 |
|---|---|---|
| k1 家族(vanilla、b1、f1、f2、g1、g2、h1、d1–d3 的基损失) | ℓ = log p − log q(采样点) | **梯度期望无偏**:∇E_p[ℓ] = E[∇log p·ℓ] + E[∇log p],被 sg 丢掉的第二项在 on-policy 期望下为零。有限批内非零 → 只是方差;OPD+ 的"objective 有偏"指的是**报告的目标值**,不影响梯度期望 |
| 非 k1 臂(b2 的 FKL、b3 的 FKL 侧、c1/c2 的截断 KL、e1 的 rank loss) | 一般 f-散度 / 非似然差形式 | **OPD+ 的偏差指控实质适用**:sg(−ℓ) 作优势是启发式代理,其梯度不等于 ∇E_p[ℓ] 的无偏估计。此外 PPO 裁剪 + old_log_prob 滞后 + token-mean 归一在所有臂上引入与 OPD+ 无关的既有偏离(PPO 文献已知) |

## 3. 裁决:保留底座,记录而非修正

三条理由,按重要性:

1. **审计保真**:受审的每篇论文的声称都是在**这个**估计器(或其近亲)下做出的。
   换成 OPD+ 修正后再审,审的就不是文献的 OPD,而是另一个算法 —— 那是方法论文
   该做的事,不是审计该做的事。
2. **臂间可比性完好**:全部 17 臂同一底座、同偏同滞,偏差在臂间差分中相消;
   受影响的只是与"理想梯度"的绝对对齐,而那不是任何判决的量。
3. **campaign 已开跑**:12 臂在此底座上训练中;中途换底 = 第二批 = 第二个噪声底,
   与 0.45/0.55 决策同一算术,结论相同。

## 4. 论文里怎么写(limitations 一段,措辞已备)

> All arms share verl's canonical PG estimator (sg(−ℓ) as per-token advantage under a
> clipped surrogate), the formulation of [ThinkingMachines; Demystifying]. For the k1
> family this score-function gradient is unbiased in expectation; for divergence-valued
> arms (b2, b3, C/E axes) it is the field's standard heuristic surrogate, whose bias
> OPD+ [2606.01039] formalizes and corrects. We deliberately audit under the field's
> estimator rather than the corrected one: the claims under audit were made here.
> Between-arm comparisons share the base and are unaffected; absolute alignment with
> the idealized gradient is not a quantity any verdict uses.

## 5. 附:b3 的一条内部脚注

~~(r5 前描述)~~ b3 现为叠加式(§7),其 FKL 项无 clamp(b2 有):b3-vs-b2 仍可比但非同底,差异入臂注;
b3-vs-b1 的对照中路由到 RKL 的 token 与 vanilla 同底。b3 的三方读法不受影响。

## 6. 追加(2026-08-07 audit-r3):top-k 臂的优化器路径错配 —— 本页此前的盲区

verl 的 `distillation_loss` 有两条出口,注释自证血统:PG(TM 引文)与**直接反传
(GKD 引文,2306.13649)**。散度值 ≥0 的臂(b2/c1/c2/e1/b3)在 PG 下退化为
"advantage = −KL ≤ 0 恒负"的无差别打压 —— 无替代方向的抑制(恰是 CR-OPD 批判的形态),
预测症状:负增益、输出变短、熵塌 —— 与观察一致。**这些臂的论文原式全部为直接优化**,
故审计保真要求它们走直接分支;k1 家族(含 D 轴的采样 token 底损失)维持 PG(Demystifying
协议)。本页 §1–3 的"全臂共用 PG"论述当时未质疑 top-k 的归属,是审计盲区,记此为改。

**连带**:c1(cornell, PG 路径)的无-Mode-A 结果降级为"待直接路径复核" —— 短输出可能
部分是打压伪影。PG 版 run 保留为意外的 PG-vs-direct 消融,不作臂判决。

## 7. 更正(2026-08-07 audit-r5):b3 移出直接分支名单

§6 把 b3 归入散度值臂是对其官方代码(WLS04/EOPD)成文前的推断。实读 `core_algos.
compute_policy_loss_on_policy_distill`:EOPD 的损失是 **PG 底座(全 token 采样 k1
优势,裁剪代理)+ 高熵 token 上的叠加式 top-k 前向 KL**(`pg_loss = pg_loss +
soft_kd_loss`,固定阈值 0.8、系数 1.0)。即 b3 的正确归属是 **PG 分支 + 一项直接
反传的叠加项**,两分支各取所需;§6 的直接分支名单缩为 b2/c1/c2/e1(其论文原式
确为直接优化,不受影响)。执行:`simopd/b3_additive.py` 包装 verl `distillation_loss`
送入叠加项,r3 的 where-切换版作废。

## 8. 更正(2026-08-07 audit-r5):c1 也移出 —— 且方向与 b3 相反

LSM 官方库(hhh675597/revisiting_opd)的 `compute_opd_advantage`:renorm 截断 KL
**取负后原样作 per-token 优势**(无基线、无白化、reward_weight 默认 0),下游 PPO
裁剪代理 —— **正是 §6 判为"病态"的那个形状,却是其发表数字的生产线**。论文 §3.2
的行文(支撑集外梯度不传播)读作直接反传:论文与代码相左,审计以码为准(声称由
代码产生)。故 c1 忠实形式 = **PG**,r3 对 c1 的直接分支改判撤销;cornell PG run
(0.598,无 Mode-A)恢复为忠实结果,其"待同集群对照"的告诫不变。§6 的直接分支
名单最终为 **b2/c2/e1**(verl 自家引文与我们自研,均不受影响)。

方法论教训一并记下:r3 用"散度值≥0 ⇒ PG 下恒负打压 ⇒ 必是伪影"的**先验推理**
批量改判了五个臂;r5 逐个对官方码后,五个里有两个(b3、c1)的原作者恰恰把"病态"
形状发表了出来。审计的对象是文献做了什么,不是文献该做什么 —— 恒负优势在
token-mean 聚合 + 裁剪下是"按 KL 大小差异化压制",其有效性本身就是 D6/机理面板
该测量的经验问题,不是审计可以代答的。


## 9. 记录(2026-08-07 终审):verl 首微批归一化怪癖(全臂同偏,不修)

`global_batch_info` 由 `ppo_loss` 在**蒸馏损失之后**填充,故每个进程的第一个
micro-batch 按微批 token 数归一(≈N_micro 倍权重),此后各 mini-batch 的首微批
沿用上一批的计数(微漂)。全部臂(含基线)同承此偏,按 §3 同偏可比原则记录不修;
b3 的叠加项已改全批归一(losses.py),不受此怪癖影响。日志里 step-1 的损失尺度
异常即此。
