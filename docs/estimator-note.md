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

b3 的 FKL 侧 token 与 b2 完全同底,故 b3-vs-b2 的对照**同偏可比**;
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
