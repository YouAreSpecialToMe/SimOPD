# 逐臂溯源审计 r4(2026-08-07)—— 论文原式 / 原码 / 我们的实现 三方对表

> 方法:方向性语义优先(r3 的教训:错的从来不是算式,是"谁被选、谁被压、α在哪边、
> 走哪条优化分支")。核验手段注明:**[数]**=数值对拍已过 **[码]**=对读源码/原仓
> **[文]**=本轮取原文核对 **[构]**=构造性一致(直接调用被引实现)。

| 臂 | 判定 | 依据与备注 |
|---|---|---|
| vanilla `k1_rec` | ✅ 一致 | [构] 直接调 verl `kl_penalty(k1)`,仅加面板;PG=Demystifying 协议 |
| a1 `gkd_mix` | ✅ 一致@λ=0.5(待彩排) | [数] 四路径;**λ 语义映射已注记**:`SIMOPD_GKD_LAMBDA`=P(off-policy)=1−λ_GKD(GKD 的 λ 是学生数据份额);0.5 两义相同,≠0.5 时按此换算 |
| a2 `coldstart` | ✅ 配方级一致 | Rethinking recipe;档位已修;保留题纪律 |
| b1 `skew_kl` | ✅ 一致 | [文·真论文] DistiLLM SRKL_α=KL(q‖αq+(1−α)p),α 在学生侧,α=0.1;我们 logaddexp(s+logα, t+log(1−α)) 逐项吻合;界 −logα ✓ |
| b2 `forward_kl_topk` | ✅ 一致 | [构] verl 原装核+后处理;r3 起走**直接分支**(verl 注释引 GKD);clamp₀ 记录 |
| b3 `eopd_gate` | ✅ 一致(2 处记录内选择) | [数] 双侧分布级(r4 重写:采样 k1 在直接反传下自身退化);路由方向=EOPD(低熵RKL/高熵FKL);阈值取批分位 = 记录内选择 |
| c1 `lsm_renorm` | ✅ 一致(码锚) | [码] 对齐 EasyOPD `kl_renorm_topk`;[数] 5e-7;**直接分支**;PG 旧 run → `pgab` 消融 |
| c2 `qb` [自研] | ✅ 设计=实现 | [数] 预算自适应 6/9/12@8;直接分支 |
| d1 `tip` | ✅ 一致 | [数] soft-OR/p98裁/0.5留存 逐位精确;PG+采样k1 底=正确分支 |
| d2 `selectkd` | ✅ 一致(声明变体) | **[文] SpecKD 确证:损失只落 ACCEPTED**(top1∈teacher-topk),greedy 变体硬掩码=原文;β 软降权变体未实现且已声明 |
| d3 `teachability` | ✅ 一致(含记录下界) | 分歧×兼容、取高;兼容=交集下界(记录) |
| e1 `pl_rank` [自研] | ✅ 设计=实现 | [数] 后缀 logsumexp、失序单调;直接分支 |
| f1 `softlog` | ✅ 一致 | sign·log1p,奇变换=压 Δℓ;Demystifying 赢家式 |
| f2 `hard_clip` | ✅ 一致 | verl `loss_max_clamp` ±10 + 命中率面板 |
| g1 `verified_only` | ✅ 一致(家族成员) | [码] n=1 GRPO 单例特判→优势=verifier 原始分;缩放掩码≡过滤;RG-OPD 家族声明;2607.23731 必读可能加 caveat |
| g2 `fire` | ✅ **r4 修复后一致** | **[文] 原缺标题后半**:Eq.5–8 逐式实现([数] 0.0 对拍,逐轨迹 mean(w̃)=1);迁 top-k 路径取老师熵;微批种群/截断熵两偏离记录;PG 家族(FiRe 自身即优势式) |
| h1 `firstseg` | ✅ 一致(声明式形变) | 原文截 rollout(改采样分布),我们 loss-mask 形式 —— 防 A 轴混淆的**有意**形变,臂注声明 |
| i0 / i1 | ✅ 构造性 / 设计期 | i0=换打分器;i1 待手术 V1–V4 |

**底座级(r3)**:散度值臂(b2/c1/c2/e1/b3)→ 直接分支;k1 家族(含 D 轂底损失)→ PG。
**处置**:PG 误配的在跑 run(b2/c2/e1)与 g2 的 stage-1 旧排队 run,完结后一律迁
`*_ablation`,正名重发(指纹自动分批);c1 头条待直接路径复核。

---

# r5:逐臂"原文+原码"双对照(2026-08-07 起,按轴推进)

> r4 对的是论文原式;本轮加上**各论文自己的源代码**作第三方证词。每臂一张三方表。

## A 轴(rollout 来源与日程)

### vanilla(λ=1 锚)
构造性一致:损失=verl `kl_penalty(k1)` 本体(仅加面板);PG 形式=TM 博客与
Demystifying 的正典估计器(verl 注释自证血统)。A 轴的 λ=1 端点由它承担。

### a1_gkd_mix0.5 —— GKD (2306.13649) × TRL `trl/experimental/gkd/gkd_trainer.py`
| 项 | 论文 Algorithm 1 | TRL 正典实现 | 我们 (gkd_mix.py) | 判定 |
|---|---|---|---|---|
| λ 语义 | u≤λ → **student 生成**(λ=student fraction) | `random.random()<=lmbda` → 学生在环生成 | `SIMOPD_GKD_LAMBDA`=P(off-policy)=**1−λ_GKD**,映射注记在码 | ✅ @0.5 两义重合;≠0.5 需换算(已注记) |
| 掷币粒度 | **per-batch** | per-batch | per-(prompt, step) 确定性 hash | ⚠️ 记录偏离:期望同 λ,分层更细且可复现;每访新币的语义保持(同一 prompt 跨步可换分支) |
| off-policy 数据 | 固定 (X,Y):ground-truth **或 teacher 生成** | dataset batch;`seq_kd=True` 时 teacher 在环生成 | **预生成 teacher 响应缓存**(τ=1.0、seeded、学生模板) | ✅ 论文"固定 teacher 生成集"变体本体(TRL 的 seq_kd 是在环变体;我们数据无金标 CoT,teacher 生成是忠实可用项) |
| off-policy 损失 | 与 on-policy **同一散度** | 同一 `generalized_jsd_loss`(非 CE) | 同一 verl 蒸馏路径 | ✅ 结构一致 |
| 散度 | 菜单:FKL/RKL/JSD(.1/.5/.9),任务相关 | β 插值实现 | 协议 RKL(k1)= 菜单内选项 | ✅ 一臂一旋钮:λ 单独隔离,散度归 B 轴 |
| 优化器 | 直接反传 | 直接反传 | PG(sg-优势);单 epoch ratio≡1 | ⚠️ 记录偏离(estimator-note 论证,预注册) |
| 温度 | γ=1(学生在环采样) | `args.temperature` | rollout τ=1.0 | ✅ |

状态不变:needs(等 gen_offpolicy 预生成 + 3 步 GPU 彩排)。

### a2_coldstart —— Rethinking (2604.13016) × thunlp/OPD 官方库
| 项 | 论文 | 官方库 | 我们 | 判定 |
|---|---|---|---|---|
| 配方顺序 | off-policy SFT(teacher rollouts)→ OPD | vllm_rollout → LlamaFactory full-SFT → on_policy_distillation.sh | gen_coldstart → verl sft_trainer → OPD | ✅ 同序(SFT 载具不同=工程,非协议) |
| **SFT 数据过滤** | 未明说 | **`--enable-rejection-sampling true`** | verifier 拒采(--keep-all 留作消融) | ✅ **升级:从"通行解读"变为其官方库确证**;G 轴纪律(verifier 只滤不进输入)与其一致 |
| **SFT 题目与 OPD 隔离** | "**deduplicating against the SFT prompt subset**",OPD 用剩余 ~30K | 同 | reserved slice,OPD 用 remainder | ✅ **升级:我们的"防重看题"设计正是其原版做法** |
| 规模比例 | SFT 200K 响应 : OPD ~30K 题 | 同 | SFT ≤12k 响应(3000 题×≤4): OPD ~9.5k 题 | ⚠️ 记录偏离:档位总量 12.5k 题装不下其比例;配方形状保持,数量按档位缩放 |
| 学生 | 其发布 ckpt 名为 Qwen3-1.7B-SFT(1.7B 学生) | 同 | 1.7B-Base | ✅ 尺寸同档(巧合但可引) |


## B 轴(散度)

### b1_skew_kl —— DistiLLM (2402.03898) × 官方库 `jongwooko/distillm/distillm/losses.py`
| 项 | 论文 | 官方码 `skewed_reverse_kl` | 我们 | 判定 |
|---|---|---|---|---|
| 形式 | SRKL_α=KL(q‖αq+(1−α)p) | `mixed = lam*student + (1-lam)*teacher`,外侧=student,`lam=0.1` | logaddexp(s+logα, t+log(1−α)),外侧=student,α=0.1 | ✅ **逐符号一致**(侧向+数值) |
| 计算域 | 全词表和(离线) | 全词表 | 采样位单样本估计:外侧=q 恰是 rollout 分布,E_{y~q}[·]=SRKL **无偏** | ⚠️ 记录翻译(与 vanilla 的 k1 同类);SKL 不可这样估(外侧是 teacher,没人采样它)—— 这也是选 SRKL 入臂的原因 |
| 界 | — | — | 积分元 ≤ −log α(mix ≥ αq) | ✅ 附带成立 |

r5 顺带修正:我们旧 docstring 头行写 `SKL_a(p||q)` 而 DistiLLM 记号里 p=teacher,读者会误判为 skew **forward**(其混合大头在 student 侧,方向相反)—— α 侧向陷阱本审计的靶子,已改写并注明两码对齐。

### b2_forward_kl —— verl 原装(实现即出处)
✅ as-shipped:verl `forward_kl_topk`(截断 FKL + clamp₀,verl 自己的选择,已记录);r3 起走直接分支(verl 注释引 GKD)。无外部库可再对。

### b3_eopd_gate —— EOPD (2603.07079) × 官方库 `WLS04/EOPD` —— **判决翻转,r5 重写**
| 项 | 官方码(core_algos + dp_actor) | r3 版(错) | r5 版(现) |
|---|---|---|---|
| 组合方式 | **叠加**:`pg_loss = pg_loss + soft_kd_loss` | where(gate, fkl, rkl) 硬切换 | ✅ 叠加(b3_additive 包装器送过 PG detach) |
| RKL 底座 | **全 token 采样 k1 优势,PG 裁剪代理**(=我们 vanilla) | 分布式 renorm RKL,直接分支 | ✅ 采样 k1 + PG 分支 |
| 阈值 | **固定绝对值 0.8**(`soft_kd_entropy_threshold`) | 批分位 0.5 | ✅ 固定 0.8(SIMOPD_B3_ENT_THRESH) |
| FKL 形式 | top-k 截断,Σp(log p−log q),无 renorm 无 clamp,按全体有效 token 归一 | clamp₀ 版 | ✅ 原式(clamp₀ 是 b2/verl 的选择,不带入) |
| 老师熵 | 全词表(ref 通道预计算) | top-k 截断 | ⚠️ 仍截断(打分老师只给 top-k;只会低估→0.8 门少开,b3_gate 记实际比例,teacher_mass 记尾量)|

注:2026-08-06 的登记预埋了应变条款("switch-vs-add is the internal ablation if the
full text says otherwise")—— r5 执行的正是该条款。r3 把 b3 归直接分支系成文前推断,
estimator-note §7 已更正;b2/c1/c2/e1 的直接分支归属不受影响(各自论文确为直接优化)。
CPU 验证:kernel 逐位对拍(raw k1/截断 FKL/固定阈门控)、STASH 项 0.00e+00、
梯度流回 student_logits 且未门控 token 零梯度、包装器叠加与先清残留语义。

## C 轴(词表支撑)

### c1_lsm_topk32_renorm —— LSM (2603.25562) × 官方库 `hhh675597/revisiting_opd` × EasyOPD 参考实现
| 项 | 论文 | 官方码 | 我们 | 判定 |
|---|---|---|---|---|
| 方向 | Eq.8:π̂_θ 加权 log(π̂_θ/q̂) = **KL(student‖teacher)** | `student_probs_norm*(student−teacher)`,student 在外 | `kl_divergence(log_p=stu_n)`(verl 语义:log_p 在外) | ✅ 三方同向(EasyOPD on-policy 分支亦同) |
| 双侧 renorm | Eq.7 支撑集内独立 softmax | `log_softmax(ref_logits_k)` 双侧 | 双侧 logsumexp 归一;r4 对 EasyOPD 数值 5e-7 | ✅ |
| k | **32**(Table A1 默认,全实验) | 同 | 32 | ✅(EasyOPD 框架默认 256 是框架值,非论文值) |
| clip | — | `clip_log_ratio` ±5 **默认 False** | 无 | ✅(默认侧一致;其内部选项记录) |
| **优化器** | §3.2 行文读作直接反传 | **`compute_opd_advantage`:原始 −KL 作优势,无基线无白化,PPO 裁剪** | r3 曾改直接;**r5 回 PG(以码为准)** | ⚠️→✅ **论文≠代码,判决以码;r3 改判撤销** |

**连带**:cornell c1 PG run(0.598,无 Mode-A)恢复为忠实结果;直接版从必做重跑降为
可选 paper-form 消融。estimator-note §8 记方法论教训:r3 的先验批量改判,五臂里两臂
(b3、c1)的原作者恰恰发表了"病态"形状 —— 审计对象是文献做了什么,不是该做什么。

### c2_quantile_budget [自研]
无外部出处,审计=设计↔实现:分位预算自适应(6/9/12@8)r2 数值精确;直接分支为
r3 注册的自研选择,LSM 之发现不外推(自研臂的"忠实"即注册文本)。✅

## D 轴(token 选择)—— 三臂全有参数级修正

### d1_tip —— TIP (2604.14084) × 官方库 `HJSang/OPSD_OnPolicyDistillation`
| 项 | 论文 | 官方库 | 我们(r5 后) | 判定 |
|---|---|---|---|---|
| 分数 | soft-OR:s=ĥ+δ̂−ĥδ̂,**纯批级 min-max**(论文自陈离群敏感为局限) | **选择器不在库里**(库=均匀 token 基建 + 直接分布式 KL;另有 entropy 加权 multinomial 工具) | soft-OR + 纯 min-max;**p98 熵裁剪删除**(r5:那是我们替被审方法悄悄修 bug) | ✅ 修正后一致(论文为准) |
| 选择 | 确定性 top-ρ,**ρ=0.5 主配置** | — | 同 | ✅ |
| 底损失 | ℒ=1/|T|·Σ D_KL(P_S‖P_T)(分布式直接) | 直接分布式 | 协议采样 k1-PG + 按选中数 rescale(≡1/|T| 归一) | ⚠️ 记录翻译:一臂一旋钮,受审的是**选择器**;归一等价 |

### d2_selectkd —— SelecTKD (2510.24021),无官方码(截至 2026-08-07)
| 项 | 论文 | 我们 r5 前 | r5 后 |
|---|---|---|---|
| 接受规则 | student argmax ∈ teacher top-**k=5**(默认且消融最优) | ∈ top-**32**(载荷宽度,窗宽 6×,TAR 虚高) | ✅ 前 5 列(载荷有序,可切片) |
| 拒绝 token | **β=0.01 降权**(默认;掩码是变体) | 硬掩码 + rescale | ✅ V_t∈{β,1} 直乘,全批归一,无 rescale |
| 目标 D | KL 族不限(KL/RKL/SKL/SRKL) | 采样 k1 RKL | ✅ 族内成员(声明) |
| Spec-k 变体 | 学生采 k 个候选按 min(1,p/q) 验收 | 未实现 | 声明未实现(greedy 变体臂) |

### d3_teachability —— TA-OPD (2605.26844) × 官方库 `wyy-code/TA-OPD`(tip_compat.py)
| 项 | 论文+官方码 | 我们 r5 前 | r5 后 |
|---|---|---|---|
| 归一 | **batch_quantile 默认:clip((z−Q05)/(Q95−Q05),0,1)**(码:`opd_metric_q_low/high`=0.05/0.95;min-max 只是备选) | 纯 min-max | ✅ `_robust_norm` |
| 兼容性 | teacher 质量落在 student **top-16**(默认;扫 8/16/32);码 `Cmass=Σ t_probs[s_ids]` | student top-32 | ✅ top-16(SIMOPD_TEACH_K);交集下界告诫保留 |
| 预算 | **论文推荐 5%**(其多 seed 聚合脚本名即 ratio005;码内框架默认 1.0=不选) | 50%(与 d1 预算对齐的自选) | ✅ 0.05 —— 忠实优先于配平,账本报实际预算 |
| 底损失 | 码走 slime 的 KL-as-advantage(PG);**论文明文认可采样位 k1 形式** | 采样 k1-PG | ✅ **忠实,非翻译** |

CPU 验证:_robust_norm 手算、d2 β 权重直乘 + TAR@5<TAR@载荷宽(窗变严实证)、
d3 5% 预算 + 按选中数 rescale、d1 无裁剪 minmax,全部逐位对拍;shadow 面板适配
selectkd 的权重集(≥1 为其集合)。
