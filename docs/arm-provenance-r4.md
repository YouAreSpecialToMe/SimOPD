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

## E 轴(支撑内对齐目标)

### e1_pl_rank [自研] —— PL 机械源 PLD (2506.12542, NeurIPS25),r5 修正引用口径
| 项 | PLD 原式(其 §4.2) | 我们 | 判定 |
|---|---|---|---|
| PL 似然 | −s_{π*_k} + log Σ_{ℓ≥k} e^{s} 后缀式 | 同一后缀 logcumsumexp 机械 | ✅ 这一层才是"源自 PLD"的部分 |
| 项加权 | **每项 × 教师置信 q^T**(其核心贡献,借此包摄 CE) | **无权** —— 置信加权会把 value 混进 rank 项,而本臂设计点是 rank/value 分离 | ⚠️ 有意不引入,登记声明 |
| 排列 | **真标签置首** + 教师 logit 降序 | token 级无标签;纯教师序(top-1 自然居首) | ⚠️ 不可用,声明 |
| 归一 | 无 | ÷k(跨臂损失尺度可比) | ⚠️ 我们的选择,声明 |
| 组合 | 独立成损失 | + value-KL 锚(系数=内部消融) | ⚠️ 自研添加,防火墙声明 |

处置:desc 从 "form from PLD" 改为 "PL machinery as in PLD" —— 借的是后缀 logsumexp
似然机械,不是其损失。四处分歧全部入册;实现与既有数值验证(失序单调 0.34<1.10<2.84)
不变;直接分支属自研注册选择,维持。防火墙声明(自研臂同协议受审)照旧。

## F 轴(信号调节)—— 源 = Demystifying 2607.13399(协议锚;无官方码,其实现为 verl)

### f1_soft_log
| 项 | 论文 | 我们 | 判定 |
|---|---|---|---|
| 公式 | Δℓ̃ = sign(Δℓ)·log(1+|Δℓ|),施于逐 token 优势 | 同式施于 k1 损失(=优势取负) | ✅ 变换为奇函数,与取负可交换,严格等价 |
| "赢家"口径 | **差距适中时**优;差距大时硬裁优 | 我们档位 2.4× 适中 | ✅ 条件性入 desc(原 desc 无条件,已收紧) |
| 面板 | — | raw/compressed 双记录 + shrink_ratio | ✅ 附加可见性 |

### f2_hard_clip
| 项 | 论文 | 我们 | 判定 |
|---|---|---|---|
| 公式 | clip(Δℓ, c_min, c_max),**未给数值** | verl loss_max_clamp,±10 对称 | ⚠️ **±10 是我们的注册选择**(登记已明示;clip_hit_rate 让咬合度可见,不作先验辩护) |
| 施加对象 | 优势 | 损失(对称裁剪为奇函数,同 f1 论证) | ✅ 等价 |
| 进入条件 | 差距大时的补救 | 预注册为 D5 见 Mode B 才入,因 stock 顺带冒烟 | ✅ |

两臂实现零改动;本轮修正均为登记口径(f1 的条件性、f2 的阈值归属)。

## G 轴(轨迹门控)

### g1_verified_only —— r5 收窄口径:[自研朴素成员],非 RG-OPD 实现
| 项 | RG-OPD(其 Eq.2 + `UoC-tail/RG-OPD`) | 我们 | 判定 |
|---|---|---|---|
| 规则 | **方向对齐门**:A>0 且 L_T>L_S+δ 才蒸;A≤0 且 L_T<L_S−δ **也蒸(负教学)** | 过验证全保,未过全弃 | ⚠️ 非其方法 —— desc 改为"G 轴朴素正典成员,最近发表亲属 RG-OPD(规则不同)" |
| 底损失 | top-50 RKL + 尾修正 | 协议 k1 | 同上,不同臂 |
| 信号 | GRPO 优势 | verl n=1 GRPO 单例→verifier 原始分(断言防换) | ✅ 机制同族 |

**必读账结清(2607.23731 Outcome-Confounded Local Supervision)**:outcome 级过滤
不定位 token 级信号——过滤后仍有 ~68% 响应 token 质量属"失败中同意"。措辞红线入
`verdict.py CAVEATS`:g1 判决只能说"轨迹选择有益/有害",**永不声称"信号纯化"**。

### g2_fire_likelihood —— FiRe (2606.02684) × 官方库 `YuYingLi0/FiRe-OPD`(dp_actor.py + 启动脚本)
九项对照:八项吻合 —— 过滤统计 sum/len、`traj_skip_percentile=20`、cT=(1−H_T/max)、
cS=H_S/max(均 valid-token max)、`entropy_alpha/beta=1.0`、w detach 后乘 RKL 优势、
PG+PPO 裁剪(`only_reverse_kl_advantages=True`)、全词表老师熵(我们截断=已记录偏离)。
第九项**论文≠代码**:Eq.8 逐轨迹均值归一 vs 码内**全体训练 token 均值归一**——照 c1
先例以码为准,r5 已改([数] 0.00e+00 重验;逐轨迹均值不再强制为 1,均匀自信的轨迹
保有 >1 权重)。其 rollout-IS 修正(threshold 5.0)属其基建,不入臂。

## H 轴(监督窗口)

### h1_first_segment —— ESR "Less is More" (2605.27028),无官方码
| 项 | ESR 论文 | 我们 r5 前 | r5 后 |
|---|---|---|---|
| 窗口 | **N=100 默认**,稳健区间 50–200(其老师中位输出 ~1150 → 约 9% 的切割) | K=512 —— 远超其测试域,对本档位响应长度几乎不切,**臂近乎失效** | ✅ K=100(字面引进默认值,d3/d2 同例);firstseg_covered_frac 报本地覆盖率 |
| 形式 | **rollout 截断**(生成停在 N,改采样分布 + 省 rollout 成本) | loss-mask(生成不动,只动监督窗) | ⚠️ 我们的声明式形变(防 A 轴混淆);**其 §5.3 确认从未测过 loss-mask 变体** —— 判决措辞:测的是"固定分布下的监督局部性",非 ESR 全配方 |
| 底损失 | 标准 OPD 反向 KL | 协议 k1(同族) | ✅ |
| 归一 | 窗内求和 | 按保留数 rescale(≡窗内均值,防学习率混淆) | ✅ 等价类,登记声明 |

**处置**:在跑的 h1(K=512)完结后随 migrate_stale 迁移(已入默认名单),K=100 版正名重发。

---

# r5 封卷总表(2026-08-07,八轴全毕)

| 臂 | r5 结果 | 一句话 |
|---|---|---|
| vanilla | ✅ 构造性 | verl kl_penalty 本体;A 轴 λ=1 锚 |
| a1_gkd | ✅ 双证 | 论文 Alg.1 + TRL;缓存=论文固定 (X,Y) 本体;掷币粒度/PG 估计=记录偏离 |
| a2_coldstart | ✅ 升级 | 拒采 + 题目隔离经官方库确证为其原版;规模比例=记录偏离 |
| b1_skew_kl | ✅ 逐符号 | 官方 skewed_reverse_kl 同侧同值;采样估计=无偏翻译;记号陷阱修正 |
| b2_forward_kl | ✅ as-shipped | verl 即出处;直接分支(verl 自家 GKD 引文) |
| b3_eopd_gate | 🔄 **翻转重写** | 官方码=PG 底座+叠加 FKL、固定阈 0.8;b3_additive 送项过 detach |
| c1_lsm | 🔄 **翻转复活** | 论文≠代码 #1:官方码原始 −KL 作优势=PG;cornell run 复为忠实结果 |
| c2_qb [自研] | ✅ | 设计=实现 |
| d1_tip | 🔧 参数 | p98 裁剪=我们无据代修,删;官方库无选择器,论文为权威 |
| d2_selectkd | 🔧 参数 | 验证窗 32→5(TAR 虚高源)、硬掩码→β=0.01 降权;无官方码 |
| d3_teachability | 🔧 参数 | 官方码 Q05/Q95 归一、K=16;预算 50%→论文推荐 5% |
| e1_pl_rank [自研] | 📝 引用 | "form from PLD"→"PL machinery";置信加权等四分歧=设计声明 |
| f1_soft_log | 📝 口径 | 施加侧奇函数严格等价;"winner"条件化(适中差距) |
| f2_hard_clip | 📝 归属 | 论文无数值;±10=我们的注册值,clip_hit_rate 作证 |
| g1_verified_only | 📝 收窄 | 非 RG-OPD(其为方向门+负教学);2607.23731 措辞红线入 CAVEATS |
| g2_fire | 🔧 参数 | 论文≠代码 #2:Eq.8 逐轨迹归一 vs 码内批均归一,以码为准;余八项吻合 |
| h1_first_segment | 🔧 参数 | K 512→100(ESR 默认;原值对本档近失效) |
| i0 | ✅ 构造性 | 换打分器 |
| i1 | 设计期 | 待手术 V1–V4 |

**统计**:2 判决翻转(b3、c1)· 5 参数级修正(d1/d2/d3/g2/h1)· 4 登记口径修正
(e1/f1/f2/g1)· 2 论文≠代码裁决(c1 优化器、FiRe 归一;b3 的 "augmenting" 歧义
亦由码定)· 2 出处升级(a2 两项自选变原版、d3 底损失获论文明文认可)。

**方法论沉淀**(论文 discussion 素材):(1) 论文与其官方代码相左时以码为准 —— 数字
是代码生产的;本审计三次用到该准则,方向各异。(2) 审计者不得代被审方法修 bug
(d1 的 p98)、也不得替其温和化(b3/c1 的"病态"恒负优势是发表形式)。(3) 引用
只许声称借用的那一层(e1、g1)。(4) 参数的字面引进是默认约定(d2 k=5、d3 5%、
h1 N=100),本地语境差异交由面板报告。

**执行侧**:陈旧名单 = {b3, g2, d1, d2, d3, h1}(migrate_stale 默认);跑完迁移后
daemon 自动以忠实配置重发。cornell 零动作(c1 已复活)。i1 手术与 a1 彩排照旧排期。
