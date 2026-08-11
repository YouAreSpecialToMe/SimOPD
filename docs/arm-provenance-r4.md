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

## r5 增补(2026-08-07):g3_kdrl —— 覆盖对表补的第 20 臂

来源:survey 2606.22793 "outcome/verifier coupling" + nick7nlp §4.3 RL-augmented
objectives 与我们 8 轴 diff 后的唯一可审计缺口 —— G 轴原来只有"奖励作过滤"(g1),
缺"奖励作共目标"。

### g3_kdrl —— KDRL (2506.02208),无官方码(记录)
| 项 | KDRL 论文 | 我们 | 判定 |
|---|---|---|---|
| 目标 | J = J_GRPO − β·KL^k2 | verl combine 分支:policy_loss + coef·distill(use_task_rewards=True) | ✅ **同构,零新机械** |
| KD 项 | k2 = ½R²(R=log π_T−log π_θ),学生 rollout 上**直接反传**;消融 k2>k3 | `k2_kdrl`(带 Δℓ 面板的 k2,面板取带符号 k1)+ USE_POLICY_GRADIENT=False | ✅ [数] 逐位手算 |
| β | **固定 2e-3 主默认**;退火 5e-3→1e-3(δ=5e-5/步)为变体 | DISTILLATION_LOSS_COEF=0.002;退火=预注册内部消融 | ✅ 字面引进默认 |
| RL 项 | GRPO,规则奖励,组采样 | 协议 n=1 → GRPO 组基线退化为原始 verifier 分(与 g1 门控同信号) | ⚠️ 记录偏离(协议级) |
| regime | R1 系长 CoT(DeepScaleR-1.5B) | 审计档位 | ⚠️ 每臂皆然,记一次 |

与 g1 构成 G 轴的干净对照:**同一 verifier 信号,滤 vs 加**。campaign.tsv 第 3 波
any 行入列;run_opd_baseline.sh 新旋钮(use_task_rewards / distillation_loss_coef)
入指纹。注册表现为 20 臂(18 可跑)。

## r5 增补修订(2026-08-07 同日,两次用户裁定):g3 → **j1_kdrl,J 轴,n=8 迷你 cell**

用户两问定型:(1) "n=1 的 RL 项对吗" —— 不对到值得改设计:真 GRPO 在单例组优势恒 0,
verl 特判后 RL 项退化为"只奖成功、失败零梯度",KDRL 奖励信号的负压半边消失。忠实微格
从 V 波触发**前置为即刻执行**。(2) "不一定归 G" —— 对:G=奖励**过滤**蒸馏信号,
KDRL=奖励作**独立共目标**,增设 **J 轴(奖励耦合)**;g1↔j1 跨轴构成"滤 vs 加"对照。
改名发生在任何 run 存在之前(不违"永不改名"——该规则护的是工件)。

**Cell 设计**:{vanilla_n8, j1_kdrl},n=8 × 32 prompts = 256 seqs/步(~2× 主表
token/步,~28–30h/泳道);cell 内一旋钮(目标),n 偏离压在 cell 边界;
verdict.py 新增 BASE_OVERRIDES:j1 对 vanilla_n8 判,vanilla_n8 自身对 vanilla 判
(白捡组采样旋钮在纯 OPD 下的读数 —— Demystifying 的 prompt-diversity 主张顺带受审)。
ROLLOUT_N 入指纹。注册表 21 臂(19 可跑)。


---

# r6 终审封卷(2026-08-07):六路对抗性全代码审查

> 范围 = 全仓每一行(损失栈/注入接缝/发射编排/测量栈/数据预生成/配置文档),
> 六个独立审查代理并行,发现须附具体失败场景与复现;修复分六批提交
> (6c3889f / 3c03ff8 / ddef03a / 96d8d43 / 83c52cd / 本批)。完整证词在各批
> commit message;此处只录**若不修会怎样**的账:

**发射拦截级(任一即毁战役)**:_lane 吞掉臂拒绝→搁置臂以 vanilla 冒名记 OK;
LANE_TAG 入指纹→一切断点续跑必熔断;MAX_CKPT_KEEP=2→套件曲线物理不存在;
16-token 前缀键在真数据 635 组碰撞(252 组答案冲突)→a1 预生成必崩/若删断言则静默串题;
a2 SFT 逐行必崩(模板 think 块)→或用忽略开关则条件化错位;
套件 32,768 预算在学生 ckpt 上引擎拒启;verdict 在噪声底缺席时先发 PROMOTE。

**臂判决级**:哑教师行毒化 TIP 归一(d1 退化纯熵选择,shadow_tip 同污);
批统计种群含 prompt(d3 的 5% 预算量错种群);FiRe 过滤在 16k 单序列微批恒不触发;
b3 叠加项微批归一→有效系数 ~128× 且随长度漂移、且首微批静默丢项(未接线);
Δℓ 跨臂面板六臂尺度污染;工件无时代标识→mtime 决定哪个数进论文(fixture:
touch 一个旧文件,suite_acc 0.449→0.283 无警告)。

**过程之误(审计的审计)**:zmq marker 名笔误被幂等测试当场抓获;S8 变量名笔误
被复查抓获;两次 replace 断言反写导致的静默空转被逐条落盘纪律终结——
与被审代码同样的教训:**验证行为,不验证标签**。

**遗留待用户拍板**:W 波上限(8k 守约 vs 16k 重推,EXPANSION-PLAN 已挂旗)。
**遗留执行项**:a1/a2 预生成按新键/新数据集重跑(cornell 两个陈旧 job 已撤);
__pilot8k 迁移程序(含旧指纹 RESUME=force 一次性记录);16k 单泳道 probe。

## r6 增补(2026-08-07):f3_power —— F 族有界成员由替转审

用户裁定将 F 谱系的替补(casefile 原记"Box–Cox 有界变换")提为正式臂。审计(仅论文;
摘要链接 EIT-NLP/PowerOPD 但仓库未公开,记录同 d2/h1):

| 项 | PowerOPD 论文 | 我们 | 判定 |
|---|---|---|---|
| 形式 | **r^α = sg[π_T^α − π_θ^α]**(α 次幂概率差,非 log-ratio 的变换!)天然有界 [−1,1];α→0 极限 = log-ratio | `exp(α·s)−exp(α·t)` 取负作优势,[数] 逐位 + 极限验证 | ✅ |
| 优化器 | **原式即 PG 奖励**(∇J=Σ r·∇log π) | PG 分支 | ✅ 构造性忠实 |
| α | **无默认**:按每指标挑 {0.1…500} 最优(多重比较,记 caveat) | **钉 α=1**(族正典成员 r=p_T−p_θ,零任意性);SIMOPD_POWER_ALPHA 预注册消融 | ⚠️ 审计选择,已声明 |
| 记录偏离 | lr 5e-7 / rollout ≤1,024 tok / 1.5k 步 | 协议 1e-6 / 16k / 250 定点 | ⚠️ 逐条入 note |

面板纪律沿用 F6(Δℓ 取原始 k1);`power_dead_frac` 监测高 α 死 token 比例。
注册表 **22 臂(19 可跑)**;S 波 57 runs / 126 卡(a1 入列 132)。

## r6 增补(2026-08-07):后补批添 b4_jsd + b5_k2(B 轴补全)

**b4_jsd**(文献补席,计划 §2 的 B 轴承诺兑现):GKD 广义 JSD-β,TRL 正典插值式
(r5 A 轴时已逐行核:m=βT+(1−β)S,jsd=β·KL(T‖m)+(1−β)·KL(S‖m));β=0.5 对称正典点,
renorm top-k 双侧分布式(教师外侧半边不可采样估计;截断记录),直接分支(GKD)。
[数] 逐位 0.0、JSD∈[0,ln2]、β=0.5 对称、**缩放极限**数值验证(jsd/β→FKL、
jsd/(1−β)→RKL;端点本身退化为零——首版 note 把两端写反,被自己的极限测试当场纠正)。

**b5_k2**(【自研】防火墙臂,估计器阶梯):k2=½(log-ratio)² 独立成目标,直接分支,
零新码(复用 k2_kdrl 模式、无 task rewards)。动机即本战役方法论主线:r3/r5 最大
修正类全是"估计器/优化器形式",而名册从未单独变动过估计器。k1(vanilla)↔k2(b5)
同目标异估计器,任何判决差即估计器效应;k3(GRPO 默认)留替补。KDRL 消融 k2>k3
为其主张来源。

后补批现员:a3(门控)+ b4 + b5(就绪);全并行 +18 卡,不入首发算术。
注册表 **25 臂(21 可跑)**。

## r6 增补(2026-08-07):后补批添 c3_intersection + c4_pi_tail_budget(C 轴补全)

**c3_intersection**(thunlp/OPD 官方码逐行锚,本地 clone):其 `intersection` 策略 =
逐候选优势 A = −(S−T)·w̃(w̃ = 交集上重归一学生概率;默认 reward_weight_mode=
student_p、kl_estimator=k1),3D PPO 代理在其自身代码于 ppo_epochs=1 时化简为
**L = −Σ sg(A)·log π** —— 我们实现该化简形于直接分支(忠实其出货路径;协议单
epoch 下两形等价)。交集集合对称,教师侧载荷即够(only_stu/union 需其双前向架构,
接缝外声明出局)。[数] 化简式逐位、空交集零损失。机制上是 Rethinking overlap
甜区主张的探针。

**c4_pi_tail_budget**(【自研】头条构造臂):把头条定理的误差控制量 π(S̄) 直接做成
预算旋钮 —— 逐 token 取教师秩序最小前缀使学生质量 ≥ 1−ε(ε=0.05),其上做 c1 式
renorm RKL,直接分支。[数] 预算/π-tail 手算、**最小前缀性**、ε→1 预算→1、ε→0
记 miss。c4_budget / c4_pi_tail / c4_eps_missed 三面板把定理量变成一等公民序列。
与 c2 之别:c2 批阈值钉平均预算,c4 逐 token 钉误差项本身。

后补批现员五名:a3(门控)+ b4 + b5 + c3 + c4(就绪);全并行 +30 卡。
注册表 **27 臂(23 可跑)**。

## r6 增补(2026-08-07):d1 分解旋钮落地 + D 轴影子扩展挂账

**已落**(52933b8):`SIMOPD_TIP_MODE ∈ {soft_or, entropy_only, divergence_only}`,
默认 soft_or(d1 本体分毫不动)。两个消融配置预注册:entropy_only@ρ=0.5 对 TIP
"熵保留 50% 平全量"之主张,divergence_only@ρ=0.1 对其"自信错 <10% 近平全量"。
零新 kernel,走 SIMOPD_ 指纹捕获自动分批。[数] 三模式选择集逐位手算吻合;fixture
内熵支/散度支 Jaccard=0.33(分解非同义反复)。d1 判决升维:soft-OR 行不行 →
哪一支在扛。消融 run 属 V-wave 机制波,不入主判决表。

**挂账**(用户裁定"2先加代办事项",任务单 #7):五篇【替】选择器(SEAD 2606.28562、
Evidence 2606.22830、Rock 2605.09253、Blockwise 2606.24084、Position-Bias
2606.22600)r5 式快读 → 判据可由教师 top-k 载荷 + 学生 logits 零额外前向算出者,
入 `_shadow_panel` 只记不训;V-wave 触发器(vs d1/d2/d3 Jaccard>0.8)免费出冗余
判决,正交者为晋升候选(晋升=修正案,非自动成臂)。不可算者出局并记因。

## r6 增补(2026-08-07):后补批添 e2_set_coverage + e3_zvalue(E 轴阶梯补全)+ 全臂 τ 面板

E 轴的独特结构是支撑内信息阶梯:全值(c1)> 仿射不变值 > 全序(e1)> 集合成员。
两个空档一次补齐,并把阶梯本身变成全臂可测的量。

**e2_set_coverage**(【自研】由替转审——casefile 挂【替】多时的 set-coverage,
关底档):损失 = −log Σ_topk π + PL_ANCHOR_COEF×value 锚(与 e1 同一 env 同一
系数:e1↔e2 把"序→集合"孤立为唯一动项,锚登记为 E 轴公共加项)。结构项 =
−log(1−π_tail):c4 把定理量当面板,e2 把它当损失(跨轴呼应入臂注)。对支撑内
分布有意无差别——若训不动或塌掉,那本身就是"成员信息不足以驱动 OPD"的判决。
[数] 组合式逐位、退化下(全质量压单支撑 token)损失恰余锚项。直接分支。

**e3_zvalue**(【自研】中间档;z 形式源 Logit Standardization KD,2403.01427,
CVPR24 highlight,视觉离线域,形式源非审计对象):支撑内两侧 z-分(其"z=自适应
温度"预处理)后走轴锚 KL 形。保间距比例、弃平移与尺度;z(log-prob)≡z(logit)
(逐 token 常数平移,std 平移不变),top-k log-prob 载荷即够。无锚(结构项本身
是 value 匹配器,生值锚会把尺度走私回来;不对称入册:跨档读法 c1 纯生值↔e3 纯 z、
e1↔e2 同锚)。判读三分支:c1≈e3≈e1→序数够;c1≈e3>e1→间距比例在扛(自适应
温度党赢);c1>e3→绝对尺度不可弃。[数] 手算逐位、仿射匹配损失恒零(档语义
不变量)、乱序显著为正。直接分支。

**全臂免费面板**:`rank_kendall_tau`(学生序 vs 教师序,tau-a,+1=已对齐)入
`_overlap_diagnostics` 与 c1 的 inline 站点(该臂绕开共享函数,漏加即哑——站点
注释早有预警,本次双站点数值对拍)+ OVERLAP_KEYS 传递。与既有 student_mass
(覆盖质量)合读:每个 top-k 臂上"序先收敛还是质量先收敛"直接可测——E 轴判决
的跨臂旁证,机理面板新一等公民。

**出局声明**:成对边际族(RankNet/BiLD 类)——夹在 z-值与全序之间,与两邻档
分离度低;top-1 argmax CE——极端锐化端点,协议指标下熵塌/长度混淆风险,无 OPD
主张持有者,且 e1 的 top-1 居首性质已部分覆盖其动机;PLD 置信加权——r5 已登记
的有意不引入(value 混入 rank 项毁分离设计)。

后补批现员七名:a3(门控)+ b4 + b5 + c3 + c4 + e2 + e3(就绪);全并行 +42 卡。
注册表 **29 臂(25 可跑)**。跨轴对照网:c1↔e3↔e1↔e2 同支撑(k=32)四档、
c4_pi_tail 面板 ↔ e2 损失同量。

## r6 增补(2026-08-07):G 轴路由族补全 —— FIRE_MODE 旋钮 + g4 镜像 + g5 旗舰

G 轴地图按路由函数 r(A, L_T, L_S) 重画后,在审成员全是复合体或家族边缘:g2 滤×权
合体,g1 只是 RG-OPD 家族的朴素正成员(r5 收窄),旗舰本尊无臂。三件补齐:

**SIMOPD_FIRE_MODE ∈ {both, filter_only, reweight_only}**(d1 旋钮同款,零卡):
filter_only 保滤置 w≡1,reweight_only 保权全不弃;归一化种群随动(全保时=全响应
token)。g2 判决升维:标题的哪一半在扛。[数] 三模式关系式逐位(纯滤=回放式、
纯权=无回放系数、both=复合)+ 非法模式拒斥。g2 env 显式登记 both。

**g4_failure_only**(【自研】镜像):k1_verified_only 谓词反转(keep = Σadv≤0),
同内核纪律(缩放掩码、验证答案不进输入、优势断言)。符号家族三点括号
{g1:+, g4:−, vanilla:全}。全过验证批无更新,与 g1 全败批对称,gate_keep_frac 可见。
2607.23731 红线镜像同守(CAVEATS 双登记)。[数] 与 g1 保留集互补、空批零、回放系数
逐位。PG 分支(k1 家族)。

**g5_rgopd_gate**(家族旗舰入审,RG-OPD 2607.04037):Eq.2 原样 —— 轨迹级门
g=1[(A>0∧L_T>L_S+δ)∨(A≤0∧L_T<L_S−δ)],L 为掩码**求和** logprob,δ=0 其默认
(论文无 δ 消融;SIMOPD_RGOPD_DELTA 字面引进)。负教学支 = **同一目标只做门控**
(其 §3 明文 same reverse-KL objective;r5 账"也蒸"读法坐实)。one-knob 纪律照 D 轴
注册:门是臂,其 top-50 RKL+尾修正底不搬(g5-vs-g1/g4 审规则不审底)。
rgopd_pos/neg_kept_frac 面板拆负教学份额。[数] 四象限真值表、δ=0 边界(gap=0 严格
不等号全弃)、回放一致。PG 分支。**源况**:论文正文取到(fetch 2026-08-07);
`UoC-tail/RG-OPD` 库 404(论文所引地址),门规则以论文 Eq.2 为锚已足,库若复活
补一次对码。**编号**:g3 已焚(g3_kdrl→j1_kdrl 改名史),名册永久跳过。

SG-OPD(2606.09304,注意其含相位教师采样第二组件)与 ReNIO(2606.23104)维持
【替】,待 r5 式快读定夺(用户未开闸,不入挂账任务)。

后补批现员九名:a3(门控)+ b4 + b5 + c3 + c4 + e2 + e3 + g4 + g5(就绪);
全并行 +54 卡。注册表 **31 臂(27 可跑)**。G 轴对照网:{g1,g4,vanilla} 符号括号
⊂ g5 方向门,g1↔j1 滤加对照跨 J 轴不变。

## r6 增补(2026-08-07):H 轴证伪括号 —— h2 镜像 + h3 预算对照 + k1 族位置面板

h1 单臂的结构性混淆:若前 100 token 监督追平全量,分不清**位置论**(ESR 的主张,
信号真集中前段)与**预算论**(伪影,随便 K 个 token 都够)。括号拆之:

**h2_last_segment / h3_random_segment**(【自研】,同建 _window_kernel):h1 内核
换窗谓词——h2 取末 K(变长右端点,response_mask 长度定位),h3 取均匀随机偏移
连续 K 窗(on-policy 轨迹每步全新,无持久指派可保,逐轨迹逐微批抽,run 种子定,
登记)。括号纪律:**共享 SIMOPD_FIRST_SEGMENT_K=100**(预算配平 by construction,
D_RETENTION 同款)、同连续性、同回放归一、同 F6(delta_ell 面板记未回放 raw)——
位置是唯一动项。判读:h1≈full 且 h2/h3 低 → 前段集中坐实;三者≈full → 预算伪影
(位置论死、杠杆活);h1≪full → loss-mask 形直接证伪;h2≈full>h1 → 反向集中。
短响应(len≤K)全员同塌全监督,covered_frac 各报。[数] h2 谓词/回放逐位、h3 窗
连续/界内/大小/种子复现/偏移覆盖、h1 回归不动。PG 分支(k1 族)。

**位置面板**(零卡,收口点 `_signal_quantiles` → k1 族 9 调用点 + top-k registry
全覆盖):位置分箱 mean|signal|,箱缘 [0,100)/[100,500)/[500,2k)/[2k,∞),首缘恰
= 括号窗宽。vanilla 的曲线即"前段集中"主张的直接测量;门控臂传入 kept 种群,
面板语义 = 该臂受训种群的位置谱(登记)。空箱不发射。位置 = 填充列索引,h1 约定
同源(响应张量右填充,0 列 = 首响应 token)。

**边界入册**:Are Full Rollouts Necessary(2605.31490)与 ESR 真形同属生成视界类
(截 rollout 改采样分布)——若升审必须仿 n8 前例开自带对照迷你 cell,不进主表。
出局:结构分段(需数据侧掩码)、课程化 K(调度域)、位置衰减软权(Prefix-fade
落点,"可取未取")。ADWIN/Prefix-Guided/Prune-OPD 维持【替】待读。

后补批现员十一名:a3(门控)+ b4 + b5 + c3 + c4 + e2 + e3 + g4 + g5 + h2 + h3
(就绪);全并行 +66 卡。注册表 **33 臂(29 可跑)**。

## r6 记录(2026-08-07):J 轴补全审查 —— 用户裁定有意不扩

八轴补全扫描(A/B/C/D/E/F/G/H 均已过)轮至 J 时,用户裁定到此为止,理由是成本
结构:J 臂依 cell 纪律必须带 n=8(4 卡泳道 ×3 seed,单臂 = 常规臂两倍卡),而
j1(KDRL 统一目标)+ vanilla_n8 对照已足以回答"奖励作独立共目标是否值得"的一阶
问题;g1↔j1 的跨轴"滤 vs 加"对照不受影响。J 轴扩员(如 β 扫描、其他奖励耦合形)
= 修正案域。入册为有意划界,非覆盖缺口。

**八轴补全扫描就此收卷**:D(TIP_MODE 旋钮)、E(e2/e3 + τ 面板)、G(FIRE_MODE +
g4/g5)、H(h2/h3 + 位置面板)本轮新增;A/B/C/F 前轮已补(a3/b4/b5/c3/c4/f3)。
终账:**33 臂(29 可跑)**,后补批 11 员 +66 卡,三个预注册分解旋钮
(TIP_MODE/FIRE_MODE/JSD_BETA)+ 两个全名册免费面板(rank_kendall_tau、位置分箱)。

## r6 记录(2026-08-08):S 波本地分工终版 —— 每机 4 臂,vanilla 出让远程

用户两连裁定:本地 24 卡不跑 n8 cell(缓期 wave 8 hold)、不跑 vanilla("协同者跑
vanilla 我们跑臂"),每机 4 臂满泳道。终版:m1=d1/d2/d3+b3,m2=f1/f2/f3+b1,
m3=g1/g2+h1+e1(12 臂×3=36 run);远程主池=vanilla+b2/c1/c2,后补池 10 臂不变。
方法论处置:判决配对发生在分析期,地板晚于臂不构成问题——但**地板在别的集群**构成
c1 类跨集群比较。入册:本地 12 臂的判决在"本地 vanilla 回补"(3 run,泳道腾空或
新卡到位即除)之前一律带跨集群 caveat;认领 vanilla 的协作站点以其为自家地板。

## r6 复审(2026-08-09):A 轴 —— 一条 r5 结论撤销,一条 r4 论证撤销

r5 之后 A 轴代码大改(全前缀键、a2 自定义 SFT 数据集、16k 修正案)且 a3 从未逐项
对表,故整轴重审。源:GKD 正文(HTML 取)、Rethinking 正文、thunlp/OPD 本地 clone
(`scripts/infer/vllm_rollout.py`、`LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml`)。

### 撤销 #1(r4 的 seam 注,非 r5 表):"PG 与直接反传梯度重合" —— **不成立**

**归属先说清**:r5 的 a1 表那一行本就把优化器判为 ⚠️ 记录偏离(论文=直接/TRL=直接/
我们=PG),没有错。过头的是 **r4 写进 arms.yaml `seam:` 的那句**"the ratio is
identically 1 and the gradients coincide"——宽容读法(裁剪代理≡未裁剪 PG)成立但与
GKD 无关,字面读法(≡GKD 的直接损失)不成立。撤的是后者。

**锚点复核(以码为准的规矩要求)**:r5 的锚是 TRL `trl/experimental/gkd/
gkd_trainer.py`,本轮重读该文件:掷币 `random.random() <= self.lmbda` 在
`training_step()` 内**逐 batch**;`generalized_jsd_loss` 在 **(B,S,V) 全词表 logits**
上计算;loss **直接反传**,无优势/无 stop-grad/无 PG;off-policy 分支用**同一个** jsd
(非 CE);从不 detach 采样分布。GKD 正文("we do not backpropagate through the
student's sampling distribution"、逐 batch 掷币)与之完全一致。**两证同向,故本撤销
不是"以论文压码",而是由码加固。**

我们是采样点 k1 + PG。
两者不是同一估计器:direct = mean ∇log π_S,PG = mean(ℓ·∇log π_S)。且对采样点损失,
direct 分支会把采样 token 的概率**往下压**——所以 PG 是采样估计器下唯一合理选择,
但"梯度重合"这句必须撤。
**连带的实质性后果**:采样点 k1 的方向随**采样者**而定。on-policy 时 E[ℓ]=RKL;
off-policy(教师采样)时 E[ℓ]=−FKL,优势 A=−ℓ 的期望为 +KL(π_T‖π_S)≥0 ⇒ off-policy
份额实际是**按 log-ratio 加权的教师文本行为克隆**,而非 GKD 的 supervised-KD 项。
**定量对照**(把差别钉死):最小化 FKL 的采样估计 = 最大化 E_{y~π_T}[log π_S] =
**教师文本上的普通 SFT**(梯度 = ∇log π_S 的无权均值);我们的 PG 给的是**同一批梯度
按 (log π_T − log π_S) 加权**。即 a3 = 加权版 SFT,忠实形 = 无权 SFT(采样估计)或
全词表 FKL(GKD 的 token 级 KD 本体)。更一般的一句话:**在单样本估计器下,散度方向
不是自由旋钮,它由谁采样决定**——这正是 a1/a3 不能声称复现 GKD 目标的根因。
判决口径改为:a1/a3 度量的是**数据来源的剂量反应(在本协议估计器下)**,明确不声称
复现 GKD 目标。忠实变体(λ 混合作用在分布式 top-k 散度上、方向由 D 固定)= 修正案
候选 a4_gkd_dist,本轮不建。

### 撤销 #2(r5/a2 "出处升级"):其 rejection sampling 是**有效性**过滤,非正确性
`is_valid_output` 逐行读:必须含 `\boxed`;拒绝重复行(≥5 次)、100 字符 n-gram 重复
(≥3)、>5000 字时的连续块重复;每槽重试 ≤3。**不查答案对错**。r5 把它读作 verifier
拒采并据此"升级"为原版确证——撤销。执行:`gen_coldstart_data.py --filter` 三档,
**默认 validity(按其阈值逐项移植)**,`verifier`(正确性)降为我们的消融,`none` 保留。

### 新增对表:a2 的 SFT 超参(r5 时无从对)
其 `qwen3_base_full_sft.yaml`:Qwen3-1.7B-Base 全参 SFT、**1 epoch**、**lr 1e-5**、
cosine、warmup_ratio 0.05、cutoff_len 14336、bf16。我们:epochs 2→**1**、lr/调度/
warmup 由未登记的 verl 默认改为**显式钉死同值**;cutoff 17408 vs 14336 = 16k 协议缩放
(其 resp cap 7168↔14336,我们 16384↔17408,比例一致),登记偏离。

### 新增声明:a2 的 think 模板不一致(他们内部就不一致)
他们教师 rollout `--enable-thinking false`、OPD 阶段 `enable_thinking=False`,但
**SFT 配置 `enable_thinking: True`**。我们两阶段一律 non-thinking,使冷启动条件化与
OPD/评测完全一致(`ColdstartSFTDataset` 的 loss 边界即生成前缀之后)。登记为**有意
偏离**:我们审的是配方形状,不是其模板组合。

### 新增守卫:门控臂的 env 停车位
a1/a3 的旋钮停在 `env2_pending`,翻 stock 时若忘改名,`gkd_mix` 因缺 `SIMOPD_GKD_CACHE`
静默不装 ⇒ **两个 vanilla 顶着 a1/a3 的名字跑**。arm_lint 新增拒绝规则(自测:临时翻
stock 立即报错)。

**A 轴现状**:a1/a3 门控(等 cornell 预生成 + 3 步彩排);a2 门控(等 SFT ckpt),
其配方三项按本次复审改齐。判决语言的三处收窄已入臂注。

(合流注 2026-08-11,本章节 cherry-pick 自 cornell 线 659bc29:上句 "a2 门控" 是
cornell 视角。舰队侧 a2 为 stock 且三 seed 已近跑完 —— 但其阶段 1 产物建于本复审
之前(正确性过滤 + 2 epoch + verl 默认 lr),按本复审口径,这三行是 `--filter
verifier` 消融格;validity 配方格待按钉死超参重建阶段 1 后补跑。跨设备 bug 章节
未随本次合流带入 —— 舰队线已独立修复同一处,`_stat_mask` 直接带 device 出生。)

## r6 登记(2026-08-11):B 轴 β 阶梯兑现 —— b4_jsd_b0.1 / b4_jsd_b0.9

**性质**:不是新臂,是 b4_jsd 落地时(2026-08-07)随臂登记的**内部消融**的执行。
`SIMOPD_JSD_BETA` 当时即写入臂注,依据是 GKD 自陈最优 β 任务相关({0.1, 0.5, 0.9}),
故单一 β 不足以代表该散度。零新代码:同一 `jsd_topk` 核、同一 renorm top-32 支撑、
同一 DIRECT 分支,**β 是三档间唯一变量**。指纹自动分档(`SIMOPD_JSD_BETA` 在
`run_opd_baseline.sh` 的指纹集内,仅 SHADOW/PI_TAIL_WIDTHS/LANE_TAG 豁免),
ckpt 目录按 run_id 天然隔离。

**数据存在之前登记的两个对立预测**(16k 首轮 B 轴读数提出的问题):FKL 成分在一种
形式下致命(b3 把 top-k FKL 叠加进采样点 PG 底座 → 0.003,全场唯一塌了不返),在
另一种形式下只是平庸(b2 全局截断 FKL,长期 ~0.46),而对称中点 b4 稳定(0.524@250)。

- **H1(形式假说)**:信号一旦是**分布式**的,方向即为二阶效应 ⇒ 三档落点相近,
  塌陷的归因落在 C 轴(采样点 vs 分布式),B 轴方向不背这口锅。
- **H2(方向假说)**:方向仍主导 ⇒ 三档单调铺开,b4@0.5 只是两种失效模式的平均。

两者预测相反,故该阶梯可判决。β→0 的缩放极限是 renorm FKL(b0.1 = mass-covering
端),β→1 是 renorm RKL(b0.9 = mode-seeking 端);b0.9 承担更锋利的一读:采样点
RKL 家族恰是晚期塌陷的那批,若 mode-seeking 本身即病因,b0.9 应为阶梯最差档。

**排期**:wave 9,2 卡行(勿置于四卡箱),队列语义保证其排在全部 ≤8 波之后。
成本按 b4_jsd 实测外推(29.9 wall-h/seed、179 GPU·h/臂)≈ **358 GPU·h + 60 格
离线 suite**。台账、verdict 名册、arm_lint 分支真值表(两档均 direct)同步已改。

## r6 登记(2026-08-11):c2 钉扎粒度阶梯 —— 附一条既成事实的偏离补记

**偏离补记(先于消融、独立成立)**:c2 的 "batch 级" τ 实际算在**打包微批**的响应
token 种群上(`_stat_mask` 限定;与 g2 的微批种群偏离同型)。16k 响应饱和 17408
打包上限后,一个微批只装得下一条序列 —— 所以已完成的 c2 三行是一条**粒度随训练
漂移**的混合曲线:早期(短响应,微批多序列)= 跨序列钉扎,后期 = 事实上的逐序列
钉扎。CPU 对拍第 5 项证实:单序列微批下 sequence 档与 batch 档逐位相等。

**旋钮**:`SIMOPD_QB_SCOPE ∈ {fixed, sequence, batch}`(默认 batch = 出货行为
逐位不变,对拍第 1 项;非法值 import 时拒斥)。同核、同 top-64 载荷、同 direct
分支、同 margin 机械 —— **keep 规则是三档间唯一变量**。c2 本体 env 有意不加该
键:加了会翻已完成行的指纹。

**数据存在之前登记的对立预测**:

- **P1(漂移无害)**:perseq ≈ c2(batch)⇒ 跨序列再分配从不重要,已完成 c2 行
  的混合曲线可安全解读;自适应的活性成分是位置间分配。
- **P2(跨序列流动是活性成分)**:perseq < c2 ⇒ "难题该多花词表预算"是真效应,
  c2 头名的一部分来自批级信息流,论文措辞降级为"批级自适应";同时已完成 c2
  行的前后段不可比,需按漂移分段读。

**fixed8 的双重身份**:零自适应端点,兼补上缺失的**等预算固定对照** —— 现行
c2-vs-c1 混着预算大小(8 vs 32)与分配策略两个变量。读法:fixed8↔c2 = 等预算下
纯自适应效应;fixed8↔c1-direct(缺口格 C1)= 定策略下纯预算规模效应。副产品:
perseq 档的损失恢复逐样本可分解(支撑不再依赖同批打包了谁)—— 对批级阈值的
标准审稿质疑的预防性回答。

**执行**:wave 10(排在 β 阶梯 wave 9 之后),2 卡行,成本按 c2 实测外推
≈ 2×300 GPU·h + 60 suite 格。CPU 对拍六项全过(逐位复现出货路径、fixed 恰 B、
端到端逐段分位、stat 掩码/纯 prompt 段/+inf、稠密回退 loud、跨档不变量)。

## r6 登记(2026-08-11):B/F 压缩剂量线(回溯分析,零新 run)+ β 阶梯补注册 H3 + Fisher 去混淆

**压缩剂量线(回溯分析登记)**。五臂同支撑(采样点)、同分支(全 PG)、同预算,
每臂只动信号量级的变换,构成压缩强度的单调梯子;lock(≥90% 截断)严格单调:

| 臂 | 量级变换 | 名义上界 | lock |
|---|---|---|---|
| vanilla | 恒等 | ∞ | 122 |
| f2_hard_clip | clip ±10 | 10 | 198 |
| f1_soft_log | sign·log1p | ~log | 208 |
| b1_skew_kl | skew 混合 | ≈2.30 | 247 |
| f3_power | π_T−π_θ | 1 | 从不(截断 0.010) |

命题:**采样点信号的无界量级驱动长度失控,压缩按剂量推迟之**。执行纪律:x 轴
不用名义上界,用已埋的 `_signal_quantiles` 面板**实测**各臂 |signal| 有效分位
(p99);若实测量级排序与名义排序不符,本命题即被否证 —— 这是可证伪登记,不是
事后叙事。两端各有失败模式:零压缩端 = 终止塌陷;全压死端(f3)= 不撞帽仍跌
0.107 的"第二失败模式"(信号强度换终止安全的代价候选)。

**Fisher 去混淆**。塌陷文档 §5 的 USE_POLICY_GRADIENT Fisher p=0.017,在
EXPECT_PG 真值表下与"采样点 vs top-k"几乎完美共线;仅有的两个交叉格双双背叛
估计器解释:b5(采样点×direct)lock 125 ≈ vanilla 122,c1(top-k×PG)截断
0.008。受控内对照:换估计器移动 lock 3 步,换量级上界移动 +125 步。判决语言:
**"PG 与失控的相关几乎全部由支撑/目标形式承载;机理杠杆是量级(本剂量线)与
终止压力(b5↔j1 对照)"**。

**β 阶梯补注册 H3(有界性假说)**。JSD 对任意 β∈(0,1) 有界(≤log 2),FKL/RKL
无界 —— b4 安全 vs b2 撞帽(0.867)可能是量级而非方向。三方裁决结构,判决量
lock/截断曲线优先:
- **H3(有界性)**:三档截断曲线全平(b4@0.5 同款)⇒ b2 的帽归给无界量级,
  方向效应只出现在分数;
- **H2(方向)**:b0.1 截断曲线向 b2 滑动 ⇒ 方向经由长度动力学起作用;
- **H1(形式)**:三档分数也相近 ⇒ 分布式形式主导,方向与量级皆二阶。

**b3 机理补完(不动点)**。转变窗 50–100(熵 0.648→0.071,全场唯一塌陷点梯度
尖峰 18.5@100);此后策略退化为无终止重复文本,教师在其上熵极低 ⇒ **门自关**
(阈 0.8),教师 top-k 恰好也预测重复 ⇒ 学生质量 0.9997、|Δℓ|→0.03、verifier
→0.000:**一个近零损失的退化不动点,破坏由门控 FKL 项在转变窗内完成**。分析期
最后一颗钉子:b3_gate 面板在 50–100 窗内的实开率。

## r6 登记(2026-08-11):C 轴二次重审 —— 量级统一假说 + C1 消融对立预测 + c1 漂移事实

**量级统一假说(M1 扩展,零新 run)**。top-k 家族里恰好两个撞帽者(e2 0.961、
b2 0.867)都是**无界损失**(−log(1−π_tail) 与截断 FKL 的 −log π 项),而 renorm-KL
族(c1/c2/c4 及 b4 的有界 JSD)与化简直接形(c3)全部终止安全 —— C 轴的安全可能
不由"支撑宽度"承载,而由"实践中温和的信号量级"承载,与 B/F 压缩剂量线同源。
执行:M1 的 signal_quantiles 实测扩展到 top-k 臂;预测:e2/b2 的 |signal| p99
显著重尾于 c 族。若实测不重尾,支撑-安全需要另立机制。

**C1 消融(c1-direct)对立预测,先于数据登记**。c1 的晚期漂移事实:熵 1.9→0.66
(崖在 175–200 后)、蒸馏损失全程单调上行 0.25→0.45(29 臂中最干净的一条上行)、
教师 top-32 质量 0.84→0.96 与熵崖同步 —— 正是 PG 病理(优势恒非正的无方向压制,
向教师 top-1 过锐化)的慢动作签名。两个可判预测:
- **P-PG(漂移归 PG)**:c1-direct 稳熵(参照 c2/c4 的 0.4–0.6 平台)、KL 平或降、
  终值上移 —— "文献默认最差"改判为"LSM 的 PG 选择拖累了它自己的方法";
- **P-K(漂移归 fixed-k)**:c1-direct 复现熵崖 + KL 上行 —— 固定 teacher-rank
  截断本身携带过锐化,c2/c4 的自适应边是解药,headline 定理再加一票。

**事实登记(C 轴画像)**:c2/c4 是全 29 臂**训练期 verifier 得分最高的两臂**
(0.128 / 0.120@250)—— 自适应支撑不止终止安全,rollout 质量也最高;C 族的早期
兴奋恢复是全场唯一不复发的(c2 0.385@50→0.016,c4 0.258→0.018,c3 0.104→0.008,
对照 40/46 复发棘轮);c1 的 τ1.0 rollout 长度 1.3k 为全场唯一低于基座(3.2k 均值)
且方向朝教师(1.6k 均值)的臂 —— "忠实长度"读法;步时 c1 115 / c3 375 / c4 505 /
c2 780 s/step。**matched-compute 粗检**:c2 花掉 c1 全程算力(≈7.6 wall-h)时约在
step 35,读数已 ≈0.62 > c1 终值 0.524 —— "自适应>固定"大概率过算力关(正式计算
归 compute ledger 交付物,此处仅登记粗检方向)。

## r6 登记(2026-08-11):监督支撑族(D∪H 合并读法)—— 位置剂量线 + 桥格 + d3 种群偏离

**合并声明**:仅分析/论文层。run_id、axis 字母、引用谱系(d 审 TIP/SelecTKD/
TA-OPD,h 审 ESR)、指纹一律不动。两轴机械上同为 `1[t∈Keep]·Δℓ·rescale`,
区别只在 Keep 的来源;合并后是三因子空间:位置 × 判据 × 预算。

**位置剂量线(现有数据,回溯读数登记)**。监督质量落在序列的**哪里**控制终止,
落**多少**不控制:

| 窗 | 臂 | 截断@末 |
|---|---|---|
| 前段 K=100(~0.6%)| h1 | 0.151(安全)|
| 随机连续 K=100 | h3 | 0.182 |
| 判据散选 50% / TAR / 5% | d1 / d2 / d3 | 0.990 / 0.997 / 1.000 |
| 尾段 K=100 | h2 | lock@37(灾难)|

预算反驳:d3 只教 5% 照样 1.000 撞帽,h1 教 0.6% 全程安全。中段排序被三重混淆
(散选 vs 连续窗 × 判据 × 预算)—— 桥格负责拆。与压缩剂量线正交:那条变量级,
这条变位置;两条都是零新 run 的登记读数。

**桥格 DH1(random-scatter @50% 与 @5%)**。D 轴从未有过 random-at-matched-budget
对照 —— d1-vs-vanilla 只回答"选一半行不行",回答不了"TIP 判据比随机选一半好多少",
而后者才是选择器文献的主张。桥格判据=随机 ⇒ 不需要 KEEP_SAMPLED 载荷 ⇒ 2 卡行
(d 族为判据付 4 卡)。三个读数:① 判据 vs 随机 @ 等预算(d1↔r50、d3↔r5);
② 散选 vs 连续窗(r50/r5 ↔ h3),位置剂量线中段去混淆;③ 成本判决 —— 若
random-scatter 追平判据臂,d 族的全部 4 卡溢价买到零。

**d3 种群偏离补记**(c2/g2 同型,已入臂注):batch_quantile 归一化种群 = 打包
微批;16k 饱和后微批单序列,出货行从跨序列归一漂移为事实上的逐序列归一。

## r6 登记(2026-08-11):E 阶梯纯化 —— 锚系数归零的两格(wave 11)+ 对立预测

**裁定(用户)**:值锚污染了信息消融仪器 —— e1/e2 出货形态是 "X + 0.1·值" 混合,
阶梯的正面结论("序/集信息够不够")一句都不合法,只有 a fortiori 负读数幸存
(e2 带补贴仍撞帽;纯 e3 胜过带补贴的 e1)。锚的原始动机(防 margin 塌伤采样评测)
在机理框架下不成立:margin 塌不塌该在评测里测量,不该在训练信号里预防。
`SIMOPD_PL_ANCHOR_COEF` 注册时即为内部消融轴 —— 本登记执行 coef=0,零新代码,
指纹自动分格;锚在 coef=0 下仍作为面板照打(训练不用,测量保留)。

**e1_pl_rank_a0(纯序)对立预测**:
- **P-order**:a0 ≈ e1(0.572)⇒ 序独扛,锚是装饰,"保序不保值"获直接证据;
  M-I 侧顺带回答 e1 的终止安全(截断 0.008)属于序损失本身;
- **P-anchor**:a0 落入 vanilla 带或撞帽 ⇒ 值补贴一直承重,"序够了"死;
  {a0, e1, c1-direct} 改读为值强度剂量线 0 / 0.1 / 1.0 —— 阶梯不报废,换名。

**e2_set_coverage_a0(纯集 = 近安慰剂)对立预测**:结构项 −log(mass) 在 16k 数据
里 step ~50 即饱和(mass 0.998,损失 ~0.002),故此格过了开局就是**近零信号对照**
—— 名册上最接近安慰剂的臂。它把出货 e2 的死因劈成两半:
- **P-frozen**:策略几乎不动,停在基座带附近、不撞帽 ⇒ e2 的帽是它那份
  弱训练信号**造成**的(0.1 锚成了破坏者 —— 若此,支撑内弱值匹配的失稳效应
  本身升格为 M-II 发现);
- **P-drift**:照样撞帽 ⇒ 长度吸引子几乎不需要训练压力就可达 —— 一条很强的
  M-I 陈述(on-policy 采样噪声自身足以滚进吸引子)。
两支皆有信息量。coverage / anchor 面板照打。

**排期**:wave 11(排在 QB 阶梯 wave 10 之后),2 卡行。成本:e1@0 ≈ 140 GPU·h
(按 e1 外推);e2@0 若 P-frozen 成立会显著短于本体的 282。集档修复格(有界/
非饱和集损失)维持提案状态,**必须无锚出生**。

## r6 登记(2026-08-11):F 轴收尾 —— M6 挂账 + α 梯降为条件触发提案

**M6(零 run)**:f3@100 vs f3@250 的 textdump 对读 —— 全名册唯一不能赖给计分器
的下降(从不撞帽、长度熵健康),退化的到底是什么值得读文本;`power_dead_frac` /
`clip_hit_rate` / `shrink_ratio` 三块装了没读的表并入 M1 面板收割。

**α 梯(SIMOPD_POWER_ALPHA)降为条件触发提案,不排卡**。存在理由仅一条:压缩
剂量线五档是五种函数形式,跨形式批评("变的是形式不是量级")唯一的构造性反驳
是形式固定的 α 参数线(α→0 严格退化为 vanilla)。但它带一个实质混淆:小 α 下
π_T^α−π_θ^α ≈ α·Δℓ —— 与有效学习率同旋钮,内插点难与低 lr vanilla 区分。且无
任何在册判决依赖它。**触发条件**(照 f2 的 D5/Mode-B 先例):M1 实测量级排序
解释不了 lock 排序,或跨形式批评实际出现;启用时必须自带 lr 混淆处理方案。
在此之前,量级假说的裁决走 M1 实测 + wave 9(H3)+ wave 11。

## r6 登记(2026-08-11):G 轴配额格(用户设计)—— g1_quota / g4_quota,status: needs

**代数前置**:出货的 mask + T/T_keep 缩放 + token-mean **恒等于**保留 token 上的
条件均值 —— 改归一化约定是空操作。crater 的病根是 T_keep 随通过率坍缩(饥荒窗
通过率 0.5–2.1%,放大器 50–200×,样本 ~5 条轨迹)。本设计钉的是**样本量**:
DAPO 式动态采样,采到 K 条 verifier pass 才更新。vendored verl v1 已带机械
(`algorithm.filter_groups` + `max_num_gen_batches`,trainer_base.py:210),缺一个
自定义谓词("按通过数配额"),小接缝非手术。

**五个钉死的决定**:K=16(健康后期一步 ~23 条 pass 的量级);max_num_gen_batches=8,
配额不满照常更新并记录 K_actual(优雅降级,无静默截断);**pass 禁止跨步缓存**
(缓存即 off-policy,泄漏进 A 轴;步内补采仍 on-policy);补批抽新 prompt(DAPO
惯例;pass 条件化的易题偏置是固有的,照实记录);g4 对称同改(fail 全程 ~90%,
配额几乎不咬合,近乎免费 —— 存在意义是让符号括号保持单旋钮)。

**对立预测(g1 ↔ g1_quota = M4 的因果对照,地位对齐 N1 之于 M-I)**:
- **P-famine**:无 crater、早期曲线平滑、suite@25 不再是全场地板 ⇒ 饥荒-放大器
  机理坐实,且"g1 的 crater 可免费修掉"成为 recipe 级结论;
- **P-not**:crater 照现 ⇒ M4 的放大器账错,深 U 另有机理。

**账本**:饥荒期生成成本 ∝ 1/通过率 —— 生成 token 与监督 token 在此剧烈分叉,
compute ledger 的第一个活案例。成本估计:g1_quota ~300 GPU·h(240 + 饥荒溢价),
g4_quota ~357。**解锁** = 谓词落地(舰队侧)+ CPU 对拍 + 3 步彩排 + wave-12 tsv
加行;旋钮停 `env2_pending`(SIMOPD_GQ_QUOTA=16 / SIMOPD_GQ_MAX_GEN_BATCHES=8),
lint 拒绝忘改名的 stock 翻转。g2/g5 不动:其 recipe 即受审对象。

## r6 登记(2026-08-11):H 族补全对(用户设计)—— h4 散选 + h5 生成截断,wave 12

**h4_random_scatter**:h3 去掉连续性 —— 同 K=100、同 kernel 工厂、同分支,
连续性是对 h3 的唯一变量。M-I 细化预测已登记:散选掩码**每条**轨迹都触及响应
尾部(CPU 对拍 20/20 次),h3 的窗只偶尔落在尾部 —— 若尾部监督剂量驱动长度
失控,h4 的截断曲线必须高于 h3。P-budget(h4≈h3 ⇒ 连续性无关)vs
P-window(h4<h3 或更早 lock ⇒ 局部连贯/尾部剂量承重)。对拍:逐行恰 K、
短行全保(族约定)、非连续、逐次新抽。

**h5_gen100**:ESR 的**原式**(rollout 截断在 100,全部监督)—— 零新代码,
MAX_RESPONSE_LENGTH 本就 env 可调且入指纹。h1 当年有意偏离的正是这个形式
(截断改变采样分布,满预算下与 A 轴混淆);h1↔h5 把那条 recorded deformation
变成**被测量的配对**:同 K,mask vs truncation,差 = 分布改变 + ~50× 生成成本。
**大声记录**:in-loop val 同受 100 帽,h5 的 in-loop 列是 greedy@100 预算,
与主表不可比 —— 裁判是离线 suite(32,768 预算)。P-skeleton(h5 在 suite 上
≈ h1 ⇒ 效率头条 + ESR 原式获证)vs P-mask(h5 < h1 ⇒ 定分布的 loss-mask 形
承重)。M-I 双向开放登记:训练 rollout 永不接近 16k 帽(吸引子不可达)但
100-token 截断 rollout 几乎无 EOS(出生即停止监督饥饿)—— eval 时长度行为是
本格的第二读数,ckpt textdump 便宜。协议注:响应帽是 §3.8 协议常数,本臂为
逐臂登记偏离(ESR paper-form),指纹自动分格。

**排期**:wave 12(m15/m16/m18,2 卡);G 配额格解锁改记 wave 13。成本:
h4 ≈ h3 的 210 GPU·h;h5 估 ~15–30 GPU·h(生成 1/50)—— 若 P-skeleton 成立,
h5 将同时是全场最便宜和 per-token 效率最高的臂。
