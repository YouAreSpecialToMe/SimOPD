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

