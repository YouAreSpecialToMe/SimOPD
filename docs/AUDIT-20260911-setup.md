# 2026-09-11 复核:重训的 setup 与各臂实现("vanilla 这回特别好"是不是哪里错了)

用户 09-11 的问题:重训里 vanilla 很好、十来条臂贴着它走,是不是 setup 或某个方法的实现有问题。
本文只记录能从 git 里的数据与代码核到的事;跑在集群上的运行时环境(每个 run 的 `run_manifest.json`)
要等轨迹包 tier1 到了再核。可复现:`scripts/analysis/arm_activation_check.py`;第 2 节的表由本文生成脚本重算。

## 0 结论

1. **vanilla 不是异常好,它是在重现。**与 08 月 corr 波(同臂名、同配置、旧集群旧栈)按相同评测步配对,
   26 条共同臂的平均偏移 **−0.001**,`vanilla_corr` 自身 −0.001,24/26 条在 ±0.02 内(§2)。"特别好"是相对
   legacy vanilla(k1_rec 未修 eos,三种子塌到 composite .247)—— 那正是 08-24 的判决"修正载体治好晚期塌缩",
   这次在新集群、verl `aebd1f8` + vLLM 0.26 上原样复现。
2. **29 条臂的旋钮都在运行时留下了自己的仪表**(§3):每条臂 `metrics_full` 里有载体没有的专属键,值在动;
   H 轴分段臂的 Δℓ 位置份额是硬证据(h1 = 1.00/0/0/0,h2 集中在末段)。没有"登记了没生效"的臂。
3. **贴着载体走的臂,是旋钮忠实但在修正载体上没东西可咬**(§4):f2 只截 0.3% 的 token(它自己的设计注释
   就写着 p99.7),d2 保留 97.7% 的 token(采样 token 几乎都在教师 top-5),c4_hq 的 agreed-freeze 每 5 万
   token 触发一次,n2 的校准项按全 batch token 数归一、结构上只有 ~1/L 的权重(实测 2e-6),f3 的 a=1 是
   登记的正则成员(loss = p_θ − p_T,不是恒等)。这些旋钮在 legacy 载体上有大把 |Δℓ|≈25 的错配终止 token 可咬,
   修了之后就没有了 —— **这是审计要写的结论,但判决表必须带"咬合度"一列**,否则"没开"和"没用"分不开。
4. **唯一的 setup 缺口:`_n0` 十条没有契约对照。**A 轴 4 条与 H 预算线 6 条相对载体差的不只是旋钮,还有
   `SIMOPD_STOP_IDS=151645` + `SIMOPD_EOS_IDS=151643,151645`(rollout 会在 im_end 上停,在环评测同样),
   `arms.yaml` 里没有"载体 + v2 契约、无其他旋钮"的臂。A 轴 −0.05..−0.14 是旋钮与契约的合并效应。
   **建议补 `vanilla_v2`**(250 步 ≈ 70 GPU·h),它同时是 H 线的正确基线。

## 1 查了什么

| 检查 | 方法 | 结果 |
|---|---|---|
| 重现性 | 08 月 corr dump(`docs/data/training_metrics_corr_*`)vs `results/20260909`,同臂同步配对 | §2,mean −0.001 |
| 旋钮激活 | 每臂 `metrics_full` 的专属键 + 咬合度仪表 + Δℓ 位置分箱 | §3 |
| 咬合度 | 读 `losses.py` / `topk_losses.py` / `gkd_mix.py` 里对应仪表的定义 | §4 |
| 静态一致 | `arms.yaml` 的 `USE_POLICY_GRADIENT` vs `arm_lint.EXPECT_PG`;启动器默认值 vs README 共同配置 | 29/29 一致;batch 128 / mini 128 / n=1 / 16384 / lr 1e-6 / temp 1.0 / 每 25 步 一致 |
| 内核电池 | `scripts/*_battery.py`,本机 CPU torch | §6 |
| 同一 run 伪装成两条臂? | `vanilla_te` vs `vanilla_corr` 逐步比对 | 250 步里连续量 0 步完全相等,相关 .88–.98(同 seed 同题、不同采样),是两条 run |

## 2 逐臂重现:重训 − 08 月波(同一评测步配对,in-loop MATH500 greedy)

| 臂 | 共同评测点 | 到步 | Δacc(重训 − 08 月波) |
|---|---|---|---|
| `c4_state` | 3 | 100 | -0.082 |
| `h2_last_segment_corr` | 1 | 50 | -0.062 |
| `a5_aggrevate_n0` | 2 | 50 | -0.026 |
| `d2_selectkd_corr` | 5 | 175 | -0.016 |
| `h1_first_segment_corr` | 8 | 200 | -0.013 |
| `c4_pi_tail_budget_corr` | 6 | 175 | -0.008 |
| `c3_intersection_corr` | 7 | 200 | -0.008 |
| `a1_gkd_mix0.5_n0` | 5 | 125 | -0.008 |
| `n2_corr` | 4 | 125 | -0.006 |
| `b1_skew_kl_corr` | 5 | 150 | -0.006 |
| `e2_set_coverage_a0_corr` | 5 | 125 | -0.004 |
| `c5_union_fkl` | 4 | 100 | -0.002 |
| `vanilla_corr` | 10 | 250 | -0.001 |
| `c5_union_rkl` | 3 | 75 | +0.001 |
| `c4_hq` | 4 | 100 | +0.002 |
| `g6_seqmean_corr` | 2 | 50 | +0.002 |
| `h4_random_scatter_corr` | 4 | 100 | +0.004 |
| `a3_offpolicy_n0` | 4 | 125 | +0.005 |
| `h3_random_segment_corr` | 3 | 75 | +0.005 |
| `f2_hard_clip_corr` | 4 | 100 | +0.006 |
| `f3_power_corr` | 4 | 100 | +0.007 |
| `c2_quantile_budget_corr` | 4 | 100 | +0.008 |
| `b2_forward_kl_corr` | 3 | 75 | +0.009 |
| `d3_teachability_corr` | 6 | 175 | +0.011 |
| `a4_dagger_anneal_n0` | 3 | 75 | +0.015 |
| `g1_verified_only_corr` | 4 | 125 | +0.136 |

`c4_state`(−0.08)只有 3 个共同点、都在 100 步前;`g1`(+0.14)是深 U 的谷底时机不同。其余全在噪声带(单点 SE 0.022)。

`vanilla_corr` 逐步:

| step | 08 月波 | 重训 |
|---|---|---|
| 25 | 0.612 | 0.608 |
| 50 | 0.620 | 0.604 |
| 75 | 0.624 | 0.632 |
| 100 | 0.630 | 0.622 |
| 125 | 0.632 | 0.638 |
| 150 | 0.630 | 0.636 |
| 175 | 0.652 | 0.662 |
| 200 | 0.686 | 0.624 |
| 225 | 0.622 | 0.664 |
| 250 | 0.648 | 0.660 |

## 3 逐臂激活表(`arm_activation_check.py` 输出)

| 臂 | 登记旋钮(相对载体) | 专属键数 | 咬合度仪表 | 末窗值 [全程范围] | Δℓ/loss 位置份额 0–100 / 100–500 / 500–2k / 2k+ |
|---|---|---|---|---|---|
| `vanilla_corr` | (载体) | 0 | — | — | 0.23 / 0.29 / 0.26 / 0.23 |
| `vanilla_te` | mode=k1_rec; TERM_EVENT=1 | 0 | eos_dl_at_stop | -1.87 [-8.8..-0.1] | 0.23 / 0.29 / 0.26 / 0.23 |
| `c2_quantile_budget_corr` | mode=qb_quantile_budget; QB_MARGIN=max; QB_TARGET_BUDGET=8; TERM_EVENT=1; USE_POLICY_GRADIENT=False | 16 | qb_captured_mass | 0.999 [0.71..1] | 0.24 / 0.29 / 0.25 / 0.22 |
| `c4_pi_tail_budget_corr` | mode=pi_tail_budget; DISTILLATION_TOPK=34; PI_TAIL_EPS=0.05; TERM_EVENT=1; USE_POLICY_GRADIENT=False | 17 | c4_budget | 5.77 [5.4..23] | 0.25 / 0.29 / 0.25 / 0.21 |
| `c4_hq` | mode=pi_tail_budget; DISTILLATION_TOPK=34; C4_HQ=1; USE_POLICY_GRADIENT=False | 19 | c4_hq_freeze | 2.15e-05 [8.5e-06..0.0006] | 0.24 / 0.28 / 0.25 / 0.23 |
| `c4_state` | mode=pi_tail_budget; DISTILLATION_TOPK=34; C4_HQ=1; C4_REP=1; USE_POLICY_GRADIENT=False | 20 | c4_rep_freeze | 0.0516 [0.032..0.52] | 0.25 / 0.28 / 0.24 / 0.23 |
| `c2_qb_fixed8_corr` | mode=qb_quantile_budget; QB_MARGIN=max; QB_SCOPE=fixed; QB_TARGET_BUDGET=8; TERM_EVENT=1; USE_POLICY_GRADIENT=False | 16 | qb_captured_mass | 0.243 [0.22..0.89] | 0.35 / 0.36 / 0.23 / 0.06 |
| `c2_qb_perseq_corr` | mode=qb_quantile_budget; QB_MARGIN=max; QB_SCOPE=sequence; QB_TARGET_BUDGET=8; TERM_EVENT=1; USE_POLICY_GRADIENT=False | 16 | qb_captured_mass | 0.999 [0.68..1] | 0.24 / 0.29 / 0.25 / 0.21 |
| `c5_union_rkl` | mode=union_rkl; DISTILLATION_TOPK=34; USE_POLICY_GRADIENT=False | 20 | un_p_imend | 2.81e-14 [1.6e-16..2e-08] | 0.29 / 0.31 / 0.26 / 0.14 |
| `c5_union_fkl` | mode=union_fkl; DISTILLATION_TOPK=34; USE_POLICY_GRADIENT=False | 20 | un_p_imend | 8.96e-10 [6e-14..1.4e-08] | 0.22 / 0.29 / 0.28 / 0.21 |
| `n2_corr` | mode=k1_termcal; EOS_AUX_COEF=1.0; TERM_EVENT=1 | 1 | eos_aux_term | 2.11e-06 [2.6e-07..0.00015] | 0.29 / 0.27 / 0.24 / 0.19 |
| `b1_skew_kl_corr` | mode=skew_kl_a0.1; TERM_EVENT=1 | 0 | — | — | 0.21 / 0.27 / 0.27 / 0.24 |
| `b2_forward_kl_corr` | mode=forward_kl_topk; DISTILLATION_TOPK=34; TERM_EVENT=1; USE_POLICY_GRADIENT=False | 4 | teacher_mass_min | 0.335 [0.029..0.47] |  |
| `c3_intersection_corr` | mode=intersection_topk; DISTILLATION_TOPK=34; TERM_EVENT=1; USE_POLICY_GRADIENT=False | 15 | c3_inter_size | 20.6 [18..21] | 0.22 / 0.32 / 0.26 / 0.20 |
| `d2_selectkd_corr` | mode=selectkd_verify; DISTILLATION_TOPK=34; SELECTKD_BETA=0.01; SELECTKD_K=5; TERM_EVENT=1 | 3 | d_selected_frac | 0.977 [0.93..0.99] | 0.25 / 0.28 / 0.25 / 0.21 |
| `d3_teachability_corr` | mode=teachability_select; DISTILLATION_TOPK=34; D_RETENTION=0.05; TEACH_K=16; TERM_EVENT=1 | 3 | d_selected_frac | 0.05 [0.05..0.14] | 0.30 / 0.26 / 0.22 / 0.22 |
| `e2_set_coverage_a0_corr` | mode=set_coverage_anchor; DISTILLATION_TOPK=34; PL_ANCHOR_COEF=0; TERM_EVENT=1; USE_POLICY_GRADIENT=False | 16 | e2_coverage | 1 [0.68..1] | 0.24 / 0.52 / 0.23 / 0.01 |
| `f2_hard_clip_corr` | LOSS_MAX_CLAMP=10.0; TERM_EVENT=1 | 1 | clip_hit_rate | 0.00318 [0.0017..0.0064] | 0.29 / 0.27 / 0.24 / 0.20 |
| `f3_power_corr` | mode=k1_power; POWER_ALPHA=1.0; TERM_EVENT=1 | 1 | power_dead_frac | 0.00108 [0.00096..0.042] | 0.31 / 0.25 / 0.22 / 0.22 |
| `g6_seqmean_corr` | LOSS_AGG_MODE=seq-mean-token-mean; TERM_EVENT=1 | 0 | — | — | 0.38 / 0.30 / 0.24 / 0.08 |
| `g1_verified_only_corr` | mode=k1_verified_only; TERM_EVENT=1 | 1 | gate_keep_frac | 0.0304 [0..0.089] | 0.40 / 0.28 / 0.23 / 0.09 |
| `h1_first_segment_corr` | mode=k1_firstseg; FIRST_SEGMENT_K=100; TERM_EVENT=1 | 1 | firstseg_covered_frac | 0.024 [0.019..0.099] | 1.00 / 0.00 / 0.00 / 0.00 |
| `h2_last_segment_corr` | mode=k1_lastseg; FIRST_SEGMENT_K=100; TERM_EVENT=1 | 1 | lastseg_covered_frac | 0.0138 [0.0089..0.077] | 0.01 / 0.21 / 0.36 / 0.42 |
| `h3_random_segment_corr` | mode=k1_randseg; FIRST_SEGMENT_K=100; TERM_EVENT=1 | 1 | randseg_covered_frac | 0.00909 [0.0079..0.091] | 0.05 / 0.18 / 0.29 / 0.48 |
| `h4_random_scatter_corr` | mode=k1_randscatter; TERM_EVENT=1 | 1 | randscatter_covered_frac | 0.00848 [0.0078..0.091] | 0.24 / 0.32 / 0.31 / 0.13 |
| `a1_gkd_mix0.5_n0` | DISTILLATION_TOPK=66; EOS_IDS=151643,151645; GKD_CACHE=$STORE/simopd_math/gkd_offpolicy.parquet; GKD_LAMBDA=0.5; STOP_IDS=151645 | 8 | gkd_lambda_realized | 0.503 [0.38..0.62] | 0.26 / 0.27 / 0.25 / 0.23 |
| `a3_offpolicy_n0` | DISTILLATION_TOPK=66; EOS_IDS=151643,151645; GKD_CACHE=$STORE/simopd_math/gkd_offpolicy.parquet; GKD_LAMBDA=1.0; STOP_IDS=151645 | 8 | gkd_lambda_realized | 1 [1..1] | 0.22 / 0.27 / 0.26 / 0.25 |
| `a4_dagger_anneal_n0` | DISTILLATION_TOPK=66; EOS_IDS=151643,151645; GKD_CACHE=$STORE/simopd_math/gkd_offpolicy.parquet; GKD_SCHEDULE=mode=linear,start=1.0,end=0.0,warmup=0,decay=250; STOP_IDS=151645 | 8 | gkd_lambda_realized | 0.674 [0.52..1] | 0.28 / 0.27 / 0.23 / 0.21 |
| `a5_aggrevate_n0` | DISTILLATION_TOPK=66; A5_KEYS=$STORE/simopd_math/gkd_offpolicy.parquet.dry; A5_TMAX_SCHEDULE=mode=linear,start=0,end=16384,warmup=0,decay=250; EOS_IDS=151643,151645; STOP_IDS=151645 | 14 | gkd_mixed | 74.6 [0..1.1e+02] | 0.38 / 0.28 / 0.21 / 0.13 |

注:专属键数 = 该臂 metrics_full 里有而 vanilla_corr 里没有的键(内核分支跑了的直接证据);咬合度 = 旋钮实际改动的信号比例,读法见脚本 BITE 表;位置份额 = 末窗内 |Δℓ|(k1 族)或 |loss|(top-k 族)按位置分箱的相对份额,H 轴分段臂应只在自己的段里有份额。

## 4 咬合度分级

- **咬得狠、行为确实变了**:`c2_qb_fixed8`(教师 top-8 覆盖掉到 .24 → 熵爆)、`d3`(只留 5%)、`g1`(放行 3%)、
  `h1–h4`(位置掩码)、A 轴(λ 0.5 / 1.0 / 退火,`gkd_cache_miss` 的 ~500 是评测步的 val 题,不是 bug)、
  `c3`(交集 20/34 列)、`c5`(并集;`un_p_imend` 纹丝不动是预注册的负结果)、`e2`(覆盖=1 后损失 → 7e-4,
  等于停训 → 低熵漂移;`PL_ANCHOR_COEF=0` 是登记的 a0 消融)。
- **忠实但结构上小**:`f2`(0.3%)、`d2`(2.3% 的 token ×0.01)、`c4_hq`(2e-5)、`n2`(2e-6,~1/L)、`f3`(a=1)。

  **为什么触发这么少(09-11 补核,修正上一版的说法)**:它们都是**尾部干预**,而这对师生在 τ=1 下的逐 token
  Δℓ 尾部本来就薄 —— 而且**legacy 波就是这么薄**,修 eos 没有改变逐 token 分布,改变的只是每条序列末尾那一个
  停止 token 的账:

  | 量 | legacy 16k 波(未修 eos,seed 0) | 重训(修正载体) |
  |---|---|---|
  | Δℓ 均值 / p95 / p99 / p99.9(vanilla) | 0.32–0.62 / 0.06–0.53 / – / – | 0.34–0.66 / 0.08–0.55 / 0.7–1.4 / 2.5–3.0;p0.1 = −12…−15 |
  | f2 `clip_hit_rate`(clamp 10) | 0.38% → 0.17%(250 步) | 0.30% → 0.28%(100 步) |
  | d2 `d_selected_frac`(教师 top-5 内) | 0.94 → 0.99 | 0.95 → 0.98 |
  | 教师 top-k 熵 | 0.15–0.79 nats | 0.15–0.60 nats |
  | g1 `gate_keep_frac` ≈ 训练集 verifier 通过率 | 0.6%–3.5% | 0.6%–2.5% |
  | c4_hq 冻结 / token | – | 3.5e-4(短答期)→ 2e-5(9k 长答期)= 每条序列末尾 0.2–0.4 次 |

  读法:clamp=10 只碰 Δℓ < −10 的那 0.3%(教师给学生采到的 token ≤ e⁻¹⁰);教师 top-5 装下 95–99% 的学生采样。
  legacy 波里让 f2 / d2 看起来"有效"的,是每条序列**那一个**被错认的停止 token(Δℓ≈−25,不在教师 top-5):
  f2 把它从 −25 截到 −10、d2 把它 ×0.01,都是在削弱"别停"的推力 —— 这与 08-24 的 N0 规则一致(d2 在"被事件修正
  救活"的五条里)。修正后停止事件的 Δℓ ≈ −1.2(`eos_dl_at_stop`),既不过 clamp 也不出 top-5,两条臂就只剩尾部那
  0.3% / 2% 可动。c4_hq 的"少"是分母问题:按 token 算 2e-5,按序列末尾算是 20–40%。g1 的 1–4% 是数据性质
  (τ=1 采样下训练集的 verifier 通过率),与载体无关。触发率还随训练下降(d2 5%→1%,f2 0.4%→0.2%):学生越像教师,尾部越薄。
  **表里的"咬合度"应同时给两个分母:占 token 的比例,和占序列末尾的比例。**
- **只有间接证据**:`b1`(Δℓ 幅度分箱明显低于载体,skew α=0.1 应有的形状)、`g6`(位置权重分布不同);
  两者没有专属仪表,要坐实得看轨迹包里的逐 token 量。

## 5 setup 一致性

- `USE_POLICY_GRADIENT`:k1 族 True、top-k 族 False,`arms.yaml` 与 `arm_lint.EXPECT_PG` 29/29 一致。
- 启动器默认(`run_opd_baseline.sh`)= README 共同配置:`train_batch_size=128`、`ppo_mini_batch_size=128`(每批一次更新)、
  `rollout_n=1`、prompt 1024 / response 16384、lr 1e-6、temperature 1.0、`save_freq=test_freq=25`、250 步;
  载体 env 把 `DISTILLATION_LOSS_MODE=k1_termfix`、`TOPK=66` 盖在默认值之上。
- 终止符携带:所有 k1 族臂 `eos_missing=0`;`teacher_columns` 173 份摘要 `tch_lp_nan=0`。
- 停止契约:载体 `off`;`_n0` 十条 `151645`。**没有 carrier+v2 的对照**(§0 第 4 条)。

## 6 内核电池(本机 CPU)

`a5` 41/41 · `gkd_schedule` 40/40 · `h9` 27/27 · `h_horizon` 25/25 · `n2_eos` 67 项 · `stop_set` 25/25 · `traj_dump` 42/42 —— 全绿。
`c4_state_battery` / `union_battery` 依赖 verl(import 链要 ray、transformers、datasets,且 verl 用 3.10 语法):用 3.10 环境补跑,`c4_state` **31/31 ALL PASS**,`union` **22 项 ok、无失败、退出码 0**(输出里唯一含 fail 字样的一行是 torch `transformer.py:20` 的 UserWarning)。

`arm_lint.py`(同环境,103 条登记臂):59 个 PROBLEM,**0 个不是 `campaign.tsv` 账本类**;账本类分布:

```
46   PROBLEM   seed N appears 2x in campaign.tsv
  13   PROBLEM   has no campaign.tsv row
```

即老 16k 战役的三种子行重复、A 轴 `_n0` 四条没有派工行 —— 舰队发射门对这两类本就豁免(`campaign\.tsv row` 正则),PG 标志 / 注册表 / env 合法性的运行时检查对名册 29 条没有报错。

## 7 这次没查到的

- 每个 run **真实收到的 env**:只在集群 run 目录的 `run_manifest.json` / `manifest/launch_*.json` 里,轨迹包 tier1 会带回来;
  本文的"登记旋钮"列来自 `arms.yaml`。
- 内核数学与各论文的逐条对照:那是 08-07 的 audit r5 与 09-04 的全项目审查做的事,这次没重做。
- 单种子、in-loop greedy(套件分与在环分在塌缩臂上能差 0.2):同 09-09 数据集 README 的缺口。

## 8 实现与原论文像不像;为什么他们报有效、我们没有(09-11 补)

用户问:这几条臂的实现和原论文一样吗,为什么论文里有效果而我们没有。对着 `docs/arm-provenance-r4.md`(08-07 r4/r5 三方对表)、
两波数据(legacy 16k 三种子、08 月 corr 波、09 月重训)和三篇原文(arXiv 2510.24021 / 2607.13399 / 2605.26844,09-11 取)核了一遍。

**实现层**:每条外来臂都做过"论文原式 / 官方代码 / 我们"三方对表,偏离全部登记在 `arms.yaml` note 里。与本问相关的:
f2 = verl `loss_max_clamp`,论文只给 clip 形式不给阈值,±10 是我们注册的;d2 = 论文默认 k=5、β=0.01(无官方码);
d3 = 官方码的 Q05/Q95 归一、top-16、论文推荐的 5% 预算;h1 = ESR 默认 N=100,形式由截 rollout 改为 loss mask(声明形变,
h5 是原式,R6 已判 h5≡h1);c3 = thunlp 官方码的化简式;b1 = DistiLLM 官方 `skewed_reverse_kl` 逐符号;f3 = PowerOPD 原式,
α=1 是我们钉的(论文按指标挑 α)。系统性差别只有一条:所有臂的"底"统一为协议的 sampled-k1 PG + 修正载体,论文各自的底不同。
**实现对得上,差在体制。**

**为什么他们有效、我们没有 —— 四条,按证据强度排:**

1. **他们的 baseline 病着,我们的没病(⚠ §9 已收窄:他们的 baseline 是「过长但不塌」,不是撞帽塌缩)。**Demystifying 自己的表:clip 在 4B-GRPO→1.7B-Base(大 gap、长度涨到上限=Mode A)
   44.3→47.4,在 1.7B-GRPO→1.7B-Base 上 45.7→45.7,零效果;scale 在小 gap +0.6、大 gap −3.1。我们 legacy 载体带同一种病,
   f2 把 lock 从 122 推到 198,175 步 f2 0.648 vs vanilla 0.438;d2 150 步 0.641 vs 0.475 —— **比论文报的还大,是复现**。
   修好终止符后 vanilla 不塌,f2/d2 无事可做(重训 f2@100 0.632 vs 0.622,d2@175 0.666 vs 0.662;N0 审计 T2:f1/f2/f3/b1 在
   修正载体上全部 ±3%)。效果是真的,但它治的是 baseline 的病,而这个病在我们这里是终止符身份错位。
2. **分歧体制不同,旋钮咬不到东西。**SelecTKD 是 Qwen2-7B-Inst→1.5B-Inst、Mistral→Danube2(跨家族)、UltraChat 开放式对话,
   student top-1 常在 teacher top-5 之外,TAR 才有上升空间;我们 τ=1 下 95–98% token 已在教师 top-5 内(§4)。TA-OPD 的教师含
   GRPO 思考模型、DAPO 题,"分歧×兼容"要有真分歧才选得出;我们 top-k 熵 0.15–0.6 nats,5% 选出来近乎随机,只剩 1/20 监督
   → 变差(重训 d3@175 0.484;corr 波晚期 .457 vs vanilla_corr .656)。这是**真负复现**,不是"没效果"。
3. **效果量本来就小,在环量尺看不见。**论文效应量:SelecTKD +1.5–3 pp,TA-OPD 10% 预算 +0.0–3.2(四对里一对 +0.01),
   Demystifying scale +0.6。我们在环 MATH500 500 题 greedy 单种子,单点 SE 0.022,3 pp = 1.4σ;离线 suite(k=8)一格未跑。
   准确说法:**修正载体上不存在 ≥5 pp 的效果;≤3 pp 本轮设计不判**(09-01 合并 f1/f4/f5/d1/g2/g4/g5 的依据就是 ±3% 动态)。
4. **有几条"没效果"就是复现成功。**ESR/h1 的主张是"只教前 100 token 不掉点",h1@200 0.616 vs 0.624,R6 十格 h5≡h1 —— 正是其主张。
   G 族(g2/g4/g5/g6)在修正载体上 ±2%,论文效应若在也小于我们的分辨率。

| 臂 | 论文体制 | 论文效应 | legacy 载体(病 baseline) | 修正载体 |
|---|---|---|---|---|
| f2 clip | 4B-GRPO→1.7B-Base(**长度平台 11.5k,未撞帽**,§9) | +3.1(大 gap);0.0(小 gap) | @175 +0.21 | @100 +0.01 |
| d2 SelecTKD | Qwen2 Inst 对 / 跨家族,UltraChat | +1.5–3 pp | @150 +0.17 | @175 +0.00 |
| d3 TA-OPD 5% | GRPO 思考教师,DAPO | 10% 预算 +0–3.2 | 122 步塌(比 vanilla 早) | @175 −0.18 |
| h1 ESR N=100 | 1.5–32B,τ0.7,LoRA | 不掉点 + 24× | 唯一不塌的 k1 臂 | @200 −0.01(=主张) |
| b1 skew α=0.1 | GPT-2/OPT 指令跟随 | 优于 RKL | lock 247(最晚) | @150 −0.02 |
| c3 intersection | 1.7B-Base←4B-NT,n=4,7k | overlap 甜区 | 稳但 −0.09 | @200 −0.05 |

**对判决措辞的后果**:每条外来臂的判决写成"在 X 体制下",并带两个分母的咬合度(§4);"论文的效果能否复现"要在**他们的体制**里
答 —— 最便宜的两格是大 gap(4B-GRPO 或 8B 教师)上 vanilla vs f2,和一对低 TAR(跨家族)师生上 vanilla vs d2;本轮不加。

## 9 Demystifying 自己的 baseline 有没有问题(09-11;当晚读图后**重写**)

> **更正记录**:本节第一版(commit 6823632)写了两句现在**撤回**的话 ——
> ①「他们的 setup 就是我们 legacy vanilla 的 setup」;②「他们的现货老师格在我们这里就是这个伪影」。
> 撤回理由:把论文的图逐张渲染出来读之后发现,**他们所有未加调节的 OPD 曲线用的都是自训 GRPO 老师**,
> 而且**跟我们同一个学生(1.7B-Base)那条 vanilla 曲线根本没塌**。下面是重读的结果。

### 9.1 他们的配置(Appendix A 原文)

「We implement the training pipeline using verl.」·「a total batch size of 128, 1 rollout for each prompt」·
「a single epoch per rollout batch to eliminate off-policy bias」·**「The maximum sequence length is set to 16,384 tokens during training.」**·
「temperature of τ=1.0 and top-p=1.0」·训练集 Nemotron-Cascade Math·非思考前缀。**lr 与训练步数仍然全文未报。**
注意措辞是 maximum **sequence** length(我们的协议是 prompt 1024 + response 16384 分开钉),这一处不可逐字对齐,登记。

### 9.2 图里的数(SVG 渲染后读的,不是正文转述)

**Figure 9**(Teacher `Qwen3-4B-GRPO` → Student **`Qwen3-1.7B-Base`**,与我们同一个学生,横轴到 ~460 步):
y 轴刻度 0/2k/5k/7k/10k/12k。vanilla OPD(灰)从 ~1k 起、第 ~20 步先降到 ~0.7k、然后陡升,
**第 ~60 步见顶 ~12.7k,之后一路缓降到 ~11.5k,从第 250 步到第 460 步一直平**;
+Clip 顶 7.7k→回落→10.5k→~9.5k;+Logs 顶 11.0k→8.7k→10.8k→~9.5k。右图 Train Accuracy 三条**全程在涨**
(OPD ~0.11,clip/logs ~0.125)。**即:在 16,384 的帽下,他们这条 vanilla 峰值只到 ~12.7k,离帽还有 3.6k,而且是降下来平住的 —— 没有塌。**

**Figure 6**(Mode A/B 图,四栏 Accuracy / Response Length / Clip Ratio / advantage):
- **Mode A(蓝)= `Qwen3-4B-GRPO` → `Qwen3-1.7B`(chat 学生,不是 Base)**:长度从 ~11k 涨到 ~12.8k,
  **clip ratio 0.47 → 0.68–0.75**,accuracy 0.28 → 0.12,曲线在 ~150 步停。12.8k 均值配 0.75 截断率 = 16,384 帽被四分之三的 rollout 撞到,与正文「grows until hitting the predefined maximum limit」自洽。
- **Mode B(红)= `Qwen3-1.7B-GRPO` → `Qwen3-1.7B-Base`**:长度第 60 步冲到 ~10k,随后**稳在 ~8.2k 平了 550 步**;
  **clip ratio 冲到 0.49 后单调衰减,第 ~300 步起为 0.00 并保持**;accuracy 0.03 一路涨到 0.15;
  到第 ~620 步长度**瞬间掉到 0**、accuracy 掉到 0。

**Figure 4**(四老师对比)的老师是 `Qwen3-{1.7B,4B}-GRPO` 与 `Qwen3-{1.7B,4B}`,**并且整张图是「OPD w/ Clip」**,横轴到 1.1k 步。
`Qwen3-4B-Instruct-2507` 在全文只出现三处:Table 1 顶部作为**模型自身分数**、§3 pass@1024 天花板实验的老师之一(**只报 pass@k,不报任何训练动态**)、以及 Table 2 里 30B-A3B-2507 是别人方法的老师。
**Table 1 的 OPD / +Clip / +Scale 三行只在 `Qwen3-1.7B-GRPO` 与 `Qwen3-4B-GRPO` 两个自训老师下报。**

### 9.3 结论(取代第一版)

1. **不能说「他们的 baseline 病着」。**跟我们同一个学生、未加调节的那条曲线(Fig 9 灰)跑了 460 步不塌,峰值离帽 3.6k,准确率还在涨。
   更要命的对照是 Fig 6 的 Mode B:**Base 学生在他们手里 clip ratio 能衰减到 0.00 并保持 300 步以上** —— 学生正常停得下来。
   我们的 legacy Base 学生从来做不到这件事(棘轮到 1.00,`MECHANISMS.md` M-I:0/500 回头)。
2. **我们塌的那一格,他们没跑过。**`1.7B-Base ← 现货 4B-Instruct-2507 + 不加调节`,论文里没有任何训练动态。
   所以我们的结果**不与他们的数字冲突**,它是他们证据之外的一格。这对论文其实是好消息:主张从「他们错了」收缩为
   「他们的协议在换一个现货 Instruct 老师后会产生一个他们没测到的终止符伪影」,后者我们有单旋钮证据(.247→.348、截断 1.0→.10)。
3. **他们 vanilla 的曲线形状像我们的修正载体,不像我们的 legacy。**升→过冲→缓降→平台:`vanilla_corr` 是 12.4k@50 → 平 9.7k;
   他们是 12.7k@60 → 平 11.5k;我们的 legacy 是 13.3k@50 → 回落 10.2k@75 → **再冲到 16,384@125 不下来**。第三段是我们独有的。
4. **登记假说(不是结论)**:他们的两个老师都是从 Qwen3 chat 线自训 GRPO 的,而 Fig 6 Mode B 显示 Base 学生的截断率能归零 ——
   说明**他们的老师没有惩罚学生的 `<|endoftext|>`**;现货 4B-Instruct-2507 会(我们实测停止位 q_T(eot) 中位 1.4e-11,`docs/data/eos_stop_*.txt`)。
   顺着这条,他们的 **Mode B 是同一个错位的镜像候选**:老师若在很多位置都给 eot 不低的质量,采样到 eot 就拿正优势,
   学生最终锁死在「开头一段就停」—— 正是他们描述的 abrupt degeneration。他们自己的解释(1/T 稀释)也能盖住 Mode B,两者并存,
   **判据**:对一个 chat 线 GRPO 老师量 q_T(eot) 在停止位与非停止位的分布,与现货 Instruct 老师对比。需要一次前向,不需要训练。
5. **对 §8 第 1 条的影响**:§8 说「他们的 baseline 病着,我们的没病」——**前半句撤回**,改为:
   **他们报的 clip 增益(4B-GRPO 44.3→47.4)来自一条不塌但过长的 baseline(11.5k 平台),不是来自撞帽塌缩**;
   我们 legacy 上 f2 的 +0.21 比他们大一个量级,因为我们的 baseline 是撞帽塌缩的 —— 两边的 baseline 病得不一样重,
   这仍然解释「为什么修正载体上 f2 归零」,但不能再说他们的 baseline 也是撞帽的那种。
