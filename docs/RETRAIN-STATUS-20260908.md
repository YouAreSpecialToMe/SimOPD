# SimOPD 重训进度报告 — 2026-09-08

2026-09 名册（AGENT.md §2.2）的重训进度，截至 **2026-09-08 08:30 PDT**。
所有数字现取自各臂的 checkpoint 目录与 `metrics/launch_*.jsonl`。

---

## 0. 这一批实验在问什么

学生 `Qwen3-1.7B-Base` ← 教师 `Qwen3-4B-Instruct-2507` 的**在线策略蒸馏**。
被审计的现象是 **Mode-A 长度塌缩**：训练过程中响应长度从约 1k 一路涨到 16k 上限，
截断率逼近 100%。这不是开销，是被研究的对象本身。

上一波暴露出的问题是**终止符的语义错位**：学生的停止符是 `<|endoftext|>` (151643)，
教师的是 `<|im_end|>` (151645)，逐 token 比对时等于拿两个不同事件的概率做 KL。
本批的全部工作围绕这一点展开。

### 载体（carrier）：每条臂的共同底座

除 C 轴 4 条外，**所有臂都是"载体 + 恰好一个旋钮"**。载体本身就是判决基线
`vanilla_corr`：

| 旋钮 | 值 | 含义 |
|---|---|---|
| `DISTILLATION_LOSS_MODE` | `k1_termfix` | 学生采到停止符的位置，读**事件级** Δℓ = log q_T(E_T) − log p_θ(E_S)，而非 token 级 log q(eot) − log p(eot) |
| `DISTILLATION_TOPK` | 66 | 64 真 top-k + 2 列 gathered 终止符 |
| `SIMOPD_KEEP_SAMPLED` | 1 | 保留采样列（事件级读数需要） |
| `SIMOPD_GATHER_EOS` | 1 | 终止符单独成列 |
| `SIMOPD_EOS_IDS` | `151643` | 学生停止集 E_S |
| `SIMOPD_EOS_TEACHER_IDS` | `151643,151645` | 教师停止集 E_T |
| `SIMOPD_STOP_IDS` | `off` | rollout 只在 eot 停 |

"一条臂 = 载体 + 一个旋钮"是这批实验的设计纪律，也是结果可归因的前提。

### 共同训练配置

| 项 | 值 |
|---|---|
| 学生 / 教师 | `Qwen3-1.7B-Base` / `Qwen3-4B-Instruct-2507` |
| 响应上限 | 16384 token（prompt 上限 1024） |
| batch / mini-batch | 128 / 128，`rollout_n=1`，temperature 1.0，top_p 1.0 |
| 学习率 | 1e-6 |
| 存档 / 评测频率 | 每 25 步 |
| 步数 | 250（判决基线与预注册重点臂）/ 200（其余） |
| seed | 全部 s0，单 seed |

---

## 1. 总进度

| 口径 | 数值 |
|---|---|
| 实时步数 | **4252 / 6050 = 70.3%** |
| **落档步数** | **4025 / 6050 = 66.5%**（续跑从这里开始） |
| 完成臂 | **7 / 29** |
| 存档 | 165 个 checkpoint / 4.4 T |

两个口径差 227 步，是 `save_freq=25` 的固有窗口——进程被停时最后不足 25 步不落盘。

**判完成一律以 `global_step_<目标>` 目录是否存在为准**；用 `launch_*.jsonl` 的 step
最大值会漏掉收尾那一步，我前期因此少报过一次。

---

## 2. 已完成的 7 条

| 臂 | 步数 | 旋钮（相对载体的增量） | 这条臂在问什么 |
|---|---|---|---|
| **`vanilla_te`** | 250 | `k1_rec` + `TERM_EVENT=1` | **回归对照**：plain k1 + 载体 + `TERM_EVENT=1` 能否复现 `vanilla_corr`（`k1_termfix`）。两条路线此前被默认等价但**从未验证过**——它成立才能给 18 条 `_corr` 臂"载体 + 一个旋钮"的读法背书 |
| **`c2_quantile_budget_corr`** | 200 | `qb_quantile_budget`, margin=max, budget=8 | 分位数预算规则跑在修正后的终止符表示上。实测从 step 50 起 `qb_budget` 就钉死在 8.00、`captured_mass` 0.999——自适应已退化成"永远取 8 列" |
| **`c2_qb_perseq_corr`** | 200 | 同上 + `QB_SCOPE=sequence` | 预算梯子第 1 级：**每条轨迹一个 τ** |
| **`c3_intersection_corr`** | 200 | `intersection_topk`, TOPK 34, `TERM_EVENT=1` | thunlp 交集支撑跑在塌缩支撑上（Path 2 only）。已知它比载体**长 +24.8%** |
| **`c4_hq`** | 200 | `pi_tail_budget`, TOPK 34, `C4_HQ=1` | c4 + agreed-freeze（h 高**且** q_T(E_T) 高 → 退到 top-1）+ 续写条件覆盖目标。预测：停止位 m 从 0.18–0.27 降到约 0.01，T1 错停侵蚀消失，而早停教学（准确率来源）保留 |
| **`d3_teachability_corr`** | 200 | `teachability_select`, K=16, 保留率 5% | teachability = 分歧 × 兼容度，取前 5%。修正后长度 +32%，动态与载体明显不同 |
| **`h1_first_segment_corr`** | 200 | `k1_firstseg`, K=100 | 只监督**前 100 个 token** |

---

## 3. 未完成的 22 条

按剩余步数升序 —— 这就是下个窗口的推荐顺序。`ckpt` 是磁盘上最大存档步。

### 差 25 步（4 条，一个窗口能全收）

| 臂 | ckpt | 旋钮 | 这条臂在问什么 |
|---|---|---|---|
| **`vanilla_corr`** | 225/250 | 载体本身，无增量 | **判决基线**。语义对齐的最小改动：STOP token ratio → STOP **event** ratio。不加新损失、不加新梯度通道，是终止符章节的**伪影隔离格** |
| `c2_qb_fixed8_corr` | 175/200 | `QB_SCOPE=fixed` | 预算梯子第 0 级：**固定教师秩 top-8，零自适应**。三级 fixed → sequence → batch-quantile 只差 `QB_SCOPE` 一个字 |
| `c4_pi_tail_budget_corr` | 175/200 | `pi_tail_budget`, TOPK 34, ε=0.05 | π-tail 预算规则跑在修正表示上。corr 波覆盖 19 臂却跳过了 c2 和 c4；C 族唯一的 N0 证据是 c3，而它**不能外推**（c3 支撑是学生秩前缀，c4 是学生质量前缀，对"终止符坐标塌到 eot"的反应不同） |
| `d2_selectkd_corr` | 175/200 | `selectkd_verify`, k=5, β=0.01 | SelecTKD：学生提议 / 教师验证。Path 1 与 Path 2 **同时**生效 |

### 差 50–100 步（11 条）

| 臂 | ckpt | 旋钮 | 这条臂在问什么 |
|---|---|---|---|
| `f3_power_corr` | 150/200 | `k1_power`, α=1.0 | 概率尺度幂次奖励。legacy 效应最大处的 **null 对照** |
| `c5_union_fkl` | 175/250 | `union_fkl`, TOPK 34，**不加 N0** | 并集支撑上的 FKL：「**教会教师的终止符约定**」vs N0 的「翻译它」。预注册 un_p_imend 上升 |
| `c4_state` | 125/200 | `pi_tail_budget` + `C4_HQ=1` + `C4_REP=1` | c4 的**完整状态自适应控制器**（agreed-freeze + 续写目标 + rep-freeze）。预测：正确 ~1k / 准确率不变，错但停留在 ~6k，greedy-32k 截断从 20% 降到个位数 |
| `b1_skew_kl_corr` | 125/200 | `skew_kl_a0.1` | skew-KL（α=0.1）采样估计量。**B 轴 null 代表** |
| `e2_set_coverage_a0_corr` | 125/200 | `set_coverage_anchor`, anchor=0 | 集合覆盖目标（锚系数 0）。**第二条塌缩通路** |
| `a1_gkd_mix0.5_n0` | 100/200 | GKD 缓存, λ=0.5, TOPK 66, 双停止契约 | 固定混合的 GKD。与协作者的 v2 stop2 波次配对，**唯一差异是 N0** |
| `a3_offpolicy_n0` | 100/200 | GKD 缓存, **λ=1.0** | 纯离策 |
| `a4_dagger_anneal_n0` | 100/200 | GKD 缓存, λ 线性 1.0→0.0 / 250 步 | DAgger 退火。R1/R2 预注册裁决对象 |
| `g1_verified_only_corr` | 100/200 | `k1_verified_only` | 只用已验证轨迹的门。**G 轴最经典的一条** |
| `h3_random_segment_corr` | 100/200 | `k1_randseg`, K=100 | 一个**随机连续** 100-token 窗口。H 轴 null 代表 |
| `h4_random_scatter_corr` | 100/200 | `k1_randscatter` | **100 个随机散点** token。与 h3 配对回答"窗口 vs 散点" |

### 差 125–150 步（7 条）

| 臂 | ckpt | 旋钮 | 这条臂在问什么 |
|---|---|---|---|
| `c5_union_rkl` | 125/250 | `union_rkl`, TOPK 34，**不加 N0** | 并集支撑（教师 top-k ∪ 学生 top-k ∪ 精确终止符列）上的 RKL。wave-21 显示 `TERM_EVENT` 延缓熵塌缩的幅度远超停止位 6e-5 的占比，怀疑关键是**支撑几何**（损失终于看见学生自己的坐标）而非事件语义。**每步约 18 分钟，截断率 99%——这是被研究的现象，不是故障** |
| `b2_forward_kl_corr` | 75/200 | `forward_kl_topk`, TOPK 34 | 塌缩支撑上的非归一化教师 top-k 前向 KL。**唯一"分数≈但交付坏"的一条** |
| `f2_hard_clip_corr` | 75/200 | `LOSS_MAX_CLAMP=10.0` | 硬截 \|k1\| ≤ 10。文献标准做法 |
| `n2_corr` | 100/250 | `k1_termcal`, `EOS_AUX_COEF=1.0` | **N2 终止边缘校准通道**：事件级终末 Δℓ + 稠密的 stop-vs-continue BCE，跑在修正基座上 |
| `a5_aggrevate_n0` | 50/200 | T_max 线性 0→16384 / 250 步 | AggreVaTe。每步额外走一遍教师改标，因此比其他 A 轴臂慢约一倍 |
| `g6_seqmean_corr` | 50/200 | `LOSS_AGG_MODE=seq-mean-token-mean` | 逐序列均值聚合。G 轴 null 代表 |
| `h2_last_segment_corr` | 50/200 | `k1_lastseg`, K=100 | 只监督**最后 100 个 token**。与 h1 对照（h1 已完成） |

**剩余 2025 步。** 按实测约 6.6 步/lane·小时、16 条 lane 满载 ≈ 105 步/小时，需要约
**19 小时满载时间**，即 2.5–3 个夜窗。步时随响应长度增长而变慢，实际会更久。

> 排序说明：派工目前按名册优先级，不按剩余步数。今夜 `d2_selectkd_corr`（差 9 步）
> 整晚排在一条还要跑 38 步的臂后面没跑上。是否改成剩余步数优先待定。

---

## 4. 按轴看覆盖度

| 轴 | 内容 | 完成 / 总数 |
|---|---|---|
| **N（终止符语义，OURS 主线）** | 判决基线 + 回归对照 + 各族在修正基座上的重跑 | 6 / 21 |
| **C（支撑几何，OURS）** | c4 的两级控制器 + 并集支撑双向 KL | 1 / 4 |
| **A（在线策略程度）** | GKD 混合系数 λ ∈ {0.5, 1.0, 退火} + AggreVaTe | 0 / 4 |

### A 轴预演门

A 轴 4 条臂上机前要过机判彩排（3 步跑通 + 侧带判据）。**四条全 PASS，已解锁并起跑**：

```
PASS a1_gkd_mix0.5_n0
PASS a3_offpolicy_n0
PASS a4_dagger_anneal_n0
PASS a5_aggrevate_n0
```

**但第一次的 PASS 是打折的，这点必须记下来**：判据分派表按裸旋钮名写
（`a5_aggrevate)`），而名册臂名带载体后缀（`a5_aggrevate_n0`），`case` 匹配不上就掉进
默认分支——`a5` 被 a1 的 `gkd_mix` 判据判了 FAIL，而 a1/a3/a4 各自的 λ 断言
（a1 末值必须 0.5、a3 必须 1.0、a4 必须单调降）**从未执行**。修成按剥掉后缀的旋钮名
分派后重跑，四条才是按各自判据过的。**判据本身一个字没改**——这是让判定更严，不是放水。

---

## 5. 数据完整性

### 归档层：29 / 29 条臂齐整

| 文件 | 覆盖 |
|---|---|
| `traj/light.jsonl` | 29/29，7552 – 41856 行 |
| `traj/ids_<n>.parquet` | 每步一个，与实时步数一一对应 |
| `traj/summary_<n>.parquet` / `step_<n>.parquet` | 每 25 步一对 |
| `traj/div/rank*.jsonl` | 29/29 |
| `run_manifest.json` | 29/29 |
| `traj/tok_step<N>_rank0_*.parquet` | 按步分桶 |

### 教师列健康

`traj_dump` 的教师块 off-by-one 修复（读 `[P-1:P+L-1]`）在抽查的臂上都成立：

| 臂 | 采样点 | `tch_lp_nan` | `tch_lp_last` |
|---|---|---|---|
| `vanilla_te` | `summary_250` | **0.000** | −57.43 |
| `h1_first_segment_corr` | `summary_200` | **0.000** | −39.37 |
| `a1_gkd_mix0.5_n0` | `summary_100` | **0.000** | −26.16 |

修复前 `tch_lp_nan = 0.566`、`tch_lp_last` 全 NaN，`resp[i] == tch_top1_id[i]` 只有
0.046（修复后 0.769）。**这条不许回退。**

### 另外两条不许回退的修复

- **`trainer.use_v1=False`** —— 训练框架默认走 V1，而 SimOPD 所有 trainer 侧接缝都按
  类名挂在 V0 的 trainer 上。用 V1 时启动横幅照打、归档文件一个不写，`div` 步全 null。
  这是最危险的一类失败：**看起来一切正常**。
- **`SIMOPD_SHORT_RUN_OK=1`** —— 200 步是名册协议步长，不是泄漏的短跑。守卫给的三个
  逃生口里只有这个可用（另两个会改 run 名或把步数压成 3）。

---

## 6. 本轮（09-07 23:14 → 09-08 08:30 PDT）

- 实时步数 3398 → **4252**（+854 步，+14.1 个百分点）
- 新完成 3 条：`vanilla_te`(250)、`d3_teachability_corr`(200)、`h1_first_segment_corr`(200)
- A 轴解锁并起跑到 100/200（`a5` 到 50/200）
- 有效机时约 **261 GPU·小时**

### 损失（据实）

| 事件 | 代价 |
|---|---|
| 一个节点本地盘写满，7 条臂同时 ENOSPC | 4 条在跑的回退到上一存档，约 **60 步** |
| 掉队检测误杀一个健康作业 | `vanilla_corr / n2_corr / c5_union_fkl / c5_union_rkl` 回退约 20–25 步（`c5_union_rkl` 每步 18 分钟，这条最贵） |
| 掉队检测误杀 A 轴预演门 | 白跑 35 分钟，重来 |
| 时限被算错，作业起来 80 秒即死 | 4 条 lane 白加载一次模型 |

后三项都是我引入的判据缺陷，不是环境问题，已全部修掉：掉队检测改为只对训练作业、
且按"还没收工的 lane 数"算应占卡；时限只对已在运行的作业下调；磁盘加了开工清扫与
容量预检（不够就当场拒跑，而不是跑三小时再死）。派工也从贪心装箱改成按空闲容量摊平
（同样 17 条臂，并发从 9 条 lane 提到 16 条）。

---

## 7. 未决事项

### 阻塞性

| 项 | 现状 | 影响 |
|---|---|---|
| checkpoint 异地备份 | 未配置 | **4.4 T / 165 个 checkpoint 只有一份**。这批重训要补的，正是上一批 308 个 checkpoint 随集群消失。同步脚本已就位，配上凭据即自动传（仓库须 private） |
| 实验跟踪 | 离线模式 | 阻塞 AGENT.md §3 的旧波次重导出 |
| **评测** | **0 项** | 165 个 checkpoint **零评测**。评测专列已重写进 git 但从未真跑过；节点紧张时选了训练优先 |

### 待决定

1. **名册是 29 条还是缺 4 条？** AGENT.md §2.2 表头写 ≈3950 GPU·h，§2.2b 写
   训练 1380 + 评测 509 = 1888，两者差约 465 GPU·h / 4 条臂。
2. **`h5_gen100_n0` 跑不跑？** 它的停止契约（`STOP_IDS: "151645"`）与判决基线的 `off`
   不是同一个契约，所以我没把它注册进 `verdict.py` 的裁决台账。
3. **派工是否改成剩余步数优先**（见 §3 末）。

### 已知残余风险

- 判"lane 收工"靠收尾标记，而 lane 是**先放卡、后写标记**。今夜出现过一次单次误记
  （下一轮自行清零，没升级成取消）。若某条 lane 收尾超过一个巡检周期，仍可能误杀。
  更稳的判法是「已报结果的臂数 == 分配给它的臂数」。
- 训练框架取端口有 TOCTOU 竞态（拿到端口号后先关 socket 再重绑）。已把 lane 错峰
  从 15 秒提到 45 秒压低撞车概率；根治要改 vendored 代码（它自己另有安全版本）。

---

## 8. 下个窗口先做什么

1. 收掉差 25 步的四条：**`vanilla_corr`**、`c2_qb_fixed8_corr`、
   `c4_pi_tail_budget_corr`、`d2_selectkd_corr`。
   其中 `vanilla_corr` 是判决基线——它不完成，`verdict.py --base vanilla_corr` 无从谈起。
2. 然后是 `f3_power_corr`（差 50）与 `c5_union_fkl` / `c4_state` / `b1_skew_kl_corr` /
   `e2_set_coverage_a0_corr`（各差 75）。
3. A 轴四条已解锁，各差 100–150 步，可与上面并行。
4. **`vanilla_te` 已完成 250 步**——它是给全部 `_corr` 行背书的回归对照，
   建议优先把它与 `vanilla_corr` 的对比跑出来，因为后续所有 `_corr` 结论都依赖
   「`k1_rec` + `TERM_EVENT=1` ≡ `k1_termfix`」这个从未验证过的等价性。
