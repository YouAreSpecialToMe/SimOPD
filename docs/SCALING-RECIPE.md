# SimOPD 组合方法与模型规模实验

更新：2026-09-21。代码基于 `0917` 的 `60ab0cacd6ff2cabb4fb4382cc966a63f4b1ccc2`。
已实现并通过 CPU 数值测试；尚未进行 GPU 联调或提交训练。

## 实验问题与矩阵

固定组合方法，在不同 teacher / student 大小下，与同配置 Vanilla 比较。
每个模型对只跑两个方法，不以组件消融作为启动前提。

| Teacher / Student | 1.7B Base | 4B Base | 8B Base |
|---|---|---|---|
| Qwen3-4B | Vanilla / Ours | Vanilla / Ours | Vanilla / Ours |
| Qwen3-8B | Vanilla / Ours | Vanilla / Ours | Vanilla / Ours |
| Qwen3-32B | Vanilla / Ours | Vanilla / Ours | Vanilla / Ours |

**9 个模型对 × 2 个方法 = 18 runs / seed。** 先跑 seed 0；若整套补 seed 1、2，共 54 runs。
种子重复不包含在默认 Slurm 数组中。

Teachers 使用 `Qwen/Qwen3-{4,8,32}B`，students 使用
`Qwen/Qwen3-{1.7,4,8}B-Base`。统一 student-only rollout、non-thinking。
教师为冻结模型，对学生生成的序列评分。

主矩阵 4B teacher 选同代 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)，
不是历史 `Qwen3-4B-Instruct-2507`。后者可另作衔接实验，但没有加入本次 18 runs；
旧 32.82 baseline 不能代替新模型对的配对 Vanilla。大小相同的 Base 学生与后训练教师仍然是不同模型；
4B teacher → 8B student 也保留，以检查方法是否依赖教师更大。

## 固定方法

| 方面 | Ours |
|---|---|
| 生成来源 | Student-only，生成完整响应 |
| 响应筛选 | Failure-only：二值 verifier 判为 0 的响应 |
| 位置权重 | SelecTKD：student top-1 在 teacher top-5 中为 1，否则为 0.01 |
| 词表支持集 | Quantile 动态 budget，目标平均约 8 项，强制保留 teacher top-1 |
| 分布目标 | Reverse SKL，α = 0.1 |
| 信号裁剪 | 局部 SKL credit 对称裁剪到 [-10, 10] |

SelecTKD 已替换随机 100-token window；成功响应仍需生成、验证和教师评分。
第一版也仍返回 66 列教师候选（含终止事件 carrier），随后在 loss 中选择支持集。
因此这些组件不直接保证 rollout 或 teacher forward 更便宜；速度结论必须实测。

### 支持集与 selector

对候选词表项 j，margin 为 `max(p_full(j), q_full(j))`。
在每个 packed microbatch 的有效 response prediction rows 上计算分位阈值，目标平均 budget = 8。
这一 population 包括成功、失败响应，以及 selector 接受、不接受的位置；先计算支持集，再做两类加权。
并列分数和强制保留 top-1 可使实际平均支持集超过 8。

SelecTKD 使用动态筛选前、按教师概率排序的候选 top-5 与原学生 top-1；
终止事件沿用仓库的合并规则。它不使用裁剪后的支持集来重新定义 top-5。
支持集和 selector 均停止梯度。

Prediction rows 与固定 verl 的 `no_padding_2_padding` 一致：
`[sequence_end - response_length - 1 : sequence_end - 1]`。
排除 prompt-only rows 和最后的 dummy row；这与旧 `_stat_mask` 的最后 rl 行不完全相同，
本组合不宣称逐位复现历史 quantile arm。当前仅支持 SP=1。
不同模型的动态 packing 可能改变分位阈值 population，因此应保存实际 support size，不能把所有差异只归因于参数量。

### SKL 和 clip 的精确定义

在选中支持集 S 上分别重归一化学生 p 与冻结教师 q：

\[
m_j=\alpha p_j+(1-\alpha)q_j,\qquad
D=\sum_{j\in S}p_j\log\frac{p_j}{m_j},\qquad \alpha=0.1.
\]

使用在 p=q 时为零的局部 credit：

\[
c_j=\log\frac{m_j}{p_j}+\alpha\frac{p_j}{m_j}-\alpha,
\qquad \bar c_j=\operatorname{clip}(c_j,-10,10).
\]

反向传播的 surrogate 为

\[
L_{\rm surrogate}=-\sum_{j\in S}\operatorname{sg}(p_j\bar c_j)\log p_j.
\]

`sg` 表示停止梯度。不触发 clip 时，这一梯度等于 D 对学生参数的完整梯度，
包括 m 对学生的依赖；已用 float64 梯度对照验证。
触发 clip 后，这是明确修改过的更新方向。日志 forward loss 保留原 D，不能将其误读为未修改的 D 梯度。
同时记录 clip fraction、裁剪前后 credit 绝对值。

这与历史 sampled SKL + PG 不是同一梯度估计器，也不把 clamp 一个 KL 标量当作 credit clip。
Vanilla 保留固定 verl 默认的 sampled-loss `loss_max_clamp=10`；
Ours 显式设置 `loss_max_clamp=null`，防止内部 credit clipping 后再次裁剪标量 loss。

### Failure-only 的归一化

令 f_i 是响应失败指示量，v_it 是有效位置 mask，s_it ∈ {1, 0.01} 是 SelecTKD 权重。
完整更新使用

\[
\frac{\sum_{i,t} f_i v_{it}s_{it}L_{it}}
     {\sum_{i,t} f_i v_{it}}.
\]

分母是失败响应的有效 token 数；SelecTKD 仅作用于分子，不按被接受位置数重新放大。
driver 在 DP / microbatch 拆分前计算全局失败比例，并传入归一化因子，
与 verl 的全局 token-mean 相结合。局部 rank 没有失败响应时仍参与同步，贡献零梯度。
完整 batch 全部成功时跳过 optimizer 和 scheduler，避免零 loss 下仍有权重衰减。

筛选直接读取 `token_level_scores` 的二值 verifier outcome，不使用归一化 advantages。
缺失、非有限、非 0/1 的结果，以及空响应均拒绝运行。
250 是 rollout iteration 预算；全成功而跳过的更新不补做，应记录实际更新次数。

## 对照、评测与记录

配对固定：同一 student / teacher、数据、seed、batch=128、n=1、LR=1e-6、
250 rollout iterations、prompt cap=1024、response cap=16384、采样 temperature=1/top-p=1。
训练单次 full-batch update、epoch=1、token-mean；均启用相同终止事件处理。
每 50 iterations 保存与在线验证一次，保留全部 checkpoint。
在线验证是 MATH500 greedy，不能替代完整离线 benchmark composite。

离线统一评估 AIME24、AIME25、AMC23、Minerva、MATH500，所有方法使用相同 checkpoint 网格与 decoding / sample count。
沿用仓库的完整离线评测入口及锁定配置；此启动器不自动提交离线评测作业。
主比较用预先固定的最终 step 250，peak 仅作为附加统计，不能给不同方法不同挑选范围。
每个未训练学生另测一次起点；不把历史 1.7B 的 step-0 数值搬给 4B / 8B。

需要保存 accuracy、response length、entropy、clip fraction、failure fraction、support size、
SelecTKD 接受率、训练步耗时、显存峰值及 GPU-hours。
`recipe_*` 内核指标是有效响应位置上的诊断，包含被 failure gate 排除的成功响应；
不能把它们当成失败位置专属统计。轨迹归档可用于额外分组分析。

选择 SKL 的有限依据：现有 seed-0 step-200 离线 composite 为 34.54，Vanilla 为 34.25；
历史三种子 peak 的 SKL 33.81±0.78 与 JSD 33.83±0.26 基本持平。
因此固定 α=0.1 是缩小实验选择的决定，不宣称已证明 SKL 最优，也没有组合增益的实测结论。
同一步数数据摘录见 [evidence_at_200.csv](scaling/evidence_at_200.csv)，
来源文件摘要与排除的示意数据见 [sources.json](scaling/sources.json)。

## 32 卡部署

暂按既有集群文档的 **H100 80GB、每节点 8 卡** 制定：

| 每个训练任务 | 配置 |
|---|---|
| Student actor + rollout | 4 卡；FSDP，rollout TP=1 |
| Teacher scoring | 4 卡；32B 用 TP=4，4B/8B 用 TP=1 |
| 合计 | 8 卡 / run |
| 32 卡并行 | 4 runs；18 runs 至少 5 个调度波次 |

此配置尚未通过 GPU 显存测试。特别是 8B student、16K response 的训练峰值需要确认。
启动器要求可见恰好 8 卡且每卡至少 75 GiB；若实际硬件不同，应调整配置后重新 profile。
先在 32B teacher → 8B student 上各跑 Vanilla / Ours 的独立 25-step profile，
观察 warmup 后 step 时间与峰值，再提交正式矩阵。profile 不写入正式 run 目录。
必要时调整 token packing / offload / TP，并同步修改两种方法；只调整单臂会损害效率对比。

不能由参数量直接推算完工时间。测得每个 run 的训练步均耗时 t_r 秒后：

`training GPU-hours ≈ Σ_r 8 × 250 × t_r / 3600`

再加初始化、在线验证、存盘和离线评测开销。除以 32 只是理想并发下界，
32B 任务较慢以及最后不满四任务的波次会增加实际 wall time。

## 使用方式

使用项目已固定的 GPU 环境与 verl commit
`3d36367e83d7130f0b658b6be46db96d85ff9ce7`。
先在 GPU 节点缓存上述模型，并准备 `DATA_DIR` 中的 `train.parquet` / `math500.parquet`，
以及现有可写 `CKPT_ROOT`。默认离线加载模型。

```bash
# 生成清单（不会启动训练）
python scripts/scaling.py plan --out configs/scaling_seed0.csv

# 查看完整启动配置（默认 dry-run）
python scripts/scaling.py run --teacher 32 --student 8 --method ours

# 在已分配的 8-GPU 节点执行独立 profile
python scripts/scaling.py run --teacher 32 --student 8 --method ours --profile --execute
python scripts/scaling.py run --teacher 32 --student 8 --method vanilla --profile --execute

# Slurm: 设置当前 checkout 和已部署的 GPU 虚拟环境绝对路径
export SIMOPD_ROOT="$PWD"
export SIMOPD_VENV=/path/to/gpu/venv
# DATA_DIR / CKPT_ROOT 可由环境或项目 simopd_env.sh 设置
# 按集群补充 --partition / --account / --time
sbatch slurm/scaling.sbatch
```

Slurm 默认 `0-17%4`，只启动 seed 0。每个任务独占一个 8-GPU 节点。
它不会把 scaling arms 放入旧 campaign 的 lane roster。
已有集群若使用 DLC 而非 Slurm，可在分配好的 8 卡容器中调用同一个 `--execute` 入口。

## 验证状态

CPU 测试命令：`python -m pytest -q tests`，依赖 PyTorch、PyYAML、pytest。
19 项通过：完整 SKL 梯度、clip 极端值、支持集与 prediction 对齐、终止事件 carrier、
SelecTKD、跨 shard 的 failure 归一化、全成功跳过更新，以及模型矩阵与配对控制。
Shell 语法与 `git diff --check` 通过。

尚未验证真实 verl / Ray / vLLM 进程间传递、CUDA、FSDP 和显存；CPU 测试中的少量框架 stub 不代表集群联调。
没有提交训练、没有新的准确率或速度结果。
