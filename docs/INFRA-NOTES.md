# SimOPD Infra 笔记 v1(2026-07-31 代码勘察结论)

## v1.1 补充:集群实测(2026-07-31,unicorn/Cornell)

- **partition**:`nlplarge-sasha-highpri`(srun 会自动转到 `-interactive` 变体;
  PriorityTier=30,PreemptMode=REQUEUE,QoS 无每用户 TRES 上限 —— 2 卡纪律靠自觉)。
  节点 `nlplarge-compute-01`:8× A100-SXM4-80GB,256 CPU,2TB RAM,Ubuntu 24.04。
- **驱动 570.124.06 = CUDA 12.8 上限** → cu130 轮子(vLLM≥0.18 的 PyPI 默认)不能跑。
  解:**vLLM GitHub release 的 `+cu129` 轮子**(cu12 小版本兼容)+
  torch `2.11.0+cu129`(download.pytorch.org/whl/cu129)。sglang 备选被否:
  sglang 0.5.8 钉 transformers==4.57.1,与 verl 主线 (>=5.5.3) 冲突。
- **flash-attn 无 torch2.11 预编译轮** → 计算节点自编译(nvcc 12.8 在
  /usr/local/cuda-12.8;sm80 only;slurm/build_flash_attn.sbatch)。
  过渡期 smoke 用 sdpa + use_remove_padding=False(verl 的 flash_attn import 全是惰性)。
- **存储**:home NFS 全局 97%(~1TB 空);/share/rush 100% 满;用户 ~/.cache 占 915G
  (待清理)。**节点本地 /scratch 55T(3.6T 空)** → RAY_TMPDIR、checkpoints 放
  /scratch/zz865/;节点 /tmp 仅 15G,勿用。
- **配额实态(关键)**:QoS `nlplarge-share` 的 **MaxTRESPerAccount = cpu=64 /
  gpu=2 / mem=500G**(整个 rush 账号在此 partition 的总额)。即:**同一时刻只能有
  一个 2-GPU OPD run**;job 模板定为 `gpu:2 + 12c + 150G`(给并行的 hablf 留 4c/160G)。
  计划 §7.5 的"6-8 路并行"在这里不成立 → 见下一条。
- **竞争**:hablf_* 链(另一项目,每 job 1 GPU/2c/80G/6h)与 OPD 共享上述配额,
  两个 hablf GPU job 即占满 gpu=2。实测我们 sbatch 的 priority(~17.9k)高于
  hablf(~17.5k),GPU 释放时 slurm 会优先保留给 OPD job。仲裁权在人:不动 hablf 队列。
- **并行解法(W2 关键)**:`rush` partition = rush-compute-[01-03],**16× RTX A6000
  48G(sm86)+ 5× Titan RTX(sm75,FA2 不支持,不用)**,未见账号 TRES 上限。
  0.6B←1.7B 筛选 run 2×A6000 可容 → **筛选并行池放 rush,A100 对留给锚点/终验/4B
  teacher 档**。flash-attn 因此编 sm80+sm86 双架构;rush 节点驱动版本待探测确认。
- **模型缓存**:Qwen3 0.6B-Base/1.7B-Base/1.7B/4B/4B-Base/8B-Base/
  **4B-Instruct-2507**(Demystifying 现货 teacher,锚点用)均已入 HF cache。
- **协议实录**:Demystifying setup 抽取见 docs/PROTOCOL-demystifying.md
  (训练集 = nvidia/Nemotron-Cascade-RL-Math 14,476 题;PG 化 Δℓ_t advantage =
  verl `use_policy_gradient=True` + `loss_mode=k1`)。

## 结论:以 verl 主线原生 OPD 为基座

三个库勘察完(verl 主线 / thunlp/OPD / EasyOPD),裁定:

| 库 | 角色 | 理由 |
|---|---|---|
| **verl 主线** `verl/trainer/distillation/` | **基座** | 原生 OPD 完整:teacher 资源池 + 损失注册表 + 双训练模式;文档 2026-05-26 更新,活跃维护 |
| thunlp/OPD(Rethinking 官方库) | 锚点配方参考 | verl fork + slurm 脚本;含 token_reward_direct 等 adv 估计器变体;**注意其 N_RESPONSES=4、DeepMath-103K、resp 7168** —— 与 Demystifying 的 n=1/16k 不同,我们的协议跟 Demystifying |
| EasyOPD | 思路采石场 | 方法集大半在我们管辖权外(跨词表/vision/judge);但 5-hook 架构、`kl_renorm_topk` 开关(opcd.yaml)、sod 步级实现值得抄;环境钉死 Py3.11+CUDA12.4+H20 验证,不当基座 |

## 事故复盘:vanilla 首跑全损(job 674764,2026-07-31)

**现象**:300 步基线跑到 **229 步被 24h 硬超时杀掉**,`save_freq=300` → **零 checkpoint**;
Ray stdout 缓冲被 SIGKILL 丢弃 → **零指标**(step 行一条没落盘)。24 GPU·h 无产出。

**步时退化实测**(唯一幸存数据,tqdm 进度条):
step 1 = 103s → step 81 = 98s → **step 121 = 454s** → step 201 = **763s** → step 229 = 566s。
80→120 步之间 4-7× 变慢 = 长度通胀(**Mode A**)发作;结合 shakedown 的 val 时长
(step0 21min → step25 3.8min → step100 21min)构成 U 型:先坍缩后通胀。

**这是科学结果不是 bug**:vanilla(sampled-token RKL + PG,8k 帽)在 0.6B←1.7B 上
100 步内复现 Demystifying 的 Mode A。F 轴(软压缩)由此有了活体靶子。

**三条纪律(已落地)**:
1. `SAVE_FREQ=50`(不是 300)—— 长跑必须随时可断可续;
2. **wandb 必开**(`trainer.logger=["console","wandb"]`,已设为默认)—— console 日志
   在 Ray 缓冲里,进程被杀即全丢,wandb 逐步上报杀不掉;
3. 时限 24h → **72h**(partition MaxTime=UNLIMITED,之前是我自己设小了)。

**预算重估(影响 W2 排程)**:Mode A 下 300 步 ≈ 37-42h/run,不是原估的 8-12h。
18 臂串行 = 一个月,不可行 → 并行池成为刚需(rush A6000 或 PAI-DLC,
见 docs/DEPLOY-PAI-DLC.md)。若并行度仍不够,备选是把筛选步数从 300 降到 150
(Mode A 拐点在 100 步前已可见)—— 属协议参数变更,需预注册台账记录,待定。

## verl 免费送的(对照我们的轴)

- **teacher 服务全套**:独立 teacher 资源池,vLLM/SGLang 副本,`prompt_logprobs` 打分
  (prompt+response 前缀 + 1 个 dummy token),`max_logprobs` 自动抬到 ≥ topk;
  多 teacher 路由(`teacher_key`)。约束:池大小必须 == Σ(num_replicas × per_replica_world_size)。
- **损失注册表** `DISTILLATION_LOSS_REGISTRY` + 装饰器注册 —— 我们所有自研臂的插槽:
  - `forward_kl_topk`(topk 默认 32,**未重归一化**,负损失 clamp_min 0)
  - 采样 token 估计器族:`kl/k1/abs/mse/k2/low_var_kl/k3`(默认 k3)
- **双训练模式**:`use_policy_gradient`(TML PG-OPD,-loss 当 advantage 进 policy loss)
  vs 直接反传(GKD);`use_task_rewards` 可叠 PPO 任务奖励(= G 轴混合臂的现成半件)。
- **F 轴半件**:`loss_max_clamp`(硬 clip 已有!)、`log_prob_min_clamp`。
- **飞行记录仪半建成**:`overlap_ratio`、`overlap_token_advantage`、student/teacher mass
  已在打点 —— 代码注释直接引用 Rethinking 2604.13016。

## 缺口 → 接缝图(我们要写的)

| 臂 | 接缝 | 规模 |
|---|---|---|
| LSM 截断 RKL top-k(带重归一化开关) | 新注册 loss,仿 `fsdp/losses.py:compute_forward_kl_topk` | ~80 行 |
| 分位预算支撑(margin q/π/max) | 同上文件族;teacher 已回 top-K 张量 (bsz,seq,topk),把 K 开到 64–128 当候选池,batch 内直方图求 τ | ~120 行 |
| PL-rank + value 锚 / set-coverage | 同一注册表,新 loss fn | ~100 行 |
| skew-KL / JSD | 估计器族里加两个 loss_mode | ~40 行 |
| TIP / SelecTKD / Teachability token 选择 | 自定义 loss fn 内做权重掩码(model_output 有 log_probs,熵可算;SelecTKD 需 student top-1 vs teacher top-k 比对,张量都在) | 各 ~60 行 |
| 软 log 压缩 | 估计器路径上的逐 token 变换(硬 clamp 旁边加一档) | ~20 行 |
| verified-only 门控 | rollout 后按 verifier reward 过滤 batch(verl reward 管线现成,加一个 trainer 侧 filter) | ~50 行 |
| 冷启动 | 独立 SFT 阶段(verl sft 或 LlamaFactory),非 OPD 环内 | 配置活 |

**总侵入面 ≈ 500–600 行,全部走注册表/装饰器,不改 verl 核心。**

## 0.6B 槽位落地(对照 §7.5)

verl 的 teacher 池是**独立 GPU 组**(不与 actor 同卡),所以最小槽 =
actor(训练+rollout hybrid)1 卡 + teacher 池 1 卡(1.7B 或 4B 均 TP=1 稳进 80G,
`gpu_memory_utilization` 默认 0.5 可调)= **2 卡/run 证实可行**;16 卡 = 8 路并行。
终验档(1.7B←4B)同构,2–3 卡。

## W1 清单

- [ ] `pip install -e verl` + 环境(vLLM 版本对齐 verl 要求)
- [ ] 跑通官方 OPD 示例(GSM8K 小配置,验证 teacher 池起得来)
- [ ] 复现锚点:1.7B-Base ← 4B,math,n=1/batch128/16k,对 Demystifying 中档曲线
- [ ] 把 verl 已有 overlap 打点接到我们的飞行记录仪面板
- [ ] 按接缝图开工自研臂(先 LSM + 软压缩两个最小件,验证注册表路径)
- [ ] 从 EasyOPD 抄 `kl_renorm_topk` 语义 + hook 设计笔记

## 风险备忘

- teacher 打分吞吐:每 run 每步 ~128×(≤8k) token 的 prefill,1 卡 4B 应该够,W1 实测;
- nested tensor 格式(thd/bshd)在 Qwen3 上的坑:文档提到 Qwen3.5 不支持 THD,
  0.6B/1.7B/4B 用 bshd 保险;
- thunlp fork 与主线 API 已分叉,抄配方不抄代码。
