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

## 事故复盘 2:改动正在运行的脚本(2026-08-01,彩排 c1_lsm 假 FAIL)

**现象**:c1_lsm 报 FAIL,日志只有一行
`run_opd_baseline.sh: line 110: _tokens}: command not found`。

**根因**:**bash 按字节偏移量边读边执行脚本**。我在彩排正跑着 c1 时编辑了
`run_opd_baseline.sh`(加了 4 行注释 ≈250 字节),bash 于是在旧偏移处恢复,
正好落在 `${max_num_tokens}` 中间。**kernel 代码完全无辜,是一个自造的假阳性。**
危险之处在于它长得像真失败 —— 差一点就去 debug 那个没问题的 top-k dispatch。

**结构性修复**:campaign / DSW lane 启动时把 `scripts configs src` 拷进
`$SNAP` 快照目录并从那里运行(PYTHONPATH 也指向快照)。这样 40 小时的正式
campaign 期间工作区可以随便改,不影响在跑的 run;快照目录名带 job id 与 git sha,
顺带成了"这个 run 到底跑的哪份代码"的凭证。

**纪律**:不要编辑正在被执行的 shell 脚本;要改就改工作区,run 用快照。

## 版本矩阵(2026-08-03 实测可用组合)

| | 版本 | 说明 |
|---|---|---|
| Python | **3.12.7** | |
| torch | **2.11.0+cu129** | 钉死。cu129 而非 PyPI 默认的 cu130:cu130 要驱动 ≥580,cu129 在 ≥525 上都能跑 |
| torchvision | **0.26.0+cu129** | 与 torch 严格配对 |
| torchaudio | 2.11.0(+cu129 亦可) | 纯文本训练用不到,加载失败不影响 |
| vLLM | **0.26.0+cu129** | 钉 torch==2.11.0 |
| transformers | 5.10.4 | verl 主线要求 >=5.5.3,<5.11 |
| **flash-attn** | **2.8.3.post1** | 无 torch2.11 预编译轮,必须源码编译 |
| CUDA 工具链 | nvcc 主版本须与 torch 一致(12.x ↔ cu129) | 12.9 vs 12.8 只是小版本,torch 仅警告 |
| GPU 架构 | **sm80**(A100)→ `TORCH_CUDA_ARCH_LIST=8.0` | A10=8.6 / L20=8.9 / H20·H800=9.0 |

**仓库自带编好的 wheel**:`deploy/dsw/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl`
(54MB,在 Cornell 机器上对着 torch 2.11.0+cu129 / cp312 / cxx11abiTRUE 编的,含 sm80+sm86)。
`setup.sh` 检测到就用 `--no-deps` 装,A100 直接可用。**之所以要自带,是因为官方没有 torch2.11
的预编译轮** —— Dao-AILab 的 cp312 轮子只到 torch 2.10。

**任何情况下都要 `--no-deps`。** 它把 `torch` 声明成无上限依赖,直接 pip install
会让 torch 被升到最新(实测:2.11.0+cu129 → 2.13.0+cu129),torchvision 与 vLLM 随之
失配,刚编好的扩展也对着一个已不存在的 torch,报 undefined symbol。**症状像 flash-attn
的问题,病因是它把 torch 换掉了。**

```bash
source simopd/bin/activate
SITE=$(python -c 'import site;print(site.getsitepackages()[0])')
rm -rf $SITE/flash_attn $SITE/flash_attn_2_cuda*.so $SITE/flash_attn-*.dist-info
FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=32 \
  pip install --no-deps --no-build-isolation --no-cache-dir flash-attn==2.8.3.post1
python -c "import torch, flash_attn_2_cuda; print('ok', torch.__version__)"   # torch 必须先导入
```

没有 flash-attn 也能跑:`USE_REMOVE_PADDING=False`,慢一些,其余相同。

### 迁移列的依赖(2026-08-03 加)

| 包 | 装法 | 为什么 |
|---|---|---|
| evalplus | **`--no-deps` 装 0.3.1** | 依赖闭包里有 `google-generativeai`,会把 **protobuf 6.33 拉回 5.29**,vllm/ray 随之失配。我们只用它的数据 + 单测执行,不用它的生成后端 |
| appdirs / tempdir / wget / termcolor / tree-sitter(+python) / multipledispatch | 手写列全 | 上面 `--no-deps` 之后 evalplus 真正需要的运行时依赖 |
| langdetect / nltk / immutabledict | 正常装 | IFEval checker 的依赖(absl 已有) |
| instruction_following_eval | **仓库内 `third_party/`** | Google 官方 checker **没有 PyPI 发行版**;PyPI 上的 `ifeval`(0.0.1)是无关第三方上传 |

`--dry-run --report` 先看清单再装,配一份把 torch/vllm/transformers/numpy/protobuf/ray
钉死的 constraints —— 装完复验这六个没动。这条纪律是 flash-attn 那次换掉 torch 换来的。

**nltk 3.10 的导入守卫会误伤我们**:它会拦截"源路径落在 CWD 内"的传递依赖导入,
而我们的 venv 就在 `<repo>/simopd`,于是从仓库根目录跑时**每一个** site-package 都像
CWD 导入,`import nltk` 直接炸在 `regex` 上。它自带的提示(用 `-P` / `PYTHONSAFEPATH`)
在这个布局下**没用** —— 那是控制 CWD 搜不搜,而这里是 origin 落在 CWD 内。
正解是官方开关 `NLTK_DISABLE_IMPORT_SECURITY=1`,且必须在首次 `import nltk` 之前设。
代码里已在 `transfer_eval` / `fetch_assets` 内就地设好。

**evalplus 的 ground truth 很大**:HumanEval+ 255MB、MBPP+ 793MB 的 pkl,
`fetch_assets.py` 在 setup 阶段预热 —— 它是执行全部 canonical solution **并记录耗时**
算出来的,而那些耗时随后就是单测时限的基准,放到 campaign 里现算等于让基准在满载节点上测。

## 事故复盘 3:代码评测会因机器负载假失败(2026-08-03)

同一批 HumanEval+ canonical solution,**load≈20 时 160/164,load≈6 时稳定 163/164** ——
输入、代码、随机性都没变,三个失败是假的。

病因:evalplus 每个单测的上限是 `max(min_time_limit, 4×参考耗时)`,平凡函数
(`string_sequence` / `make_a_pile` / `tri`)配上 extreme 输入时,4×参考耗时远小于
`min_time_limit`,于是落到 **1.0s 地板**;CPU 争抢单独就能击穿它。

为什么这条对本项目致命:**争抢源就是我们自己** —— 四条训练泳道占满节点时评代码,
某个臂会看起来"把代码能力搞坏了",而其实什么都没发生。这正是迁移列要防的那种假判决。

处置:默认 `min_time_limit` 抬到 **4.0**(`SIMOPD_EVALPLUS_MIN_TIME_LIMIT` 可调);
ground truth 预热到 setup;`transfer_eval.py --selfcheck` 换机器时自证。
代价是超时用例本身变慢(地板抬高 4 倍),这个交换值得。

**HumanEval/32(`find_zero`)在任何时限下都挂** —— 牛顿法 tol 1e-5 在更难的多项式上
真的不收敛。**163/164 是 harness 自身的参考天花板**,不是我们的缺陷,别去"修"它。

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

## 事故复盘 4:发布器把续跑当成另一次尝试(2026-08-19)

`exp_publish.py`(在 simopd_data,不在仓库)两处都用「last step 最大的那次尝试胜出」
挑日志。它防的是死掉的双胞胎:一条 lane 起来跑 8 步就死、另一条跑到 250,不能让前者
的 1–8 步污染后者。同一条规则遇上**续跑**就砍曲线 —— 08-18 舰队重排后
b4_jsd_b0.1/b0.9、f2_clip2.3、h4_random_scatter 六个 run 从 step 50 的 bank 接着跑,
新日志只记 51 步往后,于是 1–50 整段被丢掉:11 条 (arm,seed) 曲线从 ~60 行掉到 13 行,
in-loop 表里 s25/s50 两列从三 seed 退成 `·1`,而表面上一切正常 —— 最大 step 还涨了。

改成按 step 合并:同一个 run 的所有尝试按日志 mtime 从旧到新叠,step 撞了新的赢。
双胞胎那个顾虑照样成立(幸存者后写,盖掉双胞胎的前几步),续跑则正确接上。
另外把日志 glob 从 `logs/m*/lane*.log` 放宽到 `logs/[mj]*/lane*.log` —— DLC 舰队
(j4d*/j5d*)的 70 份 lane 日志此前整批不可见,昨天所有训练都在那里。

教训与复盘 2 同源:**「更远」不等于「更全」**。发布器做取舍时,取舍规则要跟着数据的
产生方式走,而不是跟着某个单调量走;这次的检验是拿上一版 CSV 逐 (arm,seed) 比行数和
起点,而不是只看最大 step。

## 风险备忘

- teacher 打分吞吐:每 run 每步 ~128×(≤8k) token 的 prefill,1 卡 4B 应该够,W1 实测;
- nested tensor 格式(thd/bshd)在 Qwen3 上的坑:文档提到 Qwen3.5 不支持 THD,
  0.6B/1.7B/4B 用 bshd 保险;
- thunlp fork 与主线 API 已分叉,抄配方不抄代码。
