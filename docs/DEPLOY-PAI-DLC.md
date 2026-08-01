# 在阿里云 PAI-DLC 上部署 SimOPD 实验(v1,2026-08-01)

> 前提:已有 PAI 工作空间 + 专有资源组(卡)。
> 目标:**同一份臂定义在本地 slurm 和 PAI-DLC 上跑出同一个协议** —— 审计项目里
> 硬件可以换,协议不能换。

## 0. 形态映射

我们的一个 run = 单机 2 卡(actor 1 + teacher 池 1),Ray 在容器内本地起。
所以 **不需要多机分布式**,DLC 侧就是最简单的一种作业:

| 我们的概念 | DLC 对应物 |
|---|---|
| 一个臂(一个 run) | 一个 `pytorchjob`,`--workers=1 --worker_gpu=2` |
| slurm 的 `sbatch` | `dlc submit pytorchjob`(见 `deploy/pai/submit_dlc.sh`) |
| `.venv` + flash-attn 自编 | 自建镜像(见 `deploy/pai/Dockerfile`),推到 ACR |
| `~/.cache/huggingface`、`~/data`、`/scratch/.../ckpt` | 一个 NAS 数据集,挂到 `/mnt/data` |
| 并行度(本地被账号配额卡死在 gpu=2) | 资源组内并发提交多个 job = W2 贪心轮的并行池 |

Ray 是**容器内单机**模式,不要用 DLC 的多机 worker 去拆 actor/teacher ——
verl 的 teacher 资源池是同一个 Ray 集群内的独立 GPU 组,拆到两个 DLC worker 上
反而要自己接 Ray 组网,没有收益。

## 1. 镜像(一次性)

`deploy/pai/Dockerfile` 复刻了 2026-07-31 在 Cornell 集群验证通过的整套栈:
torch 2.11.0+cu129 / vLLM 0.26.0+cu129(GitHub wheel)/ verl 主线 / flash-attn 源码编译。

```bash
# 在能连外网的构建机上(不需要 GPU,nvcc 编译不需要卡)
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.0;9.0" -t simopd:v1 -f deploy/pai/Dockerfile .
docker tag  simopd:v1 registry-vpc.cn-<region>.aliyuncs.com/<namespace>/simopd:v1
docker push registry-vpc.cn-<region>.aliyuncs.com/<namespace>/simopd:v1
```

✅ **卡型已确认(2026-08-01):A100 = sm80,与本地集群同架构。**
于是本地已编好的 `~/wheels/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl`
(编的就是 sm80;86)**可以直接复用** —— 拷进构建上下文即可省掉 14 分钟编译:
`cp ~/wheels/flash_attn-*.whl deploy/pai/`。Dockerfile 有则用之,无则源码编译。
同架构还有个额外好处:**跨集群硬件差异从"未知风险"降级为"同代同架构"**,
§6 的 vanilla 双跑校验从必需品变成便宜的保险。

✅ **显存已确认:A100-80G**,与本地 `nlplarge-compute-01`(A100-SXM4-80GB)同款。
`gpu_memory_utilization`(actor 0.6 / teacher 0.85)按 80G 调的,**原样搬过去即可,
一个数都不用改**。

若 PAI 给的是 PCIe 版(gn7e)而非 SXM4:**对我们无影响** —— actor 与 teacher 都是
TP=1,两者之间走 Ray/HTTP 而非跨卡 NCCL 集合通信,NVLink 用不上;剩下的只是
HBM 带宽 1935 vs 2039 GB/s 与 TDP 300W vs 400W,量级在 5-15% 吞吐,不改任何结论。

剩下唯一要扫一眼的是驱动:cu129 轮子需要驱动 ≥ 525(CUDA 12.x 小版本兼容),
PAI 的 A100 节点基本都满足,上机 `nvidia-smi` 确认即可。

用 `registry-vpc.*` 内网域名拉镜像(免公网流量、快很多)。

## 2. 存储(一次性)

建一个 **NAS 数据集**(PAI 控制台 → 数据集 → NAS),挂载路径用默认 `/mnt/data`,
拿到形如 `d-xxxxxxxxxxxx` 的 ID。里面按这个布局放:

```
/mnt/data/
├── hf_cache/        # HF_HOME:Qwen3 全家(0.6B/1.7B/4B/4B-Instruct-2507…)约 40GB
├── simopd_math/     # train.parquet(Nemotron 14,476)+ math500.parquet
├── ckpt/            # 各 run 的 checkpoint(0.6B 每个约 4-5GB)
└── wandb/           # 离线模式时的落盘目录
```

灌数据两条路:① 开一个 DSW 实例挂同一个 NAS,在里面 `hf download`(国内建议
配 HF 镜像端点或走 ModelScope);② 本地打包 rsync 到 NAS。
**模型务必预置好并 `HF_HUB_OFFLINE=1`** —— 训练作业里现拉模型既慢又容易失败。

## 3. 提交一个臂

```bash
export WORKSPACE_ID=...   RESOURCE_ID=quota-xxxx
export IMAGE=registry-vpc.cn-<region>.aliyuncs.com/<ns>/simopd:v1
export DATA_SOURCES=d-xxxxxxxxxxxx

EXPERIMENT_NAME=vanilla_s0 ./deploy/pai/submit_dlc.sh
EXPERIMENT_NAME=lsm_topk32 DISTILLATION_LOSS_MODE=forward_kl_topk ./deploy/pai/submit_dlc.sh
```

`submit_dlc.sh` 的环境变量接口和 `slurm/baseline.sbatch` **完全一致**
(EXPERIMENT_NAME / TOTAL_TRAINING_STEPS / MAX_RESPONSE_LENGTH / ACTOR_LR …),
两边跑的是同一个 `scripts/run_opd_baseline.sh`,协议不会漂。

关键参数:
- `--worker_gpu=2`(我们的标准槽位);
- `--job_max_running_time_minutes=0`(不限时)。**本地就是栽在 24h 硬超时上的**,
  Mode A 长度通胀会让 300 步远超 24 小时,这里千万别设小;
- `--priority`(1-9)在资源组内排队用;
- `--envs` 透传协议旋钮。

## 4. 上机前必查清单

| # | 检查 | 为什么 / 不过关怎么办 |
|---|---|---|
| 1 | `python -c "import flash_attn"` + 一次前向 | 架构不匹配会在第一个训练步才炸,别等长跑 |
| 2 | `df -h /dev/shm` | **Ray object store 吃共享内存**,容器默认 64MB 会直接起不来;不足时调大 shm 配额,或降 `RAY_object_store_memory` |
| 3 | `ls /mnt/data/hf_cache` + `nvidia-smi` | 挂载和卡型二连确认 |
| 4 | `curl -s -o /dev/null -w "%{http_code}" https://api.wandb.ai` | 通就在线记录;不通改 `WANDB_MODE=offline` 落 NAS 事后 `wandb sync`(thunlp 那套用 SwanLab,也是国内可用的替代) |
| 5 | 先跑 `TOTAL_TRAINING_STEPS=10 TEST_FREQ=-1` 的冒烟 | 和我们本地 smoke 同一套验收:双资源池起得来、k1+PG 出数、10 步跑完 |

## 5. 抢占与断点(重要)

资源组被抢占 / 作业重启时,没存 checkpoint 就等于全丢 ——
**2026-07-31 本地那次 24 小时长跑就是这么白烧的**(`save_freq=300`,229 步被杀,
零 checkpoint、零指标)。所以:

- `SAVE_FREQ=50`(0.6B 每个 ckpt 约 4-5GB,300 步 6 个 ≈ 30GB,NAS 完全放得下);
- **wandb 必开**:verl 的 console 日志走 Ray stdout 缓冲,进程被 SIGKILL 时缓冲直接丢,
  曲线全无;wandb 是逐步上报,杀了也留得下;
- verl 支持从 `default_local_dir` 断点续训,重交同名作业即可接上。

## 5.5 槽位与并行度(卡怎么算)

**一个 run = 2 张 A100,这是硬下限。** verl 把 teacher 池注册成独立的 Ray 资源池
(`resource_pool_spec["teacher_pool"]`,`n_gpus_per_node` 必须为正整数),
**结构上不允许 teacher 与 actor 共卡**,不改代码就省不掉这张。

| 档位 | 卡数 | 依据 |
|---|---|---|
| 筛选 0.6B ← 1.7B(W2 主力,~18 臂) | **2**(actor 1 + teacher 1) | 已实测:smoke/shakedown/基线全是这个配置 |
| 失配档 0.6B ← 4B-Instruct-2507 | **2** | 4B bf16 ≈ 8GB,单卡 vLLM 宽裕 |
| 锚点/终验 1.7B-Base ← 4B(16k) | **2**(80G 下大概率够,待实测) | 0.6B@8k 时 actor 峰值 reserved 仅 23GB/80GB;1.7B@16k 粗估 50-70GB,80G 卡放得下,首跑盯 `max_memory_reserved` |

显存余量说明:0.6B@8k 的 actor 只用到 80G 的约 29%,**余量很大但换不成卡** ——
瓶颈是 verl 的资源池结构,不是显存。

**所以:并行度 = 总卡数 ÷ 2。** 这个数直接决定 W2 排期(按当前 ~40h/run 的
Mode A 实测;贪心轮之间必须串行,轮内可并行):

| PAI 总卡数 | 并行 run | R1(约 18 臂)墙钟 |
|---|---|---|
| 8 | 4 | 5 批 × 40h ≈ 8 天 |
| 16 | 8 | 3 批 × 40h ≈ 5 天 |
| 36+ | 18 | 1 批 ≈ 1.7 天 |

若卡数不足以把 R1 压进 3-4 天,**备选是把筛选步数 300 → 150**(Mode A 拐点在
100 步前已可见),属协议参数变更,需台账记录。

## 6. 与本地集群的分工建议

| 用途 | 放哪 | 理由 |
|---|---|---|
| W2 贪心轮筛选(~18 臂 × 300 步) | **PAI**(并发提交) | 本地账号配额锁死 gpu=2,只能串行;这是迁 PAI 的唯一真正理由 |
| 复现锚点(1.7B←4B-2507,16k) | 本地 A100 对 | 长跑单条,不抢并行度;A100 与 Demystifying 环境更近 |
| 离线评测(AMC23 avg@32、pass@8 面板) | 任一,推荐本地 | 纯推理,1 卡即可 |

**跨集群纪律**:同一个臂只在一个集群上跑完(避免半程换硬件)。
硬件侧本来是大风险,但 PAI 与本地**都是 A100-80G、同架构 sm80、同一个镜像栈**,
风险已降到很低 → vanilla 双跑校验保留为**便宜的保险**(一条 run),
把 pass@1 差值记进台账;若差值落在噪声底内,后续可直接混用两边结果。

## 7. 待你确认的账号侧信息

- 卡型(决定 `TORCH_CUDA_ARCH_LIST`)与单作业可申请的卡数;
- 资源组 ID / 工作空间 ID;
- 共享内存上限(见清单 #2);
- 是否能连 HuggingFace / wandb(决定离线策略)。

参考文档:
[dlc submit 命令与参数](https://help.aliyun.com/zh/pai/developer-reference/commands-used-to-submit-jobs) ·
[DLC 是什么](https://help.aliyun.com/zh/pai/what-is-dlc) ·
[DLC 中挂载 OSS/NAS](https://help.aliyun.com/zh/pai/user-guide/use-cloud-storage-for-a-dlc-job) ·
[读写 NAS 数据](https://help.aliyun.com/zh/pai/user-guide/use-nas) ·
[数据集挂载](https://help.aliyun.com/zh/pai/read-and-write-dataset-data)
