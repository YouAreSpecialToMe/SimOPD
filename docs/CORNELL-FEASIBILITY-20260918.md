# 这 650 步能不能在 Cornell 跑(2026-09-18)

交接包里 10 条臂还差 650 步。问题是 Cornell 的 unicorn 集群能不能接。
结论:**只能在 A100 80 G 上跑,A6000 48 G 起不来**,而 A100 现在全被我们自己的
verification 项目占着。

## 1 卡的盘子

| 节点 | 卡 | 架构 | 显存 | 当时空闲 |
|---|---|---|---|---|
| `rush-compute-01` | 5× TITAN RTX | Turing sm_75 | 24 G | 5 |
| `rush-compute-02` | 8× A6000 | Ampere sm_86 | 48 G | 0 |
| `rush-compute-03` | 8× A6000 | Ampere sm_86 | 48 G | 1 |
| `nlplarge-compute-01` | 8× A100-SXM4 | Ampere | 80 G | 0 |

集群开了 `PrivateData=jobs`,`squeue` 只看得到自己的作业,所以"谁占着"查不到,
但 `scontrol show node` 的节点级占用是准的。

`rush` 分区:`PriorityTier=20`、`MaxTime=UNLIMITED`、`PreemptMode=REQUEUE`,
`rush` 账户 QoS 是 `normal`,关联行里没有 gres/gpu 上限 —— 没有配额卡我们,
纯粹是抢不抢得到。

**TITAN RTX 那 5 张对训练没用**:Turing 不支持 bf16,而协议钉死学生 FSDP bf16
混合精度。做推理对照可以(vLLM 会静默降 fp16),进表不行 —— 同一张 benchmark
表里混进不同数值路径的格子。

## 2 A6000 为什么起不来

存档文件名是 `model_world_size_1_rank_0.pt` / `optim_world_size_1_rank_0.pt` ——
当初按 **FSDP world size 1** 存的,verl 只能按同样 world size 载回来。所以学生的
静态态**不能拆到多卡分摊**,必须整个压在一张卡上。

`run_opd_baseline.sh` 自己的账本(1.7B bf16 混合精度):3.2 权重 + 6.3 fp32 master
+ 12.7 AdamW 双矩 + 3.2 梯度 = **25.3 G**。存档大小对得上:落盘
7.57 G(fp32 master)+ 12.82 G(AdamW)= 20.4 G。

vLLM 的 `gpu_memory_utilization` 是**引擎初始化时按总显存预留的比例**,不是用量上限:

| | A100 80 G(当初跑的) | A6000 48 G |
|---|---|---|
| 学生静态态(不可分摊) | 25.3 G | 25.3 G |
| vLLM 预留 `0.45 ×` 总显存 | 36.0 G | 21.6 G |
| 小计 | 61.3 / 80 | **46.9 / 48** |
| **留给全部激活** | **18.7 G** | **1.1 G** |

而按钉死的 `ppo_max_token_len_per_gpu=17408`,**单个全词表 logits 张量就是**
17,408 × 151,936 × 2 B = **4.93 GiB** —— 一个张量就是剩余预算的 4.5 倍。
第一次前向就死,不是"慢一点但能跑"。

## 3 每个能修它的旋钮都在续训指纹里

`scripts/run_opd_baseline.sh:389` 把这些哈希进 `simopd_fingerprint.txt`:

```
ngpus  tws  vllmmem  tokpergpu   ← 要塞进 48 G 必须动的,全在这里
student teacher loss pg topk taskrw dcoef rolloutn arm extra simopd
bs mini lr prompt resp data pad
```

不符即 `FATAL: ... from a DIFFERENT config` 退出。`RESUME=force` 能过,但它把新
指纹写回去 —— 就是"两段配置拼进一条曲线,日志里没有任何东西说明这件事"。
交接 README 也写了同一句:不要绕过它。

**指纹抓不到的一件事**:它只哈希配置,**不哈希库版本**。集群现在是
vllm 0.17.1 / torch 2.10.0+cu128 / transformers 4.57.6 / flash_attn 2.8.3,
verl 没装;名册钉的是 verl `aebd1f8` + vLLM 0.26。`campaign.sh --fingerprint`
会比对这些,但它**不参与续训闸门** —— 在版本不同的环境里续训,指纹会放行,
差异会静默骑在曲线上。真要在这边续训,这一项得手工核对。

## 4 要多少机时(实测,不是估的)

取各臂自己 `perf/time_per_step` 的末窗中位数 × 2 卡/lane:

| 臂 | 还差 | 末段步时 | GPU·h |
|---|---|---|---|
| `c5_union_rkl` | 100 | 1071 s | 59.5 |
| `g6_seqmean_corr` | 75 | 970 s | 40.4 |
| `e2_set_coverage_a0_corr` | 75 | 927 s | 38.6 |
| `h2_last_segment_corr` | 100 | 692 s | 38.5 |
| `c5_union_fkl` | 50 | 1005 s | 27.9 |
| `a5_aggrevate_n0` | 100 | 391 s | 21.7 |
| `h4_random_scatter_corr` | 50 | 720 s | 20.0 |
| `h3_random_segment_corr` | 50 | 596 s | 16.6 |
| `g1_verified_only_corr` | 25 | 831 s | 11.5 |
| `a4_dagger_anneal_n0` | 25 | 238 s | 3.3 |
| **合计** | **650** | | **278** |

步时随长度涨,278 是下界,预算按 300–350 更稳。8 张 A100 开 4 条 lane 约
**1.5–2 天**;只有 2 张就是 6 天。

## 5 A100 的档期

`nlplarge-compute-01` 是唯一的 A100 节点,8 张全是我们 verification 项目的作业。
按 2026-09-17 的读数,`p2novgate` 最先放 2 张,`p2stack` 最晚,后面还排着一个
14 天上限的 `p2stack`。那是 ICLR 正文 9/25 的数据。**截稿前 A100 基本不是我们的。**

## 6 现在就能开工的一半

补评测**不碰指纹**:冻结 checkpoint 上跑 benchmark,没有优化器态,1.7B bf16
只要 ~3.4 G 权重。**A6000 绰绰有余。** 规模见交接 README §B —— 99 格待评,
93 个纯推理权重 597 GB,单卡约 470 GPU·h。

不能用那 5 张 TITAN RTX:Turing 无 bf16,vLLM 静默降 fp16,不报错但混进
不同数值路径,进不了同一张表。

## 7 复现

```
scontrol show node <node> | tr ' ' '\n' | grep -E 'Gres|AllocTRES|CfgTRES'
sinfo -p rush -N -o '%N %t %C %G'
sacctmgr -n show assoc account=rush format=account,user,qos,grptres%30
python3 scripts/analysis/step_times.py   # 末窗步时 → GPU·h
```

非交互 ssh 要先 `export PATH=/usr/local/slurm/current/bin:$PATH`。
