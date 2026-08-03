# SimOPD

**Auditing the on-policy distillation (OPD) variant zoo → a minimal effective recipe.**

对现有 OPD 变体文献(zoo 462 条目、核心受审池 ~55 篇)做统一协议审计:
每个 trick 判为 **必需 / 无用 / 有副作用**,双向贪心蒸出最小有效配方 **SimOPD**。
方法论血统:LitePPO(Tricks or Traps, arXiv:2508.08221)之于 GRPO/DAPO。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/SimOPD-plan.md](docs/SimOPD-plan.md) | 实验计划 v3.1:定位、7 轴、诊断 D1'–D6、三阶段流程、判决规则、硬件预算、里程碑 |
| [docs/SimOPD-casefile.md](docs/SimOPD-casefile.md) | 案卷:动物园普查、管辖权裁定、8 轴参赛名单(审/替/落)、代表选择准则 |
| [docs/INFRA-NOTES.md](docs/INFRA-NOTES.md) | infra 勘察:verl 主线原生 OPD 基座裁定、缺口→接缝图、槽位布局、W1 清单;v1.1 集群实测补充 |
| [docs/PROTOCOL-unified.md](docs/PROTOCOL-unified.md) | **统一实验协议(预注册)**:10 篇受审论文 setup 调研表 + 锁定协议 + 各臂实现来源(代码复用图)+ 显式偏离清单 |
| [docs/PROTOCOL-demystifying.md](docs/PROTOCOL-demystifying.md) | Demystifying 协议实录(锚点依据):模型/数据/超参抽取 + 未决项 |
| [docs/METRICS.md](docs/METRICS.md) | **Metrics 规范(预注册)**:判决层/副作用面板/机理飞行记录仪/效率层 + 节奏总表 + 实施清单 |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | **Benchmark 选型(预注册)**:文献频次表 + 分档套件 + 数据源/统一评分器 + 卫生检查 + 不选清单 |
| [docs/DEPLOY-PAI-DLC.md](docs/DEPLOY-PAI-DLC.md) | 阿里云 PAI-DLC 部署:形态映射、镜像/存储/提交三步、上机前必查清单、抢占与断点、跨集群分工纪律 |

## 快速事实

- **模型**:student Qwen3-0.6B-Base 主力;teachers Qwen3-1.7B / 4B 现货(零 GRPO 自训);
  终验档 + 复现锚点 = 1.7B-Base ← 4B(Demystifying 阶梯现成格)
- **域**:math(MATH500 / AMC23)+ code(HumanEval+/MBPP+)+ IFEval。
  **选择只在 math**(两篇锚点论文亦然);**迁移列逐臂**评三域,只报副作用不参与晋级
- **硬件**:8–16 × A100-80G;2 卡/run(actor 1 + teacher 池 1),16 卡 8 路并行
- **时间线**:实验 3–4 周冲刺 + 1 周 buffer,目标 ICLR 2027
- **竞品边界**:Rethinking(2604.13016)/ Demystifying(2607.13399)是机理研究线;
  本项目是文献审计线 —— 它们不是竞品,是案卷

## 在 PAI-DSW 上跑(8×A100-80G 交互实例)

```bash
git clone git@github.com:YouAreSpecialToMe/SimOPD.git && cd SimOPD
bash deploy/dsw/setup.sh                      # 建 ./simopd 虚拟环境 + 装依赖 + 模型 + 数据(venv+pip,不需要 uv)
#   装之前会给包源赛跑,挑实测最快的(SIMOPD_RACE=0 跳过;显式设 TORCH_FIND_LINKS/UV_DEFAULT_INDEX 则不赛)
bash deploy/dsw/doctor.sh                     # 体检:一屏看清哪里坏了、怎么修
bash deploy/dsw/run_parallel.sh --rehearsal   # 每臂 3 步,先便宜地验一遍
bash deploy/dsw/run_parallel.sh               # 正式 campaign
```

一个 run = **2 卡**(actor + teacher 池;verl 把 teacher 池注册成独立 Ray 资源池,
不与 actor 共卡),所以 8 卡 = **4 条泳道并行**。泳道之间只共享文件系统:各自的
`CUDA_VISIBLE_DEVICES`、各自的 Ray 临时目录、各自的日志。

⚠ 泳道清理**不能**用 `ray stop --force` —— 它是全机范围的,会连带杀掉另外三条泳道。
`_lane.sh` 按泳道私有临时目录精确清理。

⚠ DSW 实例**停止**会丢掉进程(nohup 只扛掉线,扛不住停机)。checkpoint 每
`SAVE_FREQ` 步落在 workspace 卷上,停机后是续跑而不是重跑 —— 这条纪律是
2026-07-31 那次 24 GPU·时全损换来的(见 INFRA-NOTES 事故复盘)。

`setup.sh` 结束时会生成 **`simopd_env.sh`** 并挂进 `~/.bashrc` —— **新开的 shell 自动激活
venv 并带好全部变量**(HF_HOME / HF_ENDPOINT / HF_HUB_DISABLE_XET / DATA_DIR / CKPT_ROOT /
WANDB_DIR / PYTHONPATH)。当前 shell 立刻生效:

```bash
source simopd_env.sh
```

幂等:重复 source 不会叠加 PYTHONPATH,也不会顶掉已激活的其他 venv。
想摘掉:`sed -i '/# >>> simopd >>>/,/# <<< simopd <<</d' ~/.bashrc`

## 监控实验

```bash
python scripts/watch.py              # 全部 run 一屏
python scripts/watch.py --watch 60   # 每 60 秒刷新
python scripts/watch.py --run vanilla_s0   # 单个 run 的 val 轨迹 + 长度/步时
```

读的是**日志里 verl 的 step 行**而不是 wandb —— 两个集群格式一致,wandb 离线或被墙也照常работа。
四个健康告警都是这个项目真实踩过的坑,不是通用模板:

| 告警 | 含义 | 来历 |
|---|---|---|
| `STALLED` | 久无新 step | 一次 18 分钟静默启动,实为每进程 14.5s 的导入税 |
| `MODE-A` | 长度上涨 + 步时恶化 | 300 步跑成 >24h、在第 229 步被杀 |
| `NO-CKPT` | 过了 SAVE_FREQ 仍无 checkpoint | 同一次:24 GPU·时零产出 |
| `DISK` | checkpoint 逼近容量 | verl 默认全留,17 run ≈850GB |

`age` 和 `src` 两列用来分辨**作废的 run** —— 被取消的作业在日志里和活着的长得一模一样,
拿作废数据下判断比没数据更糟。

## 跨域迁移列(逐臂)

```bash
python scripts/transfer_eval.py --selfcheck    # 换机器先跑:验代码沙箱(不需要 GPU)
bash scripts/eval_transfer.sh vanilla_s0       # 单臂 final ckpt
bash scripts/eval_transfer.sh --all            # 所有已出 checkpoint 的臂
```

math 上训练、code/IF 上评测:量的是**副作用**(这个 trick 有没有把别的域弄坏),
1083 题 greedy ≈ **0.25 GPU·时**,对 18 GPU·时的训练是 1.5% 开销。
**只报告不晋级** —— 选择仍然纯在 math。

它**测不了**"trick 在 code 域是否有效":那个问题的核心量是 student 尾质量 π(S̄),
而迁移评测里没有 teacher、没有蒸馏,π(S̄) 根本不存在。见 plan §0。

用各自的**官方 harness**(evalplus / Google `instruction_following_eval`),
因为这一列的价值就在于能和已有迁移列的两篇(FiRe、Teachability)对得上。

⚠ **代码题会因机器负载假失败**:同一批 canonical solution,load≈20 时 160/164、
load≈6 时 163/164。默认已把单测时限地板从 1.0s 抬到 4.0s
(`SIMOPD_EVALPLUS_MIN_TIME_LIMIT`),但**别在四条泳道满载时跑代码评测**。

## 第三方代码(不入库,本地 clone)

```bash
git clone --depth 1 https://github.com/volcengine/verl.git      # 基座:主线原生 OPD
git clone --depth 1 https://github.com/thunlp/OPD.git           # Rethinking 官方库(锚点配方参考)
git clone --depth 1 https://github.com/lds-ustc/EasyOPD.git     # 采石场(hook 设计 / renorm 开关)
```

## 状态

- [x] 立项、范围决策(OPD only / general 域 / 独立项目)
- [x] 竞品精读与定位(审计 genre)
- [x] 案卷:普查 + 8 轴参赛名单
- [x] infra 勘察与基座裁定(verl 主线)
- [ ] W1:环境 + verl OPD 示例跑通 + 复现锚点(进度闸门)
- [ ] W2–W3:贪心 R1–R4(300 步早筛)
- [ ] W4:Phase 3 三域终审 + 配方消融,结果冻结
