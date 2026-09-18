# SimOPD

**Auditing the on-policy distillation (OPD) variant zoo → a minimal effective recipe.**

对现有 OPD 变体文献(zoo 462 条目、核心受审池 ~55 篇)做统一协议审计:
每个 trick 判为 **必需 / 无用 / 有副作用**,双向贪心蒸出最小有效配方 **SimOPD**。
方法论血统:LitePPO(Tricks or Traps, arXiv:2508.08221)之于 GRPO/DAPO。

---

## ⚠ 2026-09 名册重训的交接包(`0917` 分支,09-18 已合并进 `ch-dev`)

上一轮执行方已停机。`0917` = `ch-dev@fea5556` + 那一轮实际在跑的 Slurm 自动化 + 一份交接手册;
09-18 合并回 `ch-dev`,两条分支现在内容一致。

**还剩什么、先做哪些、谁在跑哪条 → [docs/REMAINING-20260918.md](docs/REMAINING-20260918.md)**
(逐臂待办表从实物现算、按优先级的顺序、认领方式、等拍板的三件事)。

**落档 5400/6050 步(89.3%)、19 条臂训练完成、155 份评测 parquet。
剩 650 步训练和 114 格评测。**

权重和数据不在 git 里,在两个公开的 HF 仓库:

| 仓库 | 内容 | 大小 |
|---|---|---|
| [Jerrycool/opd-ckpt](https://huggingface.co/Jerrycool/opd-ckpt) | `resume/` 10 条臂的完整存档(带优化器) + `eval/` 24 条臂的推理权重 | 865 GiB |
| [Jerrycool/opd](https://huggingface.co/datasets/Jerrycool/opd) | `runs/` 训练侧产物 + `evals/` 155 份评测 + `assets/` A 轴离策缓存 | 13.6 GiB |

### 五步跑起来

```bash
git clone https://github.com/YouAreSpecialToMe/SimOPD.git && cd SimOPD && git checkout 0917
cp simopd_env.example.sh simopd_env.sh && $EDITOR simopd_env.sh   # 路径、分区、SIMOPD_SUITE_K=8
bash deploy/dsw/setup.sh                                          # verl 钉在 deploy/dsw/verl.pin
huggingface-cli download Jerrycool/opd-ckpt              --local-dir dl_ckpt
huggingface-cli download Jerrycool/opd --repo-type dataset --local-dir dl_data
. ./simopd_env.sh && bash scripts/assemble_handoff.sh dl_ckpt dl_data
```

最后一步会把两个仓库合回同一个目录、把 `eval/` 的权重重新嵌套成评测端要的形状,
然后逐臂自检。**不要手工 `cp`** —— 停止契约和 checkpoint 分在两个仓库,而且两种
权重的目录形状不一样,手工摆几乎必错。自检不过就别往下走。

摆好之后:

```bash
bash slurm/campaign_tick.sh                             # 逐臂剩余步数
RUNS="a4_dagger_anneal_n0:0" LANES=1 STEPS=200     sbatch slurm/retrain_lane_node.sbatch               # 续训(最短的一条,只剩 25 步)
python scripts/eval_refill_exp.py --grid light --write  # 扫出缺的评测格
EVAL_GRID=light sbatch slurm/retrain_eval_farm.sbatch   # 起评测专列
```

### 详细手册

**[docs/HANDOFF-20260917.md](docs/HANDOFF-20260917.md)** —— 12 节,事无巨细:
硬件前提、从零装环境、`simopd_env.sh` 逐项、资产准备、实验设计速览、
**逐臂剩余步数表**、评测网格与缺口、自动化层的数据流与闸门文件、
**这一轮踩过的 12 个坑**(每条都给了根因与修复位置)、已知缺口、验收清单。

上手顺序:只想快点跑起来看 §1–§5 和 §12;想知道某处为什么这么写看 §10。

### 三个必须知道的点

1. **verl 版本钉死在 `deploy/dsw/verl.pin`**(`3d36367e`)。`sitecustomize.py` 的八个
   钩子按名字挂 verl/vLLM 的内部符号,换版本钩子会**静默**挂空 —— 横幅照打、
   归档一个文件不写。这批 checkpoint 全部产自这个 commit。
2. **`SIMOPD_SUITE_K=8` 必须设。** 已有 155 份评测是 avg@8,而脚本默认 32;
   `eval_suite.py` 按每题采样数过滤产物,两者不会混进同一个 composite。
3. **`trainer.use_v1=False` 不能动。** verl 默认 `use_v1=true`,V1 下 SimOPD 的三处
   trainer 接缝全部失效且不报错。

---

## 协作者

**2026-09 名册的收尾工作**见 **[docs/REMAINING-20260918.md](docs/REMAINING-20260918.md)**(含认领方式)。

下面这份是 **2026-08 16k campaign** 的远程协作契约(29 臂 × 3 seed、`vanilla` 基线),
对 2026-09 名册不适用,留作历史:名册中 `remote` 标记的臂开放认领,契约与交付见
**[docs/COLLABORATORS.md](docs/COLLABORATORS.md)**。

## 结果与分析(16k campaign,29 臂 × 3 seed)

| 文档 | 内容 |
|---|---|
| [docs/campaign_16k_report.md](docs/campaign_16k_report.md) | **实况报告**:in-loop 曲线(mean±std)、效率、post-eval 大表、参照上下界与 GRR、中途工程变更与事故账 |
| [docs/training-dynamics.md](docs/training-dynamics.md) | **训练过程指标**:response length / 截断率 / entropy / grad norm / 蒸馏内部量 / 单步耗时随训练的变化 —— 图 + 表 + 可下载 CSV |
| [docs/late-training-collapse.md](docs/late-training-collapse.md) | **后期掉分的成因**:配对诊断证明这是"不会停"而非"不会算";29 臂横断证据与唯一显著的配置轴 |
| [docs/data/](docs/data/) | 训练指标原始导出(逐 seed 逐 step,43 个指标,wide-form CSV) |

## 文档

| 文档 | 内容 |
|---|---|
| [docs/ARMS-GUIDE.md](docs/ARMS-GUIDE.md) | **从这里开始**:基础 OPD setting(目标函数、锁定协议)+ 每个臂的直观/数学讲解 + 执行脚本 + 本集群(Group_GY 7 机)适配清单 |
| [docs/SimOPD-plan.md](docs/SimOPD-plan.md) | 实验计划 v3.1:定位、7 轴、诊断 D1'–D6、三阶段流程、判决规则、硬件预算、里程碑 |
| [docs/SimOPD-casefile.md](docs/SimOPD-casefile.md) | 案卷:动物园普查、管辖权裁定、8 轴参赛名单(审/替/落)、代表选择准则 |
| [docs/HANDOFF-20260917.md](docs/HANDOFF-20260917.md) | **交接手册(2026-09 名册重训)**:环境、资产、逐臂剩余步数、评测缺口、Slurm 自动化层、12 个已知坑 |
| [docs/REMAINING-20260918.md](docs/REMAINING-20260918.md) | **还没跑完的(给接手的合作者)**:650 步训练 + 114 格评测的逐臂表、建议顺序、认领、必须知道的 9 条、等拍板的 3 件 |
| [docs/EVAL-LEDGER-20260918.md](docs/EVAL-LEDGER-20260918.md) | 155 份评测入账:覆盖哪些格、`\boxed{}` 风险解除、在环 val 与离线评测不可比 |
| [docs/CORNELL-FEASIBILITY-20260918.md](docs/CORNELL-FEASIBILITY-20260918.md) | 续训在 Cornell 能不能跑:只能用 80 G 卡的显存算术、指纹钉住的旋钮、实测机时 |
| [docs/INFRA-NOTES.md](docs/INFRA-NOTES.md) | infra 勘察:verl 主线原生 OPD 基座裁定、缺口→接缝图、槽位布局、W1 清单;v1.1 集群实测补充 |
| [docs/PROTOCOL-unified.md](docs/PROTOCOL-unified.md) | **统一实验协议(预注册)**:10 篇受审论文 setup 调研表 + 锁定协议 + 各臂实现来源(代码复用图)+ 显式偏离清单 |
| [docs/PROTOCOL-demystifying.md](docs/PROTOCOL-demystifying.md) | Demystifying 协议实录(锚点依据):模型/数据/超参抽取 + 未决项 |
| [docs/METRICS.md](docs/METRICS.md) | **Metrics 规范(预注册)**:判决层/副作用面板/机理飞行记录仪/效率层 + 节奏总表 + 实施清单 |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | **Benchmark 选型(预注册)**:文献频次表 + 分档套件 + 数据源/统一评分器 + 卫生检查 + 不选清单 |
| [docs/DEPLOY-PAI-DLC.md](docs/DEPLOY-PAI-DLC.md) | 阿里云 PAI-DLC 部署:形态映射、镜像/存储/提交三步、上机前必查清单、抢占与断点、跨集群分工纪律 |

## 快速事实

- **模型**(2026-08-04 依实测重定):**Qwen3-1.7B-Base ← Qwen3-4B-Instruct-2507**,
  全现货、零 GRPO 自训。**筛选档 = 终验档 = 复现锚点,合为同一格**(Demystifying 现成格)。
  非思考天花板实测 4B-2507 **0.896** > 8B 0.792 > 1.7B 0.702 —— **关掉 thinking 之后,
  更大的老师不等于更强的老师**
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
bash deploy/dsw/envtest.sh                    # 单臂 3 步,失败自动 triage —— 换机器先跑这个
LANES=1 bash deploy/dsw/run_parallel.sh --rehearsal "vanilla:0"   # 加泳道包装
bash deploy/dsw/run_parallel.sh --rehearsal   # 加 4 路并发
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

## 出错了怎么看

```bash
python scripts/triage.py                # logs/ 里最新那个
python scripts/triage.py /tmp/x.log --ray   # 顺带挖 Ray worker 日志
```

verl 的日志里真报错和几千行噪声混在一起。实测**健康日志 83%、失败日志 75% 是装饰**
(`set -x` 回显、Ray 前缀、TransferQueue perf 块、tqdm 重绘)。triage 砍掉这些,
并且内建两条这个栈的规矩:

- **多 run 日志里第一个 traceback 通常不是病因** —— 真实 rehearsal 日志上,13 个臂 OK、
  1 个 FAIL,而文件里第一个 traceback 属于一个**成功的** run(wandb 关闭竞态)。
  所以按 `#### RUN: ####` 分段,**FAIL 段内的排最前,teardown 竞态排最后**。
- **`rpc_code: 14` 意味着 actor 进程死了**,那段 traceback 既不说是哪个 worker、
  也不说死因 —— `--ray` 直接去各泳道的 Ray temp dir 读 worker 日志。

Ray/hydra/asyncio 的中转帧折叠成计数(把真正出错那行挤出屏幕的就是它们),
并对本项目真撞过的四种症状给修法提示。

## 监控实验

```bash
python scripts/watch.py              # 全部 run 一屏
python scripts/watch.py --watch 60   # 每 60 秒刷新
python scripts/watch.py --run vanilla_s0   # 单个 run 的 val 轨迹 + 长度/步时
```

读的是**日志里 verl 的 step 行**而不是 wandb —— 两个集群格式一致,wandb 离线或被墙也照常工作。
四个健康告警都是这个项目真实踩过的坑,不是通用模板:

| 告警 | 含义 | 来历 |
|---|---|---|
| `STALLED` | 久无新 step | 一次 18 分钟静默启动,实为每进程 14.5s 的导入税 |
| `MODE-A` | 长度上涨 + 步时恶化 | 300 步跑成 >24h、在第 229 步被杀 |
| `EARLY-STOP-DUE` | 预注册早停规则触发 | 实测 85% 机时买不到 pass@1;`--enforce` 执行并记账 |
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
- [ ] W2–W3:贪心 R1–R4(150 步上限 + 预注册早停)
- [ ] W4:Phase 3 三域终审 + 配方消融,结果冻结
