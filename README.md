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
- **域**:math(MATH500 / AMC23)+ code(HumanEval+/MBPP+)+ IFEval;筛选在 math
- **硬件**:8–16 × A100-80G;2 卡/run(actor 1 + teacher 池 1),16 卡 8 路并行
- **时间线**:实验 3–4 周冲刺 + 1 周 buffer,目标 ICLR 2027
- **竞品边界**:Rethinking(2604.13016)/ Demystifying(2607.13399)是机理研究线;
  本项目是文献审计线 —— 它们不是竞品,是案卷

## 在 PAI-DSW 上跑(8×A100-80G 交互实例)

```bash
git clone git@github.com:YouAreSpecialToMe/SimOPD.git && cd SimOPD
bash deploy/dsw/setup.sh                      # 环境 + 模型 + 数据,幂等可重跑
python scripts/arm.py check                   # 15/16 臂应报 runnable
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
