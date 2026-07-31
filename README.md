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
| [docs/INFRA-NOTES.md](docs/INFRA-NOTES.md) | infra 勘察:verl 主线原生 OPD 基座裁定、缺口→接缝图、槽位布局、W1 清单 |

## 快速事实

- **模型**:student Qwen3-0.6B-Base 主力;teachers Qwen3-1.7B / 4B 现货(零 GRPO 自训);
  终验档 + 复现锚点 = 1.7B-Base ← 4B(Demystifying 阶梯现成格)
- **域**:math(MATH500 / AMC23)+ code(HumanEval+/MBPP+)+ IFEval;筛选在 math
- **硬件**:8–16 × A100-80G;2 卡/run(actor 1 + teacher 池 1),16 卡 8 路并行
- **时间线**:实验 3–4 周冲刺 + 1 周 buffer,目标 ICLR 2027
- **竞品边界**:Rethinking(2604.13016)/ Demystifying(2607.13399)是机理研究线;
  本项目是文献审计线 —— 它们不是竞品,是案卷

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
