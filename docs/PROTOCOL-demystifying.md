# Demystifying (2607.13399) 实验协议实录(2026-07-31 从 arXiv HTML 抽取)

> 用途:复现锚点(W1 进度闸门)与我们协议对齐的唯一依据。
> 全文精读待做;本页先钉住锚点必需的事实。

## 论文身份

- 标题:Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations
- 作者:Rui Wang, Hongru Wang, Yi Chen, Boyang Xue, Tianqing Fang, Wenhao Yu, Kam-Fai Wong

## Setup(与计划 §1 对照)

| 项 | 论文值 | 我们 |
|---|---|---|
| student | **Qwen3-1.7B-Base**(主)、Qwen3-1.7B | 0.6B-Base 主力;1.7B-Base 终验档 ✓ |
| teachers | Qwen3-1.7B-GRPO、Qwen3-4B-GRPO(自训!)、**Qwen3-4B-Instruct-2507**(现货)、Qwen3-8B | 零自训纪律 → 锚点 teacher 用 **Qwen3-4B-Instruct-2507** |
| 训练集 | **Nemotron-Cascade Math**(= HF `nvidia/Nemotron-Cascade-RL-Math`,14,476 题,字段 problem/answer/source)+ 严格非思考约束 | 同 |
| batch | 128 | ✓ |
| n rollout/prompt | 1(消融 2、8) | ✓ |
| max len | 16,384(训练) | 筛选 8k 帽,锚点/终审 16k |
| τ / top-p | 1.0 / 1.0 | ✓ |
| lr | **未在 setup 节标明** —— 全文核验待做 | 暂用 1e-6(verl 例默认),锚点前必须钉死 |
| 评测 | MATH500 + Minerva (pass@1);AMC'23、AIME 24–26、HMMT'25 (avg@32) | 筛选 MATH500 pass@1 + AMC23 avg@32 |
| loss | reverse KL(student→teacher),sampled-token | ✓ vanilla 基线 |

## 信号调节(F 轴两臂的论文形式)

- 硬 clip:`clip(Δℓ_t, c_min, c_max)`
- 软 log 压缩:`sign(Δℓ_t)·log(1+|Δℓ_t|)`(论文赢家;我们 F 轴复核臂)

## 锚点配置结论(W1)

**Qwen3-1.7B-Base ← Qwen3-4B-Instruct-2507,Nemotron-Cascade-RL-Math,
n=1 / batch 128 / 16,384 / τ=1.0 / top-p=1.0,reverse-KL sampled-token。**
对表:论文中该格子的 MATH500 曲线。

## 二次抽取(2026-07-31,全文 HTML 定向查询)

- **公式化:PG 化**。"OPD is implemented via policy gradient with Δℓ_t as the
  per-token advantage",PPO 式 clipped ratio。= verl `use_policy_gradient=True`。
- 每 rollout batch 只优化一个 epoch("single epoch per rollout batch to
  eliminate off-policy bias")✓ 计划已对齐。
- 长度病理机制:ā = 1/T Σ a_t 的序列均值化让 student 用长度稀释负 advantage。
- 非思考约束 = 给 rollout 前缀一个(空)think block;具体模板句法论文未给。
- **lr / optimizer 细节 / 训练步数:论文缺失(ABSENT)**。锚点只能做曲线形状+
  终点区间对齐,不能逐点对齐;lr 取 1e-6(verl 例默认)并在台账记录。

## 未决(全文精读时补)

- [x] "空 think block" 模板已实测确认(2026-07-31):Qwen3 tokenizer(Base 同款)
  `apply_chat_template(..., enable_thinking=False)` 渲染出
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` —— 与论文"prefix a block"1:1 对应;
  verl 侧用 `data.apply_chat_template_kwargs.enable_thinking=False` 即可。
- [ ] 评测生成参数(温度、max tokens、AMC/AIME avg@32 的采样温度)
- [ ] 曲线横轴刻度(step? epoch? token 数)—— 决定锚点对表方式
