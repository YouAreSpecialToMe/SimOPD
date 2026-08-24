# docs/data —— 结果数据(仓库里唯一的"数据"层)

原始产物不进仓库,也进不来:检查点 **33 TB**、逐样本 parquet **14 GB / 4833 份**,
都留在共享盘 `/mgfs/shared/Group_GY/changhao/simopd_data/`(`ckpt/simopd/`、`evals/`)。
这里只放**结论级**的表 —— 分析、作图、写论文只需要它们,且不依赖 wandb 在线或集群可达。

## 本轮归档(2026-08-24)

| 文件 | 是什么 | 谁生成 |
|---|---|---|
| `post_eval_cells.csv` | 逐 (arm, seed, step, benchmark):avg@k / pass@k / 长度分布 / 截断率 / finish_reason / 测量契约 | `scripts/extract_post_eval.py --roster ALL` |
| `post_eval_bystep.csv` | 逐 (arm, seed, step) 汇总:composite 与四组分,**只在五个基准齐全时**才算 | 同上 |
| `inloop_wave_dynamics.csv` | 训练动态逐 step:val_acc / 长度 / 熵 / 梯度 / 损失 + 终止塌缩仪表(eos_p/q_at_stop、entropy_student/teacher、delta_ell 分位、overlap_*) | `scripts/analysis/export_wave_dynamics.py`(读 wandb) |
| `ckpt_inventory.csv` | 每个 run 在盘上存到第几步、有哪些步 | 盘上直接列 |

口径注意:
* composite = eval_suite.py 的等权宏平均(AIME24+25 合池 / AMC23 / Minerva / MATH500),
  缺一个基准的格**不参与**,绝不用更小的宏集补齐。
* run 名里的后缀是**测量契约**,不是噪音:`*.legacy_stop`、`*.renorm-defect-20260822`
  与干净 run 是不同的东西,arm 列保留后缀,不要合并。
* 同名 EXPERIMENT_NAME 可能有多个 wandb run(每次开机一个 id);动态表按 step 合并、
  后创建的 run 胜(续跑接续)。
* `has_text` 说明该格的 parquet 有没有存回答原文 —— 终止符类分析要挑有原文的格。

其余文件是更早的专项探针(eos_stop_*、a2_coldstart_probe*、inloop_*_vs_*、
n0_effect_audit 等),生成脚本见 `scripts/analysis/`。
