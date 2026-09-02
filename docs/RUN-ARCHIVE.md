# 每个 run 落盘什么(分析归档)

> 2026-09-02 起,`scripts/run_opd_baseline.sh` 默认(`SIMOPD_ARCHIVE=1`)把复盘一条曲线所需的
> 全部原始量写在 run 自己的 ckpt 目录里,`scripts/ckpt_sync.py` 随权重一起送上 HuggingFace。
> 起因:2026-08-24 旧集群失联,308 个 ckpt 与整个修正波的 rollout 一起没了,我们手里只剩
> eval parquet —— "学生在终止符上的 Δℓ 怎么走、教师想不想停、正文里何时开始出现 im_end"
> 全都无从复盘。这一层保证那种损失不再可能。

## 目录布局(`$CKPT_ROOT/simopd/<experiment>/`)

```
global_step_<n>/actor/huggingface/   权重(bf16 + tokenizer;ckpt_sync 传这个)
global_step_<n>/actor/…              FSDP 分片 + optimizer(只对精确 resume 有用;--with-optimizer 点名才传)
latest_checkpointed_iteration.txt    verl 的 resume 指针
simopd_fingerprint.txt               resume 指纹(臂定义的 sha1,见启动器 resume 段)
simopd_stop_contract.txt             停机契约 pin(off / 151643,151645)
run_manifest.json                    最新一次启动的 run 定义(见下)
manifest/launch_<ts>.json            每次启动一份(重启/续跑各留一条)
metrics/launch_<ts>.jsonl            verl "file" logger:wandb 拿到的每个标量,逐步一行 {"step","data"}
traj/meta.json                       契约、E_S/E_T、K、P、T、列说明(一次性)
traj/light.jsonl                     每步每序列一行的轻摘要
traj/summary_<n>.parquet             每 25 步:整批每序列一行的标量摘要(含教师量)
traj/step_<n>.parquet                每 25 步:前 32 条整序列(ids + 逐 token 列)
traj/_verl_text/                     仅 SIMOPD_TRAJ_TEXT=1(verl 自己的剥特殊符文本,每步整批,~1 GB/run,默认不写、不传)
val_gen/<step>.jsonl                 在环 math500 生成(verl 文本 dump,每 test_freq 一次)
```

`metrics/` 一次启动一个文件(verl 的 FileLogger 以 `wb` 打开会截断),读的时候按 step 拼接、后写覆盖先写。

## 轨迹列(`traj/`)

| 列 | 在哪 | 含义 |
|---|---|---|
| `step seq uid` | 全部 | 训练步、批内序号、prompt 的 uid(GRPO 分组键) |
| `resp_len score adv` | 全部 | 响应长度、token_level_scores 之和、advantage(GRPO 每 token 同值,取首位) |
| `last_id last_is_stop truncated` | 全部 | 末 token id;是否在记录的终止符集合里;`resp_len == T` 且末 token 非终止符 = 撞帽 |
| `n_stop_body` | 全部 | 末 token 之前正文里出现终止符 id 的次数(im_end 混进正文的信号) |
| `rep4` | 全部 | 重复 4-gram 占比,`1 - 唯一/总数`(→1 = 死循环) |
| `stu_lp_sum stu_lp_last` | 全部 | 学生 `old_log_probs`(rollout 后当前策略)对采样 token 的 Σ 与末位 |
| `tch_lp_sum tch_lp_last tch_lp_nan` | summary | 教师对采样 token 的 logprob:Σ、末位、"采样列不在教师块里"的位置数 |
| `dl_mean dl_last dl_max` | summary | Δℓ = stu_lp − tch_lp 的均值 / 末位 / 最大;自然停止时 `dl_last` 就是终止事件上的 Δℓ |
| `tch_top1_last_id tch_top1_last_lp` | summary | 末位置上教师块里 logprob 最大的 id(教师在结尾最想出什么)|
| `p_last_<id>` | summary | 末位置上教师给每个记录的终止符 id 的概率(不在块里 = NaN)|
| `prompt_ids response_ids gt` | step_ | 无损 id 与 ground truth |
| `stu_lp tch_lp tch_top1_id tch_top1_lp tch_lp_<id>` | step_ | 逐 token 列(list),与 `response_ids` 等长 |

教师列全部**按 id 在教师块里查找**,不依赖列布局(top-K 的排序、`SIMOPD_KEEP_SAMPLED` 保留的采样列、
`SIMOPD_GATHER_EOS` 追加的精确终止列都能找到);找不到记 NaN 而不是 0。事件级修正(`TERM_EVENT=1`)
的量可离线重构:停机位上 log p_S(E_S) / log q_T(E_T) 由各终止符列 logsumexp;`meta.json` 记着
E_S、E_T 与本 run 的契约。

记录的终止符集合 = rollout 停机集 ∪ E_S ∪ E_T(去重保序),通常就是 `{151643, 151645}`。

## `run_manifest.json`

`experiment / project / launched_at / host / cuda_visible_devices / ckpt_dir / fingerprint /
stop_contract / resumed_from / arm_args(hydra 覆盖)/ extra_overrides / eval_protocol.suite_k /
git.simopd|verl {sha, dirty} / python / torch / env{…}`。`env` 收 `SIMOPD_* DISTILLATION_* MAX_* TOTAL_*
TRAIN_* PPO_* ROLLOUT_* … WANDB_RUN_GROUP WANDB_TAGS` 等前缀;**名字含 KEY/TOKEN/SECRET/PASSWORD 的一律不收**
(这个文件会上 HF)。

## 开关与开销

| 变量 | 默认 | 说明 |
|---|---|---|
| `SIMOPD_ARCHIVE` | 1 | 0 = 整层关掉(不进指纹) |
| `SIMOPD_TRAJ_DIR` | `$ckpt_dir/traj` | 启动器给;手动设了就用你的 |
| `SIMOPD_TRAJ_EVERY` | 25 | 整批摘要 + 整序列的间隔(与 save_freq 同拍) |
| `SIMOPD_TRAJ_N` | 32 | 每次存几条整序列(≤0 整批,慎用) |
| `SIMOPD_TRAJ_LIGHT` | 1 | 每步 light.jsonl(不碰教师块,毫秒级) |
| `SIMOPD_TRAJ_TEXT` | 0 | 1 = 放行 verl 自己的文本 dump |
| `LOGGER` | 未设 → `["console","wandb","file"]` | 显式给了就照用(不再加 file) |

体量:light 每步 ~40 KB;summary 每 25 步 ~50 KB;step_ 每 25 步 ~10 MB(32 条 × 16k × 6 列);
一个 200 步的 run 约 100 MB;metrics 几 MB。全部随 `ckpt_sync.py` 上 HF(`<run>@aux`),
`traj/_verl_text/` 除外。dump 在 driver 上、每 25 步花 1–3 s(教师块 [B, P+T, K] 的 numpy 扫描)。

这些都**不进 resume 指纹**:它们改的是"记什么",不是 loss 算什么(与 `SIMOPD_SHADOW` 同类);
hydra 侧走单独的 `ARCHIVE_ARGS`,不混进 `ARM_ARGS`。

## 读法

```python
import pandas as pd, glob, json
d = "<ckpt_dir>/traj"
light = pd.read_json(f"{d}/light.jsonl", lines=True)                       # 每步每序列
summ  = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(f"{d}/summary_*.parquet")))
full  = pd.read_parquet(f"{d}/step_200.parquet")                            # 逐 token
meta  = json.load(open(f"{d}/meta.json"))
# 停止率 / 截断率 / 死循环率随步走
light.groupby("step")[["last_is_stop", "truncated"]].mean(), light.assign(loop=light.rep4 > .5).groupby("step").loop.mean()
# 终止事件上的 Δℓ 与教师在结尾想不想停
summ[summ.last_is_stop].groupby("step")[["dl_last", "p_last_151643", "p_last_151645"]].mean()
```

失败语义:落盘失败**绝不**弄挂训练步,但会在 stderr 喊一次(`[simopd] traj_dump 写盘失败`);
`run_manifest.py` 失败只打 `WARN`。安装时 `SIMOPD_TRAJ_DIR` 有值而 verl 未配 `rollout_data_dir`
会静默无事(启动器保证两者同出),这正是 `traj_dump.install()` 要喊的那种"看着武装了其实没有"。
