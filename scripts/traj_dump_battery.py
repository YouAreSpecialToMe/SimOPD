#!/usr/bin/env python3
"""CPU battery for simopd.traj_dump: padding is stripped from both sides, the
terminator survives to disk, sampling honours EVERY/N, a write failure never kills
the step (but does shout), and install() refuses to bind silently when verl moved.
The teacher block is fabricated in verl's REAL layout (logits convention, dummy last row,
pad id = 151643) and the teacher columns are asserted against planted values -- the
2026-09-08 off-by-one ([P:P+L]) turns every one of those assertions red. refuse_v1() must
raise while the archive is on. No GPU, no ray, no verl beyond a stub.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

tmpdir = tempfile.mkdtemp()
os.environ["SIMOPD_TRAJ_DIR"] = os.path.join(tmpdir, "traj")
os.environ["SIMOPD_TRAJ_EVERY"] = "5"
os.environ["SIMOPD_TRAJ_N"] = "2"
os.environ["SIMOPD_TRAJ_MOD"] = "1"      # 子集规则 seq_key % MOD == 0:合成序列的 key 都不被 8 整除,N 的断言要它为 1
os.environ["EXPERIMENT_NAME"] = "battery_arm"
os.environ["SIMOPD_STOP_IDS"] = "off"

from simopd import traj_dump  # noqa: E402

PASS = 0


def ok(cond, msg):
    global PASS
    if cond:
        PASS += 1
        print(f"ok  {msg}")
    else:
        print(f"FAIL: {msg}")
        sys.exit(1)


EOT, IM_END, PAD = 151643, 151645, 151643
REAL_R = [4, 6, 2]                          # 第 0 条自然停止,第 1 条撞帽,第 2 条短
K = 4                                       # 教师块宽度(真跑是 66+1;布局同形)


def PLANT(j):
    """教师对响应第 j 个采样 token 的 logprob(按位置区分,读错一位就对不上)。"""
    return -0.25 * (j + 1)


def make_batch(n=3, plen=4, rlen=6):
    """左 padding 的 prompt + 右 padding 的 response,和 verl 的真实布局同形。"""
    prompts = torch.full((n, plen), PAD, dtype=torch.long)
    responses = torch.full((n, rlen), PAD, dtype=torch.long)
    attn = torch.zeros((n, plen + rlen), dtype=torch.long)
    real_r = REAL_R
    for i in range(n):
        prompts[i, plen - 2:] = torch.tensor([11 + i, 22 + i])       # 只有末 2 个是真 token
        attn[i, plen - 2:plen] = 1
        L = real_r[i]
        body = [100 + i, 200 + i, 300 + i, 400 + i, 500 + i, 600 + i][:L]
        if i == 0:
            body[-1] = EOT                  # 停在学生自己的终止符
        if i == 2:
            body[-1] = IM_END               # 吐出教师的终止符(在旧契约下不停机)
        responses[i, :L] = torch.tensor(body)
        attn[i, plen:plen + L] = 1
    # 教师块 [n, plen+rlen, K],照 verl 的真实布局造(见 traj_dump._teacher_seq 的注释):
    #   * logits 约定:第 p 行 = 教师对「位置 p+1 的 token」的分布(vLLM 第 0 项被丢弃、整体左移一位),
    #     所以响应 token j 的行在 plen-1+j;
    #   * 末个真实位置 plen+L-1 是 verl 补的哑行(id 全 0、lp 全 0.0)—— 老切法 [P:P+L] 末位读到的就是它;
    #   * 两侧 padding 照 teacher_manager._pad_teacher_outputs:id = pad_token_id(Qwen 的 pad 就是
    #     151643 = eot!)、lp = 0.0。读进 padding 的代码会把 eot 的概率读成 1.0。
    # 列 0 放采样 token(lp 最大 → 也是 top1),列 1 放另一个终止符,列 2/3 放旁观 id。
    tids = torch.full((n, plen + rlen, K), PAD, dtype=torch.long)
    tlp = torch.zeros((n, plen + rlen, K), dtype=torch.float32)
    for i in range(n):
        L = real_r[i]
        # plen-2 行预测第 2 个真 prompt token(在 plen-1 位):真行,但与 response 无关
        tids[i, plen - 2] = torch.tensor([22 + i, IM_END, 7000, 8000])
        tlp[i, plen - 2] = torch.tensor([-0.5, -3.0, -4.0, -5.0])
        for j in range(L):
            tok = int(responses[i, j])
            other = EOT if tok == IM_END else IM_END
            tids[i, plen - 1 + j] = torch.tensor([tok, other, 7000 + j, 8000 + j])
            tlp[i, plen - 1 + j] = torch.tensor([PLANT(j), -2.0 - j, -3.0 - j, -4.0 - j])
        tids[i, plen + L - 1] = 0
        tlp[i, plen + L - 1] = 0.0
    b = types.SimpleNamespace()
    b.batch = {"prompts": prompts, "responses": responses, "attention_mask": attn,
               "response_mask": (torch.arange(rlen).unsqueeze(0) < torch.tensor(real_r).unsqueeze(1)),
               "token_level_scores": torch.tensor([[1.0] + [0.0] * (rlen - 1)] * n),
               "teacher_ids": tids, "teacher_logprobs": tlp,
               "old_log_probs": torch.full((n, rlen), -0.5)}
    return b


import pandas as pd  # noqa: E402

# ---------------------------------------------------- 采样:EVERY / N 生效 ---
traj_dump._write(make_batch(), 5)
p5 = os.path.join(os.environ["SIMOPD_TRAJ_DIR"], "battery_arm", "step_5.parquet")
ok(os.path.exists(p5), "step 5 (5 的倍数) 落盘了")
traj_dump._write(make_batch(), 7)
ok(not os.path.exists(os.path.join(os.path.dirname(p5), "step_7.parquet")),
   "step 7 不是 EVERY 的倍数 -> 不落盘")
p7 = os.path.join(os.path.dirname(p5), "ids_7.parquet")
ok(os.path.exists(p7), "step 7 不是 EVERY 的倍数,但整批无损 id 每步都落(SIMOPD_TRAJ_IDS_EVERY 默认 1)")
ok(len(pd.read_parquet(p7)) == 3, "ids_7 是整批 3 条,不受 SIMOPD_TRAJ_N=2 限制")
df = pd.read_parquet(p5)
ok(len(df) == 2, f"SIMOPD_TRAJ_N=2 只存 2 条(实得 {len(df)})")

# ------------------------------------------- 教师列按 verl 的真实布局对齐 ---
# 2026-09-08 集群报告:此前 _teacher_seq 读 [P:P+L],把 token j 对到教师对 token j+1 的分布、末位读到 verl
# 的哑行 —— tch_lp_nan 0.566、tch_lp_last 全 NaN、resp==top1 只有 0.046。下面这组在老切法下全红。
for r in (df.iloc[0], df.iloc[1]):
    L = int(r.resp_len)
    ok(list(r.tch_top1_id) == list(r.response_ids),
       f"seq{int(r.seq)}: 教师 top1 与采样 token 逐位相等(得 {list(r.tch_top1_id)} vs {list(r.response_ids)})")
    ok(not np.isnan(np.asarray(r.tch_lp, dtype=float)).any(), f"seq{int(r.seq)}: 采样列 logprob 无 NaN")
    ok(np.allclose(np.asarray(r.tch_lp, dtype=float), [PLANT(j) for j in range(L)]),
       f"seq{int(r.seq)}: 采样列读到的是种下的值(得 {list(r.tch_lp)})")
    ok(np.allclose(np.asarray(r.tch_top1_lp, dtype=float), [PLANT(j) for j in range(L)]),
       f"seq{int(r.seq)}: top1 的 logprob 也是种下的值")
summ = pd.read_parquet(os.path.join(os.path.dirname(p5), "summary_5.parquet"))
ok(len(summ) == 3 and int(summ.tch_lp_nan.sum()) == 0,
   f"summary: 整批 tch_lp_nan == 0(得 {summ.tch_lp_nan.tolist()};KEEP_SAMPLED 下非零 = 错位)")
ok(np.isfinite(summ.tch_lp_last).all() and np.allclose(summ.tch_lp_last, [PLANT(L - 1) for L in REAL_R]),
   f"summary: tch_lp_last 有限且 = 末位种值(得 {summ.tch_lp_last.tolist()})")
ok(summ.tch_top1_last_id.tolist() == summ.last_id.tolist(), "summary: 末位教师 top1 = 末 token")
ok(np.allclose(summ.dl_last, -0.5 - summ.tch_lp_last), "summary: dl_last = stu_lp_last - tch_lp_last")
ok(np.isfinite(summ.dl_mean).all(), "summary: dl_mean 有限(采样列每一位都找到了)")

# ------------------------------------------------------- padding 两侧剥净 ---
r0 = df.iloc[0]
ok(list(r0.prompt_ids) == [11, 22], f"prompt 左 padding 剥净 (得 {list(r0.prompt_ids)})")
ok(len(r0.response_ids) == 4 and int(r0.resp_len) == 4, "response 右 padding 按 response_mask 剥净")
ok(PAD not in list(r0.response_ids)[:-1], "正文里没有混进 pad")

# ------------------------------------------- 终止符活着到盘上(本模块的存在理由) ---
ok(int(r0.last_id) == EOT, f"末 token 就是 eot 151643(得 {int(r0.last_id)})")
ok(int(df.iloc[1].last_id) != EOT, "撞帽那条的末 token 不是终止符")
# STOP_IDS=off 不表示"没有 stop token":rollout_stop_set() 恒含学生自己的 model_eos,
# off 只是不追加。所以旧契约下 eot 结尾本来就是自然停止,而 im_end 不是。
ok(r0.stop_set == "151643", f"契约 off 时 stop_set 只有学生自己的 eos(得 {r0.stop_set})")
ok(bool(r0.last_is_stop), "契约 off 时 eot 结尾仍判为自然停止(它是 model_eos)")
ok(not bool(df.iloc[1].last_is_stop), "撞帽那条不算自然停止")

# 契约打开后 im_end 也进 stop_set —— 这正是 v2 双终止符契约要改的东西
os.environ["SIMOPD_STOP_IDS"] = "151643,151645"
traj_dump._write(make_batch(), 10)
d2 = pd.read_parquet(os.path.join(os.path.dirname(p5), "step_10.parquet"))
ok(d2.iloc[0].stop_set == "151643,151645", f"stop_set 随契约记录 (得 {d2.iloc[0].stop_set})")
ok(bool(d2.iloc[0].last_is_stop), "v2 契约下 eot 结尾仍是自然停止")
ok("tch_lp_151645" in d2.columns and np.isfinite(np.asarray(d2.iloc[0].tch_lp_151645, dtype=float)).all(),
   "v2 契约下 im_end 列存在且逐位有限")
ok(np.allclose(np.asarray(d2.iloc[0].tch_lp_151645, dtype=float), [-2.0 - j for j in range(4)]),
   f"im_end 列读到的是种下的值(得 {list(d2.iloc[0].tch_lp_151645)})")
s2 = pd.read_parquet(os.path.join(os.path.dirname(p5), "summary_10.parquet"))
ok(np.allclose(s2.p_last_151645, [np.exp(-2.0 - 3), np.exp(-2.0 - 5), np.exp(PLANT(1))]),
   f"p_last_151645 = exp(末位 im_end 列)(得 {s2.p_last_151645.tolist()};第 2 条以 im_end 结尾,取采样列)")

# 第 2 条以 im_end 结尾:整批存下来才能看见它 —— 这正是文本管线看不见的那一类
os.environ["SIMOPD_TRAJ_N"] = "0"
traj_dump._write(make_batch(), 15)
d3 = pd.read_parquet(os.path.join(os.path.dirname(p5), "step_15.parquet"))
ok(len(d3) == 3, "SIMOPD_TRAJ_N=0 存整批")
ok(int(d3.iloc[2].last_id) == IM_END,
   "学生吐出的 <|im_end|> 出现在盘上 —— 解码文本管线会把它剥掉")
os.environ["SIMOPD_TRAJ_N"] = "2"

# ------------------------------------------------------ 写盘失败不弄挂训练 ---
class Boom(dict):
    def __getitem__(self, k):
        raise RuntimeError("模拟坏 batch")


calls = {"n": 0}
# 2026-09-02 起 verl 自己那份剥特殊符的文本 dump 默认不写(每步整批约 1 GB/run),包装器只在
# SIMOPD_TRAJ_TEXT=1 时才调原函数 —— 这个用例要证明的是「写盘失败不弄挂训练、原函数照常」,
# 所以显式打开;默认关的那一面下面另测一次。
os.environ["SIMOPD_TRAJ_TEXT"] = "1"


def fake_orig(self, batch, rei, timing, d):
    calls["n"] += 1
    return "verl 原函数被调用了"


mod = types.ModuleType("verl.trainer.ppo.ray_trainer")


class RayPPOTrainer:
    global_steps = 20
    _log_rollout_data = fake_orig


mod.RayPPOTrainer = RayPPOTrainer
sys.modules["verl.trainer.ppo.ray_trainer"] = mod
traj_dump.install()
ok(getattr(RayPPOTrainer._log_rollout_data, "_simopd_traj_dump", False), "install() 完成包裹")

bad = types.SimpleNamespace()
bad.batch = Boom()
import io  # noqa: E402
from contextlib import redirect_stderr  # noqa: E402

buf = io.StringIO()
with redirect_stderr(buf):
    out = RayPPOTrainer._log_rollout_data(RayPPOTrainer(), bad, {}, {}, "/tmp/x")
ok(out == "verl 原函数被调用了", "写盘失败后 verl 原函数照常执行(训练不中断)")
ok("轨迹不会被保存" in buf.getvalue(), "写盘失败会喊,不静默")

traj_dump.install()
ok(True, "install() 幂等,不会二次包裹")

# ------------------------------------------------- verl 挪走函数时必须报错 ---
del RayPPOTrainer._log_rollout_data
try:
    traj_dump.install()
    ok(False, "verl 挪走 _log_rollout_data 时应当抛错")
except RuntimeError as e:
    ok("静默失效" in str(e), "verl 挪走函数 -> 抛错而不是静默无事发生")

# ------------------------------------------------- V1 trainer:归档开着就拒绝 ---
# verl 默认 trainer.use_v1=true,归档钩子挂在 legacy trainer 上 —— 09-08 集群报告:横幅照打、一个文件不写。
try:
    traj_dump.refuse_v1("battery")
    ok(False, "归档开着时 refuse_v1() 应当抛错")
except RuntimeError as e:
    ok("use_v1=False" in str(e), "归档开着 + V1 trainer -> 抛错,不许静默武装")

# --------------------------------------------------------- 未设环境 = 无事 ---
del os.environ["SIMOPD_TRAJ_DIR"]
ok(traj_dump.refuse_v1("battery") is None, "未开归档时 refuse_v1() 放行(没归档就没什么可拒绝)")
ok(traj_dump.enabled() is False, "未设 SIMOPD_TRAJ_DIR 时 enabled()=False")
ok(traj_dump.install() is None, "未设环境时 install() 直接返回,零开销")

print(f"traj_dump battery {PASS}/{PASS} pass")
