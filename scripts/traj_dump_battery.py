#!/usr/bin/env python3
"""CPU battery for simopd.traj_dump: padding is stripped from both sides, the
terminator survives to disk, sampling honours EVERY/N, a write failure never kills
the step (but does shout), and install() refuses to bind silently when verl moved.
No GPU, no ray, no verl beyond a stub.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402

tmpdir = tempfile.mkdtemp()
os.environ["SIMOPD_TRAJ_DIR"] = os.path.join(tmpdir, "traj")
os.environ["SIMOPD_TRAJ_EVERY"] = "5"
os.environ["SIMOPD_TRAJ_N"] = "2"
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


def make_batch(n=3, plen=4, rlen=6):
    """左 padding 的 prompt + 右 padding 的 response,和 verl 的真实布局同形。"""
    prompts = torch.full((n, plen), PAD, dtype=torch.long)
    responses = torch.full((n, rlen), PAD, dtype=torch.long)
    attn = torch.zeros((n, plen + rlen), dtype=torch.long)
    real_r = [4, 6, 2]                      # 第 0 条自然停止,第 1 条撞帽,第 2 条短
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
    b = types.SimpleNamespace()
    b.batch = {"prompts": prompts, "responses": responses, "attention_mask": attn,
               "response_mask": (torch.arange(rlen).unsqueeze(0) < torch.tensor(real_r).unsqueeze(1)),
               "token_level_scores": torch.tensor([[1.0] + [0.0] * (rlen - 1)] * n)}
    return b


import pandas as pd  # noqa: E402

# ---------------------------------------------------- 采样:EVERY / N 生效 ---
traj_dump._write(make_batch(), 5)
p5 = os.path.join(os.environ["SIMOPD_TRAJ_DIR"], "battery_arm", "step_5.parquet")
ok(os.path.exists(p5), "step 5 (5 的倍数) 落盘了")
traj_dump._write(make_batch(), 7)
ok(not os.path.exists(os.path.join(os.path.dirname(p5), "step_7.parquet")),
   "step 7 不是 EVERY 的倍数 -> 不落盘")
df = pd.read_parquet(p5)
ok(len(df) == 2, f"SIMOPD_TRAJ_N=2 只存 2 条(实得 {len(df)})")

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

# --------------------------------------------------------- 未设环境 = 无事 ---
del os.environ["SIMOPD_TRAJ_DIR"]
ok(traj_dump.enabled() is False, "未设 SIMOPD_TRAJ_DIR 时 enabled()=False")
ok(traj_dump.install() is None, "未设环境时 install() 直接返回,零开销")

print(f"traj_dump battery {PASS}/{PASS} pass")
