"""训练 rollout 轨迹落盘 —— 存 token id,不是解码文本。

为什么不用 verl 自带的 `trainer.rollout_data_dir`:它的 `_log_rollout_data` 用
`batch_decode(..., skip_special_tokens=True)`,终止符在写盘那一刻就被剥掉了。2026-08-21
查"b2_forward_kl 到底停在 eot 还是 im_end"时撞上同一堵墙的 eval 版本:那些
`finish_reason=stop` 的回答明明采样到了终止符,存下来的文本里却连 `<|endoftext|>`
都没有,于是"正文含 im_end = 0/1500"这个数什么也不证明。整个战役的头号机制是两个
终止符的身份错配,而落盘管线恰好把终止符抹掉了。

所以这里包住 verl 的那次调用,额外写一份 id 版。id 是无损的:重新分词解码文本会二次
丢失(特殊符号没了、边界可能重切),而下游的账本/支撑/重复段分析全都要真 id。

采样是强制的,不是可选:整批全存是 256 seq x 16k tok x 250 步 ~ 1e9 token/run。
默认每 25 步(与 save_freq 同拍)存前 8 条,一个 run 约 1.3M token,几 MB。

环境变量:
  SIMOPD_TRAJ_DIR    输出目录。未设 = 完全不装(零开销)。
  SIMOPD_TRAJ_EVERY  每多少步存一次(默认 25)。
  SIMOPD_TRAJ_N      每次存多少条序列(默认 8;<=0 表示整批,慎用)。

产物:$SIMOPD_TRAJ_DIR/<experiment>/step_<n>.parquet,每行一条序列:
  step seq prompt_ids response_ids resp_len score last_id last_is_stop stop_set
`last_id` 就是"这条轨迹以什么结尾"的直接答案;`last_is_stop` 说明它是自然停止还是撞帽。

注意:verl 只在配置了 trainer.rollout_data_dir 时才调 _log_rollout_data,所以
run_opd_baseline.sh 在 SIMOPD_TRAJ_DIR 有值时会把它一并传下去。两者缺一,这里静默无事发生
—— 而"静默无事发生"正是 h9 中继烧掉 66 步的那个形状,所以 install() 在只配了一半时会喊。
"""
import os
import sys

_MARK = "_simopd_traj_dump"
_state = {"warned": False, "wrote": 0}


def enabled():
    return os.environ.get("SIMOPD_TRAJ_DIR", "").strip() != ""


def _cfg():
    return (os.environ["SIMOPD_TRAJ_DIR"].strip(),
            int(os.environ.get("SIMOPD_TRAJ_EVERY", "25") or 25),
            int(os.environ.get("SIMOPD_TRAJ_N", "8") or 8))


def _rows(batch, step, n_want):
    """(step, seq, prompt_ids, response_ids, ...) —— 去掉两侧的 padding,只留真 token。"""
    import torch

    tb = batch.batch
    prompts, responses = tb["prompts"], tb["responses"]
    B, T = responses.shape
    n = B if n_want <= 0 else min(n_want, B)

    # 响应侧真实长度:优先 response_mask;没有就从 attention_mask 的响应段推。
    if "response_mask" in tb:
        rm = tb["response_mask"]
        rm = rm.to_padded_tensor(False) if getattr(rm, "is_nested", False) else rm
        lens = rm.bool().sum(-1)
    elif "attention_mask" in tb:
        lens = tb["attention_mask"][:, -T:].bool().sum(-1)
    else:
        lens = torch.full((B,), T, dtype=torch.long)

    scores = None
    if "token_level_scores" in tb:
        scores = tb["token_level_scores"].sum(-1).float().cpu().tolist()

    from simopd import eos_gather
    try:
        stop_ids = eos_gather.rollout_stop_set()
    except Exception:
        stop_ids = []

    out = []
    for i in range(n):
        L = int(lens[i])
        r = responses[i, :L].cpu().tolist()
        # prompt 是左 padding 的:用 attention_mask 的 prompt 段截掉 pad。
        p = prompts[i].cpu().tolist()
        if "attention_mask" in tb:
            pm = tb["attention_mask"][i, : prompts.shape[1]].bool().cpu().tolist()
            p = [t for t, keep in zip(p, pm) if keep]
        last = r[-1] if r else None
        out.append(dict(
            step=step, seq=i, prompt_ids=p, response_ids=r, resp_len=L,
            score=(scores[i] if scores else float("nan")),
            last_id=last,
            last_is_stop=bool(last is not None and last in stop_ids),
            stop_set=",".join(map(str, stop_ids)) if stop_ids else "",
        ))
    return out


def _write(batch, step):
    root, every, n_want = _cfg()
    if every > 0 and step % every != 0:
        return
    import pandas as pd

    exp = os.environ.get("EXPERIMENT_NAME", "run")
    d = os.path.join(root, exp)
    os.makedirs(d, exist_ok=True)
    rows = _rows(batch, step, n_want)
    path = os.path.join(d, f"step_{step}.parquet")
    tmp = path + ".tmp"
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, path)     # 原子:读者永远看不到半截文件
    _state["wrote"] += 1
    if _state["wrote"] <= 2 or step % (every * 10) == 0:
        n_stop = sum(1 for r in rows if r["last_is_stop"])
        print(f"[simopd] traj_dump: step {step} -> {path} ({len(rows)} 条,"
              f"自然停止 {n_stop},末 id {[r['last_id'] for r in rows[:4]]})",
              file=sys.stderr, flush=True)


def install():
    """包住 verl 的 _log_rollout_data:它照常写它剥了特殊符的文本,我们额外写 id 版。"""
    if not enabled():
        return
    mod = sys.modules.get("verl.trainer.ppo.ray_trainer")
    if mod is None:
        return                      # 还没 import 到;sitecustomize 的钩子会再来一次
    cls = getattr(mod, "RayPPOTrainer", None)
    fn = getattr(cls, "_log_rollout_data", None) if cls else None
    if fn is None:
        raise RuntimeError("traj_dump: RayPPOTrainer._log_rollout_data 不存在 —— verl 挪了位置;"
                           "SIMOPD_TRAJ_DIR 已设置却什么都不会写,静默失效")
    if getattr(fn, _MARK, False):
        return

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        try:
            _write(batch, int(getattr(self, "global_steps", -1)))
        except Exception as e:
            # 落盘失败绝不能弄挂训练步;但必须响 —— 静默的落盘等于没落盘,
            # 而只有到分析的时候才发现文件不在,那时轨迹已经永远没了。
            if not _state["warned"]:
                _state["warned"] = True
                import traceback
                traceback.print_exc()
                print(f"[simopd] traj_dump 写盘失败({e!r}):这个 run 的轨迹不会被保存,"
                      f"训练继续", file=sys.stderr, flush=True)
        return fn(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

    setattr(_log_rollout_data, _MARK, True)
    cls._log_rollout_data = _log_rollout_data
    root, every, n_want = _cfg()
    print(f"[simopd] traj_dump armed: {root} (每 {every} 步 x {n_want} 条,存 token id)",
          file=sys.stderr, flush=True)
