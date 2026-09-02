"""训练 rollout 轨迹落盘 —— 存 token id 与逐 token 对数概率,不是解码文本。

为什么不用 verl 自带的 `trainer.rollout_data_dir`:它的 `_log_rollout_data` 用
`batch_decode(..., skip_special_tokens=True)`,终止符在写盘那一刻就被剥掉了。2026-08-21
查"b2_forward_kl 到底停在 eot 还是 im_end"时撞上同一堵墙的 eval 版本:那些
`finish_reason=stop` 的回答明明采样到了终止符,存下来的文本里却连 `<|endoftext|>`
都没有,于是"正文含 im_end = 0/1500"这个数什么也不证明。整个战役的头号机制是两个
终止符的身份错配,而落盘管线恰好把终止符抹掉了。

所以这里包住 verl 的那次调用,写 id 版。id 是无损的:重新分词解码文本会二次丢失
(特殊符号没了、边界可能重切),而下游的账本/支撑/重复段分析全都要真 id。

2026-09-02 扩成"分析归档":旧集群连同 308 个 ckpt 和整个修正波的 rollout 一起失联后,
我们手里只剩 eval parquet,训练时到底发生了什么(学生在终止符上的 Δℓ 怎么走、教师想
不想停、正文里何时开始出现 im_end)全都无从复盘。这一层把复盘所需的原始量在训练时
就落在 run 自己的 ckpt 目录里,随权重一起上云:

  step_<n>.parquet     抽样 N 条整序列:prompt/response id、逐 token 学生 logprob(old_log_probs)、
                       教师采样列 logprob、教师 top1 id/logprob、每个终止符 id 的教师 logprob 列
  summary_<n>.parquet  整批每序列一行:长度/分数/结尾/截断/重复率/Σlogprob/Δℓ 均值与末位/
                       末位教师 top1/末位各终止符概率/正文里终止符出现次数
  light.jsonl          每一步每序列一行的轻摘要(不碰教师块,毫秒级):长度/结尾/分数/重复率
  meta.json            一次性:契约、E_S/E_T、K、P、T、列说明

采样与开销:整批全存是 256 seq x 16k tok x 250 步 ~ 1e9 token/run,不可能。默认每 25 步
(与 save_freq 同拍)存前 32 条整序列 + 整批摘要,一个 run 十几到几十 MB;light 每步几十 KB。

环境变量:
  SIMOPD_TRAJ_DIR      输出目录。未设 = 完全不装(零开销)。run_opd_baseline.sh 默认给 $ckpt_dir/traj。
  SIMOPD_TRAJ_NOSUB    =1 不再套 <experiment> 子目录(启动器已把目录定到 run 自己的 ckpt 目录下)。
  SIMOPD_TRAJ_EVERY    整批摘要 + 抽样整条的间隔(默认 25)。
  SIMOPD_TRAJ_N        每次存多少条整序列(默认 32;<=0 整批,慎用)。
  SIMOPD_TRAJ_LIGHT    =1(默认)每步追加 light.jsonl;=0 关。
  SIMOPD_TRAJ_TEXT     =1 才放行 verl 自己那份剥了特殊符的文本(每步整批,~1 GB/run);默认 0。

注意:verl 只在配置了 trainer.rollout_data_dir 时才调 _log_rollout_data,所以
run_opd_baseline.sh 在 SIMOPD_TRAJ_DIR 有值时会把它一并传下去。两者缺一,这里静默无事发生
—— 而"静默无事发生"正是 h9 中继烧掉 66 步的那个形状,所以 install() 在只配了一半时会喊。
"""
import json
import os
import sys
import time

_MARK = "_simopd_traj_dump"
_state = {"warned": False, "wrote": 0, "meta": False, "tch_warned": False}


def enabled():
    return os.environ.get("SIMOPD_TRAJ_DIR", "").strip() != ""


def _cfg():
    e = os.environ
    return dict(
        root=e["SIMOPD_TRAJ_DIR"].strip(),
        every=int(e.get("SIMOPD_TRAJ_EVERY", "25") or 25),
        n=int(e.get("SIMOPD_TRAJ_N", "32") or 32),
        light=(e.get("SIMOPD_TRAJ_LIGHT", "1").strip() != "0"),
        text=(e.get("SIMOPD_TRAJ_TEXT", "0").strip() == "1"),
        nosub=(e.get("SIMOPD_TRAJ_NOSUB", "0").strip() == "1"),
    )


def _dir(cfg):
    if cfg["nosub"]:
        return cfg["root"]
    return os.path.join(cfg["root"], os.environ.get("EXPERIMENT_NAME", "run"))


# ---- 终止符集合(全部 best-effort:落盘永远不能因为一个 env 解析而挂) ----------------
def _stop_sets():
    """-> (rollout 停机集, E_S, E_T)。E_S/E_T 只在 N0 臂有意义;取不到就退到停机集。"""
    from simopd import eos_gather
    try:
        roll = list(eos_gather.rollout_stop_set())
    except Exception:
        roll = []
    try:
        es = list(eos_gather.stop_ids())
    except Exception:
        es = list(roll)
    try:
        et = list(eos_gather.teacher_ids())
    except Exception:
        et = list(es)
    return roll, es, et


def _all_stop_ids(roll, es, et):
    out = []
    for i in list(roll) + list(es) + list(et):
        if i not in out:
            out.append(int(i))
    return out


# ---- 张量 -> numpy(driver 上的 batch 在 CPU,普通 dtype 时零拷贝) --------------------
def _np(t):
    import numpy as np
    import torch

    if t is None:
        return None
    if getattr(t, "is_nested", False):
        t = t.to_padded_tensor(0)
    if isinstance(t, torch.Tensor):
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _extract(batch, want_teacher):
    """把 dump 用得到的量从 DataProto 拿成 numpy。缺哪个键就 None,列自然缺席,不报错。"""
    import numpy as np

    tb = batch.batch
    responses = _np(tb["responses"]).astype(np.int64, copy=False)
    prompts = _np(tb["prompts"]).astype(np.int64, copy=False)
    B, T = responses.shape
    P = prompts.shape[1]
    attn = _np(tb["attention_mask"]).astype(bool, copy=False) if "attention_mask" in tb else None
    if "response_mask" in tb:
        rmask = _np(tb["response_mask"]).astype(bool, copy=False)
    elif attn is not None:
        rmask = attn[:, -T:]
    else:
        rmask = np.ones((B, T), dtype=bool)
    lens = rmask.sum(-1).astype(np.int64)
    scores = (_np(tb["token_level_scores"]).sum(-1).astype(np.float64)
              if "token_level_scores" in tb else np.full(B, np.nan))
    old_lp = _np(tb["old_log_probs"]).astype(np.float32, copy=False) if "old_log_probs" in tb else None
    adv = _np(tb["advantages"]) if "advantages" in tb else None
    adv0 = adv[:, 0].astype(np.float64) if adv is not None and adv.ndim == 2 else np.full(B, np.nan)

    tids = tlp = None
    if want_teacher and "teacher_ids" in tb and "teacher_logprobs" in tb:
        tids, tlp = _np(tb["teacher_ids"]), _np(tb["teacher_logprobs"])
        if tids.ndim == 2:
            tids, tlp = tids[..., None], tlp[..., None]
        if tids.shape[:2] != (B, P + T) or tlp.shape != tids.shape:
            if not _state["tch_warned"]:
                _state["tch_warned"] = True
                print(f"[simopd] traj_dump: 教师块形状 {tids.shape}/{tlp.shape} 与 [B,P+T]={(B, P + T)} 不符,"
                      f"教师列不落盘", file=sys.stderr, flush=True)
            tids = tlp = None

    ntb = getattr(batch, "non_tensor_batch", None) or {}
    uid = [str(u) for u in ntb["uid"]] if "uid" in ntb else [None] * B
    gts = [None] * B
    if "reward_model" in ntb:
        try:
            gts = [str(rm.get("ground_truth"))[:200] if isinstance(rm, dict) else None
                   for rm in ntb["reward_model"]]
        except Exception:
            pass
    return dict(B=B, T=T, P=P, responses=responses, prompts=prompts, attn=attn, lens=lens,
                scores=scores, old_lp=old_lp, adv0=adv0, tids=tids, tlp=tlp, uid=uid, gts=gts)


# ---- 逐序列的量 ----------------------------------------------------------------------
def rep4(ids):
    """重复 4-gram 占比:1 - 唯一 4-gram 数 / 4-gram 总数。0 = 无重复,→1 = 死循环。"""
    import numpy as np

    a = np.asarray(ids, dtype=np.uint64)
    n = a.size - 3
    if n <= 0:
        return 0.0
    m = np.uint64(1000003)
    h = ((a[:-3] * m + a[1:-2]) * m + a[2:-1]) * m + a[3:]     # uint64 自然回绕
    return float(1.0 - np.unique(h).size / n)


def _teacher_seq(ex, b, L, stop_all):
    """序列 b 响应段的教师量(长度 L):采样列 logprob、top1 id/logprob、各终止符 id 的 logprob。
    全部按 id 匹配定位,不依赖教师块的列布局(top-K 排序 / 追加的精确终止列 / 保留的采样列)。
    找不到的 = 该 id 既不在 top-K 也没被精确 gather,记 NaN,而不是 0。"""
    import numpy as np

    P = ex["P"]
    ids = ex["tids"][b, P:P + L]            # [L, K]
    lp = ex["tlp"][b, P:P + L].astype(np.float32, copy=False)
    resp = ex["responses"][b, :L]
    ar = np.arange(L)
    valid = ids >= 0
    lp_v = np.where(valid, lp, -np.inf)

    def _col(target):
        m = (ids == target[:, None]) if hasattr(target, "shape") else (ids == target)
        m &= valid
        has = m.any(1)
        v = lp[ar, m.argmax(1)]
        v = np.where(has, v, np.nan).astype(np.float32)
        return v

    top = lp_v.argmax(1)
    out = dict(tch_lp=_col(resp), tch_top1_id=ids[ar, top].astype(np.int64),
               tch_top1_lp=lp[ar, top].astype(np.float32))
    for s in stop_all:
        out[f"tch_lp_{s}"] = _col(int(s))
    return out


def _light_row(ex, b, step, stop_all):
    import numpy as np

    L = int(ex["lens"][b])
    r = ex["responses"][b, :L]
    last = int(r[-1]) if L else None
    last_is_stop = bool(last is not None and last in stop_all)
    row = dict(step=step, seq=b, uid=ex["uid"][b], resp_len=L, score=float(ex["scores"][b]),
               adv=float(ex["adv0"][b]), last_id=last, last_is_stop=last_is_stop,
               truncated=bool(L >= ex["T"] and not last_is_stop),
               n_stop_body=int(np.isin(r[:-1], stop_all).sum()) if L > 1 else 0,
               rep4=rep4(r))
    if ex["old_lp"] is not None and L:
        s = ex["old_lp"][b, :L]
        row["stu_lp_sum"] = float(s.sum())
        row["stu_lp_last"] = float(s[-1])
    return row


def _full_rows(ex, step, n_want, stop_all):
    """抽样整序列(含逐 token 列)+ 整批摘要。"""
    import numpy as np

    B, T, P = ex["B"], ex["T"], ex["P"]
    n = B if n_want <= 0 else min(n_want, B)
    rows, summ = [], []
    for b in range(B):
        L = int(ex["lens"][b])
        base = _light_row(ex, b, step, stop_all)
        tch = _teacher_seq(ex, b, L, stop_all) if (ex["tids"] is not None and L) else None
        s = dict(base)
        if tch is not None:
            t_lp, s_lp = tch["tch_lp"], (ex["old_lp"][b, :L] if ex["old_lp"] is not None else None)
            s["tch_lp_sum"] = float(np.nansum(t_lp))
            s["tch_lp_nan"] = int(np.isnan(t_lp).sum())          # 采样列不在教师块里的位置数
            s["tch_lp_last"] = float(t_lp[-1])
            s["tch_top1_last_id"] = int(tch["tch_top1_id"][-1])
            s["tch_top1_last_lp"] = float(tch["tch_top1_lp"][-1])
            if s_lp is not None:
                dl = s_lp - t_lp
                ok = np.isfinite(dl)          # 采样列不在教师块里的位置是 NaN;全 NaN 时不让 numpy 喊
                s["dl_mean"] = float(dl[ok].mean()) if ok.any() else float("nan")
                s["dl_last"] = float(dl[-1])
                s["dl_max"] = float(dl[ok].max()) if ok.any() else float("nan")
            for sid in stop_all:
                s[f"p_last_{sid}"] = float(np.exp(tch[f"tch_lp_{sid}"][-1]))   # NaN 传播
        summ.append(s)
        if b < n:
            p = ex["prompts"][b]
            if ex["attn"] is not None:
                p = p[ex["attn"][b, :P]]
            row = dict(base, prompt_ids=p.tolist(), response_ids=ex["responses"][b, :L].tolist(),
                       gt=ex["gts"][b])
            if ex["old_lp"] is not None:
                row["stu_lp"] = ex["old_lp"][b, :L].tolist()
            if tch is not None:
                for k, v in tch.items():
                    row[k] = v.tolist()
            rows.append(row)
    return rows, summ


# ---- 写盘 ----------------------------------------------------------------------------
def _write_meta(d, cfg, ex, roll, es, et, stop_all):
    if _state["meta"] or os.path.exists(os.path.join(d, "meta.json")):
        _state["meta"] = True
        return
    meta = dict(
        experiment=os.environ.get("EXPERIMENT_NAME", ""),
        written_at=time.strftime("%FT%TZ", time.gmtime()),
        rollout_stop_set=roll, E_S=es, E_T=et, stop_ids_recorded=stop_all,
        stop_contract=os.environ.get("SIMOPD_STOP_IDS", ""),
        term_event=os.environ.get("SIMOPD_TERM_EVENT", ""), gather_eos=os.environ.get("SIMOPD_GATHER_EOS", ""),
        loss_mode=os.environ.get("DISTILLATION_LOSS_MODE", ""), topk=os.environ.get("DISTILLATION_TOPK", ""),
        B=ex["B"], T=ex["T"], P=ex["P"], K=(int(ex["tids"].shape[-1]) if ex["tids"] is not None else None),
        every=cfg["every"], n_full=cfg["n"], light=cfg["light"],
        columns=dict(
            stu_lp="学生(old_log_probs,rollout 后当前策略)对采样 token 的 logprob,逐 token",
            tch_lp="教师对采样 token 的 logprob(教师块里按 id 找到的列;NaN = 不在 top-K 也未被 gather)",
            tch_top1_id="教师块里 logprob 最大的 id(教师在该位最想出什么)",
            tch_lp_STOPID="教师对该终止符 id 的 logprob 列;p_last_STOPID = 末位置上的 exp",
            dl="Δℓ = stu_lp - tch_lp;dl_last = 序列末 token(自然停止时就是终止事件)",
            n_stop_body="末 token 之前正文里出现终止符 id 的次数(im_end 混进正文的信号)",
            rep4="重复 4-gram 占比(→1 = 死循环)",
            truncated="resp_len == T 且末 token 不是终止符(撞帽)",
            note="事件级修正(TERM_EVENT)的量可离线重构:停机位上 log p_S(E_S)/log q_T(E_T) 由各终止符列 logsumexp",
        ),
    )
    tmp = os.path.join(d, "meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(d, "meta.json"))
    _state["meta"] = True


def _write(batch, step):
    cfg = _cfg()
    is_full = cfg["every"] > 0 and step % cfg["every"] == 0
    if not is_full and not cfg["light"]:
        return
    d = _dir(cfg)
    os.makedirs(d, exist_ok=True)
    ex = _extract(batch, want_teacher=is_full)
    roll, es, et = _stop_sets()
    stop_all = _all_stop_ids(roll, es, et)
    _write_meta(d, cfg, ex, roll, es, et, stop_all)

    if is_full:
        import pandas as pd

        rows, summ = _full_rows(ex, step, cfg["n"], stop_all)
        for name, data in ((f"step_{step}.parquet", rows), (f"summary_{step}.parquet", summ)):
            path = os.path.join(d, name)
            tmp = path + ".tmp"
            pd.DataFrame(data).to_parquet(tmp)
            os.replace(tmp, path)          # 原子:读者永远看不到半截文件
        light_rows = summ                  # 摘要已含轻摘要的全部字段,light 不重复算
        _state["wrote"] += 1
        if _state["wrote"] <= 2 or step % (cfg["every"] * 10) == 0:
            n_stop = sum(1 for r in summ if r["last_is_stop"])
            print(f"[simopd] traj_dump: step {step} -> {d} (整批 {len(summ)} 条摘要,自然停止 {n_stop},"
                  f"截断 {sum(1 for r in summ if r['truncated'])},整序列 {len(rows)} 条,"
                  f"末 id {[r['last_id'] for r in rows[:4]]})", file=sys.stderr, flush=True)
    else:
        light_rows = [_light_row(ex, b, step, stop_all) for b in range(ex["B"])]

    if cfg["light"]:
        keep = ("step", "seq", "uid", "resp_len", "score", "adv", "last_id", "last_is_stop",
                "truncated", "n_stop_body", "rep4", "stu_lp_sum", "stu_lp_last")
        lines = [json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) for r in light_rows]
        with open(os.path.join(d, "light.jsonl"), "a") as f:
            f.write("\n".join(lines) + "\n")


def install():
    """包住 verl 的 _log_rollout_data:我们写 id 版;它那份剥了特殊符的文本只在 SIMOPD_TRAJ_TEXT=1 时放行。"""
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
        if _cfg()["text"]:
            return fn(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
        return None

    setattr(_log_rollout_data, _MARK, True)
    cls._log_rollout_data = _log_rollout_data
    cfg = _cfg()
    print(f"[simopd] traj_dump armed: {_dir(cfg)} (每 {cfg['every']} 步整批摘要 + {cfg['n']} 条整序列,"
          f"light={'on' if cfg['light'] else 'off'},verl 文本={'on' if cfg['text'] else 'off'})",
          file=sys.stderr, flush=True)
