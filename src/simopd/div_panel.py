"""学生—教师分歧面板:在学生自己的轨迹上,逐 token 算 forward KL / reverse KL / JSD / TV,
按序列落盘。每个 top-k 臂统一算,不依赖臂自己的核。

为什么在 worker 上:只有 actor worker 同时握着学生的完整分布(logits)与教师的 top-K 块;
driver 上只有采样列。所以这里包住 verl 的 compute_topk_loss(logit-processor 阶段,逐 token
产出 div_* 键,随其它核输出一起被引擎导出成 nested tensor),再包住注册表里的最终调用
(那里有 response_mask / responses / 由 traj_dump 注入的 simopd_step),按序列聚合后追加到
$SIMOPD_TRAJ_DIR/div/rank<r>.jsonl;dump 步上还写逐 token 的 parquet(2026-09-02 起默认整批,
SIMOPD_DIV_MOD=1;>1 时按 seq_key % MOD == 0 抽样)。

支撑与尾桶(所有臂同一定义,与臂自己的 SUPPORT_MODE / TERM_EVENT 折叠无关):
  S = 教师块里的 id 集合(top-K ∪ 精确 gather 的终止符列 ∪ 保留的采样列,按 id 去重),
  q̂ = (q|S, 1-Σ_S q), p̂ = (p|S, 1-Σ_S p) 各自是 S∪{tail} 上的分布。
  FKL = KL(q̂‖p̂)   RKL = KL(p̂‖q̂)   JSD = ½KL(q̂‖m)+½KL(p̂‖m), m=(p̂+q̂)/2   TV = ½Σ|p̂-q̂|
  qS = Σ_S q(教师块覆盖的教师质量)  pS = Σ_S p(学生落在块上的质量)  agree = 1[argmax_S p == argmax_S q]
  这是真 KL 的下界(数据处理不等式:tail 归并只会减小散度)。

内存:不再物化 [N, V] 的 log_softmax(那是 KEEP_SAMPLED 族的 18.6GiB 杀手);逐 token 分块
logsumexp(SIMOPD_DIV_CHUNK,默认 512 行)+ 在教师 id 上 gather。开销约等于多做一次 logsumexp。

环境变量:SIMOPD_TRAJ_DIR 有值且 SIMOPD_DIV_PANEL!=0 时启用;SIMOPD_TRAJ_EVERY 与 traj_dump 共用
(dump 步);逐 token 落盘的序列子集由 SIMOPD_DIV_MOD 单独控制(默认 1 = 整批;traj_dump 的 step_
子集 SIMOPD_TRAJ_MOD 是它的子集,按 seq_key 仍对得上)。

产物:
  div/rank<r>.jsonl                  每 micro-batch 追加,每序列一行:step seq_key uid n_tok
                                     {fkl,rkl,jsd,tv,qS,pS,agree}_mean, {..}_last, {fkl,rkl,jsd}_tail256
  div/tok_step<n>_rank<r>_<i>.parquet  dump 步、整批(或 key % SIMOPD_DIV_MOD == 0)的序列:逐 token 的七列
失败语义:面板任何一步失败只在 stderr 喊一次,训练照常。"""
import json
import os
import sys
import time

_PAD_LOGPROB_THRESHOLD = -1e15        # 与 topk_losses 同值:verl 只沿序列维 pad(0.0),此阈值只是不变量
PANEL_KEYS = ("div_fkl", "div_rkl", "div_jsd", "div_tv", "div_qS", "div_pS", "div_agree")
_TAIL = 256
_state = {"warned_panel": False, "warned_dump": False, "installed": False, "step": None, "tok_i": 0,
          "wrapped": {}, "rows": 0}


def enabled():
    return (os.environ.get("SIMOPD_TRAJ_DIR", "").strip() != ""
            and os.environ.get("SIMOPD_DIV_PANEL", "1").strip() != "0")


def _cfg():
    e = os.environ
    return dict(root=e.get("SIMOPD_TRAJ_DIR", "").strip(),
                every=int(e.get("SIMOPD_TRAJ_EVERY", "25") or 25),
                mod=int(e.get("SIMOPD_DIV_MOD", "1") or 1),
                chunk=int(e.get("SIMOPD_DIV_CHUNK", "512") or 512))


# ---- logit-processor 阶段:逐 token 面板 ----------------------------------------------
def _lse_chunked(logits, chunk):
    """logits [1, N, V](任意 dtype)-> float32 [1, N] 的 logsumexp,分块以免物化 [N, V]。"""
    import torch

    n = logits.shape[1]
    out = torch.empty((1, n), dtype=torch.float32, device=logits.device)
    for s in range(0, n, chunk):
        out[:, s:s + chunk] = torch.logsumexp(logits[:, s:s + chunk].float(), dim=-1)
    return out


def compute_panel(student_logits, teacher_topk_log_probs, teacher_topk_ids, chunk=512):
    """-> dict 七个 [1, N] float32 张量(与核输出同形,prompt 位也算,response_mask 在最终调用里裁)。"""
    import torch
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    t_lp = teacher_topk_log_probs.values().unsqueeze(0)
    t_id = teacher_topk_ids.values().unsqueeze(0)
    if get_ulysses_sequence_parallel_world_size() > 1:
        t_lp = slice_input_tensor(t_lp, dim=1)
        t_id = slice_input_tensor(t_id, dim=1)
    assert t_lp.shape[:2] == t_id.shape[:2] == student_logits.shape[:2], (t_lp.shape, student_logits.shape)
    return _panel_from_block(student_logits, t_lp.float(), t_id.long(), chunk)


def _panel_from_block(student_logits, t_lp, t_id, chunk):
    import torch

    with torch.no_grad():
        lse = _lse_chunked(student_logits, chunk)                              # [1, N]
        s_lp = torch.gather(student_logits, -1, t_id).float() - lse.unsqueeze(-1)   # [1, N, K]
        valid = t_lp > _PAD_LOGPROB_THRESHOLD
        # 按 id 去重(保留的采样列 / gather 的终止列可能与 top-K 重复):排序后标记后一个重复者
        sid, order = torch.sort(t_id, dim=-1)
        dup_sorted = torch.zeros_like(valid)
        dup_sorted[..., 1:] = sid[..., 1:] == sid[..., :-1]
        dup = torch.zeros_like(valid)
        dup.scatter_(-1, order, dup_sorted)
        use = valid & ~dup
        neg_inf = torch.full_like(t_lp, -float("inf"))
        t_lpm = torch.where(use, t_lp, neg_inf)
        s_lpm = torch.where(use, s_lp, neg_inf)
        q = t_lpm.exp()                                                        # 0 where !use
        p = s_lpm.exp()
        qS = q.sum(-1).clamp(max=1.0 - 1e-6)
        pS = p.sum(-1).clamp(max=1.0 - 1e-6)
        lqt = (1.0 - qS).clamp_min(1e-9).log()
        lpt = (1.0 - pS).clamp_min(1e-9).log()
        d = torch.where(use, t_lp - s_lp, torch.zeros_like(t_lp))              # log q - log p on S
        fkl = (q * d).sum(-1) + lqt.exp() * (lqt - lpt)
        rkl = (-(p * d)).sum(-1) + lpt.exp() * (lpt - lqt)
        lm = torch.logaddexp(t_lpm, s_lpm) - 0.6931471805599453                # log m on S
        lmt = torch.logaddexp(lqt, lpt) - 0.6931471805599453
        jsd = 0.5 * ((q * torch.where(use, t_lpm - lm, torch.zeros_like(lm))).sum(-1) + lqt.exp() * (lqt - lmt)) \
            + 0.5 * ((p * torch.where(use, s_lpm - lm, torch.zeros_like(lm))).sum(-1) + lpt.exp() * (lpt - lmt))
        tv = 0.5 * ((p - q).abs().sum(-1) + (lpt.exp() - lqt.exp()).abs())
        agree = (t_lpm.argmax(-1) == s_lpm.argmax(-1)).float()
        out = dict(div_fkl=fkl.clamp_min(0.0), div_rkl=rkl.clamp_min(0.0), div_jsd=jsd.clamp(0.0, 0.6931471805599453),
                   div_tv=tv.clamp(0.0, 1.0), div_qS=qS, div_pS=pS, div_agree=agree)
        return {k: torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0) for k, v in out.items()}


def _wrap_compute_topk_loss(orig):
    def compute_topk_loss(config, distillation_config, data, student_logits, data_format):
        outputs = orig(config, distillation_config, data, student_logits, data_format)
        if enabled():
            try:
                outputs.update(compute_panel(student_logits, data["teacher_logprobs"], data["teacher_ids"],
                                             _cfg()["chunk"]))
            except Exception as e:
                if not _state["warned_panel"]:
                    _state["warned_panel"] = True
                    import traceback
                    traceback.print_exc()
                    print(f"[simopd] div_panel 逐 token 面板失败({e!r}):这个 run 没有分歧面板,训练继续",
                          file=sys.stderr, flush=True)
        return outputs

    compute_topk_loss._simopd_div_wrapped = True
    return compute_topk_loss


# ---- 最终调用阶段:按序列聚合、落盘 ------------------------------------------------------
def aggregate(vals, mask, resp, step, uids=None, tail=_TAIL):
    """纯 numpy,可离线单测。vals: {key: [B, T] float}, mask: [B, T] bool, resp: [B, T] int。
    -> (rows, keys):每序列一行的标量;keys = 每序列的 seq_key。"""
    import numpy as np
    from simopd.seqkey import np_key

    B, T = mask.shape
    lens = mask.sum(-1).astype(np.int64)
    rows, keys = [], []
    for b in range(B):
        L = int(lens[b])
        key = np_key(resp[b, :L]) if L else 0
        keys.append(key)
        row = dict(step=step, seq_key=key, uid=(uids[b] if uids is not None else None), n_tok=L)
        if L == 0:
            rows.append(row)
            continue
        m = mask[b]
        for k, v in vals.items():
            short = k[4:] if k.startswith("div_") else k
            x = v[b][m]                        # 按 mask 取(response 段一般是前 L 位)
            row[f"{short}_mean"] = float(x.mean())
            row[f"{short}_last"] = float(x[-1])
            if short in ("fkl", "rkl", "jsd"):
                row[f"{short}_tail{tail}"] = float(x[-tail:].mean())
        rows.append(row)
    return rows, keys


def _rank():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0") or 0)


def _sp_rank0():
    try:
        from verl.utils.ulysses import get_ulysses_sequence_parallel_rank
        return get_ulysses_sequence_parallel_rank() == 0
    except Exception:
        return True


def _dump(model_output, data):
    import numpy as np
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding

    if not _sp_rank0():
        return
    cfg = _cfg()
    mask = data["response_mask"]
    mask = (mask.to_padded_tensor(False) if getattr(mask, "is_nested", False) else mask).bool()
    resp = data["responses"]
    resp = getattr(resp, "data", resp)
    if getattr(resp, "is_nested", False):
        resp = resp.to_padded_tensor(0)
    vals = {}
    for k in PANEL_KEYS:
        v = no_padding_2_padding(model_output[k], data)
        vals[k] = v.detach().float().cpu().numpy()
    mask_np = mask.cpu().numpy()
    resp_np = resp.cpu().numpy().astype(np.int64)
    step = tu.get_non_tensor_data(data=data, key="simopd_step", default=None)
    uids = None
    try:
        u = data.get("uid", None)
        if u is not None:
            uids = [str(x) for x in (u.tolist() if hasattr(u, "tolist") else list(u))]
            if len(uids) != mask_np.shape[0]:
                uids = None
    except Exception:
        uids = None
    rows, keys = aggregate(vals, mask_np, resp_np, step, uids)

    d = os.path.join(cfg["root"], "div")
    os.makedirs(d, exist_ok=True)
    r = _rank()
    with open(os.path.join(d, f"rank{r}.jsonl"), "a") as f:
        f.write("\n".join(json.dumps(x) for x in rows) + "\n")
    _state["rows"] += len(rows)

    # dump 步 + 采样子集:逐 token
    if step is not None and cfg["every"] > 0 and int(step) % cfg["every"] == 0:
        if _state["step"] != step:
            _state["step"], _state["tok_i"] = step, 0
        sel = [b for b, k in enumerate(keys) if mask_np[b].any() and k % cfg["mod"] == 0]
        if sel:
            import pandas as pd
            tok = []
            for b in sel:
                m = mask_np[b]
                row = dict(step=int(step), seq_key=keys[b], n_tok=int(m.sum()))
                for k in PANEL_KEYS:
                    row[k[4:]] = vals[k][b][m].astype(np.float32).tolist()
                tok.append(row)
            path = os.path.join(d, f"tok_step{int(step)}_rank{r}_{_state['tok_i']}.parquet")
            _state["tok_i"] += 1
            pd.DataFrame(tok).to_parquet(path + ".tmp")
            os.replace(path + ".tmp", path)
    if _state["rows"] <= len(rows) or (step is not None and int(step) % (cfg["every"] * 10) == 0 and _state["tok_i"] <= 1):
        fk = [x.get("fkl_mean") for x in rows if x.get("fkl_mean") is not None]
        print(f"[simopd] div_panel rank{r}: step {step} +{len(rows)} 行 -> {d} "
              f"(fkl_mean 均值 {sum(fk) / max(len(fk), 1):.3f})", file=sys.stderr, flush=True)


def _with_div_dump(fn):
    def wrapped(config, distillation_config, model_output, data):
        losses, metrics = fn(config, distillation_config, model_output, data)
        if enabled() and all(k in model_output for k in PANEL_KEYS):
            try:
                _dump(model_output, data)
            except Exception as e:
                if not _state["warned_dump"]:
                    _state["warned_dump"] = True
                    import traceback
                    traceback.print_exc()
                    print(f"[simopd] div_panel 按序列落盘失败({e!r}):div/ 不会有记录,训练继续",
                          file=sys.stderr, flush=True)
        return losses, metrics

    wrapped.__name__ = getattr(fn, "__name__", "fn")
    wrapped._simopd_div_wrapped = True
    return wrapped


def install():
    """包 verl.trainer.distillation.losses 的 compute_topk_loss(当前绑定的,通常已是 simopd 的
    分发器)与 get_distillation_loss_fn(查表时再包,晚登记的臂也覆盖)。幂等。"""
    if _state["installed"] or not enabled():
        return
    import verl.trainer.distillation.losses as vl

    if not getattr(vl.compute_topk_loss, "_simopd_div_wrapped", False):
        vl.compute_topk_loss = _wrap_compute_topk_loss(vl.compute_topk_loss)
    orig_get = vl.get_distillation_loss_fn
    if not getattr(orig_get, "_simopd_div_wrapped", False):
        def get_distillation_loss_fn(loss_name):
            fn = orig_get(loss_name)
            if getattr(fn, "_simopd_div_wrapped", False):
                return fn
            w = _state["wrapped"].get(id(fn))
            if w is None:
                w = _with_div_dump(fn)
                _state["wrapped"][id(fn)] = w
            return w
        get_distillation_loss_fn._simopd_div_wrapped = True
        vl.get_distillation_loss_fn = get_distillation_loss_fn
    _state["installed"] = True
    cfg = _cfg()
    print(f"[simopd] div_panel armed: {os.path.join(cfg['root'], 'div')} (每序列 FKL/RKL/JSD/TV/qS/pS/agree,"
          f"逐 token 于 step%{cfg['every']}==0 且 key%{cfg['mod']}==0)", file=sys.stderr, flush=True)
