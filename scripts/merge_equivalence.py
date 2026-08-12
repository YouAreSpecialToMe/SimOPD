#!/usr/bin/env python3
"""CPU equivalence battery for a topk_losses merge.

Written for the 2026-08-12 `main -> ch-dev` merge, which brought four fixes onto
the streaming-lse rewrite. The claim that has to hold before repinning a fleet
whose 84 banked rows were trained on the pre-merge snapshot:

    every kernel's `distillation_losses` is unchanged; only diagnostic keys move.

That is what makes runs on either side of the pin comparable, and it is exactly
the property the merge could silently break -- the grafted stat mask feeds the
shadow panel, which shares helpers with the loss path.

Usage (on a box with the project venv; no GPU needed):

    ./simopd/bin/python scripts/merge_equivalence.py               # HEAD vs merge base's ch-dev
    ./simopd/bin/python scripts/merge_equivalence.py --ref 023cef4 # explicit baseline

Compares the working-tree module against `git show <ref>:src/simopd/topk_losses.py`
loaded side by side under a different module name, on identical synthetic inputs.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import types

os.environ.setdefault("SIMOPD_SHADOW", "1")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stub_config(topk=32):
    """The kernels read config.distillation_loss.{topk,loss_mode} and
    config.log_prob_min_clamp only -- a namespace is enough, and keeping it a
    stub is deliberate: a real hydra config would let a default drift in."""
    dl = types.SimpleNamespace(topk=topk, loss_mode="lsm_topk_renorm",
                               loss_max_clamp=None, log_prob_min_clamp=None)
    return types.SimpleNamespace(distillation_loss=dl, log_prob_min_clamp=-20.0,
                                 strategy="fsdp")


class _NT:
    """A stand-in for verl's nested teacher payload.

    The production tensor is jagged-nested and the kernels reach for TWO things on
    it: `.offsets()` (sequence spans, used by _stat_mask and qb's per-sequence tau)
    and the ordinary packed [1, T, k] view. torch.nested's jagged tensors do not
    expose offsets() the same way in every version, and a plain dense tensor
    exposes it not at all -- which is how the first run of this battery quietly
    tested the FALLBACK path (stat mask unavailable) instead of the stat-mask path
    the merge actually changed. So the harness supplies offsets explicitly.

    The surface the kernels actually use is small and worth stating: `is_nested`
    (asserted), `values()` (the packed [T, k] view), `offsets()` (spans) and
    `device`. Everything else falls through to the packed tensor.
    """

    def __init__(self, packed, offs):
        self._t = packed          # [1, T, k]
        self._offs = offs

    is_nested = True

    def offsets(self):
        return self._offs

    def values(self):
        return self._t.squeeze(0)

    # everything else behaves like the packed tensor
    def __getattr__(self, k):
        return getattr(self._t, k)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        unwrap = lambda x: x._t if isinstance(x, _NT) else x
        args = tuple(unwrap(a) for a in args)
        kwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}
        return func(*args, **kwargs)


def make_batch(seed=0, seqs=(7, 11, 5), vocab=257, k=32, prompt=3, keep_sampled=False):
    """Packed micro-batch in the shape the kernels see: [1, T, V] student logits,
    a teacher top-k payload carrying real sequence offsets, and a response_mask
    whose per-sequence true count is < the sequence length, so _stat_mask builds a
    genuine response-tail mask instead of falling back to the full population.

    keep_sampled widens the teacher payload to k+1 (the D/b3/g2 family asserts on
    that width: verl drops the sampled token's teacher logprob without it)."""
    g = torch.Generator().manual_seed(seed)
    total = sum(seqs)
    width = k + 1 if keep_sampled else k
    student_logits = torch.randn(1, total, vocab, generator=g, dtype=torch.float32)

    lp_rows, id_rows = [], []
    for n in seqs:
        logits = torch.randn(n, vocab, generator=g, dtype=torch.float32)
        lp = torch.log_softmax(logits, dim=-1)
        v, i = torch.topk(lp, k=width, dim=-1)
        lp_rows.append(v)
        id_rows.append(i)
    offs = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqs), 0)))
    teacher_lp = _NT(torch.cat(lp_rows, 0).unsqueeze(0), offs)
    teacher_id = _NT(torch.cat(id_rows, 0).unsqueeze(0), offs)

    rm = torch.zeros(len(seqs), max(seqs), dtype=torch.bool)
    for r, n in enumerate(seqs):
        rm[r, prompt:n] = True          # last (n - prompt) rows are the response
    data = {"teacher_logprobs": teacher_lp, "teacher_ids": teacher_id,
            "response_mask": rm}
    return student_logits, teacher_lp, teacher_id, data


def make_empty_response_batch(**kw):
    """The draw that killed c4's seeds 1 and 2 at step 1 on every relaunch: a
    micro-batch in which the stat mask selects nothing."""
    sl, t_lp, t_id, data = make_batch(**kw)
    data["response_mask"] = torch.zeros_like(data["response_mask"])
    return sl, t_lp, t_id, data


# (label, function, needs the k+1 KEEP_SAMPLED payload)
KERNELS = [
    ("c1  compute_reverse_kl_topk", "compute_reverse_kl_topk", False),
    ("c2  compute_quantile_budget_topk", "compute_quantile_budget_topk", False),
    ("c3  compute_intersection_topk", "compute_intersection_topk", False),
    ("c4  compute_pi_tail_budget_topk", "compute_pi_tail_budget_topk", False),
    ("e1  compute_pl_rank_topk", "compute_pl_rank_topk", False),
    ("e2  compute_set_coverage_topk", "compute_set_coverage_topk", False),
    ("e3  compute_zvalue_topk", "compute_zvalue_topk", False),
    ("b3  compute_eopd_gate_topk", "compute_eopd_gate_topk", True),
    ("b4  compute_jsd_topk", "compute_jsd_topk", False),
    ("g2  compute_fire_components", "compute_fire_components", True),
]


def call(mod, fname, batch, cfg):
    sl, t_lp, t_id, data = batch
    fn = getattr(mod, fname, None)
    if fn is None:
        return None
    return fn(student_logits=sl, teacher_topk_log_probs=t_lp, teacher_topk_ids=t_id,
              config=cfg, data_format="packed", data=data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None,
                    help="git ref for the baseline module (default: HEAD)")
    ap.add_argument("--old-file", default=None,
                    help="baseline module as a file, instead of resolving --ref through git "
                         "(lets the battery run on a box that has the venv but not this branch)")
    ap.add_argument("--keep-sampled-only", action="store_true",
                    help="run ONLY the k+1 KEEP_SAMPLED kernels (b3/g2). SIMOPD_KEEP_SAMPLED is "
                         "read into a module global at import, so one process cannot cover both "
                         "families -- run this pass with the env set, the default pass without.")
    ap.add_argument("--new-file", default=None,
                    help="candidate module as a file (default: the working tree's)")
    a = ap.parse_args()

    if a.old_file:
        ref, old_path = f"file:{os.path.basename(a.old_file)}", a.old_file
    else:
        ref = a.ref or subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        blob = subprocess.check_output(
            ["git", "-C", REPO, "show", f"{ref}:src/simopd/topk_losses.py"], text=True)
        old_path = os.path.join(tempfile.mkdtemp(prefix="simopd_eq_"), "topk_losses_old.py")
        with open(old_path, "w") as fh:
            fh.write(blob)

    new_path = a.new_file or os.path.join(REPO, "src/simopd/topk_losses.py")
    sys.path.insert(0, os.path.join(REPO, "src"))
    new = load_module(new_path, "tl_new")
    old = load_module(old_path, "tl_old")

    cfg = stub_config()
    print(f"baseline ref : {ref[:12]}")
    print(f"torch        : {torch.__version__}  (CPU, fp32)")
    print()

    want_ks = a.keep_sampled_only
    kernels = [k for k in KERNELS if k[2] == want_ks]
    env_ks = os.environ.get("SIMOPD_KEEP_SAMPLED", "0") == "1"
    if want_ks != env_ks:
        print(f"REFUSING: --keep-sampled-only={want_ks} but SIMOPD_KEEP_SAMPLED={env_ks}. "
              f"The flag and the env must agree or every kernel asserts on payload width.")
        return 2
    print(f"pass         : {'KEEP_SAMPLED (k+1 payload)' if want_ks else 'plain top-k'}"
          f"  -- {len(kernels)} kernels")
    print()

    LOSS_KEY = "distillation_losses"
    fails = 0
    print(f"{'kernel':38} {'loss maxdiff':>14}  {'diag keys':>22}  verdict")
    print("-" * 96)
    for label, fname, ks in kernels:
        batch = make_batch(seed=1, keep_sampled=ks)
        try:
            o = call(old, fname, batch, cfg)
        except Exception as e:
            print(f"{label:38} {'--':>14}  baseline raised {type(e).__name__}: {e}")
            fails += 1
            continue
        try:
            n = call(new, fname, batch, cfg)
        except Exception as e:
            print(f"{label:38} {'--':>14}  MERGED raised {type(e).__name__}: {e}")
            fails += 1
            continue
        if o is None or n is None:
            print(f"{label:38} {'--':>14}  {'absent':>22}  SKIP")
            continue

        lo, ln = o[LOSS_KEY].double(), n[LOSS_KEY].double()
        d = (lo - ln).abs().max().item()
        ok = d == 0.0
        # which non-loss keys changed, and did the key SET change
        changed = []
        for k in sorted(set(o) | set(n)):
            if k == LOSS_KEY:
                continue
            if k not in o or k not in n:
                changed.append(f"{k}(+/-)")
            elif not torch.equal(o[k].double().nan_to_num(), n[k].double().nan_to_num()):
                changed.append(k)
        note = ",".join(changed)[:22] if changed else "identical"
        verdict = "PASS" if ok else "**LOSS MOVED**"
        if not ok:
            fails += 1
        print(f"{label:38} {d:14.3e}  {note:>22}  {verdict}")

    print()
    print("--- empty-response micro-batch (the c4 step-1 draw) ---")
    for label, fname, ks in kernels:
        batch = make_empty_response_batch(seed=2, keep_sampled=ks)
        try:
            call(new, fname, batch, cfg)
            print(f"  {label:38} survives")
        except Exception as e:
            print(f"  {label:38} RAISES {type(e).__name__}: {e}")
            fails += 1

    print()
    print("--- stat mask contract ---")
    sl, t_lp, t_id, data = make_batch(seed=3)
    total = int(t_lp.shape[1])
    m = new._stat_mask(t_lp, data, total)
    mo = old._stat_mask(t_lp, data, total)
    if m is None or mo is None:
        print("  FAIL: stat mask came back None -- the battery is testing the FALLBACK "
              "path, not the path the merge changed")
        fails += 1
    else:
        dev_ok = m.device == t_lp.device
        same = torch.equal(m, mo)
        print(f"  exercised            : {int(m.sum())} of {m.numel()} packed rows selected "
              f"(a real response-tail mask, not a fallback)")
        print(f"  device follows input : {m.device} == {t_lp.device}  {'OK' if dev_ok else 'FAIL'}")
        print(f"  bitwise vs baseline  : {'OK' if same else 'FAIL'}")
        if not dev_ok or not same:
            fails += 1

    print()
    print("--- entropy chunking is elementwise-invariant ---")
    lp = torch.log_softmax(torch.randn(1, 40, 257, generator=torch.Generator().manual_seed(4)), dim=-1)
    ref_ent = -(lp.exp() * lp).sum(dim=-1)
    for ch in (4096, 1024, 256, 7):
        got = new._student_entropy(lp, chunk=ch)
        d = (got - ref_ent).abs().max().item()
        print(f"  chunk={ch:<5} maxdiff {d:.3e}  {'OK' if d <= 1e-6 else 'FAIL'}")
        if d > 1e-6:
            fails += 1

    print()
    print("RESULT:", "ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
