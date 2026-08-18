"""Additive direct terms (b3's soft-KD, N2's EOS channel), delivered past the PG detach.

EOPD's official loss (WLS04/EOPD core_algos) is `pg_loss = pg_loss + soft_kd_loss`:
a PG base on every token plus a directly-differentiated forward-KL term on
high-teacher-entropy tokens. verl's `distillation_loss` offers one branch or the
other -- whatever the registry function returns is either detached into advantages
(PG) or aggregated directly -- so a loss that needs BOTH cannot be expressed from
inside the registry. This wraps `distillation_loss` itself: b3's registry function
computes the normalized additive term (with its pathwise gradient intact), leaves it
in STASH, and the wrapper adds it to whatever the original returned.

The stash is cleared BEFORE each inner call, not after: an exception between fill
and pop would otherwise leave a stale graph-bearing tensor to be added to the NEXT
micro-batch's loss -- backward through a freed graph, an error whose traceback
points nowhere near here. Single-threaded per rank (verl's actor update loop), so a
module dict is a safe mailbox; it never crosses process or thread boundaries.
"""

import sys

_MARK = "_simopd_b3_additive"
STASH: dict = {}


def install():
    import verl.trainer.distillation.losses as L

    fn = L.distillation_loss
    if getattr(fn, _MARK, False):
        return

    def distillation_loss(config, distillation_config, model_output, data):
        STASH.clear()
        loss, metrics = fn(config, distillation_config, model_output, data)
        # Every stashed term is added, whichever registry fn left it: b3's
        # "soft_kd", N2's "eos_aux" (2026-08-19). Pop as we go so nothing carries.
        for key in list(STASH):
            loss = loss + STASH.pop(key)
        return loss, metrics

    setattr(distillation_loss, _MARK, True)
    L.distillation_loss = distillation_loss
    print("[simopd] b3_additive armed on verl distillation_loss", file=sys.stderr, flush=True)


install()
