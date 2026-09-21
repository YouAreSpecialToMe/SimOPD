"""CPU tests; no claim that mocks replace the required cluster rehearsal."""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simopd.composed import (MODE, adaptive_support, apply_failure_gate, clipped_skl,
                             failure_metadata, install_trainer, prediction_mask,
                             selectkd_weights)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5])
def test_unclipped_gradient_matches_exact_divergence(alpha):
    torch.manual_seed(10)
    z = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    q = torch.randn(3, 7, dtype=torch.float64).log_softmax(-1).requires_grad_()
    keep = torch.tensor([[1, 0, 1, 1, 0, 1, 0]] * 3, dtype=torch.bool)
    actual, _ = clipped_skl(z.log_softmax(-1), q, keep, alpha, 100)
    p_ref = z.masked_fill(~keep, -torch.inf).softmax(-1)
    q_ref = q.detach().masked_fill(~keep, -torch.inf).softmax(-1)
    mix = alpha * p_ref + (1 - alpha) * q_ref
    reference = (p_ref[keep] * (p_ref[keep].log() - mix[keep].log())).sum()
    torch.testing.assert_close(actual.sum(), reference)
    grad = torch.autograd.grad(actual.sum(), z, retain_graph=True)[0]
    expected = torch.autograd.grad(reference, z)[0]
    torch.testing.assert_close(grad, expected, atol=1e-12, rtol=1e-10)
    assert q.grad is None


def test_clip_changes_gradient_and_is_finite_at_extremes():
    z = torch.tensor([[0., -80., -2.]], requires_grad=True)
    q = torch.tensor([[-80., 0., -2.]]).log_softmax(-1)
    keep = torch.ones_like(z, dtype=torch.bool)
    unclipped, _ = clipped_skl(z.log_softmax(-1), q, keep, clip=100)
    clipped, diag = clipped_skl(z.log_softmax(-1), q, keep, clip=1)
    a = torch.autograd.grad(unclipped.sum(), z, retain_graph=True)[0]
    b = torch.autograd.grad(clipped.sum(), z)[0]
    assert torch.isfinite(b).all() and not torch.allclose(a, b)
    assert diag['recipe_clip_frac'].item() > 0
    assert diag['recipe_clipped_credit_abs'].item() <= 1.00001


def test_reverse_kl_boundary_avoids_zero_times_infinity():
    z = torch.tensor([[0., -150.]], requires_grad=True)
    q = torch.tensor([[-150., 0.]]).log_softmax(-1)
    loss, diag = clipped_skl(z.log_softmax(-1), q, torch.ones_like(z).bool(), alpha=0)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(torch.autograd.grad(loss.sum(), z)[0]).all()
    assert diag['recipe_clip_frac'].item() == 1


def test_singleton_support_and_equal_distributions_have_zero_gradient():
    z = torch.randn(2, 6, requires_grad=True)
    for keep, q in [(torch.ones_like(z, dtype=torch.bool), z.detach().log_softmax(-1)),
                    (torch.tensor([[1, 0, 0, 0, 0, 0]] * 2).bool(), torch.randn_like(z).log_softmax(-1))]:
        loss, _ = clipped_skl(z.log_softmax(-1), q, keep)
        grad = torch.autograd.grad(loss.sum(), z, retain_graph=True)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-7, rtol=0)


def test_budget_ignores_prompt_and_dummy_rows_and_keeps_top1():
    p = torch.tensor([[[-2., -3., -4., -5.], [-1., -2., -3., -4.], [-9., -8., -7., -6.]]])
    valid = torch.tensor([[False, True, False]])
    a = adaptive_support(p, p, valid, 2)
    p[:, 0] = 100
    p[:, 2] = -100
    b = adaptive_support(p, p, valid, 2)
    assert torch.equal(a[valid], b[valid]) and b[..., 0].all()
    assert b[valid].sum() == 2
    assert adaptive_support(p, p, torch.zeros_like(valid), 2)[..., 0].all()


def test_prediction_mask_matches_verl_left_shift():
    payload = torch.nested.as_nested_tensor([torch.zeros(6, 4), torch.zeros(4, 4)], layout=torch.jagged)
    rm = torch.tensor([[1, 1, 1], [1, 0, 0]])
    actual = prediction_mask(payload, rm, 10)
    assert actual.nonzero()[:, 1].tolist() == [2, 3, 4, 8]
    with pytest.raises(ValueError):
        prediction_mask(payload, rm, 8)


def test_selectkd_uses_original_top5_not_adaptive_support():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    weight, accepted = selectkd_weights(ids, torch.tensor([5, 6]))
    torch.testing.assert_close(weight, torch.tensor([1., .01]))
    assert accepted.tolist() == [1., 0.]


def test_gate_gradient_matches_global_failure_token_mean_across_shards():
    scores = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [1., 0., 0.]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]])
    failed, scale, count = failure_metadata(scores, mask)
    assert count == 4 and scale == 7 / 4
    z = torch.randn(4, 3, requires_grad=True)
    weights = torch.tensor([[1., .01, 1.]] * 4)
    # Two DP ranks, uneven microbatches. Each contributes dp_size/global_tokens;
    # the final DP average must exactly recover the full failure-token objective.
    shards = []
    for indices in [[0], [1], [2, 3]]:
        local = apply_failure_gate(z[indices] * weights[indices], {
            'simopd_failure': failed[indices],
            'simopd_failure_scale': torch.full((len(indices),), scale)})
        shards.append((local * mask[indices]).sum() * 2 / mask.sum())
    actual = sum(shards) / 2
    expected = (z * weights * mask * failed[:, None]).sum() / count
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, z, retain_graph=True)[0],
                               torch.autograd.grad(expected, z)[0])


@pytest.mark.parametrize('scores', [torch.tensor([[float('nan'), 0.]]), torch.tensor([[.5, 0.]])])
def test_invalid_verifier_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        failure_metadata(scores, torch.ones_like(scores))


def test_missing_driver_metadata_fails_closed():
    with pytest.raises(RuntimeError):
        apply_failure_gate(torch.ones(2, 3), {})


def test_trainer_hook_skips_optimizer_for_all_success(monkeypatch):
    monkeypatch.setenv('SIMOPD_COMPOSED', '1')
    class DP:
        @staticmethod
        def from_single_dict(data, meta_info):
            return types.SimpleNamespace(meta_info=meta_info)
    monkeypatch.setitem(sys.modules, 'verl', types.SimpleNamespace(DataProto=DP))
    ns = types.SimpleNamespace
    class Trainer:
        calls = 0
        config = ns(actor_rollout_ref=ns(actor=ns(ppo_epochs=1, ppo_mini_batch_size=2,
                                                 loss_agg_mode='token-mean'), rollout=ns(n=1)))
        distillation_config = ns(distillation_loss=ns(loss_mode=MODE, use_policy_gradient=False,
                                                     use_task_rewards=False, loss_max_clamp=None))
        def _update_actor(self, batch):
            self.calls += 1
            assert 'simopd_failure' in batch.batch
            return ns(meta_info={'metrics': {}})
    install_trainer(Trainer)
    install_trainer(Trainer)  # idempotent
    trainer = Trainer()
    batch = ns(batch={'response_mask': torch.ones(2, 2),
                      'token_level_scores': torch.tensor([[0., 1.], [1., 0.]])})
    out = trainer._update_actor(batch)
    assert trainer.calls == 0 and out.meta_info['metrics']['actor/recipe/skipped_all_success'] == 1
    batch.batch['token_level_scores'][0] = 0
    out = trainer._update_actor(batch)
    assert trainer.calls == 1 and out.meta_info['metrics']['actor/recipe/failure_fraction'] == .5


def test_packed_kernel_matches_dense_reference(monkeypatch):
    # Only the process-group query is mocked; tensor packing, support, selector,
    # streamed log-normalizer and backward all run through production code.
    ulysses = types.ModuleType('verl.utils.ulysses')
    ulysses.get_ulysses_sequence_parallel_world_size = lambda: 1
    ulysses.slice_input_tensor = lambda x, dim: x
    monkeypatch.setitem(sys.modules, 'verl.utils.ulysses', ulysses)
    from simopd import topk_losses as T
    from simopd.composed import compute_composed_topk
    monkeypatch.setattr(T, 'TERM_EVENT', False)
    monkeypatch.setattr(T, '_carrier_on', lambda: False)
    torch.manual_seed(21)
    z = torch.randn(1, 13, 20, requires_grad=True)
    teacher_lp, teacher_ids = torch.randn(13, 20).log_softmax(-1).topk(8, dim=-1)
    lp = torch.nested.as_nested_tensor([teacher_lp[:6], teacher_lp[6:]], layout=torch.jagged)
    ids = torch.nested.as_nested_tensor([teacher_ids[:6], teacher_ids[6:]], layout=torch.jagged)
    config = types.SimpleNamespace(distillation_loss=types.SimpleNamespace(topk=8))
    monkeypatch.setenv('SIMOPD_QB_TARGET_BUDGET', '3')
    data = {'response_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
    out = compute_composed_topk(z, lp, ids, config, 'thd', data)
    assert all(x.shape == (1, 13) for x in out.values())
    p = z.log_softmax(-1).gather(-1, teacher_ids.unsqueeze(0))
    valid = prediction_mask(lp, data['response_mask'], 13)
    keep = adaptive_support(p, teacher_lp.unsqueeze(0), valid, 3)
    ref, _ = clipped_skl(p, teacher_lp.unsqueeze(0), keep)
    weights, _ = selectkd_weights(teacher_ids.unsqueeze(0), z.detach().argmax(-1))
    torch.testing.assert_close(out['distillation_losses'], ref * weights)
    actual = torch.autograd.grad(out['distillation_losses'][valid].sum(), z, retain_graph=True)[0]
    expected = torch.autograd.grad((ref * weights)[valid].sum(), z)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_packed_kernel_with_terminal_carrier_and_dummy_row(monkeypatch):
    ulysses = types.ModuleType('verl.utils.ulysses')
    ulysses.get_ulysses_sequence_parallel_world_size = lambda: 1
    ulysses.slice_input_tensor = lambda x, dim: x
    monkeypatch.setitem(sys.modules, 'verl.utils.ulysses', ulysses)
    from simopd import topk_losses as T, eos_gather as EG
    from simopd.composed import compute_composed_topk
    monkeypatch.setattr(T, 'TERM_EVENT', True)
    monkeypatch.setattr(T, '_carrier_on', lambda: True)
    monkeypatch.setattr(EG, 'enabled', lambda: True)
    monkeypatch.setattr(EG, 'n_extra', lambda: 2)
    monkeypatch.setattr(EG, 'teacher_cols', lambda: [0, 1])
    # Six ordinary candidates, two terminal coordinates, one sampled column.
    row = torch.tensor([.10, .09, .08, .07, .06, .05, .20, .25, .10]).log()
    q = row.repeat(5, 1)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9]).repeat(5, 1)
    q[-1] = 0
    ids[-1] = 0  # actual verl trailing dummy contract
    nq = torch.nested.as_nested_tensor([q], layout=torch.jagged)
    ni = torch.nested.as_nested_tensor([ids], layout=torch.jagged)
    z = torch.zeros(1, 5, 12, requires_grad=True)
    with torch.no_grad():
        z[..., 6] = 3  # student EOS matches merged teacher termination event
    monkeypatch.setenv('SIMOPD_QB_TARGET_BUDGET', '2')
    cfg = types.SimpleNamespace(distillation_loss=types.SimpleNamespace(topk=8))
    rm = torch.tensor([[1, 1, 1]])
    out = compute_composed_topk(z, nq, ni, cfg, 'thd', {'response_mask': rm})
    valid = prediction_mask(nq, rm, 5)
    assert (out['recipe_selectkd_accept'][valid] == 1).all()
    assert torch.isfinite(out['distillation_losses']).all()
    grad = torch.autograd.grad(out['distillation_losses'][valid].sum(), z)[0]
    assert torch.isfinite(grad).all()
    assert not grad[:, -1].any()  # final dummy never updates the model
