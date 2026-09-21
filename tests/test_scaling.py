import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scaling', ROOT / 'scripts/scaling.py')
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


def test_all_nine_model_pairs_have_two_matched_runs():
    rows = scaling.matrix()
    assert len(rows) == 18 and len({x['run_id'] for x in rows}) == 18
    for teacher in scaling.TEACHERS:
        for student in scaling.STUDENTS:
            pair = [x for x in rows if x['teacher'] == teacher and x['student'] == student]
            assert {x['method'] for x in pair} == {'vanilla', 'ours'}
            assert all(x['actor_gpus'] + x['teacher_gpus'] == 8 for x in pair)
            assert all(x['teacher_gpus'] % x['teacher_tp'] == 0 for x in pair)


def test_launch_preserves_controls_and_enables_all_components():
    common = ['STUDENT_MODEL', 'TEACHER_MODEL', 'NGPUS_PER_NODE', 'TEACHER_WORLD_SIZE',
              'SIMOPD_ROLLOUT_TP', 'SIMOPD_TEACHER_TP', 'SIMOPD_STOP_IDS',
              'SIMOPD_TERM_EVENT', 'TRAIN_BATCH_SIZE', 'PPO_MINI_BATCH_SIZE',
              'ACTOR_LR', 'MAX_RESPONSE_LENGTH', 'TOTAL_TRAINING_STEPS']
    v, _ = scaling.launch_config('32', '8', 'vanilla', 0)
    o, _ = scaling.launch_config('32', '8', 'ours', 0)
    assert all(v[k] == o[k] for k in common)
    assert o['SIMOPD_COMPOSED'] == '1' and o['USE_POLICY_GRADIENT'] == 'False'
    assert o['SIMOPD_COMPOSED_ALPHA'] == '0.1' and o['SIMOPD_COMPOSED_CLIP'] == '10'
    assert o['SIMOPD_QB_TARGET_BUDGET'] == '8'
    assert o['SIMOPD_SELECTKD_K'] == '5' and o['SIMOPD_SELECTKD_BETA'] == '0.01'
    assert v['SIMOPD_COMPOSED'] == '0'
    assert v['LOSS_MAX_CLAMP'] == '10.0'  # pinned verl default, made explicit
    assert o['LOSS_MAX_CLAMP'] == 'null'  # internal credit clipping, never clamp the scalar loss


def test_profile_cannot_be_mistaken_for_formal_run():
    env, _ = scaling.launch_config('4', '1.7', 'ours', 0, profile=True)
    assert env['TOTAL_TRAINING_STEPS'] == '25'
    assert env['EXPERIMENT_NAME'].endswith('_profile')
    assert env['PROJECT_NAME'] == 'simopd_scaling_profile'
    assert env['SAVE_FREQ'] == '-1'
