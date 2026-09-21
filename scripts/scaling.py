#!/usr/bin/env python3
"""Plan or launch one explicitly allocated scaling run (dry-run by default).

One job uses 8 GPUs: 4 actor/rollout + 4 teacher. 32 GPUs fit four jobs.
The conservative layout assumes 80GB-class GPUs; it needs a GPU rehearsal.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEACHERS = {s: f"Qwen/Qwen3-{s}B" for s in ("4", "8", "32")}
STUDENTS = {s: f"Qwen/Qwen3-{s}B-Base" for s in ("1.7", "4", "8")}
ARMS = {"vanilla": "scaling_vanilla", "ours": "scaling_ours"}


def matrix(seed=0):
    rows = []
    # Schedule each pair together; cross-size effects need all nine cells.
    for teacher in TEACHERS:
        for student in STUDENTS:
            for method in ARMS:
                rows.append(dict(run_id=f"t{teacher}_s{student}_{method}_seed{seed}",
                                 teacher=teacher, student=student, method=method, seed=seed,
                                 teacher_model=TEACHERS[teacher], student_model=STUDENTS[student],
                                 actor_gpus=4, teacher_gpus=4, rollout_tp=1,
                                 teacher_tp=4 if teacher == "32" else 1,
                                 rollout_iterations=250, status="planned_not_submitted"))
    return rows


def launch_config(teacher, student, method, seed, profile=False):
    arm = next(x for x in yaml.safe_load((ROOT / 'configs/arms.yaml').read_text())['arms']
               if x['run_id'] == ARMS[method])
    run = f"t{teacher}_s{student}_{method}_seed{seed}" + ("_profile" if profile else "")
    env = {k: str(v) for k, v in arm['env'].items()}
    env.update(STUDENT_MODEL=STUDENTS[student],
               TEACHER_MODEL=TEACHERS[teacher], NGPUS_PER_NODE="4", TEACHER_WORLD_SIZE="4",
               SIMOPD_ROLLOUT_TP="1", SIMOPD_TEACHER_TP="4" if teacher == "32" else "1",
               SIMOPD_MODEL_EOS_ID="151643", SIMOPD_SHORT_RUN_OK="1" if profile else "0",
               EXPERIMENT_NAME=run, PROJECT_NAME="simopd_scaling_profile" if profile else "simopd_scaling",
               TOTAL_TRAINING_STEPS="25" if profile else "250", TRAIN_BATCH_SIZE="128",
               PPO_MINI_BATCH_SIZE="128", ACTOR_LR="1e-6", ROLLOUT_N="1",
               DISTILLATION_LOSS_COEF="1.0", TOTAL_EPOCHS="3",
               MAX_PROMPT_LENGTH="1024", MAX_RESPONSE_LENGTH="16384",
               PPO_MAX_TOKEN_LEN_PER_GPU="17408", ROLLOUT_GPU_MEM_UTIL="0.45",
               TEST_FREQ="-1" if profile else "50", SAVE_FREQ="-1" if profile else "50",
               VAL_BEFORE_TRAIN="False", MAX_CKPT_KEEP="-1", USE_REMOVE_PADDING="True",
               FSDP_PARAM_OFFLOAD="False", FSDP_OPTIMIZER_OFFLOAD="False",
               SIMOPD_ARCHIVE="1", WANDB_MODE="offline", RAY_ADDRESS="local")
    args = ["bash", str(ROOT / 'scripts/run_opd_baseline.sh'), f"data.seed={seed}",
            f"actor_rollout_ref.rollout.seed={seed}",
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"]
    return env, args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--out', type=Path)
    plan.add_argument('--seed', type=int, default=0)
    run = sub.add_parser('run')
    run.add_argument('--teacher', choices=TEACHERS, required=True)
    run.add_argument('--student', choices=STUDENTS, required=True)
    run.add_argument('--method', choices=ARMS, required=True)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--profile', action='store_true')
    run.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if args.seed < 0:
        ap.error('seed must be nonnegative')
    if args.cmd == 'plan':
        rows = matrix(args.seed)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
        else:
            print(json.dumps(rows, indent=2))
        print('18 runs/seed; 8 GPUs/run; 32 GPUs => four concurrent runs, five waves.', file=sys.stderr)
        return
    changes, command = launch_config(args.teacher, args.student, args.method, args.seed, args.profile)
    print(json.dumps({'env': changes, 'command': shlex.join(command),
                      'execute': args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    # Do not inherit other arms' knobs, prior archive paths, or a running Ray head.
    env = dict(os.environ)
    operational = {'SIMOPD_STORE', 'SIMOPD_PY', 'SIMOPD_VENV'}
    for key in list(env):
        if key.startswith('SIMOPD_') and key not in operational:
            del env[key]
    for key in ('LOSS_MAX_CLAMP', 'LOG_PROB_MIN_CLAMP', 'EXTRA_HYDRA', 'CUSTOM_REWARD_PATH', 'CUSTOM_REWARD_NAME',
                'TRAIN_FILE_BASENAME', 'VAL_FILE_BASENAME', 'LOGGER', 'VERL_FILE_LOGGER_PATH',
                'VERL_RAY_JOB_ID'):
        env.pop(key, None)
    env.update(changes)
    for key in ('DATA_DIR', 'CKPT_ROOT'):
        if not env.get(key) or not Path(env[key]).is_dir():
            ap.error(f'{key} must point to an existing directory; source your cluster environment first')
    import torch
    if torch.cuda.device_count() != 8:
        ap.error('this launcher requires exactly 8 allocated/visible GPUs; use the Slurm job allocation')
    if min(torch.cuda.get_device_properties(i).total_memory for i in range(8)) < 75 * 1024**3:
        ap.error('the initial layout requires 80GB-class GPUs; smaller cards need a different profile')
    env['PYTHONPATH'] = str(ROOT / 'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    import hashlib
    tag = hashlib.sha256((changes['EXPERIMENT_NAME'] + os.environ.get('SLURM_JOB_ID', str(os.getpid()))).encode()).hexdigest()[:12]
    env['RAY_TMPDIR'] = '/tmp/simopd_scale_' + tag
    env['SIMOPD_LANE_TAG'] = tag
    # One process per run directory, held for the entire child lifetime.
    import fcntl
    lock_path = Path(env['CKPT_ROOT']) / ('.scaling_' + changes['EXPERIMENT_NAME'] + '.lock')
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            ap.error('another process is already writing this run')
        raise SystemExit(subprocess.call(command, env=env, cwd=ROOT))


if __name__ == '__main__':
    main()
