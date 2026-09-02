#!/usr/bin/env python3
"""把"这个 run 到底是什么"写成一份 JSON,放在 run 自己的 ckpt 目录里。

    python scripts/run_manifest.py --ckpt-dir <dir> --fingerprint <sha> --arm-args "..." --extra "..."

写 <ckpt_dir>/run_manifest.json(最新一次启动)和 <ckpt_dir>/manifest/launch_<ts>.json(每次
启动一份,重启/续跑各留一条)。内容:臂的全部解析后环境(SIMOPD_* / DISTILLATION_* / 长度 /
步数 / 模型 / 数据)、hydra 覆盖、停机契约、resume 指纹、SimOPD 与 verl 的 git sha、评测协议
(SIMOPD_SUITE_K)、机器与卡。凡名字里带 KEY/TOKEN/SECRET/PASSWORD 的变量一律不收 —— 这个
文件会随 ckpt 一起上 HuggingFace。

为什么:2026-08-24 旧集群失联后,复盘一条曲线要靠 wandb 的 config 和 arms.yaml 的历史版本
互相拼;而 arms.yaml 会改、wandb 需要登录。run 的定义应该跟 run 的产物躺在一起。失败只
打印警告(由启动器 `|| echo` 兜住),永远不能拦住训练。
"""
import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time

_PREFIXES = ("SIMOPD_", "DISTILLATION_", "MAX_", "TOTAL_", "TRAIN_", "PPO_", "ROLLOUT_", "ACTOR_",
             "LOSS_", "LOG_PROB_", "USE_TASK_REWARDS", "USE_POLICY_GRADIENT", "USE_REMOVE_PADDING",
             "TEACHER_", "STUDENT_", "NGPUS", "CKPT_", "DATA_", "VAL_",
             "SAVE_", "TEST_", "LANE_", "VERL_", "CUDA_VISIBLE_DEVICES", "WANDB_RUN_GROUP",
             "WANDB_TAGS", "WANDB_PROJECT", "WANDB_MODE", "EXPERIMENT_NAME", "PROJECT_NAME",
             "SEED", "TAG", "REHEARSAL", "CUSTOM_REWARD", "LOGGER", "CKPT_SYNC_REPO")
_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)


def _git(path):
    try:
        sha = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20).stdout.strip() != ""
        return {"sha": sha or None, "dirty": dirty} if sha else None
    except Exception:
        return None


def collect(a):
    env = {k: v for k, v in os.environ.items()
           if k.startswith(_PREFIXES) and not _SECRET.search(k)}
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    m = dict(
        schema=1,
        experiment=os.environ.get("EXPERIMENT_NAME"), project=os.environ.get("PROJECT_NAME", "simopd"),
        launched_at=time.strftime("%FT%TZ", time.gmtime()),
        host=socket.gethostname(), pod=os.environ.get("SIMOPD_POD") or os.environ.get("HOSTNAME"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ckpt_dir=os.path.abspath(a.ckpt_dir), fingerprint=a.fingerprint or None,
        stop_contract=_read(os.path.join(a.ckpt_dir, "simopd_stop_contract.txt")),
        resumed_from=_read(os.path.join(a.ckpt_dir, "latest_checkpointed_iteration.txt")),
        arm_args=a.arm_args.split() if a.arm_args else [],
        extra_overrides=a.extra.split() if a.extra else [],
        eval_protocol=dict(suite_k=os.environ.get("SIMOPD_SUITE_K", "32"),
                           note="SIMOPD_SUITE_K 默认 32;2026-09 重训按 AGENT.md §2.2b 用 8"),
        git=dict(simopd=_git(root), verl=_git(os.path.join(root, "verl"))),
        python=platform.python_version(), env=dict(sorted(env.items())),
    )
    try:
        import torch
        m["torch"] = torch.__version__
    except Exception:
        pass
    return m


def _read(p):
    try:
        return open(p).read().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--fingerprint", default="")
    ap.add_argument("--arm-args", default="")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    m = collect(a)
    os.makedirs(os.path.join(a.ckpt_dir, "manifest"), exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for path in (os.path.join(a.ckpt_dir, "manifest", f"launch_{ts}.json"),
                 os.path.join(a.ckpt_dir, "run_manifest.json")):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
    print(f"[simopd] run_manifest: {os.path.join(a.ckpt_dir, 'run_manifest.json')} "
          f"({len(m['env'])} env keys, git {((m['git']['simopd'] or {}).get('sha') or '?')[:10]})",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
