#!/usr/bin/env python3
"""把新落盘的 checkpoint 权重持续同步到 HuggingFace(私有 repo)。

    export HF_TOKEN=...                    # 只从环境读,不写进任何文件
    python scripts/ckpt_sync.py --repo <org>/<name> --watch 600

为什么要有这个:2026-08-24 的教训 —— 训练器把 ckpt 只写在共享盘上,仓库里没有任何上传
路径(verl 的 FSDP checkpoint manager 的 hdfs_path 明写 Unused,训练命令只有
default_local_dir,wandb 也不传 artifact)。集群一换,308 个还没评测的 ckpt 连同 1636 个
评测格一起失联,而它们值 136 GB —— 是当时全量 33 TB 的 0.4%。这个脚本让那种损失不再可能。

只传 `actor/huggingface/`(每个约 3.4 GB,bf16 权重 + config + tokenizer):它足够跑评测,
而 FSDP 分片与 optimizer(约 25 GB)只对"精确 resume"有用,不值得占带宽。要保留 resume
能力的臂,用 --with-optimizer 单独指定。

幂等:已传过的 (run, step) 记在 repo 里的 SYNCED.json,断点续传直接跳过;
--dry 只打印将要传什么。
"""
import argparse
import json
import os
import sys
import time

CK = os.environ.get("CKPT_ROOT", "/mgfs/shared/Group_GY/changhao/simopd_data/ckpt") + "/simopd"


def find_ckpts(root, with_opt=()):
    """-> [(run, step, 本地目录, 是否整份)] ,按 (run, step) 排好序。"""
    out = []
    if not os.path.isdir(root):
        return out
    for run in sorted(os.listdir(root)):
        rd = os.path.join(root, run)
        if not os.path.isdir(rd):
            continue
        for d in sorted(os.listdir(rd)):
            if not d.startswith("global_step_"):
                continue
            try:
                step = int(d.rsplit("_", 1)[1])
            except ValueError:
                continue
            full = run in with_opt
            src = os.path.join(rd, d) if full else os.path.join(rd, d, "actor", "huggingface")
            # 只认写完的:HF 目录要有 config.json;整份要有 FSDP 分片
            done = (os.path.exists(os.path.join(src, "config.json")) if not full
                    else any(f.startswith("model_world_size_") for f in os.listdir(src)))
            if os.path.isdir(src) and done:
                out.append((run, step, src, full))
    return sorted(out, key=lambda t: (t[0], t[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF repo id,例如 myorg/simopd-ckpts")
    ap.add_argument("--root", default=CK, help="ckpt 根目录(默认 $CKPT_ROOT/simopd)")
    ap.add_argument("--watch", type=int, default=0, help="常驻,每 N 秒扫一次(0=扫一次就退)")
    ap.add_argument("--with-optimizer", default="", help="逗号分隔的 run 名:这些臂连 FSDP 分片+optimizer 一起传(保留 resume 能力)")
    ap.add_argument("--dry", action="store_true", help="只打印将要传什么")
    a = ap.parse_args()
    with_opt = {x for x in a.with_optimizer.split(",") if x}

    if not a.dry:
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            sys.exit("需要 HF_TOKEN(或 HUGGING_FACE_HUB_TOKEN)在环境里;本脚本不读也不写任何凭据文件")
        if os.environ.get("HF_HUB_OFFLINE"):
            sys.exit("HF_HUB_OFFLINE 是开的,先 unset 它 —— 否则上传会静默失败")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, private=True, exist_ok=True)

    while True:
        done = {}
        if not a.dry:
            try:
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(a.repo, "SYNCED.json")
                done = json.load(open(p))
            except Exception:
                done = {}
        todo = [c for c in find_ckpts(a.root, with_opt) if f"{c[0]}@{c[1]}" not in done]
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] 待传 {len(todo)} 个(已同步 {len(done)})", flush=True)
        for run, step, src, full in todo:
            key = f"{run}@{step}"
            dst = f"{run}/global_step_{step}/" + ("" if full else "actor/huggingface/")
            if a.dry:
                print(f"  would upload {key}  {src} -> {dst}")
                continue
            try:
                api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=a.repo,
                                  commit_message=f"sync {key}")
                done[key] = dict(path=dst, full=full, ts=time.strftime("%FT%TZ", time.gmtime()))
                tmp = "/tmp/SYNCED.json"
                json.dump(done, open(tmp, "w"), indent=1)
                api.upload_file(path_or_fileobj=tmp, path_in_repo="SYNCED.json", repo_id=a.repo,
                                commit_message=f"synced {len(done)}")
                print(f"  OK {key}", flush=True)
            except Exception as e:                       # 单个失败不拖垮整轮
                print(f"  FAIL {key}: {type(e).__name__} {str(e)[:120]}", flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
