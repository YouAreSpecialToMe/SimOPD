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

分析归档(2026-09-02):每个 run 目录里除了 global_step_* 还躺着 metrics/(逐步标量)、
traj/(轨迹 id + 逐 token logprob + 摘要)、val_gen/(在环生成)、manifest/ 与
run_manifest.json(run 的定义)、指纹与契约 pin —— 全是小文件(一个 run 几十 MB),
每轮扫描时按 run 整体同步(HF 只传有变化的文件),记在 SYNCED.json 的 "<run>@aux"。
`traj/_verl_text/`(verl 自己那份文本,每步整批)不传。--no-aux 关掉。
"""
import argparse
import json
import os
import sys
import time

CK = os.environ.get("CKPT_ROOT", "/mgfs/shared/Group_GY/changhao/simopd_data/ckpt") + "/simopd"


def find_ckpts(root, with_opt=(), settle=120):
    """-> [(run, step, 本地目录, 是否整份)],按 (run, step) 排序;只收静置满 settle 秒的。"""
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
            if not os.path.isdir(src):
                continue
            done = (os.path.exists(os.path.join(src, "config.json")) if not full
                    else any(f.startswith("model_world_size_") for f in os.listdir(src)))
            # 写完再传:verl 落盘要几分钟,半截目录传上去比不传更糟(远端有了、内容缺)。
            # 取目录树里最新的 mtime,静置 SETTLE 秒才认它写完了。
            newest = max((os.path.getmtime(os.path.join(dp, f))
                          for dp, _, fs in os.walk(src) for f in fs), default=0)
            if done and (time.time() - newest) >= settle:
                out.append((run, step, src, full))
    return sorted(out, key=lambda t: (t[0], t[1]))


AUX_PATTERNS = ["metrics/*.jsonl", "traj/*.parquet", "traj/*.json", "traj/*.jsonl",
                "val_gen/*", "manifest/*.json", "run_manifest.json",
                "simopd_fingerprint.txt", "simopd_stop_contract.txt",
                "latest_checkpointed_iteration.txt"]
AUX_IGNORE = ["global_step_*/**", "traj/_verl_text/**", "*.tmp"]


def find_aux(root, settle=120):
    """-> [(run, run_dir, newest_mtime, n_files, bytes)] 每个 run 一条;只看归档小文件。"""
    import fnmatch
    out = []
    if not os.path.isdir(root):
        return out
    for run in sorted(os.listdir(root)):
        rd = os.path.join(root, run)
        if not os.path.isdir(rd):
            continue
        newest, n, size = 0, 0, 0
        for dp, dns, fs in os.walk(rd):
            rel_dir = os.path.relpath(dp, rd)
            # 不下到 ckpt 目录和 verl 文本目录里去
            dns[:] = [d for d in dns if not d.startswith("global_step_") and not
                      (rel_dir == "traj" and d == "_verl_text")]
            for f in fs:
                rel = f if rel_dir == "." else os.path.join(rel_dir, f)
                if not any(fnmatch.fnmatch(rel, pat) for pat in AUX_PATTERNS):
                    continue
                if any(fnmatch.fnmatch(rel, pat) for pat in AUX_IGNORE):
                    continue
                fp = os.path.join(dp, f)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                newest, n, size = max(newest, st.st_mtime), n + 1, size + st.st_size
        if n and (time.time() - newest) >= settle:
            out.append((run, rd, newest, n, size))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF repo id,例如 myorg/simopd-ckpts")
    ap.add_argument("--root", default=CK, help="ckpt 根目录(默认 $CKPT_ROOT/simopd)")
    ap.add_argument("--watch", type=int, default=0, help="常驻,每 N 秒扫一次(0=扫一次就退)")
    ap.add_argument("--with-optimizer", default="", help="逗号分隔的 run 名:这些臂连 FSDP 分片+optimizer 一起传(保留 resume 能力)")
    ap.add_argument("--settle", type=int, default=120, help="目录静置多少秒才认它写完(默认 120)")
    ap.add_argument("--dry", action="store_true", help="只打印将要传什么")
    ap.add_argument("--no-aux", action="store_true", help="不同步 run 级分析归档(metrics/traj/val_gen/manifest)")
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
        todo = [c for c in find_ckpts(a.root, with_opt, a.settle) if f"{c[0]}@{c[1]}" not in done]
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
                # 校验后才记账:传完不等于传对。比对文件名与字节数,不一致就不写
                # SYNCED.json —— 下一轮会重传,而不是留下一个"以为传好了"的坑。
                local = {f: os.path.getsize(os.path.join(dp, f))
                         for dp, _, fs in os.walk(src) for f in fs}
                remote = {}
                for it in api.list_repo_tree(a.repo, path_in_repo=dst.rstrip("/"),
                                             recursive=True, expand=True):
                    if getattr(it, "size", None) is not None:
                        remote[os.path.basename(it.path)] = it.size
                bad = [f for f, sz in local.items() if remote.get(f) != sz]
                if bad:
                    print(f"  VERIFY-FAIL {key}: {len(bad)} 个文件大小对不上(如 {bad[:2]}),不记账,下轮重传", flush=True)
                    continue
                done[key] = dict(path=dst, full=full, n=len(local),
                                 ts=time.strftime("%FT%TZ", time.gmtime()))
                tmp = "/tmp/SYNCED.json"
                json.dump(done, open(tmp, "w"), indent=1)
                api.upload_file(path_or_fileobj=tmp, path_in_repo="SYNCED.json", repo_id=a.repo,
                                commit_message=f"synced {len(done)}")
                print(f"  OK {key}", flush=True)
            except Exception as e:                       # 单个失败不拖垮整轮
                print(f"  FAIL {key}: {type(e).__name__} {str(e)[:120]}", flush=True)
        # run 级分析归档:有新文件/变化就整目录同步一次(HF 端按内容哈希跳过没变的)。
        # 不做字节校验:metrics 与 light.jsonl 在 run 结束前一直在追加,校验会和写入赛跑;
        # 每轮都会再看一眼,run 结束后的最后一轮才是定稿。
        if not a.no_aux:
            for run, rd, newest, n, size in find_aux(a.root, a.settle):
                key = f"{run}@aux"
                if done.get(key, {}).get("newest", 0) >= newest:
                    continue
                if a.dry:
                    print(f"  would sync aux {run}: {n} files, {size / 1e6:.1f} MB (newest {time.strftime('%m-%d %H:%M', time.localtime(newest))})")
                    continue
                try:
                    api.upload_folder(folder_path=rd, path_in_repo=f"{run}/", repo_id=a.repo,
                                      allow_patterns=AUX_PATTERNS, ignore_patterns=AUX_IGNORE,
                                      commit_message=f"aux {run} ({n} files)")
                    done[key] = dict(newest=newest, n=n, bytes=size,
                                     ts=time.strftime("%FT%TZ", time.gmtime()))
                    tmp = "/tmp/SYNCED.json"
                    json.dump(done, open(tmp, "w"), indent=1)
                    api.upload_file(path_or_fileobj=tmp, path_in_repo="SYNCED.json", repo_id=a.repo,
                                    commit_message=f"synced {len(done)}")
                    print(f"  OK aux {run} ({n} files, {size / 1e6:.1f} MB)", flush=True)
                except Exception as e:
                    print(f"  FAIL aux {run}: {type(e).__name__} {str(e)[:120]}", flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
