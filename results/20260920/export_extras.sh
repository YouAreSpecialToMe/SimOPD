#!/usr/bin/env bash
# export_extras.sh <tag>   (run on w0 AFTER tools/export_results.sh <tag>; called by it for "final" exports). Nothing leaves the cluster.
# Adds to /home/tiger/opd/results_export/<tag>/ what results/20260909/ did not have (haotian 2026-09-20: "organize like 20260909 along with
# additional logs or sampled traj"):
#   logs/<node>/...gz          experiment logs of the four nodes, gzipped file by file: training lanes, eval workers, launch / hand-over /
#                              gate logs, GPU + disk guards, W&B uploader, cell hand-over + split logs, eval-queue refill log and failure
#                              records. NOT included: download / setup logs (dl_*, models, setup, wheels, assets), hidden caches, the W&B
#                              live-streamer debug directory (53 MB) and derived CSVs. Every file is scanned for credential-like strings
#                              first and such values are replaced by [REDACTED] (counts in the README section; none expected).
#   traj_samples/<run>/stage_XXXX.{parquet,json}   the 100 training trajectories sampled per 25-step stage per arm (= the W&B traj tables)
#   non_roster_dirs.tsv        directories in the checkpoint trees that are not roster runs (0909 has this table, written by hand there)
#   extras_manifest.tsv        path, bytes, md5 of every file added here
#   README.md                  section "Additional material" between the EXTRAS markers (replaced on every run)
set -uo pipefail
TAG=${1:?tag}; O=/home/tiger/opd; E=$O/results_export; OUT=$E/$TAG; RAW=$E/.logs_raw_$TAG; PY=$O/SimOPD/simopd/bin/python
[ -f $OUT/README.md ] || { echo "run tools/export_results.sh $TAG first"; exit 1; }
rm -rf $RAW $OUT/logs $OUT/traj_samples; mkdir -p $RAW $OUT/logs $OUT/traj_samples
FILTER=(--exclude='.*' --exclude='dl_*' --exclude='models.log' --exclude='setup.log' --exclude='wheels.log' --exclude='assets.log'
        --include='lanes/' --include='evalw/' --exclude='*/' --include='*.log' --include='*.tsv' --include='*.txt' --include='*.err' --exclude='*')
echo "[$(date +%T)] extras 1/4: logs"
for n in w0 w1 w2 w3; do
  mkdir -p $RAW/$n/evalq
  if [ $n = w0 ]; then src=$O; else src=$n:$O; fi
  rsync -a "${FILTER[@]}" $src/logs/ $RAW/$n/
  for q in evalq_exp evalq_foreign; do
    rsync -a $src/simopd_data/$q/refill.log $RAW/$n/evalq/${q}_refill.log 2>/dev/null
    rsync -a $src/simopd_data/$q/failed/ $RAW/$n/evalq/${q}_failed/ 2>/dev/null
  done
done
PYTHONPATH= $PY - $RAW $OUT/logs <<'PY'
import gzip, os, re, sys
raw, out = sys.argv[1], sys.argv[2]
PAT = [("key=value", re.compile(r"(?i)\b(api[_-]?key|\w*token|secret|passw(?:or)?d|authorization|bearer)\b(\s*[=:]\s*[\"']?)([A-Za-z0-9_\-\.]{16,})")),
       ("hf token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
       ("signed url", re.compile(r"(?i)(x-amz-(?:signature|credential|security-token)=)[^&\s\"']+")),
       ("url credentials", re.compile(r"(://)[^/\s:@]+:[^@\s/]+@")),
       ("40-hex near wandb", re.compile(r"(?i)(wandb[^\n]{0,200}?)\b[0-9a-f]{40}\b"))]
counts = {k: 0 for k, _ in PAT}; files = 0; nbytes = 0; hitfiles = set()
for d, _, fs in os.walk(raw):
    for f in sorted(fs):
        p = os.path.join(d, f); rel = os.path.relpath(p, raw)
        s = open(p, "rb").read().decode("utf-8", errors="replace")
        for name, rx in PAT:
            if name == "key=value":      # a purely numeric value is a metric (e.g. "chi2_token:0.0123456789012345"), not a credential
                _n = [0]
                def _kv(m, _n=_n):
                    if re.fullmatch(r"[0-9.eE+\-]+", m.group(3)): return m.group(0)
                    _n[0] += 1; return m.group(1) + m.group(2) + "[REDACTED]"
                s = rx.sub(_kv, s); k = _n[0]
            elif name in ("signed url", "url credentials"): s, k = rx.subn(lambda m: m.group(1) + ("[REDACTED]@" if name == "url credentials" else "[REDACTED]"), s)
            elif name == "40-hex near wandb": s, k = rx.subn(lambda m: m.group(1) + "[REDACTED]", s)
            else: s, k = rx.subn("[REDACTED]", s)
            if k: counts[name] += k; hitfiles.add(rel)
        dst = os.path.join(out, rel + ".gz"); os.makedirs(os.path.dirname(dst), exist_ok=True)
        with gzip.GzipFile(dst, "wb", compresslevel=9, mtime=0) as g: g.write(s.encode("utf-8"))
        files += 1; nbytes += os.path.getsize(dst)
open(os.path.join(out, ".redaction"), "w").write("files=%d gz_bytes=%d redactions=%s hit_files=%s\n" % (files, nbytes, "; ".join("%s: %d" % kv for kv in counts.items()), ", ".join(sorted(hitfiles)) or "-"))
print("logs: %d files -> %.1f MB gz; redactions: %s" % (files, nbytes / 1e6, counts))
PY
rm -rf $RAW
echo "[$(date +%T)] extras 2/4: sampled trajectories"
rsync -a --include='*/' --include='stage_*.parquet' --include='stage_*.json' --exclude='*' $O/wandb_inbox/traj_samples/ $OUT/traj_samples/
echo "[$(date +%T)] extras 3/4: non-roster directories"
ROSTER=$(grep -v '^#' $O/SimOPD/configs/retrain_roster.tsv | cut -f1 | sed 's/$/_s0/' | tr '\n' ' ')
printf 'dir\tn_checkpoints\tglobal_steps\tsize\tnote\n' > $OUT/non_roster_dirs.tsv
for n in w0 w1 w2 w3; do
  cmd="cd $O/simopd_data/ckpt/simopd 2>/dev/null && for d in *; do [ -d \"\$d\" ] || continue; case \" $ROSTER \" in *\" \$d \"*) continue;; esac; st=\$(ls -d \$d/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tr '\n' ',' | sed 's/,\$//'); nc=\$(ls -d \$d/global_step_* 2>/dev/null | wc -l); printf '%s\t%s\t%s\t%s\n' \"\$d\" \"\$nc\" \"\$st\" \"\$(du -sh \$d | cut -f1)\"; done"
  if [ $n = w0 ]; then r=$(bash -c "$cmd"); else r=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 $n "$cmd"); fi
  [ -n "$r" ] && echo "$r" | while IFS=$'\t' read -r d nc st sz; do
    case "$d" in rehearsal_*) note="on $n; product of the A-axis rehearsal gate (3-step machine-judged run), not in the roster";; *) note="on $n; not in the roster";; esac
    printf '%s\t%s\t%s\t%s\t%s\n' "$d" "$nc" "$st" "$sz" "$note" >> $OUT/non_roster_dirs.tsv
  done
done
echo "[$(date +%T)] extras 4/4: manifest + README section"
( cd $OUT && printf 'path\tbytes\tmd5\n' > extras_manifest.tsv && for f in non_roster_dirs.tsv $(find logs traj_samples -type f ! -name '.redaction' | sort); do printf '%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$(md5sum "$f" | cut -c1-32)"; done >> extras_manifest.tsv )
PYTHONPATH= $PY - $OUT <<'PY'
import glob, json, os, re, sys
out = sys.argv[1]; red = open(f"{out}/logs/.redaction").read().strip(); os.remove(f"{out}/logs/.redaction")
per = {n: (len(glob.glob(f"{out}/logs/{n}/**/*.gz", recursive=True)), sum(os.path.getsize(p) for p in glob.glob(f"{out}/logs/{n}/**/*.gz", recursive=True))) for n in ("w0", "w1", "w2", "w3")}
st = sorted(glob.glob(f"{out}/traj_samples/*/stage_*.parquet")); tb = sum(os.path.getsize(p) for p in st); arms = sorted({p.split("/")[-2] for p in st})
meta = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/traj_samples/*/stage_*.json"))]
nr = max(0, sum(1 for _ in open(f"{out}/non_roster_dirs.tsv")) - 1)
S = ["<!-- EXTRAS:START -->", "---\n", "## Additional material (not part of the 0909 layout)\n",
     "| path | content |", "|---|---|",
     "| `logs/<node>/lanes/*.log.gz` | training logs of the lanes that ran on the node (console output of `slurm/retrain_lane_node.sbatch` / `deploy/dsw/_lane.sh`: every step line, validations, the `<arm> -> OK` result line) |",
     "| `logs/<node>/evalw/*.log.gz` | eval worker logs, one per GPU (`evalw_<node>_gpu<k>`: the node's own queue; `evalw_<node>_foreign_gpu<k>`: cells handed over from another node) |",
     "| `logs/<node>/*.gz` | launch / hand-over / rehearsal-gate logs, GPU guard and disk guard, W&B uploader (`wandb_sync`, `wandb_fast`, `wandb_live`), trajectory sampler, cell hand-over (`offload.log`, `offload_state.tsv`, `split.log`) |",
     "| `logs/<node>/evalq/*` | eval-queue refill log and the failure records of the queues (`*_failed/<run>__<step>`: one line per failed attempt; 3 lines = the cell is skipped -- none reached 3) |",
     "| `traj_samples/<run>/stage_<step>.parquet` | 100 training trajectories per 25-step stage, sampled uniformly without replacement from the step's batch (the tables shown on W&B under `traj/<run>`): prompt, response, token counts, score, advantage, truncation / stop flags, repetition, student log-prob, entropies, per-sequence FKL / RKL / JSD / TV / top-1 agreement with the teacher |",
     "| `traj_samples/<run>/stage_<step>.json` | sampling record of that table: population size, seed, source file, sample vs population mean score |",
     "| `non_roster_dirs.tsv` | directories in the checkpoint trees that are not roster runs (same columns as in 0909) |",
     "| `extras_manifest.tsv` | path, bytes and md5 of every file listed in this section |", "",
     "Logs: " + "; ".join(f"{n} {c} files / {b/1e6:.1f} MB" for n, (c, b) in per.items()) + " (gzip, one file each; read with `zless` / `gzcat`, search with `zgrep`). "
     "Left on the servers: download and environment-setup logs, hidden cache files, the W&B live-streamer debug directory, per-item generations (`val_gen/`, `evals/*.parquet`, `traj/`) and checkpoints. "
     f"Credential scan before packing ({red}).\n",
     f"Sampled trajectories: {len(st)} stage tables over {len(arms)} runs, {sum(m.get('n', 0) for m in meta)} trajectories, {tb/1e6:.0f} MB. "
     f"Non-roster directories: {nr}.\n", "<!-- EXTRAS:END -->"]
p = f"{out}/README.md"; s = open(p).read(); s = re.sub(r"\n?<!-- EXTRAS:START -->.*?<!-- EXTRAS:END -->\n?", "\n", s, flags=re.S)
anchor = "---\n\n## Reproduce"
blk = "\n".join(S) + "\n\n"
s = s.replace(anchor, blk + anchor, 1) if anchor in s else s.rstrip("\n") + "\n\n" + blk
open(p, "w").write(s); print("README.md: extras section written")
PY
cp $O/tools/export_extras.sh $OUT/ 2>/dev/null
echo "[$(date +%T)] extras done: logs $(du -sh $OUT/logs | cut -f1), traj_samples $(du -sh $OUT/traj_samples | cut -f1), total $(du -sh $OUT | cut -f1)"
