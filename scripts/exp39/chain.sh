#!/usr/bin/env bash
# exp39 on one Mac (docs/experiments/2026-10-17-exp39-lowmem-plan.md): arm H again with micro_batch 8 (configs/
# exp39-h.yaml) from exp37-L read at 512 (runs/exp38-init-512, conversion checked), under the resource watch
# (scripts/exp39/watch.py); then H's full-number readouts (600 original tune, 200 original fit; forward in 8-row
# parts, the 16-row draws unchanged), the H - C report against the C rescue (runs/exp38-c, status merged), and by
# priority CPU latency, ScienceQA / dev16, the 7 protection panels. Deadline: start + EXP39_MINUTES (required; set
# with the critic before the run) and never later than EXP39_LATEST ("YYYY-MM-DD HH:MM:SS"); steps end 60 s before it
# (scripts/exp38/steps.sh); no retry; no decision stage follows.
#   EXP39_MINUTES=... EXP39_LATEST="..." nohup bash scripts/exp39/chain.sh > runs/exp39.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
: "${EXP39_MINUTES:?set EXP39_MINUTES}" "${EXP39_LATEST:?set EXP39_LATEST}"
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp39/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
uv run python - <<'PY' || { log "runs/exp38-init-512 is not the expected conversion"; exit 1; }
import json, sys
c = json.load(open("runs/exp38-init-512/CONVERTED.json")); g = json.load(open("runs/exp38-init-512/config.json"))
sys.exit(0 if (c.get("source_model_sha256") == "f9fc93c25b4bd96c55d41fd7c974c43408ea31034eac2c3a1ade2bcf5950da31"
               and g.get("image_size") == 512 and g.get("interpolate_pos_encoding") is True
               and all(c["files_identical"].values())) else 1)
PY
[ ! -e runs/exp39-h ] || { log "runs/exp39-h already exists"; exit 1; }
DEADLINE_FILE=runs/exp39.deadline
if [ ! -f "$DEADLINE_FILE" ]; then
  cap=$(date -j -f "%Y-%m-%d %H:%M:%S" "$EXP39_LATEST" +%s)
  end=$(( $(date +%s) + EXP39_MINUTES * 60 )); [ "$end" -lt "$cap" ] || end=$cap
  echo "$end" > "$DEADLINE_FILE"
fi
DEADLINE=$(cat "$DEADLINE_FILE")
STATUS=runs/exp39-status.txt
. scripts/exp38/steps.sh
log "exp39 H: init runs/exp38-init-512 (exp37-L f9fc93c2...); data $(shasum -a 256 data/exp36/align_mix.jsonl | cut -c1-16); deadline $(date -r "$DEADLINE" '+%F %T')"
report() {   # H - C with the C rescue; its status merged with this run's
  cat runs/exp38c-rescue-status.txt "$STATUS" > runs/exp39-status-merged.txt
  uv run python - <<'PY'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("r38", "scripts/exp38/report.py"); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
out = r.build({"H": "runs/exp39-h", "C": "runs/exp38-c"}, "runs/exp39-status-merged.txt")
c_sha = json.load(open("runs/exp38-c/numbers_tune/START.json"))["sha256"]["model.safetensors"]
if c_sha != "fee58d58abf7a5930b7df7f2a2818dbe13f93e0a4855b621dfe7ba73e4faa0a0":
    sys.exit("C is not the reviewed C rescue: %s" % c_sha)
run = json.load(open("runs/exp39-h/run.json"))
out["sources"] = {"C": "reused from the C rescue (scripts/exp38/rescue_c.sh, runs/exp38-c, model fee58d58...), not rerun",
                  "H": {"micro_batch": run["config"]["stages"][0]["params"].get("micro_batch"),
                        "readout_forward_rows": json.load(open("runs/exp39-h/numbers_tune/START.json")).get("forward_rows"),
                        "init": run["config"]["stages"][0]["init"], "data": run["config"]["stages"][0]["data"]}}
json.dump(out, open("runs/exp39-report.json", "w"), indent=1)
print(json.dumps({k: out.get(k) for k in ("outcome", "conditions", "H_minus_C", "identity", "sources")}, indent=1, default=str))
PY
}
finish() {
  final_report report > runs/exp39-report.txt 2>&1
  local rc=$?
  cat runs/exp39-report.txt
  log "final report (exit $rc)"
  [ "$rc" = 0 ] && rc=${1:-0}
  exit "$rc"
}
uv run python scripts/exp39/watch.py runs/exp39-h/stage1 "align_main.*exp39-h|configs/exp39-h.yaml" runs/exp39-watch.txt &
step train:h uv run devision-pipeline configs/exp39-h.yaml || finish 1
for split in tune:data/real_layout/captions_tune.jsonl fit:data/exp36/fit_slice.jsonl; do
  name=${split%%:*}; data=${split#*:}
  step "readout:h:$name" uv run python scripts/exp36/readout.py --checkpoint runs/exp39-h/stage1 --data "$data" \
    --out runs/exp39-h/numbers_$name --full-numbers-only --micro 8 || finish 1
done
step report:main report || finish 1
stop=0
extra() {
  local name=$1; shift
  if [ "$stop" = 1 ]; then status "$name" missing; return; fi
  step "$name" "$@"
  local rc=$?
  { [ "$rc" = 124 ] || [ "$rc" = 125 ]; } && stop=1
  return 0
}
mkdir -p runs/exp39-h/eval runs/exp39-h/cpu
extra cpu:h env OMP_NUM_THREADS=4 uv run devision-eval --checkpoint runs/exp39-h/stage1 --data data/r12/cpu_latency.jsonl \
  --data-root data --out runs/exp39-h/cpu/single.json --device cpu --no-group
for s in sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl \
         sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl dev16=data/exp22/dev16.jsonl \
         exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl \
         count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl flip=data/v6/dev_lr_pairs.jsonl \
         vsr=data/v5/dev_vsr.jsonl; do
  set=${s%%=*}; data=${s#*=}
  extra "eval:h:$set" uv run devision-eval --checkpoint runs/exp39-h/stage1 --data "$data" --data-root data \
    --out runs/exp39-h/eval/$set.json --role monitoring
done
finish 0
