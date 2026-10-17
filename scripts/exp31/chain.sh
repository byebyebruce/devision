#!/usr/bin/env bash
# exp31 (critic EXP28-RESULT-02 item 3): v0.3 and exp26B on the text force diagnosis, then scripts/exp31/report.py.
set -u
cd "$(dirname "$0")/../.."
run=runs/exp31-text-force
for m in v03:runs/r15-vision/stage2 exp26B:runs/exp27-transfer/exp26-step5000; do
  name=${m%%:*}; mkdir -p "$run/eval/$name"
  out=$run/eval/$name/text_force.json
  [ -f "${out%.json}.details.jsonl" ] && continue
  uv run devision-eval --checkpoint "${m#*:}" --data data/exp31/text_force.jsonl --data-root data --out "$out" \
    --role monitoring > "${out%.json}.log" 2>&1 || { echo "$name failed"; exit 1; }
done
uv run python scripts/exp31/report.py > "$run/report.txt" 2>&1; rc=$?
cat "$run/report.txt"; exit $rc
