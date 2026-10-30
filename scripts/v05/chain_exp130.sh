#!/usr/bin/env bash
# exp130 (docs/experiments/2026-10-28-exp130-v4-interp.md): after the exp129 chain exits, interpolate v0.4 -> exp129,
# refit temperatures, pick alpha on dev sets only (scripts/v05/exp130_select.py), then evaluate the chosen alpha on the
# 32 exp129 sets + lv_report. No alpha chosen -> stop after the dev step.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
R=runs/exp130-interp; B=runs/exp129-v4-scale-size-seed23/train; A=runs/r21-soft/train
[ -f $B/model.safetensors ] || { echo "no exp129 checkpoint"; exit 1; }
mkdir -p $R
ev(){ mkdir -p $R/$1/devsel; for s in v2/dev_coco_exist v2/dev_vqa_yesno v05/exp118/gui_dev_state v05/exp119/gui_shop_dev_state v05/exp116/gui_dev v05/exp116/ego_dev; do
  [ -f $R/$1/devsel/$(basename $s).json ] || uv run devision-eval --checkpoint $2 --data data/$s.jsonl --data-root data --device cuda --out $R/$1/devsel/$(basename $s).json --role monitoring > /dev/null || exit 1; done; }
ev v04 $A; ev a1.0 $B
for a in 0.9 0.75 0.6; do
  d=$R/a$a
  if [ ! -f $d/model.safetensors ]; then
    rm -rf $d.raw; uv run python scripts/exp17/interp.py $A $B $a $d.raw || exit 1
    uv run devision-calibrate --checkpoint $d.raw --data data/v05/exp81/calib.jsonl --data-root data --out $d || exit 1
    rm -rf $d.raw
  fi
  ev a$a $d
done
echo "$(date '+%F %T') dev done"
uv run python scripts/v05/exp130_select.py $R | tee $R/select.out
a=$(sed -n 's/^CHOSEN //p' $R/select.out)
[ "$a" = none ] || [ -z "$a" ] && { echo "$(date '+%F %T') no alpha eligible, stop"; exit 0; }
uv run python - "$a" <<'PY'
import sys, yaml
a = sys.argv[1]
y = yaml.safe_load(open("configs/exp129-v4-scale-size-seed23.yaml"))
ev = dict(y["evaluate"])
ev["checkpoint"] = "runs/exp130-interp/a%s" % a
ev["calibration_fit"] = "data/v05/exp81/calib.jsonl"
ev["history"] = list(y["stages"][0]["eval"].values())
yaml.safe_dump({"name": "exp130-interp", "device": "cuda", "stages": [], "evaluate": ev},
               open("configs/exp130-interp.yaml", "w"), sort_keys=False)
PY
uv run devision-pipeline configs/exp130-interp.yaml > runs/exp130-interp.out 2>&1
echo "$(date '+%F %T') exp130 eval exit $?"
uv run python scripts/lv_report.py $R/a$a --prev runs/r21-soft/eval >> runs/exp130-interp.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
