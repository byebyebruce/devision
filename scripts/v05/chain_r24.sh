#!/usr/bin/env bash
# r24 (docs/experiments/2026-10-29-r24-v05-plan.md): after the exp138 chain (given pids) ends, judge exp138's families
# by criterion 1 (scripts/v05/judge_board.py), build data/v05/r24 = r23's file + passing families of exp137 / exp138 +
# A-OKVQA replay (all board-filtered), train + evaluate r24-v05, then lv_report, paired comparisons, acceptance
# against v0.4 and the Decision Index public vision index.
#   override the families with R24_FAMILIES=caption,kie,... (default: exp137's passing caption,kie + exp138's)
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
[ -s runs/ext-decision-index/exp138-board-mix2/vision-scores.json ] || { echo "$(date '+%F %T') exp138 board scores missing, stop"; exit 1; }
uv run python scripts/v05/judge_board.py exp138-board-mix2 figure,depth,som,drive,hateful > runs/v05-chain/judge_exp138.json
E138=$(uv run python scripts/v05/judge_board.py exp138-board-mix2 figure,depth,som,drive,hateful --passed)
FAM=${R24_FAMILIES:-caption,kie${E138:+,$E138}}
echo "$(date '+%F %T') exp138 passing: '${E138}'; r24 families: $FAM"
uv run python scripts/data/v05/r24_mix.py --families "$FAM" > runs/r24-mix.out 2>&1 || { echo "$(date '+%F %T') r24 mix failed, stop"; exit 1; }
R=runs/r24-v05
uv run devision-pipeline configs/r24-v05.yaml > $R.out 2>&1
echo "$(date '+%F %T') r24 exit $?"
uv run python scripts/lv_report.py $R/eval --prev runs/r23-v05/eval >> $R.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
mkdir -p $R/eval/compare
for f in $R/eval/*.details.jsonl; do n=$(basename $f .details.jsonl); case $n in *.mismatched|*.reversed) continue;; esac
  for ref in r21-soft r23-v05 exp137-board-mix exp138-board-mix2; do [ -f runs/$ref/eval/$n.details.jsonl ] || continue
    uv run devision-compare runs/$ref/eval/$n.details.jsonl $f --by source > $R/eval/compare/$n.vs-$ref.txt 2>/dev/null
  done
done
uv run python scripts/v05/accept_r23.py --cand $R/eval --out runs/r24-v05/accept_vs_v0.4
echo "$(date '+%F %T') accept exit $?"
bash scripts/v05/board_run.sh $R/train r24-v05
echo "$(date '+%F %T') r24 chain done"
