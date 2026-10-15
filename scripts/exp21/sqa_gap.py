"""exp21 step 0 (read-only): bench_lv_scienceqa gap to laya-vision by ScienceQA category, r15 vs laya-vision vs r15 mismatched.

    uv run python scripts/exp21/sqa_gap.py
"""
import json, glob, collections, sys
sys.path.insert(0, "scripts")
import pyarrow.parquet as pq
from lv_report import sqa_groups
root = "data"
snap = glob.glob(root + "/raw/hf/datasets--derek-thomas--ScienceQA/snapshots/*/data/")[0]
val = pq.read_table(glob.glob(snap + "validation-*.parquet")[0], columns=["subject", "topic", "category", "skill", "lecture", "hint"]).to_pylist()
meta = {"lv-scienceqa:val-%d" % i: r for i, r in enumerate(val)}
g = sqa_groups(root)
need = g["needs_picture&subject=natural science"]
def load(p):
    return {json.loads(l)["sample_id"]: json.loads(l) for l in open(p)}
ours = load("runs/r15-vision/eval/bench_lv_scienceqa.details.jsonl")
mism = load("runs/r15-vision/eval/bench_lv_scienceqa.mismatched.details.jsonl")
lv = load("data/lv_bench/laya-vision-201m.scienceqa.details.jsonl")
# per-category over all, and over needs-picture natsci
for name, ids in [("ALL", set(ours)), ("NEED_NAT", need)]:
    c = collections.defaultdict(lambda: [0, 0, 0, 0])
    for s in ids:
        k = meta[s]["category"]
        c[k][0] += 1; c[k][1] += ours[s]["correct"]; c[k][2] += lv[s]["correct"]; c[k][3] += mism[s]["correct"] if s in mism else 0
    print("==", name, len(ids))
    print("%-45s %4s %6s %6s %6s %6s" % ("category", "n", "ours", "lv", "mism", "deficit"))
    for k, (n, o, l, m) in sorted(c.items(), key=lambda kv: kv[1][1] - kv[1][2]):
        print("%-45s %4d %6.1f %6.1f %6.1f %6d" % (k[:45], n, 100*o/n, 100*l/n, 100*m/n, o - l))
