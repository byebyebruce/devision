"""List the project files another machine needs before build_pub.sh and pub-v12*.yaml can run there: the
held-out sets (and the pictures their photo hashes come from), the v12 training pool and the round-13 binding
sources with their pictures, the retrain panels / calibration set and every evaluation set of pub-v12-a100.yaml
with their pictures, the VG->COCO id map, and the cached hashes. Paths are relative to the data root.

    uv run python scripts/data/pub/export_list.py --root data > /tmp/devision-files.txt
    rsync -a --partial --info=progress2 --files-from=/tmp/devision-files.txt data/ user@host:/path/devision/data/

rsync is resumable (re-run the same command after an interruption; finished files are skipped).
--no-pixmo leaves out the ~30k PixMo-Points pictures already downloaded here (the other machine then fetches them).
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ext"))
sys.path.insert(0, HERE)

TRAIN = ["ext/train_mix.jsonl", "r13/gqa/train.jsonl", "r13/vismin/train.jsonl", "r13/synth-r13p75/train.jsonl"]
CONFIGS = ["configs/pub-v12-a100.yaml", "configs/pub-v12.yaml"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--no-pixmo", action="store_true")
    a = p.parse_args()
    from v9_mix import held_files_all  # pyright: ignore[reportMissingImports]
    from held import extra_files
    repo = os.path.join(HERE, "..", "..", "..")
    jsonls = set(TRAIN)
    for c in CONFIGS:
        jsonls |= set(re.findall(r"data/([\w/\-.]+\.jsonl)", open(os.path.join(repo, c)).read()))
    jsonls -= {"pub/train_mix.jsonl", "pub/dev_pub.jsonl"}
    jsonls |= {os.path.relpath(f, a.root) for f in held_files_all(a.root) + extra_files(a.root)}
    out = set()
    for rel in sorted(jsonls):
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            print("missing here: %s" % rel, file=sys.stderr)
            continue
        out.add(rel)
        for line in open(path):
            if line.strip():
                img = json.loads(line).get("image")
                if img and os.path.exists(os.path.join(a.root, img)):
                    out.add(img)
    extras = ["v5/image_identity.json", "r13/identity.json", "r13/identity_hashes.json"]
    extras += [os.path.relpath(f, a.root) for f in glob.glob(os.path.join(a.root, "raw", "vg", "*"))]
    extras += [os.path.relpath(f, a.root) for f in glob.glob(os.path.join(a.root, "raw", "ext", "*.json"))]
    if not a.no_pixmo:
        extras += [os.path.relpath(f, a.root) for f in glob.glob(os.path.join(a.root, "ext", "pixmo_points", "images", "*"))]
        extras += [os.path.relpath(f, a.root) for f in glob.glob(os.path.join(a.root, "raw", "ext", "pixmo_points", "*.json"))]
    out |= {e for e in extras if os.path.exists(os.path.join(a.root, e))}
    total = sum(os.path.getsize(os.path.join(a.root, f)) for f in out)
    print("%d files, %.1f GB" % (len(out), total / 1e9), file=sys.stderr)
    for f in sorted(out):
        print(f)


if __name__ == "__main__":
    main()
