"""Public-data pack, OneJev-Data (OmniJev/OneJev-Data): a released Jev-format training set; only its single,
static-picture, English choice / yes-no questions fit our interface.

    uv run python scripts/data/pub/onejev.py --root data

Downloads the 18 parquet shards (~17.7 GB, resumable, Hugging Face cache) and keeps a row when: exactly one medium,
a picture under media/image/ (GUI screenshots, video frames and multi-picture rows dropped), question type choice
(2-11 options) or noul, the state is only the picture (rows that need a prompt or other context dropped). Options
are shown by their description when it is short, else by their id; the target is renormalised over the options.
Each upstream source keeps its own licence (licenses.csv, copied next to the output). Photos within NEAR_BITS of a
held-out photo are dropped (OneJev repackages GQA, VQAv2, A-OKVQA, ScienceQA ... pictures).
"""
import json
import random
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ext"))
sys.path.insert(0, HERE)
from common import already_built, base_args, finish, hf_file, hf_files, save_bytes  # noqa: E402
from held import PubHeld, require_history, split_dev, balance_noul  # noqa: E402

SOURCE = "onejev"
REPO = "OmniJev/OneJev-Data"


def option_text(key: str, desc) -> str:
    d = " ".join(str(desc or "").split())
    return d if d and len(d) <= 60 else key.replace("_", " ")


def convert(row: dict):
    """A OneJev row -> (sample without image path, picture bytes) or (None, reason)."""
    media = json.loads(row["media"] or "[]")
    if len(media) != 1 or media[0].get("type") != "image" or not str(media[0].get("path", "")).startswith("media/image/"):
        return None, "not one static picture"
    state = json.loads(row["state"] or "{}")
    if set(state) - {"image"}:
        return None, "needs context beyond the picture"
    q = json.loads(row["question"])
    target = json.loads(row["target"] or "{}")
    imgs = row.get("images") or []
    if not imgs or not imgs[0] or not imgs[0].get("bytes"):
        return None, "no picture bytes"
    base = {"id": "pub-onejev:" + row["id"], "source": "onejev:" + row["source"], "kind": row["task"],
            "group": json.loads(row["meta"] or "{}").get("group") or row["id"]}
    if q["type"] == "noul":
        t = {("true" if k in ("yes", "true") else "false" if k in ("no", "false") else k): float(v) for k, v in target.items()}
        if set(t) - {"true", "false"} or sum(t.values()) <= 0:
            return None, "noul target"
        s = sum(t.values())
        p = {"false": t.get("false", 0.0) / s, "true": t.get("true", 0.0) / s}
        return dict(base, questions={"q": {"type": "noul", "instructions": q["instructions"]}},
                    gold={"q": {"probabilities": p}}, answer_key=max(p, key=p.get)), imgs[0]["bytes"]
    if q["type"] != "choice":
        return None, "type %s" % q["type"]
    crit = q.get("criteria") or {}
    if isinstance(crit, list) or not 2 <= len(crit) <= 11:
        return None, "options"
    texts = {k: option_text(k, v) for k, v in crit.items()}
    if len(set(t.lower() for t in texts.values())) != len(texts) or set(target) - set(crit):
        return None, "options not distinct / target outside"
    s = sum(float(v) for v in target.values())
    if s <= 0:
        return None, "empty target"
    probs = {texts[k]: float(target.get(k, 0.0)) / s for k in crit}
    return dict(base, questions={"q": {"type": "choice", "instructions": q["instructions"],
                                       "criteria": {t: None for t in probs}}},
                gold={"q": {"probabilities": probs}}, answer_key=max(probs, key=probs.get)), imgs[0]["bytes"]


def main(argv=None):
    p = base_args(__doc__)
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    require_history(a.root)
    import pyarrow.parquet as pq
    rows, steps = [], Counter()
    for f in hf_files(a.root, REPO, "data/train-*.parquet"):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                steps["rows"] += 1
                r, data = convert(row)
                if r is None:
                    steps[data] += 1
                    continue
                path = json.loads(row["media"])[0]["path"]
                rel = os.path.join("ext", SOURCE, "images", path.split("media/image/", 1)[1])
                save_bytes(os.path.join(a.root, rel), data)
                r["image"], r["image_id"] = rel, "onejev:" + os.path.basename(rel).rsplit(".", 1)[0]
                rows.append(r)
    near = PubHeld(a.root).near_held([r["image"] for r in rows], SOURCE)
    steps["near a held-out photo"] = sum(r["image"] in near for r in rows)
    rows = [r for r in rows if r["image"] not in near]
    rows = balance_noul(rows, random.Random(a.seed))
    rows, steps["dev_questions"] = split_dev(a.root, SOURCE, rows, key=lambda r: r["group"])
    import shutil
    shutil.copy(hf_file(a.root, REPO, "licenses.csv"), os.path.join(a.root, "ext", SOURCE, "licenses.csv"))
    finish(a.root, SOURCE, rows, dict(steps), {"licence": "per upstream source: ext/onejev/licenses.csv",
                                                "sources": dict(Counter(r["source"] for r in rows).most_common())})


if __name__ == "__main__":
    main()
