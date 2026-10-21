"""exp21 real-diagram monitor sets (docs/experiments/2026-10-15-exp21-reading-plan.md), from data/v10's dev_sqa
(ScienceQA official test split; monitoring only):

  - dev_sqa_{temp,force}.jsonl: the questions as they are;
  - dev_sqa_{temp,force}_mism.jsonl: the same questions shown another picture of the same set, a fixed
    derangement (devision.train.evaluate.picture_pairs, seed 0); image_id stays the question's own picture, so
    true and mismatched records resample together (critic EXP21-R1-03);
  - sqa_text.jsonl: 24 of them (scripts/exp21/sqa_transcribed.json) on a blank picture, the transcribed values
    appended to the hint; sqa_blank.jsonl: the same 24 on the blank picture without the values.

    uv run python scripts/exp21/real_controls.py --root data
"""
import argparse
import json
import os

from PIL import Image

from devision.train.evaluate import picture_pairs

HERE = os.path.dirname(__file__)


def write(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def build(root: str) -> dict:
    out = os.path.join(root, "exp21")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    blank = "exp21/images/blank.png"
    Image.new("RGB", (256, 256), "white").save(os.path.join(root, blank))
    notes = json.load(open(os.path.join(HERE, "sqa_transcribed.json")))
    counts, text, empty = {}, [], []
    for kind in ("temp", "force"):
        rows = [json.loads(x) for x in open(os.path.join(root, "v10", "dev_sqa_%s.jsonl" % kind))]
        pairs = picture_pairs(rows, seed=0)
        mism = [dict(r, image=pairs[r["image_id"]], source=r["source"] + "-mism") for r in rows]
        assert all(m["image"] != r["image"] for m, r in zip(mism, rows))
        write(os.path.join(out, "dev_sqa_%s.jsonl" % kind), rows)
        write(os.path.join(out, "dev_sqa_%s_mism.jsonl" % kind), mism)
        counts[kind] = len(rows)
        for r in rows:
            if r["id"] in notes:
                hint = r.get("state_text", "")
                text.append(dict(r, image=blank, image_id="exp21:blank:" + r["id"], source="exp21-sqa-text",
                                 state_text=(hint + "\n" if hint else "") + notes[r["id"]]))
                empty.append(dict(r, image=blank, image_id="exp21:blank:" + r["id"], source="exp21-sqa-blank"))
    missing = set(notes) - {"_note"} - {r["id"] for r in text}
    if missing:
        raise SystemExit("transcribed questions not in dev_sqa: %s" % sorted(missing))
    write(os.path.join(out, "sqa_text.jsonl"), text)
    write(os.path.join(out, "sqa_blank.jsonl"), empty)
    return {"dev_sqa": counts, "sqa_text": len(text)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root)))
