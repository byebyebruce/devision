"""exp37 data lists (critic EXP37-PREP-01 item 1): the exp36 rows with only the real-layout picture swapped for its
S or L rendering (scripts/exp37/render.py) and a "variant" field; captions, order, families, COCO rows and every
other field unchanged. Training: data/exp37/align_mix_{S,L}.jsonl (16,000 rows, exp36's order); readout:
data/exp37/tune_{S,L}.jsonl (the 600 tune pictures) and fit_{S,L}.jsonl (exp36's 200-picture fit slice).

    uv run python scripts/exp37/build.py --root data
"""
import argparse
import hashlib
import json
import os


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def swap(row: dict, variant: str, root: str) -> dict:
    if not row.get("source", "").startswith("real-layout"):
        return row
    fam, view = row["id"].split(":")[1], row["id"].split(":")[2]
    rel = "exp37/images/%s/%s_%s.png" % (variant, fam, view)
    if not os.path.exists(os.path.join(root, rel)):
        raise SystemExit("missing %s" % rel)
    return dict(row, image=rel, variant=variant)


def build(root: str) -> dict:
    sources = {"align_mix": os.path.join(root, "exp36", "align_mix.jsonl"),
               "tune": os.path.join(root, "real_layout", "captions_tune.jsonl"),
               "fit": os.path.join(root, "exp36", "fit_slice.jsonl")}
    out = os.path.join(root, "exp37")
    m: dict = {"sources": {k: sha(v) for k, v in sources.items()}, "files": {}}
    for name, path in sources.items():
        rows = [json.loads(x) for x in open(path)]
        for v in ("S", "L"):
            dst = os.path.join(out, "%s_%s.jsonl" % (name, v))
            with open(dst, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(swap(r, v, root)) + "\n")
            m["files"]["%s_%s" % (name, v)] = {"rows": len(rows), "sha256": sha(dst)}
        a = [json.loads(x) for x in open(os.path.join(out, "%s_S.jsonl" % name))]
        b = [json.loads(x) for x in open(os.path.join(out, "%s_L.jsonl" % name))]
        strip = lambda r: {k: v for k, v in r.items() if k not in ("image", "variant")}  # noqa: E731
        assert [strip(x) for x in a] == [strip(x) for x in b] == [strip(x) for x in rows]
        m["files"]["%s_S" % name]["identical_apart_from_image_variant"] = True
    json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
