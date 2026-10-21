"""exp43 side-by-side examples (critic EXP43-PREP-02): 12 tune pictures by a fixed rule -- in sha256(family) order,
temperature: the first 3 with Sample A on the right, the first 2 with the speed row above the mass row (and A on the
left), the first 1 default; force: the first 3 with Pair 1 on the right, the first 3 default -- each with its
original (name) questions, the spatial questions, the asked object's side, the gold and the transcript.
runs/exp43-preview/examples.md.

    uv run python scripts/exp43/examples.py --root data
"""
import argparse
import hashlib
import json
import os
import importlib.util

_spec = importlib.util.spec_from_file_location("exp43_build", os.path.join(os.path.dirname(__file__), "build.py"))
assert _spec is not None and _spec.loader is not None
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    a = p.parse_args(argv)
    orig = {r["id"]: r for r in b.load(os.path.join(a.root, "real_layout", "questions_tune.jsonl"))}
    sp = {r["orig_id"]: r for r in b.load(os.path.join(a.root, "exp43", "tune_spatial_O.jsonl"))}
    tr = {r["orig_id"]: r for r in b.load(os.path.join(a.root, "exp43", "tune_transcript.jsonl"))}
    sc = b.scenes({r["family"] for r in orig.values()})
    v0 = sorted((i for i in orig if i.endswith(":v0")), key=lambda i: hashlib.sha256(orig[i]["family"].encode()).hexdigest())
    pick = []
    for form, rule, n in (("temp", lambda s: not s.get("a_left", True), 3), ("temp", lambda s: s.get("speed_first") and s.get("a_left", True), 2),
                          ("temp", lambda s: s.get("a_left", True) and not s.get("speed_first"), 1),
                          ("force", lambda s: not s.get("p1_left", True), 3), ("force", lambda s: s.get("p1_left", True), 3)):
        pick += [i for i in v0 if sc[(orig[i]["family"], "v0")][0] == form and rule(sc[(orig[i]["family"], "v0")][1]) and i not in pick][:n]
    lines = ["# exp43 examples (12 tune pictures, view v0; fixed rule in the script)", ""]
    for i in pick:
        o, s, t = orig[i], sp[i], tr[i]
        scene = sc[(o["family"], "v0")][1]
        lines += ["## %s  (%s; a_left=%s p1_left=%s speed_first=%s)" % (i, o["image"], scene.get("a_left"), scene.get("p1_left"), scene.get("speed_first")),
                  "", "| qid | name question | spatial question | side | gold |", "|---|---|---|---|---|"]
        for q in s["questions"]:
            lines.append("| %s | %s | %s | %s | %s |" % (q, o["questions"][q]["instructions"], s["questions"][q]["instructions"],
                                                     s["sides"][q], b.gold_of(o["gold"][q])))
        lines += ["", "transcript (blank picture): " + t["state_text"].split("\n", 1)[1], ""]
    os.makedirs("runs/exp43-preview", exist_ok=True)
    open("runs/exp43-preview/examples.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
