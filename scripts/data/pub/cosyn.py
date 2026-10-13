"""Public-data pack, CoSyn-400k (allenai/CoSyn-400k, ODC-BY): synthetic diagrams, chemical structures, circuits and
math figures rendered from code, with model-written QA pairs (question, explanation, answer).

    uv run python scripts/data/pub/cosyn.py --root data [--branches diagram,chemical,circuit,math]

Each answer is open-ended; only answers that turn into a closed question without inventing labels are kept:
  - yes / no                                   -> noul
  - "How many ..." with a whole number 0..10   -> choice over the candidates 0..10 (number words read as numbers)
  - a short answer (1-3 words) of a question whose picture has at least one other short answer of a different
    value (other labels in the same figure)    -> choice among those (2-5 options): the distractors are things the
                                                  figure shows, so the question cannot be solved by spotting the
                                                  only plausible word
Long answers are dropped. Many questions read text in the figure; at the 256-pixel input much of it is illegible,
so the MANIFEST reports per-branch counts and the training mix should take a small share first (exp14 style).
Pictures: extracted once to ext/cosyn/images/<branch>_<id>.png; the validation split is not used. Synthetic figures
are not compared with the held-out photos (they cannot be the same pictures); exact-pixel duplicates of held-out
diagrams are dropped.
"""
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ext"))
sys.path.insert(0, HERE)
from common import already_built, base_args, cap_per_picture, choice, finish, hf_files, noul, save_bytes  # noqa: E402
from held import balance_noul, get_held, split_dev  # noqa: E402

SOURCE = "cosyn"
REPO = "allenai/CoSyn-400k"
BRANCHES = ("diagram", "chemical", "circuit", "math")
LICENCE = "ODC-BY 1.0 (allenai/CoSyn-400k)"
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def clean(answer: str) -> str:
    return " ".join(str(answer).strip().rstrip(".").split())


def as_number(answer: str):
    a = clean(answer).lower()
    return int(a) if a.isdigit() else WORDS.get(a)


def kind_of(question: str, answer: str) -> str:
    a = clean(answer).lower()
    if a in ("yes", "no"):
        return "yesno"
    n = as_number(answer)
    if n is not None and re.match(r"(?i)\s*how many\b", question):
        return "count" if 0 <= n <= 10 else "drop"
    if a and len(a.split()) <= 3 and n is None and len(a) <= 40:
        return "short"
    return "drop"


def figure_questions(key: str, image: str, branch: str, qa: dict, rng: random.Random):
    """The closed questions of one figure (see the module doc)."""
    image_id = "%s:%s" % (SOURCE, key)
    pairs = list(zip(qa.get("question") or [], qa.get("answer") or []))
    shorts = [clean(a) for q, a in pairs if kind_of(q, a) == "short"]
    out, steps = [], Counter()
    for i, (q, a) in enumerate(pairs):
        q = " ".join(str(q).split())
        if not q:
            steps["empty question"] += 1
            continue
        k = kind_of(q, a)
        sid = "pub-%s:%s:%d" % (SOURCE, key, i)
        if k == "yesno":
            out.append(noul(sid, SOURCE, branch + ":yesno", image_id, image, q, clean(a).lower() == "yes", group=image_id))
        elif k == "count":
            n = as_number(a)
            out.append(choice(sid, SOURCE, branch + ":count", image_id, image, q, [str(m) for m in range(11)], str(n),
                              rng, group=image_id, candidates="0..10"))
        elif k == "short":
            ans = clean(a)
            others = sorted({s for s in shorts if s.lower() != ans.lower()})
            if not others:
                steps["short answer without another label in the figure"] += 1
                continue
            rng.shuffle(others)
            out.append(choice(sid, SOURCE, branch + ":label", image_id, image, q, [ans] + others[:rng.choice([1, 2, 3, 4])],
                              ans, rng, group=image_id))
        else:
            steps["long or other answer"] += 1
    return out, steps


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--branches", default=",".join(BRANCHES))
    p.add_argument("--per-picture", type=int, default=4)
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    import pyarrow.parquet as pq
    rng = random.Random(a.seed)
    held = get_held(a.root)
    rows, steps, per_branch = [], Counter(), defaultdict(Counter)
    for branch in a.branches.split(","):
        for f in hf_files(a.root, REPO, "%s/train-*.parquet" % branch):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=64, columns=["id", "image", "qa_pairs"]):
                for rid, img, qa in zip(batch.column("id").to_pylist(), batch.column("image").to_pylist(),
                                        batch.column("qa_pairs").to_pylist()):
                    steps["figures"] += 1
                    data = (img or {}).get("bytes")
                    if not data:
                        steps["figure without picture"] += 1
                        continue
                    key = "%s_%s" % (branch, re.sub(r"[^A-Za-z0-9_-]", "_", str(rid)))
                    ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
                    rel = os.path.join("ext", SOURCE, "images", key + ext)
                    save_bytes(os.path.join(a.root, rel), data)
                    if held.same_diagram(rel):
                        steps["identical to a held-out diagram"] += 1
                        continue
                    qs, s = figure_questions(key, rel, branch, qa or {}, rng)
                    steps.update(s)
                    per_branch[branch].update(r["kind"].split(":")[1] for r in qs)
                    rows += qs
    steps["questions"] = len(rows)
    rows = cap_per_picture(rows, a.per_picture, rng)
    rows = balance_noul(rows, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows, steps["dev_questions"] = split_dev(a.root, SOURCE, rows)
    finish(a.root, SOURCE, rows, dict(steps), {"licence": LICENCE, "per_branch": {b: dict(c) for b, c in per_branch.items()},
                                                "note": "many questions read figure text; check legibility at 256 px"})


if __name__ == "__main__":
    main()
