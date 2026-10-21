"""Round 13 bench (evaluation only): CountBenchQA (491 LAION web pictures, counts 2..10). Licence not stated by the
dataset card. Plan 1.5.

    uv run python scripts/data/r13/sources/countbench.py --root data [--force]

Source: vikhyatk/CountBenchQA (Hugging Face, data/test-*.parquet, pictures inside). Each row -> choice "How many
<things> are there in the image?" (the dataset's own question) among ONE common candidate set for every question,
"1" to "10" (the benchmark's counts are 2-10), in a per-question shuffled order. The options never depend on the
question's gold: the earlier gold-1..gold+2 set let "take the middle option" score 491/491 without the picture
(critic R13-I07). The MANIFEST reports the picture-blind baselines (uniform guess, the best constant answer, the
median option). The score is not comparable with the open-ended counting protocol of the paper / moondream.
Near duplicates: every picture's dHash is compared with the project's training history (scripts/data/r13/identity.py,
`near_seen`: photos without traceable ids, e.g. PixMo / Cauldron web pictures) and with every picture of the round-13
training sources already built (data/r13/<source>/train.jsonl); hits within 2 bits are kept but flagged
(`near_training`) and counted in the MANIFEST, so they can be reported separately. Round-13 sources built after
this bench drop pictures near it (base.Identity.near_held).
"""
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import already_built, base_args, finish_bench, hf_files, save_bytes  # noqa: E402

REPO = "vikhyatk/CountBenchQA"


CANDIDATES = [str(m) for m in range(1, 11)]


def candidates(n: int):
    """The same ten options for every question (n is only checked to be one of them)."""
    assert str(n) in CANDIDATES, n
    return list(CANDIDATES)


def option_baselines(rows) -> dict:
    """Accuracy of picture-blind rules that read only the options (and, for the best constant, the gold
    distribution: an upper bound for any rule that ignores the picture and the question)."""
    golds = [r["answer_key"] for r in rows]
    median = lambda opts: sorted(opts, key=int)[(len(opts) - 1) // 2]
    best = max(set(golds), key=golds.count)
    return {"uniform_guess": round(1 / len(CANDIDATES), 4),
            "median_option": round(sum(median(list(r["questions"]["q"]["criteria"])) == g for r, g in zip(rows, golds)) / len(rows), 4),
            "best_constant_answer": {"answer": best, "accuracy": round(golds.count(best) / len(rows), 4)}}


def main(argv=None):
    p = base_args(__doc__)
    a = p.parse_args(argv)
    if already_built(a.root, "r13/bench/countbench.jsonl", a.force):
        return
    import pyarrow.parquet as pq
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rng = random.Random(a.seed)
    rows = []
    for f in hf_files(a.root, REPO, "data/test-*.parquet"):
        for i, r in enumerate(pq.read_table(f).to_pylist()):
            data = r["image"]["bytes"]
            ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            rel = "r13/bench/images/countbench/%03d%s" % (i, ext)
            save_bytes(os.path.join(a.root, rel), data)
            opts = candidates(r["number"])
            rng.shuffle(opts)
            rows.append({"id": "countbench:%03d" % i, "source": "countbench", "kind": "count",
                         "image_id": "countbench:%03d" % i, "image": rel, "group": "countbench:%03d" % i,
                         "questions": {"q": {"type": "choice", "instructions": r["question"].strip(),
                                             "criteria": {o: None for o in opts}}},
                         "gold": {"q": {"probabilities": {o: float(o == str(r["number"])) for o in opts}}},
                         "answer_key": str(r["number"]), "laion_text": r["text"]})
    hits, history = near_training(a.root, rows)
    for r in rows:
        r["near_training"] = hits.get(r["image"], "")
    finish_bench(a.root, "countbench", rows, {
        "history_check": history,
        "near_training": sum(1 for r in rows if r["near_training"]),
        "near_training_by": {k: sum(1 for r in rows if r["near_training"] == k) for k in set(hits.values())},
        "candidates": CANDIDATES,
        "option_baselines": option_baselines(rows),
        "note": "choice among the common candidates 1-10 for every question (no gold-dependent options); "
                "not the open-ended protocol"})


def near_training(root: str, rows) -> tuple:
    """(picture -> "project_history" | "<r13 source>" for pictures within 2 bits of a training picture, how the
    project history was checked). Without data/r13/identity.json (a machine that never rebuilt the old rounds'
    data) only the round-13 sources and data-n2 built so far are checked, and the MANIFEST says so."""
    import json
    import numpy as np
    import identity  # pyright: ignore[reportMissingImports]
    from v9_mix import hashes  # pyright: ignore[reportMissingImports]
    mine = hashes(root, [r["image"] for r in rows])
    try:
        ident = identity.load(root)
    except FileNotFoundError:
        ident, history = None, "skipped: data/r13/identity.json not built on this machine"
    else:
        history = "identity.py near_seen"
    out = {rel: "project_history" for rel, h in mine.items() if ident is not None and ident.near_seen(h)}
    for f in sorted(glob.glob(os.path.join(root, "r13", "*", "train.jsonl"))):
        src = os.path.basename(os.path.dirname(f))
        rels = {json.loads(l)["image"] for l in open(f) if l.strip()}
        rels = {r for r in rels if "cogent" not in r}
        if not rels:
            continue
        arr = np.array(sorted(set(hashes(root, rels).values())), dtype=np.uint64)
        for rel, h in mine.items():
            if rel not in out and arr.size and int(np.bitwise_count(arr ^ np.uint64(h)).min()) <= 2:
                out[rel] = src
    return out, history


if __name__ == "__main__":
    main()
