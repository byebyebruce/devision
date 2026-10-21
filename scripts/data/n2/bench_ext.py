"""v0.4 evaluation: public benchmarks the project never used (plan docs/experiments/2026-10-21-v04-plan.md, stage 0),
converted to noul / choice samples, English, one picture.

    HF_ENDPOINT=https://hf-mirror.com uv run python scripts/data/n2/bench_ext.py --root data [name ...]

| name | source (Hugging Face) | kept |
|---|---|---|
| mmbench | lmms-lab/MMBench en/dev | 2-4 options as choice; the hint (if any) becomes state_text |
| cvbench | nyu-visionx/CV-Bench test_2d + test_3d | choice (count / relation / depth / distance) |
| mme | lmms-lab/MME test | yes / no as noul ("Please answer yes or no." removed) |
| naturalbench | BaiqiL/NaturalBench | each (question, picture) pair: yes / no as noul, two-option ones as choice |

Questions whose options repeat a text (the choice keys must be unique) are dropped and counted. Pictures are written
to <root>/n2/bench/images/<name>/. Every row keeps the benchmark's own category in `kind`, so reports can split by
it. Pictures of these sets must stay out of every data-n2 training source (they are matched by dHash when the
training sources are built). Writes <root>/n2/bench/<name>.jsonl and <name>.MANIFEST.json.
"""
import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ext"))
sys.path.insert(0, os.path.dirname(HERE))

LETTERS = "ABCDEFGH"


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip() in ("", "nan", "None")


def choice_row(rid: str, source: str, kind: str, image: str, question: str, options: List[str], gold: int,
               state_text: Optional[str] = None) -> Optional[dict]:
    """A choice sample, or None when two options have the same text."""
    opts = [o.strip() for o in options]
    if len(set(opts)) != len(opts) or len(opts) < 2:
        return None
    r = {"id": rid, "source": source, "kind": kind, "image_id": rid, "image": image, "group": rid,
         "questions": {"q": {"type": "choice", "instructions": question.strip(), "criteria": {o: None for o in opts}}},
         "gold": {"q": {"probabilities": {o: float(i == gold) for i, o in enumerate(opts)}}},
         "answer_key": opts[gold]}
    if state_text:
        r["state_text"] = state_text.strip()
    return r


def noul_row(rid: str, source: str, kind: str, image: str, question: str, truth: bool) -> dict:
    return {"id": rid, "source": source, "kind": kind, "image_id": rid, "image": image, "group": rid,
            "questions": {"q": {"type": "noul", "instructions": question.strip()}},
            "gold": {"q": {"probabilities": {"true": float(truth), "false": float(not truth)}}},
            "answer_key": "true" if truth else "false"}


YESNO_TAIL = re.compile(r"\s*(please\s+)?answer\s+(with\s+)?(yes|\"yes\")\s+or\s+(no|\"no\")\.?\s*$", re.I)


def strip_yesno(q: str) -> str:
    return YESNO_TAIL.sub("", q).strip()


def yes_no(answer: str) -> Optional[bool]:
    a = str(answer).strip().lower().rstrip(".")
    return True if a == "yes" else False if a == "no" else None


def cv_choice(choices: str, answer: str) -> Tuple[List[str], int]:
    opts = [str(c) for c in (ast.literal_eval(choices) if isinstance(choices, str) else choices)]
    return opts, LETTERS.index(answer.strip().strip("()"))


def save_picture(root: str, name: str, key: str, data: bytes) -> str:
    from common import save_bytes  # pyright: ignore[reportMissingImports]
    ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    rel = "n2/bench/images/%s/%s%s" % (name, re.sub(r"[^A-Za-z0-9_.-]", "_", key), ext)
    save_bytes(os.path.join(root, rel), data)
    return rel


def _tables(root: str, repo: str, pattern: str):
    import pyarrow.parquet as pq
    from common import hf_files  # pyright: ignore[reportMissingImports]
    for f in hf_files(root, repo, pattern):
        yield pq.read_table(f).to_pylist()


def mmbench(root: str) -> Tuple[List[dict], Counter]:
    rows, drop = [], Counter()
    for t in _tables(root, "lmms-lab/MMBench", "en/dev-*.parquet"):
        for r in t:
            opts = [r[L] for L in "ABCD" if not _missing(r.get(L))]
            if r["answer"] not in "ABCD"[:len(opts)]:
                drop["answer outside options"] += 1
                continue
            img = save_picture(root, "mmbench", str(r["index"]), r["image"]["bytes"])
            hint = None if _missing(r.get("hint")) else str(r["hint"])
            x = choice_row("mmbench:%s" % r["index"], "mmbench", str(r["category"]), img, r["question"], opts,
                           "ABCD".index(r["answer"]), hint)
            if x is None:
                drop["repeated option text"] += 1
                continue
            x["l2_category"], x["origin"] = r.get("L2-category"), r.get("source")
            rows.append(x)
    return rows, drop


def cvbench(root: str) -> Tuple[List[dict], Counter]:
    rows, drop = [], Counter()
    for t in _tables(root, "nyu-visionx/CV-Bench", "test_*.parquet"):
        for r in t:
            opts, gold = cv_choice(r["choices"], r["answer"])
            key = "%s_%s" % (r["type"], r["idx"])
            img = save_picture(root, "cvbench", key, r["image"]["bytes"])
            x = choice_row("cvbench:%s" % key, "cvbench", "%s-%s" % (r["type"], r["task"].lower()), img,
                           r["question"], opts, gold)
            if x is None:
                drop["repeated option text"] += 1
                continue
            x["origin"] = r.get("source")
            rows.append(x)
    return rows, drop


def mme(root: str) -> Tuple[List[dict], Counter]:
    rows, drop = [], Counter()
    seen: Dict[str, int] = Counter()
    for t in _tables(root, "lmms-lab/MME", "data/*.parquet"):
        for r in t:
            truth = yes_no(r["answer"])
            if truth is None:
                drop["answer not yes / no"] += 1
                continue
            qid = str(r["question_id"])
            seen[qid] += 1
            img = save_picture(root, "mme", qid, r["image"]["bytes"])
            rid = "mme:%s:%d" % (qid, seen[qid])
            rows.append(noul_row(rid, "mme", str(r["category"]), img, strip_yesno(r["question"]), truth))
            rows[-1]["image_id"] = rows[-1]["group"] = "mme:%s" % qid
    return rows, drop


def naturalbench(root: str) -> Tuple[List[dict], Counter]:
    """Each NaturalBench sample has two pictures and two questions; all four (question, picture) pairs are asked."""
    rows, drop = [], Counter()
    for t in _tables(root, "BaiqiL/NaturalBench", "data/*.parquet"):
        for r in t:
            idx = str(r["Index"])
            imgs = [save_picture(root, "naturalbench", "%s_%d" % (idx, k), r["Image_%d" % k]["bytes"])
                    for k in (0, 1)]
            qtype = str(r["Question_Type"]).lower()
            for qi in (0, 1):
                q = r["Question_%d" % qi]
                for ii in (0, 1):
                    ans = r["Image_%d_Question_%d" % (ii, qi)]
                    rid = "naturalbench:%s:q%d:i%d" % (idx, qi, ii)
                    if "yes" in qtype:
                        truth = yes_no(ans)
                        if truth is None:
                            drop["answer not yes / no"] += 1
                            continue
                        x = noul_row(rid, "naturalbench", "yes_no", imgs[ii], strip_yesno(q), truth)
                    else:
                        x = naturalbench_choice(rid, imgs[ii], q, ans)
                        if x is None:
                            drop["choice not parsed"] += 1
                            continue
                    x["image_id"] = x["group"] = "naturalbench:%s:%d" % (idx, ii)
                    x["pair"] = "naturalbench:%s" % idx
                    rows.append(x)
    return rows, drop


NB_OPTION = re.compile(r"([A-H])\s*:\s*(.*?)\s*;")


def naturalbench_choice(rid: str, image: str, question: str, answer: str) -> Optional[dict]:
    """NaturalBench writes the options after the question ("...?\nOption: A:red; B:blue;"); the answer is a letter."""
    parts = re.split(r"\n\s*Options?\s*:", question, maxsplit=1)
    if len(parts) != 2:
        return None
    opts = NB_OPTION.findall(parts[1])
    letters = [L for L, _ in opts]
    if letters != list(LETTERS[:len(letters)]) or len(letters) < 2:
        return None
    a = str(answer).strip().strip("()").rstrip(".")[:1].upper()
    if a not in letters:
        return None
    return choice_row(rid, "naturalbench", "choice", image, parts[0], [o for _, o in opts], letters.index(a))


def fits(rows: List[dict], checkpoint: str) -> Tuple[List[dict], int]:
    """Rows whose options all fit the model's input and stay distinct once cut (decide() refuses the others),
    checked with the checkpoint's own tokenizer and sequence builder; and how many were dropped."""
    from transformers import AutoTokenizer
    from devision.infer import ModelConfig, _option_count, config_path, question_item
    with open(config_path(checkpoint)) as f:
        cfg = ModelConfig.from_dict(json.load(f))
    tok = AutoTokenizer.from_pretrained(os.path.join(checkpoint, "tokenizer"))
    keep = []
    for r in rows:
        q = r["questions"]["q"]
        it = question_item(tok, cfg, r.get("state_text"), q)
        if len(it["markers"]) == _option_count(q) and it["options_distinct"] == len(it["markers"]):
            keep.append(r)
    return keep, len(rows) - len(keep)


BUILDERS = {"mmbench": mmbench, "cvbench": cvbench, "mme": mme, "naturalbench": naturalbench}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--checkpoint", default="runs/v03/model",
                   help="drop questions this model's input cannot hold (same tokenizer / budget for every round)")
    p.add_argument("names", nargs="*", default=list(BUILDERS))
    a = p.parse_args(argv)
    out_dir = os.path.join(a.root, "n2", "bench")
    os.makedirs(out_dir, exist_ok=True)
    for name in a.names:
        rows, drop = BUILDERS[name](a.root)
        rows, n_long = fits(rows, a.checkpoint)
        if n_long:
            drop["options do not fit the model input"] = n_long
        with open(os.path.join(out_dir, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        types = Counter(r["questions"]["q"]["type"] for r in rows)
        golds = Counter(r["answer_key"] for r in rows if r["questions"]["q"]["type"] == "noul")
        manifest = {"built_by": "scripts/data/n2/bench_ext.py", "questions": len(rows),
                    "pictures": len({r["image"] for r in rows}), "types": dict(types),
                    "noul_gold": dict(golds), "dropped": dict(drop),
                    "per_kind": dict(Counter(r["kind"] for r in rows).most_common())}
        with open(os.path.join(out_dir, name + ".MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        print(name, json.dumps({k: manifest[k] for k in ("questions", "pictures", "types", "noul_gold", "dropped")}),
              flush=True)


if __name__ == "__main__":
    main()
