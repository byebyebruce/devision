"""Extension pack, VisOnlyQA Train (Kamoi et al. 2024: synthetic geometry figures; pure perception: is there a
triangle / quadrilateral ABC, how many times longer / larger is one segment / region than another, how large is an
angle). Licence GPL-3.0.

    uv run python scripts/data/ext/visonlyqa.py --root data [--limit 40000] [--per-picture 3] [--force]

Source: Hugging Face ryokamoi/VisOnlyQA_Train, the five 2D splits syntheticgeometry__{triangle, quadrilateral,
length, angle, area} (10,000 questions each, 1.2 GB of parquet). The 3D splits (CLEVR / Super-CLEVR renders,
6 GB) are not used. Conversion:
  - triangle / quadrilateral, "There is (no) triangle ABC in the figure. True or False?" -> noul
    "Is there (no) triangle ABC in the figure?" followed by the source's one-sentence definition;
  - length / area / angle, five lettered options "(a) 2 (b) 0.5 ..." -> choice over the option texts
    ("0.25" ... "4", "10 degrees" ... "180 degrees"), the "estimate from the visual information" sentence dropped.
Balance: each answer text is used equally often per task by the source (2,000 each), but the point letters leak:
the same wording ("Is there a quadrilateral ABCD ...", "angle ABC") recurs and some letter combinations go with
one answer (question-only baseline by exact wording 0.60 on the true / false questions, 0.30 on the 5-option ones).
So the balancing group is the exact wording, not the template: true / false, as many yes as no per wording
(common.balance_yes_no); 5-option, no answer of a wording more frequent than its second answer
(v2_common.flatten_top). The template (task + polarity) is kept in the "template" field.
Selection: at most --per-picture questions per picture, the balance above, then whole wordings in the order of a
hash (deterministic) until --limit questions. Synthetic figures: no held-out picture can occur. Pictures of the kept questions are extracted to data/ext/visonlyqa/images/.
"""
import hashlib
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (already_built, balance_yes_no, base_args, cap_per_picture, choice, extract_picture,  # noqa: E402
                    finish, hf_files, noul)

SOURCE = "visonlyqa"
REPO = "ryokamoi/VisOnlyQA_Train"
TASKS = ("triangle", "quadrilateral", "length", "angle", "area")
COLUMNS = ["image_path", "answer", "prompt_no_reasoning", "response_options", "id", "task_category"]


def picture_key(image_path: str) -> str:
    """'images/SyntheticGeometry/syntheticgeometry_15131noise_1.jpeg' -> 'syntheticgeometry_15131noise_1'."""
    return os.path.splitext(os.path.basename(image_path))[0]


def true_false(prompt: str):
    """'There is (no) triangle EDB in the figure. True or False?\\n\\n<definition>\\n\\n<format line>' ->
    (question, negated) or None."""
    parts = [p.strip() for p in prompt.strip().split("\n\n") if p.strip()]
    m = re.match(r"There is (no|an?) (\w+) (\w+) in (?:the|this) figure\. True or False\?$", parts[0])
    if not m:
        return None
    neg = m.group(1) == "no"
    q = "Is there %s %s %s in the figure?" % ("no" if neg else m.group(1), m.group(2), m.group(3))
    definition = parts[1] if len(parts) > 2 and not parts[1].startswith("Your response") else ""
    return (q + " " + definition).strip(), neg


def lettered(prompt: str, answer: str):
    """'Line AB is X times longer than AE. Which ...? You only need to estimate ... (a) 2 (b) 0.5 ...' + 'b' ->
    (question, options, answer text) or None."""
    first = prompt.strip().split("\n")[0]
    i = first.find("(a)")
    if i < 0:
        return None
    opts = re.findall(r"\(([a-j])\)\s*(.+?)\s*(?=\([a-j]\)|$)", first[i:])
    by = dict(opts)
    if answer not in by or len(set(by.values())) != len(by):
        return None
    q = re.sub(r"\s*You only need to estimate.*$", "", first[:i].strip())
    q = q.replace("Which of the following options is", "Which is")
    if q.endswith("Which is a reasonable estimate?"):  # area: name the unknown like length does
        q = q[:-1] + " of X?"
    return q, [v for _, v in opts], by[answer]


def template(task: str, question: str) -> str:
    """The task, plus the polarity for the true / false tasks."""
    return "%s-%s" % (task, "no" if question.startswith("Is there no ") else "yes") \
        if task in ("triangle", "quadrilateral") else task


def wording(question: str) -> str:
    """The balancing group: the question as common.question_only_baseline's "wording" sees it."""
    return " ".join(question.lower().split())


def balance(rows: list) -> list:
    """Per wording: as many yes as no (true / false); no answer above the second one (options)."""
    from v2_common import flatten_top  # pyright: ignore[reportMissingImports]
    nouls = balance_yes_no([r for r in rows if r["questions"]["q"]["type"] == "noul"])
    return nouls + flatten_top([r for r in rows if r["questions"]["q"]["type"] == "choice"], min_group=2)


def first_wordings(rows: list, limit: int, seed: int) -> list:
    """Whole wordings (so their balance holds) in hash order until `limit` questions."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    out: list = []
    for g in sorted(by, key=lambda g: order(g, seed)):
        if len(out) + len(by[g]) > limit:
            continue
        out += by[g]
    return out


def convert(row: dict, image: str, rng: random.Random):
    """One source row -> (sample, None) or (None, reason)."""
    task = row["task_category"]
    sid = "ext-visonlyqa:%s" % row["id"]
    image_id = "visonlyqa:%s" % picture_key(row["image_path"])
    if list(row["response_options"]) == ["True", "False"]:
        tf = true_false(row["prompt_no_reasoning"])
        if not tf or row["answer"] not in ("True", "False"):
            return None, "unparsed true / false"
        q, _ = tf
        return noul(sid, SOURCE, "geometry-" + task, image_id, image, q, row["answer"] == "True",
                    group=wording(q), template=template(task, q)), None
    mc = lettered(row["prompt_no_reasoning"], row["answer"])
    if not mc or not 2 <= len(mc[1]) <= 10:
        return None, "unparsed options"
    q, opts, ans = mc
    return choice(sid, SOURCE, "geometry-" + task, image_id, image, q, opts, ans, rng, group=wording(q),
                  template=template(task, q)), None


def order(sid: str, seed: int) -> str:
    return hashlib.sha1(("%d:%s" % (seed, sid)).encode()).hexdigest()


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--limit", type=int, default=40000)
    p.add_argument("--per-picture", type=int, default=3)
    p.add_argument("--noisy", action="store_true", help="also keep the noisy JPEG renders (default: clean renders "
                   "only -- after the 256 letterbox their 4-6 px point letters are unreadable)")
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    import pyarrow.parquet as pq
    rng = random.Random(a.seed)
    files = hf_files(a.root, REPO, "data/syntheticgeometry__*.parquet")
    rows, steps, dropped, where = [], Counter(), Counter(), {}
    for path in files:
        t = pq.read_table(path, columns=COLUMNS).to_pylist()
        for k, r in enumerate(t):
            steps["rows_" + r["task_category"]] += 1
            s, why = convert(r, "", rng)
            if s is None:
                dropped[why] += 1
                continue
            if not a.noisy and "noise" in picture_key(r["image_path"]):
                dropped["noisy render"] += 1
                continue
            where[s["id"]] = (path, k, r["image_path"])
            rows.append(s)
    steps["converted"] = len(rows)
    rows = cap_per_picture(rows, a.per_picture, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows = balance(rows)
    steps["after_balance"] = len(rows)
    rows = first_wordings(rows, a.limit, a.seed)
    steps["after_limit"] = len(rows)
    # extract the kept rows' pictures (one read of the image column per file)
    need: dict = {}
    for r in rows:
        path, k, image_path = where[r["id"]]
        need.setdefault(path, {})[k] = r
    for path, by_row in need.items():
        col = pq.read_table(path, columns=["decoded_image"]).column("decoded_image")
        for k, r in by_row.items():
            r["image"] = extract_picture(a.root, SOURCE, picture_key(where[r["id"]][2]), col[k].as_py()["bytes"])
    rows.sort(key=lambda r: r["id"])
    finish(a.root, SOURCE, rows, dict(steps), {
        "dropped": dict(dropped), "templates": dict(Counter(r["template"] for r in rows).most_common()),
        "licence": "GPL-3.0"})


if __name__ == "__main__":
    main()
