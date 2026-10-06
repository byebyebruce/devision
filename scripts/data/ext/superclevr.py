"""Extension pack, Super-CLEVR (synthetic scenes of vehicles: cars, buses, bikes, motorbikes, planes; labels computed
from the scene, so exact). Licence MIT.

    uv run python scripts/data/ext/superclevr.py --root data [--share 0.6] [--per-picture 3] [--force]

Source: RyanWW/Super-CLEVR on Hugging Face: superCLEVR_questions_30k.json (10 questions per scene, each with its
functional program) and images.zip (30,000 scenes; the first 20,000 are the training split, the rest validation /
test and never used). A deterministic --share of the training scenes (by file name), at most --per-picture
questions each:
  - True / False -> noul (group: the question's template family, e.g. superclevr-compare_integer);
  - a number 0..10 -> choice with nearby numbers (common.count_options, kind "count");
  - an attribute word -> choice among 2-5 words of the same attribute. The attribute is the program's last step
    (query_color / query_shape / query_material / query_size) and the vocabulary is every answer word of that
    step in the question file (8 colours, 21 vehicle shapes, rubber / metal, small / large). A shape's options
    always include another vehicle of its superclass (bus, bike, motorbike, plane, car), since questions often
    name the superclass.
Balance: yes = no per template family; counts flattened (no answer above twice the mean). Output data/ext/superclevr/.
Synthetic pictures, so no held-out picture can occur.
"""
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (already_built, balance_yes_no, base_args, cap_per_picture, choice, count_options,  # noqa: E402
                    finish, flatten_answers, hf_file, keep_row, noul, out_dir, save_bytes)

SOURCE = "superclevr"
REPO = "RyanWW/Super-CLEVR"
TRAIN_SCENES = 20000
# Super-CLEVR's shape hierarchy (questions name either a shape or its superclass: "the bus", "the aeroplane")
SUPERCLASS = {**{w: "plane" for w in ("airliner", "biplane", "jet", "fighter")},
              **{w: "bus" for w in ("articulated bus", "double bus", "regular bus", "school bus")},
              **{w: "bicycle" for w in ("mountain bike", "road bike", "tandem bike", "utility bike")},
              **{w: "motorbike" for w in ("chopper", "cruiser", "dirtbike", "scooter")},
              **{w: "car" for w in ("sedan", "suv", "wagon", "minivan", "truck")}}          # README: the first 20k images are training, the next 5k validation, the last 5k test


def query_attribute(program: List[dict]) -> Optional[str]:
    """'color' for a program ending in query_color, ...; None for exist / count / compare programs."""
    last = program[-1]["type"] if program else ""
    return last[len("query_"):] if last.startswith("query_") else None


def vocabulary(questions: List[dict]) -> Dict[str, List[str]]:
    """Every answer word of each query_<attribute> step, sorted: Super-CLEVR's attribute vocabulary."""
    out: Dict[str, set] = defaultdict(set)
    for q in questions:
        attr = query_attribute(q["program"])
        if attr:
            out[attr].add(str(q["answer"]))
    return {k: sorted(v) for k, v in out.items()}


def convert(q: dict, image: str, vocab: Dict[str, List[str]], rng: random.Random):
    """One Super-CLEVR question -> (sample, None) or (None, reason)."""
    name = os.path.splitext(q["image_filename"])[0]
    sid, image_id = "ext-superclevr:%s:%d" % (name, q["question_index"]), "superclevr:%s" % name
    family = os.path.splitext(q["template_filename"])[0]
    text, ans = q["question"].strip(), q["answer"]
    if isinstance(ans, bool) or str(ans) in ("True", "False"):
        yes = ans is True or str(ans) == "True"
        return noul(sid, SOURCE, "compare" if "compar" in family else "exist", image_id, image, text, yes,
                    group="superclevr-%s" % family), None
    attr = query_attribute(q["program"])
    if attr is None:
        n = int(ans)
        if n > 10:
            return None, "count > 10"
        return choice(sid, SOURCE, "count", image_id, image, text, count_options(n, rng), str(n), rng,
                      group="superclevr-count"), None
    word = str(ans)
    others = [w for w in vocab.get(attr, []) if w != word]
    if word not in vocab.get(attr, []) or not others:
        return None, "word outside vocabulary"
    rng.shuffle(others)
    if attr == "shape":       # one distractor of the answer's own kind first: "the large aeroplane" must not
        others.sort(key=lambda w: SUPERCLASS.get(w) != SUPERCLASS.get(word))       # give the answer away
        others = others[:1] + rng.sample(others[1:], len(others) - 1)
    k = min(len(others), rng.choice([1, 2, 3, 4]))
    return choice(sid, SOURCE, attr, image_id, image, text, [word] + others[:k], word, rng,
                  group="superclevr-%s" % attr), None


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--share", type=float, default=0.6, help="share of the 20,000 training scenes used")
    p.add_argument("--per-picture", type=int, default=3)
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    questions = json.load(open(hf_file(a.root, REPO, "superCLEVR_questions_30k.json")))["questions"]
    vocab = vocabulary(questions)
    steps, dropped = Counter(), Counter()
    steps["questions_all"] = len(questions)
    train = [q for q in questions if q["image_index"] < TRAIN_SCENES]
    steps["questions_train_split"] = len(train)
    used = [q for q in train if keep_row(q["image_filename"], a.share, SOURCE)]
    names = sorted({q["image_filename"] for q in used})
    steps["scenes_used"] = len(names)

    # pictures: extracted once from images.zip (existing files skipped)
    pics = os.path.join(out_dir(a.root, SOURCE), "images")
    rel = {n: os.path.join("ext", SOURCE, "images", n) for n in names}
    todo = [n for n in names if not os.path.exists(os.path.join(a.root, rel[n]))]
    if todo:
        with zipfile.ZipFile(hf_file(a.root, REPO, "images.zip")) as z:
            for i, n in enumerate(todo):
                save_bytes(os.path.join(pics, n), z.read("images/" + n))
                if (i + 1) % 2000 == 0:
                    print("  extracted %d / %d pictures" % (i + 1, len(todo)), flush=True)
    steps["pictures_extracted_now"] = len(todo)

    rows = []
    for q in used:
        r, why = convert(q, rel[q["image_filename"]], vocab, rng)
        if r:
            rows.append(r)
        else:
            dropped[why] += 1
    steps["questions"] = len(rows)
    rows = cap_per_picture(rows, a.per_picture, rng)
    steps["after_per_picture_cap"] = len(rows)
    rows = balance_yes_no(rows)
    counts = flatten_answers([r for r in rows if r["kind"] == "count"])
    rows = [r for r in rows if r["kind"] != "count"] + counts
    steps["after_balance"] = len(rows)
    finish(a.root, SOURCE, rows, dict(steps), {
        "licence": "MIT", "route": "RyanWW/Super-CLEVR questions_30k.json + images.zip (training split only)",
        "downloads_bytes": {n: os.path.getsize(hf_file(a.root, REPO, n))
                            for n in ("superCLEVR_questions_30k.json", "images.zip")},
        "dropped": dict(dropped), "vocabulary": vocab,
        "count_answers": dict(sorted(Counter(r["answer_key"] for r in counts).items(), key=lambda kv: int(kv[0])))})


if __name__ == "__main__":
    main()
