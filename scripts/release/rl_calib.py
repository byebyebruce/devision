"""rl_calib: the routed-path calibration split (docs/experiments/2026-10-19-v04-reader-evidence-prep03.md 5.3; critic
V04-K-PREP-02). CPU only, loads no model. Newly drawn real-layout families (scripts/data/real_layout.py's scenes, value
twins and renderers, a new seed), each view one picture with the comparison question only.

Isolation: a family is drawn only if neither its scene nor its twin has a canonical case (real_layout.case_key) that any
existing data file uses -- every "case" field of every JSONL under data/ (real_layout train / tune / confirm, facts,
exp6x / exp7x, protect, ...) -- nor a transcribed ScienceQA scene; cases are also unique inside rl_calib. The planned 300
temperature + 300 magnet families: the magnet case space is small (769 cases, 646 already used), so the magnet side takes
every isolated case it can and the shortfall is recorded, never filled by relaxing the isolation. The independent
synthetic confirmation renderer (prep03 5.3) must later exclude rl_calib's cases too (MANIFEST lists them).

    uv run python scripts/release/rl_calib.py --dry-run          # counts only, nothing drawn
    uv run python scripts/release/rl_calib.py --root data        # writes data/rl_calib/ (images, questions.jsonl, MANIFEST.json)
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
from typing import Dict, Iterator, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import real_layout as rl  # type: ignore  # noqa: E402

SEED = 20261019
WANT = {"temp": 300, "force": 300}
MAX_ATTEMPTS = 400_000


def used_cases(root: str = "data") -> Set[tuple]:
    """Every canonical case any data file names (its "case" field)."""
    import ast
    out: Set[tuple] = set()
    for p in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        if os.path.getsize(p) > 300_000_000 or os.sep + "rl_calib" + os.sep in p:
            continue
        with open(p, errors="replace") as fh:
            for line in fh:
                if '"case"' not in line:
                    continue
                try:
                    c = json.loads(line).get("case")
                    if isinstance(c, str) and c.startswith("('"):
                        out.add(ast.literal_eval(c))
                except (ValueError, SyntaxError, AttributeError):
                    continue
    return out


def families(used: Set[tuple], want: Dict[str, int] = WANT, seed: int = SEED, stats: Optional[Dict] = None) -> Iterator[Tuple[int, str, Dict, Dict]]:
    """(number, form, scene, twin) of isolated new families, temperature and magnets alternating until both are full or the
    attempts run out (then the shortfall is in `stats`)."""
    rng = random.Random(seed)
    banned = rl.transcribed_keys()
    have = {"temp": 0, "force": 0}
    mine: Set[tuple] = set()
    skipped = {"transcribed": 0, "case used elsewhere": 0, "case already in rl_calib": 0}
    i = attempts = 0
    while any(have[f] < want[f] for f in have) and attempts < MAX_ATTEMPTS:
        attempts += 1
        form = "temp" if attempts % 2 else "force"
        if have[form] >= want[form]:
            continue
        s = rl.temp_scene(rng) if form == "temp" else rl.force_scene(rng)
        twin = rl.temp_twin(s) if form == "temp" else rl.force_twin(s)
        if rl.scene_key(form, s) in banned or rl.scene_key(form, twin) in banned:
            skipped["transcribed"] += 1
            continue
        keys = {rl.case_key(form, s), rl.case_key(form, twin)}
        if keys & used:
            skipped["case used elsewhere"] += 1
            continue
        if keys & mine:
            skipped["case already in rl_calib"] += 1
            continue
        mine |= keys
        have[form] += 1
        yield i, form, s, twin
        i += 1
    if stats is not None:
        stats.update(have=have, want=dict(want), shortfall={f: want[f] - have[f] for f in have}, skipped=skipped, attempts=attempts,
                     cases=sorted(repr(k) for k in mine))


def build(root: str, dry_run: bool = False, want: Dict[str, int] = WANT, seed: int = SEED) -> Dict:
    used = used_cases(root)
    out = os.path.join(root, "rl_calib")
    if not dry_run:
        os.makedirs(os.path.join(out, "images"), exist_ok=False)
    rows, stats = [], {}
    for i, form, s, twin in families(used, want, seed, stats):
        fam = "rc%05d" % i
        views = [s, twin]
        for v, scene in enumerate(views):
            iid = "rl-calib:%s:v%d" % (fam, v)
            rel = "rl_calib/images/%s_v%d.png" % (fam, v)
            _, items = (rl.temp_items if form == "temp" else rl.force_items)(scene, i, views)
            cmp_ = items[0]                                   # the comparison question only
            assert cmp_["kind"] in ("temp-question", "force-question")
            if not dry_run:
                (rl.temp_render(scene) if form == "temp" else rl.force_render(scene)).save(os.path.join(root, rel))
            hint = rl.TEMP_HINT if form == "temp" else rl.FORCE_HINT + (", but some of them are different sizes." if scene["case"] == "size" else ".")
            rows.append({"id": iid, "source": "rl-calib-" + form, "kind": form, "axis": scene["case"], "image_id": iid, "image": rel,
                         "family": fam, "group": fam, "case": repr(rl.case_key(form, scene)), "state_text": hint,
                         "questions": {"q0": {"type": "choice", "instructions": cmp_["q"], "criteria": {o: None for o in cmp_["options"]}}},
                         "gold": {"q0": {"probabilities": {o: float(o == cmp_["gold"]) for o in cmp_["options"]}}}})
    man = {"seed": seed, "want": stats["want"], "families": stats["have"], "shortfall": stats["shortfall"], "skipped": stats["skipped"],
           "attempts": stats["attempts"], "pictures": len(rows), "used_cases_elsewhere": len(used),
           "overlap_with_used": len({r["case"] for r in rows} & {repr(k) for k in used}), "cases": stats["cases"]}
    if not dry_run:
        path = os.path.join(out, "questions.jsonl")
        with open(path, "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        man["questions.jsonl"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    man = build(a.root, a.dry_run)
    print(json.dumps({k: v for k, v in man.items() if k != "cases"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
