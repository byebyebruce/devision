"""rl_calib: the routed-path calibration split (docs/experiments/2026-10-19-v04-reader-evidence-prep03.md 5.3; critic
V04-K-PREP-02). CPU only, loads no model. Newly drawn real-layout families (scripts/data/real_layout.py's scenes, value
twins and renderers, a new seed), each view one picture with the comparison question only.

Isolation: a family is drawn only if neither its scene nor its twin has a canonical case (real_layout.case_key) that any
existing data file uses -- every "case" field of every JSONL under data/ (real_layout train / tune / confirm, facts,
exp6x / exp7x, protect, ...) -- nor a transcribed ScienceQA scene; cases are unique inside rl_calib, and every case lies on
the "calib" side of a fixed hash partition whose "confirm" side is reserved for the synthetic confirmation (drawn after
the freeze). 300 temperature + 300 magnet families (1,200 pictures). The original magnet pool is nearly used up (769 cases,
646 used: 109 distance / 7 size / 7 equal left), so magnets come from a denser pool in the original units, ranges and
precision (DENSE, integer ticks; critic V04-K-PREP-02-R1-01), the base scenes 100 distance / 100 size / 100 equal; it is
drawn by real_layout's own renderer with that pool swapped in (dense_values). real_layout and the old data are unchanged.

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
FORCE_STRATA = ("distance", "size", "equal")               # base scenes, 100 families each (critic V04-K-PREP-02-R1-01)
MAX_ATTEMPTS = 400_000
# the denser magnet value pool, fixed before any model result: the original three units, ranges and decimal precision,
# generated from integer ticks (no float strings): in 0.25-5.00 by 0.05, cm 0.10-8.00 by 0.05, mm 5.5-70.2 by 0.1
DENSE = {"in": [t / 100 for t in range(25, 501, 5)], "cm": [t / 100 for t in range(10, 801, 5)], "mm": [t / 10 for t in range(55, 703)]}
PARTITION_SALT = "v04-route-calib-partition:"
CONFIRM_NEED = {"distance": 20, "size": 20, "equal": 20}   # prep03 5.3: magnets 20 per class in the synthetic confirmation


def partition(key: tuple) -> str:
    """A new case's fixed side, by hash: "calib" (rl_calib may draw it) or "confirm" (reserved for the synthetic
    confirmation, which is drawn only after the candidate is frozen)."""
    u = int(hashlib.sha256((PARTITION_SALT + repr(key)).encode()).hexdigest()[:12], 16) / 16 ** 12
    return "calib" if u < 0.5 else "confirm"


class dense_values:
    """real_layout's renderer with the dense pool: force_geometry ranks a distance in FORCE_VALUES[unit] (gap = rank
    mapping, growing with the distance), so the pool is swapped in for the drawing only and restored after."""

    def __enter__(self):
        self.saved = rl.FORCE_VALUES
        rl.FORCE_VALUES = DENSE
        return self

    def __exit__(self, *exc):
        rl.FORCE_VALUES = self.saved


def force_scene_dense(rng: random.Random, case: str) -> Dict:
    """real_layout.force_scene with the dense pool and the base case given (same draws for everything else)."""
    unit = rng.choice(list(DENSE))
    values = DENSE[unit]
    d1 = d2 = rng.choice(values)
    big = [[True, True], [True, True]]
    if case == "distance":
        while d2 == d1:
            d2 = rng.choice(values)
    elif case == "size":
        big[rng.randrange(2)][rng.randrange(2)] = False
    shape = rng.choice(["bar", "disc", "diamond"])
    poles = [[rng.choice(["NS", "SN"]) for _ in range(2)] for _ in range(2)]
    return {"case": case, "unit": unit, "d": [d1, d2], "big": big, "shape": shape, "poles": poles,
            "colours": rng.choice(rl.MAG_COLOURS), "font": rng.choice([12, 13, 13, 14, 15, 16, 18, 20]),
            "tall": rng.choice([90, 130, 170]), "width": rng.randint(580, 700), "p1_left": rng.random() < 0.7}


def force_twin_dense(s: Dict) -> Dict:
    """real_layout.force_twin with the dense pool (equal -> the next longer distance in Pair 2, a distance family)."""
    with dense_values():
        return rl.force_twin(s)


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


def magnet_universe() -> Dict[str, set]:
    """Every dense-pool magnet case by base stratum."""
    out: Dict[str, set] = {"distance": set(), "size": set(), "equal": set()}
    for unit, vals in DENSE.items():
        for i, a in enumerate(vals):
            out["equal"].add(("force", unit, ((a, "large"), (a, "large"))))
            out["size"].add(("force", unit, tuple(sorted([(a, "large"), (a, "mixed")]))))
            for b in vals[i + 1:]:
                out["distance"].add(("force", unit, ((a, "large"), (b, "large"))))
    return out


def capacity(used: Set[tuple]) -> Dict:
    """Unused dense magnet cases per stratum on each side of the partition (the confirm side must hold the synthetic
    confirmation's 20 per class), and the temperature case space against what is used."""
    uni = magnet_universe()
    out: Dict = {"magnets": {}}
    for st, cases in uni.items():
        free = [c for c in cases if c not in used]
        out["magnets"][st] = {"universe": len(cases), "used": len(cases) - len(free),
                              "free_calib": sum(partition(c) == "calib" for c in free), "free_confirm": sum(partition(c) == "confirm" for c in free)}
    n_obj = len(range(4, 47)) * len(range(300, 2000, 10))
    out["temperature"] = {"universe_approx": n_obj * (n_obj + 1) // 2, "used": sum(1 for c in used if c[0] == "temp")}
    out["confirm_sufficient"] = all(out["magnets"][k]["free_confirm"] >= v for k, v in CONFIRM_NEED.items())
    return out


def families(used: Set[tuple], want: Dict[str, int] = WANT, seed: int = SEED, stats: Optional[Dict] = None) -> Iterator[Tuple[int, str, Dict, Dict]]:
    """(number, form, scene, twin) of new families: both views' cases unused anywhere, unique in rl_calib and on the
    "calib" side of the partition; temperature by real_layout's own scene draws, magnets from the dense pool with the base
    scenes split evenly over distance / size / equal. Stops when full or out of attempts (shortfall in `stats`)."""
    rng = random.Random(seed)
    banned = rl.transcribed_keys()
    quota = {st: want["force"] // 3 + (1 if i < want["force"] % 3 else 0) for i, st in enumerate(FORCE_STRATA)}
    have = {"temp": 0, "force": 0}
    base = {st: 0 for st in FORCE_STRATA}
    mine: Set[tuple] = set()
    skipped = {"transcribed": 0, "case used elsewhere": 0, "case already in rl_calib": 0, "confirm side": 0}
    i = attempts = 0
    while any(have[f] < want[f] for f in have) and attempts < MAX_ATTEMPTS:
        attempts += 1
        form = "temp" if attempts % 2 else "force"
        if have[form] >= want[form]:
            continue
        if form == "temp":
            s = rl.temp_scene(rng)
            twin = rl.temp_twin(s)
        else:
            st = min((x for x in FORCE_STRATA if base[x] < quota[x]), key=lambda x: base[x])
            s = force_scene_dense(rng, st)
            twin = force_twin_dense(s)
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
        if any(partition(k) != "calib" for k in keys):
            skipped["confirm side"] += 1
            continue
        mine |= keys
        have[form] += 1
        if form == "force":
            base[s["case"]] += 1
        yield i, form, s, twin
        i += 1
    if stats is not None:
        stats.update(have=have, want=dict(want), shortfall={f: want[f] - have[f] for f in have}, skipped=skipped, attempts=attempts,
                     magnet_base=base, cases=sorted(repr(k) for k in mine))


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
                with dense_values():
                    (rl.temp_render(scene) if form == "temp" else rl.force_render(scene)).save(os.path.join(root, rel))
            hint = rl.TEMP_HINT if form == "temp" else rl.FORCE_HINT + (", but some of them are different sizes." if scene["case"] == "size" else ".")
            rows.append({"id": iid, "source": "rl-calib-" + form, "kind": form, "axis": scene["case"], "image_id": iid, "image": rel,
                         "family": fam, "group": fam, "case": repr(rl.case_key(form, scene)), "state_text": hint,
                         "questions": {"q0": {"type": "choice", "instructions": cmp_["q"], "criteria": {o: None for o in cmp_["options"]}}},
                         "gold": {"q0": {"probabilities": {o: float(o == cmp_["gold"]) for o in cmp_["options"]}}}})
    from collections import Counter
    man = {"seed": seed, "want": stats["want"], "families": stats["have"], "shortfall": stats["shortfall"], "skipped": stats["skipped"],
           "attempts": stats["attempts"], "pictures": len(rows), "used_cases_elsewhere": len(used),
           "magnet_base_families": stats["magnet_base"],
           "magnet_view_axes": dict(Counter(r["axis"] for r in rows if r["kind"] == "force")),
           "overlap_with_used": len({r["case"] for r in rows} & {repr(k) for k in used}),
           "partition": {"rule": "sha256(%r + repr(case)) < 0.5 -> calib, else confirm" % PARTITION_SALT,
                         "calib_cases_on_confirm_side": sum(partition(eval(c)) != "calib" for c in stats["cases"])},
           "dense_pool": {u: {"values": len(v), "first": v[0], "last": v[-1]} for u, v in DENSE.items()},
           "capacity": capacity(used), "cases": stats["cases"]}
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
