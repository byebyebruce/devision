"""exp34 semantics data without a model: golds follow the stated values and the wording (strong / weak flip,
equal stays equal), a family's two views differ only in the values (same questions, options and order), every
excluded canonical case is absent, and plan_batches keeps each question's two views in one micro batch."""
import importlib.util
import json
import os
import re

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp34_build", os.path.join(HERE, "build.py"))
assert spec is not None and spec.loader is not None
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

DATA = os.path.join(ROOT, "data")


def rows():
    return [json.loads(x) for x in open(os.path.join(DATA, "exp34", "semantics_train.jsonl"))]


def gold(r, q):
    return next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)


def test_golds_follow_values():
    for r in rows():
        t = r["state_text"]
        if r["kind"] == "temp":
            m = [(int(a), float(v.replace(",", ""))) for a, v in re.findall(r"particle (\d+) u, average particle speed ([\d,]+) m/s", t)]
            ka, kb = (x * y * y for x, y in m)
            hi = "same" if ka == kb else "sample A" if ka > kb else "sample B"
            lo = "same" if ka == kb else "sample B" if ka > kb else "sample A"
            for q, want in (("strong", hi), ("weak", lo)):
                g = gold(r, q)
                assert (b.is_same(g) if want == "same" else g == want), (r["id"], q)
        else:
            d = [float(x) for x in re.findall(r"([\d.]+) (?:in|cm|mm)", t)][:2]
            first = "same" if d[0] == d[1] else "1" if d[0] < d[1] else "2"
            for q in r["questions"]:
                g = gold(r, q)
                if first == "same":
                    assert b.is_same(g)
                elif q == "num":
                    assert ("less" if first == "1" else "greater") in g
                elif q in ("closer", "strong"):
                    assert g.endswith("Pair " + first) or ("Pair %s" % first) in g
                else:   # weak wording points at the other pair
                    assert ("Pair %s" % {"1": "2", "2": "1"}[first]) in g


def test_views_differ_only_in_values_and_batches_keep_them_together():
    by = {}
    for r in rows():
        by.setdefault((r["family"], r["kind"]), []).append(r)
    for (fam, kind), views in by.items():
        assert len(views) == 2
        a, c = views
        assert [(q, v["instructions"]) for q, v in a["questions"].items()] == [(q, v["instructions"]) for q, v in c["questions"].items()]
        for q in a["questions"]:
            la, lc = list(a["questions"][q]["criteria"]), list(c["questions"][q]["criteria"])
            if kind != "numbers":
                assert la == lc            # the same options in the same order
            assert len(la) == len(lc) == 3
    from devision.train.rlcd import plan_batches
    items = [{"group": r["group"], "qid": q, "fam": r["family"]} for r in rows() for q in r["questions"]]
    for batch in plan_batches(items, 8):
        pass
    units = {}
    for i, it in enumerate(items):
        units.setdefault((it["group"], it["qid"]), []).append(i)
    assert max(len(v) for v in units.values()) <= 8
    for batch in plan_batches(items, 8):
        s = set(batch)
        for v in units.values():
            assert s.issuperset(v) or not s.intersection(v)


def test_excluded_cases_absent():
    banned = b.excluded(DATA)
    for r in rows():
        t = r["state_text"]
        if r["kind"] == "pairs":
            nums = re.findall(r"([\d.]+) (in|cm|mm) apart", t)
            assert ("force", nums[0][1], tuple(sorted(float(n) for n, _ in nums))) not in banned
