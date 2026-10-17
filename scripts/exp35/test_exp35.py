"""exp35 without a model: number spans never include the Pair / Sample label or the unit, multi-token numbers are
covered by all their tokens (with the checkpoint's own tokenizer when present), the two token classes recombine
to the totals, and the binding counts use the right denominators."""
import importlib.util
import os

import pytest

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("exp35", os.path.join(HERE, "mlm_numbers.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

TEMP = ("Sample A: average particle speed 1,890 m/s, mass of each particle 46 u. "
        "Sample B: mass of each particle 12 u, average particle speed 980 m/s.")
FORCE = "Pair 1: two large magnets, 0.75 in apart. Pair 2: one large magnet and one small magnet, 42.8 mm apart."
TOKENIZER = os.path.join(HERE, "..", "..", "runs", "exp32-align", "stage1", "tokenizer")


def test_spans():
    s = m.number_spans(TEMP)
    assert [(x["field"], x["object"], x["text"]) for x in s] == [
        ("speed", "Sample A", "1,890"), ("mass", "Sample A", "46"), ("mass", "Sample B", "12"), ("speed", "Sample B", "980")]
    f = m.number_spans(FORCE)
    assert [(x["object"], x["text"]) for x in f] == [("Pair 1", "0.75"), ("Pair 2", "42.8")]
    assert all(FORCE[x["start"]:x["end"]] == x["text"] for x in f)


@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="needs the checkpoint tokenizer")
def test_tokens_cover_whole_numbers_only():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    for cap in (TEMP, FORCE):
        spans = m.number_spans(cap)
        ids, offs = m.encode(tok, cap, 44)
        which = m.token_spans(offs, spans)
        for k, s in enumerate(spans):
            toks = [t for t, w in enumerate(which) if w == k]
            text = tok.decode([ids[t] for t in toks]).replace(" ", "")
            assert text == s["text"], (text, s["text"])               # every token of the number, nothing more
        masked = "".join(cap[a:b] for (a, b), w in zip(offs, which) if w is not None)
        assert "Pair" not in masked and "Sample" not in masked and not any(u in masked for u in ("m/s", " u", "in", "mm"))


def test_recombine():
    parts = [{"n": 30, "accuracy": 0.5, "nll": 1.0}, {"n": 70, "accuracy": 0.9, "nll": 0.2}]
    n, a, l = m.recombine(parts)
    assert n == 100 and abs(a - 0.78) < 1e-12 and abs(l - 0.44) < 1e-12


def test_binding_denominators():
    samples = [{"family": "f1", "captions": [TEMP]},
               {"family": "f1", "captions": [TEMP.replace("46 u", "12 u", 1).replace("Sample B: mass of each particle 12 u",
                                                                                     "Sample B: mass of each particle 46 u")]}]
    right = {"real": {(i, k): True for i in range(2) for k in range(4)}}
    right["real"][(1, 1)] = False
    b = m.binding(samples, right)["real"]
    assert b["both_objects_right_when_different"]["mass"] == {"right": 1, "of": 2}
    assert b["both_objects_right_when_different"]["speed"] == {"right": 2, "of": 2}
    assert b["both_twin_views_right_when_changed"]["mass"] == {"right": 1, "of": 2}     # A and B masses swapped
    assert "speed" not in b["both_twin_views_right_when_changed"]                        # speeds unchanged
