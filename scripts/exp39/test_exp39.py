"""exp39 resource watch rules on fabricated windows (no model)."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location("exp39_watch", os.path.join(os.path.dirname(__file__), "watch.py"))
assert spec is not None and spec.loader is not None
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def test_rules():
    assert w.decide({50: 9.5}, {}, {}, 0).startswith("first window")
    assert w.decide({50: 7.0}, {50: 17.0}, {50: 1.5}, 0).startswith("first window MPS")
    assert w.decide({50: 7.0, 100: 13.0, 150: 13.0}, {}, {}, 0).startswith("two windows")
    assert w.decide({50: 7, 500: 7, 550: 30, 600: 13}, {}, {}, 0) == ""          # the window after val is not counted
    assert w.decide({50: 7, 100: 7}, {}, {}, 700).startswith("no new window")
    assert w.decide({50: 7, 500: 7}, {}, {}, 700) == ""                            # val after step 500: longer allowance
    assert w.decide({50: 7, 100: 7.5}, {50: 15}, {50: 3}, 100) == ""
