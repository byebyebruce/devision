"""exp38 report without a model: H - C on fabricated records, the four conditions, and report-only parts absent."""
import importlib.util
import json
import os

HERE = os.path.dirname(__file__)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report(tmp_path, monkeypatch):
    r = load("exp38_report", os.path.join(HERE, "report.py"))
    t36 = load("exp36_test", os.path.join(HERE, "..", "exp36", "test_exp36.py"))
    runs = {}
    for arm, right in (("H", lambda f, x: True), ("C", lambda f, x: False)):
        for split in ("tune", "fit"):
            d = tmp_path / arm / ("numbers_" + split)
            d.mkdir(parents=True)
            with open(d / "numbers.jsonl", "w") as fh:
                for x in t36.records(right, lambda f, x: False):
                    fh.write(json.dumps(x) + "\n")
        runs[arm] = str(tmp_path / arm)
    out = r.build(runs)
    assert out["invest_in_512"] and out["H_minus_C"]["whole"]["value"] == 1.0
    assert out["report_only"]["H"]["protection"] == {} and "cpu_latency_ms" not in out["report_only"]["H"]
