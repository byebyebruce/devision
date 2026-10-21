"""exp39 H - C report (critic EXP39-RUN-01, EXP39-R2-01): scripts/exp38/report.py on H = runs/exp39-h and C = the
reviewed C rescue (runs/exp38-c, model fee58d58...), with the two status files merged. When the main readouts are not
all completed the outcome is "incomplete" and the H identity is recorded only as far as it exists (missing parts are
null, never read by force); when they are complete the identities are checked strictly (exp38's checks, plus C's
model sha) and the sources are recorded.

    uv run python scripts/exp39/report.py
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(__file__)
C_SHA = "fee58d58abf7a5930b7df7f2a2818dbe13f93e0a4855b621dfe7ba73e4faa0a0"


def load_json(path: str):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def build(h: str = "runs/exp39-h", c: str = "runs/exp38-c", statuses=("runs/exp38c-rescue-status.txt", "runs/exp39-status.txt"),
          merged: str = "runs/exp39-status-merged.txt") -> dict:
    with open(merged, "w") as out:
        for p in statuses:
            if os.path.exists(p):
                out.write(open(p).read())
    spec = importlib.util.spec_from_file_location("exp38_report", os.path.join(HERE, "..", "exp38", "report.py"))
    assert spec is not None and spec.loader is not None
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    out = r.build({"H": h, "C": c}, merged)
    run = load_json(os.path.join(h, "run.json"))
    start = load_json(os.path.join(h, "numbers_tune", "START.json"))
    stage = (run or {}).get("config", {}).get("stages", [{}])[0]
    out["sources"] = {"C": "reused from the C rescue (scripts/exp38/rescue_c.sh, %s), not rerun" % c,
                      "H": {"micro_batch": stage.get("params", {}).get("micro_batch"), "init": stage.get("init"),
                            "data": stage.get("data"), "readout_forward_rows": (start or {}).get("forward_rows")}}
    if out["outcome"].startswith("incomplete"):
        return out
    c_start = load_json(os.path.join(c, "numbers_tune", "START.json")) or {}
    if c_start.get("sha256", {}).get("model.safetensors") != C_SHA:
        raise SystemExit("C is not the reviewed C rescue: %s" % c_start.get("sha256"))
    if out["sources"]["H"]["micro_batch"] != 8 or out["sources"]["H"]["readout_forward_rows"] != 8:
        raise SystemExit("H was not run with micro_batch 8 / 8-row readouts: %s" % out["sources"]["H"])
    return out


def main():
    out = build()
    json.dump(out, open("runs/exp39-report.json", "w"), indent=1)
    print(json.dumps({k: out.get(k) for k in ("outcome", "conditions", "H_minus_C", "identity", "sources", "report_only")},
                     indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
