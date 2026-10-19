"""PANEL-NUMBER-PREP-01: EXP75's two arms (N: numbers fully masked, R: the original random mask; both one pass from
EXP74's D1) read with the accepted panel probe (scripts/panel_bind: random-mask reproduction against each arm's own
training report, +/-0.003; both numbers masked on the true / fixed wrong picture), then compared -- rules written
before running (docs/experiments/2026-10-20-panel-number-plan.md):

  start    START for the two arm end points (scripts/panel_bind/report.start with this probe's end points / references)
  report   the probe report of the arms (runs/panel-number/report.json), then compare.json:
           the main reading is "both ends exact" on the 837 A != B families (strict: every token of both spans right);
           paired by family (2,000 resamples, seed 0): each arm - D1, each arm - EXP26 (D1 / EXP26 from the accepted
           PANEL-BIND-RUN-01 records, pinned by sha256 and never reloaded), N - R, and each arm's true - wrong picture;
           an arm worth a new decision diagnostic: both-exact >= 0.60, lower bound of (arm - D1) > 0 and of
           (true - wrong) > 0; number supervision credited only if N - R >= 0.10 with lower bound > 0. An arm that is not
           a completed, reproduced end point enters no comparison. Outcomes: stop (neither arm) / R only (more original
           training works) / N, no attribution (end-point progress only) / N attributed. Error types (same answer,
           swapped, other end's gold, per end, A = B) are reported as diagnostics, never as a replacement rule.
"""
import hashlib
import importlib.util
import json
import os
import sys

OUT = "runs/panel-number"
ARMS = {"n": "runs/exp75-panel-n/n", "r": "runs/exp75-panel-r/r"}
REFS = {a: (os.path.join(p, "align_report.json"), -1) for a, p in ARMS.items()}
FIXED = {"d1": ("runs/panel-bind/d1/numbers.jsonl", "e18bf3f2135cd77ee61a696071eb63b0dd87e882b484b985392bc6d433ffcb0a"),
         "exp26": ("runs/panel-bind/exp26/numbers.jsonl", "41c1c99820703b7acf38a66fef56499b6952ab620fe33eb1860328fa986d93de")}
FIXED_REPORT = ("runs/panel-bind/report.json", "d7c4e0e455fc4351fb29d728755ba3ac1c2d4bd9a96d4569b185b14beb73ba3d")
LINE, ATTRIBUTION = 0.60, 0.10


def bind():
    spec = importlib.util.spec_from_file_location("panel_bind_report", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panel_bind", "report.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def both_exact(rows):
    """family -> both ends exact on the true picture, A != B only."""
    return {r["family"]: float(r["exact"]["A"] and r["exact"]["B"]) for r in rows if r["mode"] == "real" and r["gold"]["A"] != r["gold"]["B"]}


def paired(rep, a, b):
    keys = sorted(set(a) & set(b))
    if len(keys) != len(a) or len(keys) != len(b):
        raise ValueError("the two readings are not on the same families")
    return rep.boot({k: (a[k], b[k]) for k in keys}, lambda us: sum(x - y for x, y in us) / len(us))


def fixed(rep):
    """The accepted D1 / EXP26 records, by sha256; refused when anything differs."""
    p, h = FIXED_REPORT
    if sha(p) != h or any(json.load(open(p))["endpoints"][n]["status"] != "completed" for n in FIXED):
        raise SystemExit("the PANEL-BIND-RUN-01 report is not the accepted one")
    out = {}
    for n, (path, h) in FIXED.items():
        if sha(path) != h:
            raise SystemExit("%s is not the accepted record" % path)
        out[n] = both_exact([json.loads(x) for x in open(path)])
    return out


def compare(out: str = OUT) -> dict:
    rep = bind()
    res = rep.report(out)                                  # the probe report: statuses, reproduction, statistics
    ref = fixed(rep)
    arms = {}
    for a in ARMS:
        e = res["endpoints"].get(a, {})
        arm = {"status": e.get("status", "missing")}
        arms[a] = arm
        if arm["status"] != "completed":
            continue
        rows = [json.loads(x) for x in open(os.path.join(out, a, "numbers.jsonl"))]
        be = both_exact(rows)
        arm["both_exact"] = sum(be.values()) / len(be)
        arm["minus_d1"] = paired(rep, be, ref["d1"])
        arm["minus_exp26"] = paired(rep, be, ref["exp26"])
        arm["true_minus_wrong"] = e["true_minus_wrong_both_exact"]
        arm["errors"] = {m: e["numbers"][m] for m in ("real", "shuffled")}
        arm["worth_decision_diagnostic"] = (arm["both_exact"] >= LINE and arm["minus_d1"]["ci95"][0] > 0
                                            and arm["true_minus_wrong"]["ci95"][0] > 0)
        arm["_be"] = be
    out_c: dict = {"arms": arms, "probe_status": res["status"], "line": LINE, "attribution_line": ATTRIBUTION}
    done = [a for a in ARMS if arms[a]["status"] == "completed"]
    if len(done) == 2:
        nr = paired(rep, arms["n"]["_be"], arms["r"]["_be"])
        out_c["n_minus_r"] = nr
        attributed = nr["value"] >= ATTRIBUTION and nr["ci95"][0] > 0
    else:
        attributed = None
    for a in done:
        del arms[a]["_be"]
    n_ok, r_ok = (arms[a].get("worth_decision_diagnostic") for a in ("n", "r"))
    if len(done) < 2:
        out_c["outcome"] = "incomplete: arm(s) %s not completed -- no attribution" % [a for a in ARMS if a not in done]
    elif not n_ok and not r_ok:
        out_c["outcome"] = "stop: neither arm reaches the investment line"
    elif r_ok and not n_ok:
        out_c["outcome"] = "R only: the extra original-recipe pass is effective; number supervision not credited"
    elif n_ok and not attributed:
        out_c["outcome"] = "N, not attributed: end-point progress only (N - R below its line)%s" % ("; R also passes" if r_ok else "")
    else:
        out_c["outcome"] = "N attributed: N passes and N - R >= 0.10 with lower bound > 0%s" % ("; R also passes" if r_ok else "")
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "compare.json.tmp")
    json.dump(out_c, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(out, "compare.json"))
    return out_c


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[:1] == ["start"]:
        done = {a: p for a, p in ARMS.items() if os.path.exists(os.path.join(p, "model.safetensors")) and os.path.exists(REFS[a][0])}
        if not done:
            print("no arm end point: no START (the report says incomplete)")
            return 1
        st = bind().start(OUT, done, {a: REFS[a] for a in done})
        print(json.dumps({k: v for k, v in st.items() if k != "wrong_picture"}, indent=1))
        return 0
    res = compare()
    print(json.dumps(res, indent=1))
    return 0 if not res["outcome"].startswith("incomplete") and res["probe_status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
