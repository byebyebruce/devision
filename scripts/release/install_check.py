"""Clean-install check of a checkpoint (V04-ENTRY-PENDING-01; docs/experiments/2026-10-19-v04-reader-evidence-prep03.md
5.2 step 9). CPU only.

  1. build the devision wheel from this tree (uv build) and install it, with the train extra, into a new virtual
     environment: the dependencies at the versions of uv.lock (uv export --frozen; the uv cache first, PyPI for
     whatever it lacks), then the wheel itself without dependencies;
  2. in that environment, run in isolated mode (python -I, outside this tree, so nothing is imported from the source
     tree) a fixed smoke on the checkpoint: every case through devision.load (the reading combination for a bound
     checkpoint), through devision.load(reading="off") and through a plain Decider -- answers, each question's route,
     the reads' table, the decided state and the picture identity -- plus the code it ran (evaluate.runtime_code);
  3. run the same smoke here (the source environment) and compare: the installed code was really imported from the
     new environment; module names and contents agree with the reference (the frozen IDENTITY.json "code" when given,
     else this environment's) under evaluate.code_problems; routes, tables, states and pictures are equal; every
     probability within TOLERANCE; "off" is exactly the plain Decider and reads nothing; a bound checkpoint loads "on".

    uv run python scripts/release/install_check.py --checkpoint DIR --cases cases.json [--identity <eval>/IDENTITY.json]

cases.json: [{"name", "picture" (a file), "state" (text, optional), "questions" (Jev questions)}, ...] -- e.g. two
ordinary questions, a temperature and a magnet comparison on their pictures, and the temperature question on another
picture (the mismatched control).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOLERANCE = 1e-4
MARK = "INSTALL-CHECK-RESULT "
PROBE = r'''
import json, sys
from PIL import Image
import devision
import devision.train.cli
from devision.infer import Decider, picture_identity
from devision.train.evaluate import runtime_code
a = json.loads(sys.argv[1])

def run(d):
    out = {}
    for c in a["cases"]:
        img = Image.open(c["picture"]).convert("RGB")
        diag = {}
        res = d.decide(c.get("state", ""), c["questions"], image=img, diagnostics=diag)
        diag.pop("read_seconds", None)
        out[c["name"]] = {"answers": res["answers"], "diagnostics": diag, "shown": picture_identity(img)}
    return out

on = devision.load(a["checkpoint"], device="cpu")
off = devision.load(a["checkpoint"], device="cpu", reading="off")
plain = Decider.load(a["checkpoint"], device="cpu")
res = {"status": on.reading_status, "on": run(on), "off": run(off), "plain": run(plain), "code": runtime_code(), "file": devision.__file__}
print("INSTALL-CHECK-RESULT " + json.dumps(res))
'''


def sh(cmd: List[str], **kw) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        raise SystemExit("%s failed (%d):\n%s" % (" ".join(cmd[:3]), p.returncode, (p.stderr or p.stdout)[-2000:]))
    return p.stdout


def probe(python: str, checkpoint: str, cases: List[Dict], cwd: str, isolated: bool) -> Dict:
    arg = json.dumps({"checkpoint": os.path.abspath(checkpoint), "cases": cases})
    out = sh([python] + (["-I"] if isolated else []) + ["-c", PROBE, arg], cwd=cwd)
    line = next(x for x in reversed(out.splitlines()) if x.startswith(MARK))
    return json.loads(line[len(MARK):])


def install(workdir: str, wheel_file: Optional[str] = None) -> str:
    """The new environment's python, with this tree's wheel (or the given wheel file: a package hand-over's) and the
    train extra installed as uv.lock pins it."""
    dist, req, venv = os.path.join(workdir, "dist"), os.path.join(workdir, "requirements.txt"), os.path.join(workdir, "venv")
    if wheel_file:
        os.makedirs(dist, exist_ok=True)
        shutil.copy(wheel_file, dist)
    else:
        sh(["uv", "build", "--wheel", "--out-dir", dist], cwd=ROOT)
    wheel = [os.path.join(dist, f) for f in os.listdir(dist) if f.endswith(".whl")]
    if len(wheel) != 1:
        raise SystemExit("expected one wheel in %s, found %s" % (dist, wheel))
    sh(["uv", "export", "--frozen", "--no-dev", "--extra", "train", "--no-hashes", "--no-emit-project", "-o", req], cwd=ROOT)
    sh(["uv", "venv", venv, "--python", sys.executable], cwd=workdir)
    py = os.path.join(venv, "bin", "python")
    sh(["uv", "pip", "install", "--python", py, "-r", req], cwd=workdir)      # the uv cache first, PyPI for the rest
    sh(["uv", "pip", "install", "--python", py, "--no-deps", wheel[0]], cwd=workdir)
    return py


ROLES = ("ordinary", "temperature", "magnets", "mismatched")
DOMAIN = {"ordinary": None, "temperature": "temperature", "magnets": "magnets", "mismatched": "temperature"}


def case_problems(cases: List[Dict]) -> List[str]:
    """The fixed smoke, checked before anything is installed or loaded: unique names; every role present -- "ordinary"
    (at least two questions, none routed), "temperature", "magnets" (every question of that reading domain) and
    "mismatched" ("of": a temperature case, the same state and questions on another picture: different pixels)."""
    from PIL import Image

    from devision.infer import picture_identity
    from devision.reading import domain_of
    out = []
    names = [c.get("name") for c in cases]
    if len(set(names)) != len(names) or None in names:
        out.append("case names must be present and unique: %s" % names)
    by = {c.get("name"): c for c in cases}
    for role in ROLES:
        if not any(c.get("role") == role for c in cases):
            out.append("no %s case" % role)
    for c in cases:
        role, qs = c.get("role"), c.get("questions") or {}
        if role not in ROLES:
            out.append("%s: role %r is not one of %s" % (c.get("name"), role, ROLES))
            continue
        if not qs or (role == "ordinary" and len(qs) < 2):
            out.append("%s: too few questions for a %s case" % (c["name"], role))
        bad = [k for k, q in qs.items() if domain_of(q.get("instructions", "")) != DOMAIN[role]]
        if bad:
            out.append("%s: questions %s are not of the %s domain %r" % (c["name"], bad, role, DOMAIN[role]))
        if role == "mismatched":
            of = by.get(c.get("of"))
            if of is None or of.get("role") != "temperature":
                out.append("%s: 'of' must name a temperature case" % c["name"])
            elif of.get("questions") != qs or of.get("state", "") != c.get("state", ""):
                out.append("%s: not the same state and questions as %s" % (c["name"], of["name"]))
            elif picture_identity(Image.open(of["picture"]).convert("RGB")) == picture_identity(Image.open(c["picture"]).convert("RGB")):
                out.append("%s: the same picture as %s" % (c["name"], of["name"]))
    return out


def answer_problems(questions: Dict[str, Dict], answers: Dict[str, Any]) -> List[str]:
    """One response against its request (the Jev answer contract): exactly the asked questions, each of its type;
    a noul probability finite in [0, 1]; choice probabilities over exactly the options, finite, in [0, 1], summing
    to 1, and the chosen option their arg-max."""
    out = []
    if set(answers) != set(questions):
        return ["answers %s for questions %s" % (sorted(answers), sorted(questions))]
    for k, q in questions.items():
        a = answers[k]
        if a.get("type") != q["type"]:
            out.append("question %s: answer type %r" % (k, a.get("type")))
        elif q["type"] == "noul":
            v = a.get("noul")
            if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
                out.append("question %s: noul %r" % (k, v))
        else:
            p = a.get("probabilities") or {}
            vals = list(p.values())
            if set(p) != set(q["criteria"]) or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals):
                out.append("question %s: probabilities %r" % (k, p))
            elif abs(sum(vals) - 1.0) > 1e-3:
                out.append("question %s: probabilities sum to %r" % (k, sum(vals)))
            elif a.get("choice") != max(p, key=lambda o: p[o]):
                out.append("question %s: choice %r is not the arg-max" % (k, a.get("choice")))
    return out


def same_reads(a: List[Dict], b: List[Dict]) -> bool:
    """The reads of two runs: the same questions, decoded strings, validity and values, boxes within TOLERANCE."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if set(x) != set(y) or any(x[k] != y[k] for k in x if k not in ("box", "selector_box")):
            return False
        if any(len(x[k]) != len(y[k]) or any(abs(u - v) > TOLERANCE for u, v in zip(x[k], y[k])) for k in ("box", "selector_box") if k in x):
            return False
    return True


def compare(ref: Dict, got: Dict, cases: List[Dict], reference_code: Dict, venv: str, bound: bool) -> List[str]:
    from devision.train.evaluate import code_problems
    out = []
    if not os.path.realpath(got["file"]).startswith(os.path.realpath(venv) + os.sep):
        out.append("installed devision was imported from %s, not the new environment" % got["file"])
    out += ["installed code: %s" % x for x in code_problems(reference_code, got["code"])]
    if bound and got["status"].get("mode") != "on":
        out.append("installed load is %r, not the reading combination" % got["status"])
    names = {c["name"] for c in cases}
    for side, res in (("source", ref), ("installed", got)):
        for mode in ("on", "off", "plain"):
            if set(res.get(mode, {})) != names:
                out.append("%s %s: cases %s, expected %s" % (side, mode, sorted(res.get(mode, {})), sorted(names)))
    if out and any("cases" in x for x in out):
        return out
    for c in cases:
        name, qs = c["name"], c["questions"]
        want_on = {k: ("read" if bound and DOMAIN[c["role"]] else "ordinary") for k in qs}
        for side, res in (("source", ref), ("installed", got)):
            for mode in ("on", "off", "plain"):
                r = res[mode][name]
                out += ["%s %s %s: %s" % (side, mode, name, x) for x in answer_problems(qs, r["answers"])]
                want = want_on if mode == "on" else {k: "ordinary" for k in qs}
                if r["diagnostics"].get("route") != want:
                    out.append("%s %s %s: route %r, expected %r" % (side, mode, name, r["diagnostics"].get("route"), want))
                if r["diagnostics"].get("picture") != r["shown"]:
                    out.append("%s %s %s: the decided picture is not the one shown" % (side, mode, name))
        s, g = ref["on"][name], got["on"][name]
        for k in ("table", "state", "picture"):
            if s["diagnostics"].get(k) != g["diagnostics"].get(k):
                out.append("%s: %s differs (installed vs source)" % (name, k))
        if not same_reads(s["diagnostics"].get("reads", []), g["diagnostics"].get("reads", [])):
            out.append("%s: reads differ (installed vs source)" % name)
        out += ["%s: %s" % (name, x) for x in differ(s["answers"], g["answers"])]
        off, plain = got["off"][name], got["plain"][name]
        if off["answers"] != plain["answers"]:
            out.append("%s: off is not the plain Decider" % name)
        out += ["%s off: %s" % (name, x) for x in differ(ref["off"][name]["answers"], off["answers"])]
    return out


def differ(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    out = []
    for q in a:
        pa, pb = probs(a[q]), probs(b.get(q, {}))
        if set(pa) != set(pb) or any(not abs(pa[k] - pb[k]) <= TOLERANCE for k in pa):
            out.append("question %s: probabilities differ beyond %g" % (q, TOLERANCE))
        elif a[q].get("choice") != b.get(q, {}).get("choice"):
            out.append("question %s: choice %r vs %r" % (q, a[q].get("choice"), b.get(q, {}).get("choice")))
    return out


def probs(ans: Dict[str, Any]) -> Dict[str, float]:
    if ans.get("type") == "noul":
        return {"true": ans["noul"]}
    return dict(ans.get("probabilities", {}))


INFERENCE = ("config.json", "model.safetensors", "encoder", "vision", "tokenizer", "reading")


def _check_eval():
    import importlib.util
    spec = importlib.util.spec_from_file_location("release_check_eval", os.path.join(ROOT, "scripts", "release", "check_eval.py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def identity_problems(checkpoint: str, identity: Dict, pins: Optional[Dict[str, str]] = None) -> List[str]:
    """A release check's checkpoint against a complete frozen IDENTITY.json (files, reading, code), before anything
    is installed or loaded. The frozen candidate itself goes through check_eval's own identity checks (whole tree,
    reading bundle and lineage from the pinned source); a release copy elsewhere is compared by content: every
    inference file (config, weights, encoder, vision, tokenizer, reading/) the identity holds, and no other one."""
    missing = [k for k in ("checkpoint", "files", "reading", "code") if not identity.get(k)]
    if missing:
        return ["IDENTITY.json lacks %s: not a complete frozen identity, no release check" % missing]
    ce: Any = _check_eval()
    setattr(ce, "SOURCE_PINS", pins)
    named = identity["checkpoint"]
    if os.path.realpath(named if os.path.isabs(named) else os.path.join(ROOT, named)) == os.path.realpath(checkpoint):
        now = ce.tree_hashes(checkpoint)
        changed = sorted(k for k in set(now) | set(identity["files"]) if now.get(k) != identity["files"].get(k))
        out = ["checkpoint differs from IDENTITY.json: %s" % changed[:5]] if changed else []
        try:
            if ce.reading_identity(checkpoint) != identity["reading"]:
                out.append("reading bundle / lineage are not the frozen ones")
        except SystemExit as e:
            out.append(str(e))
        return out
    now = {k: v for k, v in ce.tree_hashes(checkpoint).items() if k.split(os.sep)[0] in INFERENCE}
    want = {k: v for k, v in identity["files"].items() if k.split(os.sep)[0] in INFERENCE}
    changed = sorted(k for k in set(now) | set(want) if now.get(k) != want.get(k))
    out = ["release copy's inference files differ from IDENTITY.json: %s" % changed[:5]] if changed else []
    rj = os.path.join(checkpoint, "reading", "reading.json")
    if not os.path.exists(rj) or ce.sha256(rj) != identity["reading"]["reading_json_sha256"]:
        out.append("release copy's reading.json is not the frozen one")
    return out


def _package_bridge():
    import importlib.util
    spec = importlib.util.spec_from_file_location("release_package_bridge", os.path.join(ROOT, "scripts", "release", "package_bridge.py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha(path: str) -> str:
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def install_check(checkpoint: str, cases: List[Dict], identity: Optional[str] = None, workdir: Optional[str] = None,
                  record: Optional[str] = None, pins: Optional[Dict[str, str]] = None, bridge: Optional[str] = None) -> List[str]:
    """Problems of the checkpoint's clean install; [] when it agrees (module docstring). With `identity` (a complete
    frozen IDENTITY.json) this is the release check: the checkpoint is checked against it before anything is
    installed or loaded, and `record` (required) keeps what was checked. Without it, a development self-check only
    (its record says so; never release evidence). `bridge` (release only): a package hand-over record
    (scripts/release/package_bridge.py) -- its wheel is installed instead of building one, and the installed code must
    be the frozen modules under the recorded final version; the strict code_problems is otherwise unchanged."""
    from PIL import Image

    from devision.infer import picture_identity
    from devision.reading import bundle_state
    release = identity is not None
    if release and not record:
        raise SystemExit("a release check needs --record")
    ident = json.load(open(identity)) if identity else None
    use_bridge = bridge is not None                    # given at all: checked, whatever it holds (PACKAGE-R1-01)
    if use_bridge and not release:
        raise SystemExit("a package hand-over is checked only in a release check (--identity)")
    pb: Any = _package_bridge() if use_bridge else None
    hand: Any = None
    bwheel: Optional[str] = None
    problems = case_problems(cases) + (identity_problems(checkpoint, ident, pins) if ident is not None else [])
    if use_bridge and identity:
        try:
            hand = json.load(open(str(bridge)))
        except (OSError, ValueError) as e:
            hand = None
            problems.append("package hand-over: %s cannot be read (%s)" % (bridge, e))
        else:
            folder = os.path.dirname(os.path.abspath(str(bridge)))
            bad = pb.record_problems(hand, identity, folder)
            problems += ["package hand-over: %s" % x for x in bad]
            if not bad:
                bwheel = os.path.join(folder, hand["wheel"]["file"])
    own = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="devision-install-")
    wheel = None
    got: Dict = {}
    try:
        if use_bridge and bwheel is None and not problems:
            problems.append("package hand-over: no verified wheel")   # never fall back to building this tree
        if not problems:                                  # nothing installed or loaded for a wrong smoke / identity
            py = install(workdir, bwheel)
            dist = os.path.join(workdir, "dist")
            wheel = [os.path.join(dist, f) for f in os.listdir(dist) if f.endswith(".whl")][0]
            got = probe(py, checkpoint, cases, workdir, isolated=True)
            ref = probe(sys.executable, checkpoint, cases, ROOT, isolated=False)
            reference = ident["code"] if ident is not None else ref["code"]
            if use_bridge and ident is not None:                # only after record_problems passed (nothing installed otherwise)
                problems = ["package hand-over: %s" % x for x in pb.installed_problems(hand, ident["code"], got["code"])]
                reference = pb.bridged_reference(ident["code"], hand)
            problems += compare(ref, got, cases, reference, os.path.join(workdir, "venv"), bundle_state(checkpoint) == "complete")
        if record:
            ce: Any = _check_eval()
            rec = {"kind": "release" if release else "development self-check (not release evidence)",
                   "identity": {"path": os.path.abspath(identity), "sha256": _sha(identity), "checkpoint": ident.get("checkpoint")}
                   if identity and ident else None,
                   "checkpoint": os.path.abspath(checkpoint),
                   "inference_files": {k: v for k, v in ce.tree_hashes(checkpoint).items() if k.split(os.sep)[0] in INFERENCE},
                   "wheel": {"name": os.path.basename(wheel), "sha256": _sha(wheel)} if wheel else None,
                   "package_hand_over": {"path": os.path.abspath(bridge), "sha256": _sha(bridge)} if bridge else None,
                   "installed_code": got.get("code"),
                   "cases": [dict(c, picture_sha256=_sha(c["picture"]), pixels=picture_identity(Image.open(c["picture"]).convert("RGB"))) for c in cases],
                   "problems": problems, "verdict": "agrees" if not problems else "refused",
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with open(record, "w") as fh:
                json.dump(rec, fh, indent=1)
        return problems
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--identity", help="the candidate's complete frozen IDENTITY.json: the release check (needs --record)")
    ap.add_argument("--bridge", help="a package hand-over record (package_bridge.py make), its wheel next to it: install that wheel")
    ap.add_argument("--record", help="where to keep the result (identity, files, wheel, code, cases, verdict)")
    a = ap.parse_args(argv)
    problems = install_check(a.checkpoint, json.load(open(a.cases)), a.identity, record=a.record, bridge=a.bridge)
    kind = "release check" if a.identity else "development self-check (not release evidence)"
    print("%s: OK" % kind if not problems else "%s found:\n  %s" % (kind, "\n  ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
