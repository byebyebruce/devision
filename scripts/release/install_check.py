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
import os
import shutil
import subprocess
import sys
import tempfile
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


def install(workdir: str) -> str:
    """The new environment's python, with this tree's wheel (and the train extra) installed as uv.lock pins it."""
    dist, req, venv = os.path.join(workdir, "dist"), os.path.join(workdir, "requirements.txt"), os.path.join(workdir, "venv")
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


def compare(ref: Dict, got: Dict, reference_code: Dict, venv: str, bound: bool) -> List[str]:
    from devision.train.evaluate import code_problems
    out = []
    if not os.path.realpath(got["file"]).startswith(os.path.realpath(venv) + os.sep):
        out.append("installed devision was imported from %s, not the new environment" % got["file"])
    out += ["installed code: %s" % x for x in code_problems(reference_code, got["code"])]
    if bound and got["status"].get("mode") != "on":
        out.append("installed load is %r, not the reading combination" % got["status"])
    for name, case in ref["on"].items():
        g = got["on"].get(name)
        if g is None:
            out.append("%s: missing" % name)
            continue
        for k in ("route", "table", "state", "picture"):
            if case["diagnostics"].get(k) != g["diagnostics"].get(k):
                out.append("%s: %s differs (installed vs source)" % (name, k))
        if g["diagnostics"].get("picture") != g["shown"]:
            out.append("%s: the decided picture is not the one shown" % name)
        out += ["%s: %s" % (name, x) for x in differ(case["answers"], g["answers"])]
    for name in got["off"]:
        off, plain = got["off"][name], got["plain"][name]
        if off["answers"] != plain["answers"] or set(off["diagnostics"].get("route", {}).values()) - {"ordinary"}:
            out.append("%s: off is not the plain Decider" % name)
        out += ["%s off: %s" % (name, x) for x in differ(ref["off"][name]["answers"], off["answers"])]
    return out


def differ(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    out = []
    for q in a:
        pa, pb = probs(a[q]), probs(b.get(q, {}))
        if set(pa) != set(pb) or any(abs(pa[k] - pb[k]) > TOLERANCE for k in pa):
            out.append("question %s: probabilities differ beyond %g" % (q, TOLERANCE))
    return out


def probs(ans: Dict[str, Any]) -> Dict[str, float]:
    if ans.get("type") == "noul":
        return {"true": ans["noul"]}
    return dict(ans.get("probabilities", {}))


def install_check(checkpoint: str, cases: List[Dict], identity: Optional[str] = None, workdir: Optional[str] = None) -> List[str]:
    """Problems of the checkpoint's clean install; [] when it agrees (module docstring)."""
    from devision.reading import bundle_state
    own = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="devision-install-")
    try:
        py = install(workdir)
        got = probe(py, checkpoint, cases, workdir, isolated=True)
        ref = probe(sys.executable, checkpoint, cases, ROOT, isolated=False)
        reference = json.load(open(identity))["code"] if identity else ref["code"]
        return compare(ref, got, reference, os.path.join(workdir, "venv"), bundle_state(checkpoint) == "complete")
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--identity", help="the frozen IDENTITY.json whose code the install must match")
    a = ap.parse_args(argv)
    problems = install_check(a.checkpoint, json.load(open(a.cases)), a.identity)
    print("install check: OK" if not problems else "install check found:\n  " + "\n  ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
