"""The evaluated development package -> its final release package (critic V04-K-PACKAGE-PREP-01). CPU only, loads no
model, uploads nothing.

The frozen combination was evaluated with devision 0.3.0.dev1 (IDENTITY.json "code": every module's name and content,
the package version, the Python version). The release ships the final 0.3.0. `devision.__version__` comes only from the
package metadata, so a final wheel can hold exactly the modules that were evaluated -- but evaluate.code_problems, used
everywhere an identity is compared, rightly refuses any version difference. Instead of loosening that check, this
script writes an explicit hand-over record that only the release install check (install_check.py --bridge) consumes:

  make   --identity <eval>/IDENTITY.json --wheel dist/devision-0.3.0-py3-none-any.whl [--sdist dist/devision-0.3.0.tar.gz]
         --out <eval>/PACKAGE_BRIDGE.json
         The wheel (and sdist) must be devision at the final release of the frozen development version (0.3.0.dev1 ->
         0.3.0, nothing else), holding every frozen module with exactly its frozen content (sdist: src/devision/...),
         and every other devision/ file equal to the frozen source tree (--source: the worktree the identity's modules
         come from). The record binds the IDENTITY.json sha256, both versions, the frozen Python,
         every module's name and sha256, the wheel / sdist sha256 and uv.lock's sha256 (the dependency pins the install
         uses).
  check  --identity ... --bridge ... [--wheel ...]: record_problems below.

record_problems / installed_problems are what install_check uses: the record must be this identity's (sha256, from
version, Python, the full module map), the installed wheel the recorded file, uv.lock unchanged, and the installed
process must report the recorded final version, the frozen Python and exactly the frozen modules' contents
(evaluate.code_problems against the frozen code with only the version replaced by the recorded one). Any module
change, any other version step, another identity, wheel, Python or dependency pin is refused. Ordinary evaluation keeps
the strict code_problems; IDENTITY.json is never edited.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import tarfile
import zipfile
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KIND = "devision package hand-over (evaluated development package -> final release)"


def sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def module_name(path: str) -> Optional[str]:
    """devision/train/evaluate.py -> devision.train.evaluate; devision/__init__.py -> devision; else None."""
    if not (path.startswith("devision/") and path.endswith(".py")):
        return None
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _metadata_version(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        if not line.strip():
            break
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return {"name": out.get("Name", ""), "version": out.get("Version", "")}


def wheel_info(wheel: str) -> Dict[str, Any]:
    with zipfile.ZipFile(wheel) as z:
        meta = next((n for n in z.namelist() if n.endswith(".dist-info/METADATA")), None)
        if meta is None:
            raise SystemExit("%s: no METADATA" % wheel)
        info = _metadata_version(z.read(meta).decode())
        mods = {module_name(n): hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if module_name(n)}
        files = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("devision/") and not n.endswith("/")}
    return {"file": os.path.basename(wheel), "sha256": sha(wheel), **info, "modules": mods, "files": files}


def sdist_info(sdist: str) -> Dict[str, Any]:
    with tarfile.open(sdist) as t:
        members = [m for m in t.getmembers() if m.isfile()]
        top = members[0].name.split("/")[0] if members else ""
        pkg = next((m for m in members if m.name == top + "/PKG-INFO"), None)
        if pkg is None:
            raise SystemExit("%s: no PKG-INFO" % sdist)
        info = _metadata_version(t.extractfile(pkg).read().decode())          # type: ignore[union-attr]
        mods = {}
        for m in members:
            rel = m.name[len(top) + 1:]
            if rel.startswith("src/") and module_name(rel[4:]):
                mods[module_name(rel[4:])] = hashlib.sha256(t.extractfile(m).read()).hexdigest()   # type: ignore[union-attr]
    return {"file": os.path.basename(sdist), "sha256": sha(sdist), **info, "modules": mods}


def final_of(dev: str) -> Optional[str]:
    """The only allowed target: the final release of a development version (0.3.0.dev1 -> 0.3.0); None otherwise."""
    from packaging.version import Version
    v = Version(dev)
    return v.base_version if v.is_devrelease and v.pre is None and v.post is None and v.local is None else None


def dist_problems(kind: str, info: Dict[str, Any], frozen: Dict[str, str], target: str) -> List[str]:
    out = []
    if info["name"] != "devision":
        out.append("%s %s is %r, not devision" % (kind, info["file"], info["name"]))
    if info["version"] != target:
        out.append("%s version %r is not the final release %r of the frozen development version" % (kind, info["version"], target))
    out += ["%s module %s: content differs from the frozen one" % (kind, n) for n in sorted(frozen) if n in info["modules"] and info["modules"][n] != frozen[n]]
    out += ["%s module %s: missing" % (kind, n) for n in sorted(set(frozen) - set(info["modules"]))]
    return out


def tree_problems(info: Dict[str, Any], source: str) -> List[str]:
    """Every devision/ file of the wheel (modules the evaluation never loaded, static files) equals the frozen source
    tree's src/devision/ file, and the tree has no file the wheel lacks."""
    src = os.path.join(source, "src")
    have = {}
    for d, _, fs in os.walk(os.path.join(src, "devision")):
        for f in fs:
            if "__pycache__" in d or f.endswith(".pyc"):
                continue
            have[os.path.relpath(os.path.join(d, f), src)] = sha(os.path.join(d, f))
    out = ["wheel file %s differs from the frozen source tree" % n for n in sorted(info["files"]) if have.get(n) != info["files"][n]]
    return out + ["source file %s is not in the wheel" % n for n in sorted(set(have) - set(info["files"]))]


def frozen_modules(ident: Dict[str, Any]) -> Dict[str, str]:
    return {n: m["sha256"] for n, m in ident["code"]["modules"].items()}


def make(identity: str, wheel: str, source: str, sdist: Optional[str] = None, lock: Optional[str] = None) -> Dict[str, Any]:
    ident = json.load(open(identity))
    if not (ident.get("code") or {}).get("modules"):
        raise SystemExit("%s has no frozen code: nothing to hand over" % identity)
    code = ident["code"]
    target = final_of(code["version"])
    if not target:
        raise SystemExit("the frozen version %r is not a development release; no hand-over" % code["version"])
    frozen = frozen_modules(ident)
    w = wheel_info(wheel)
    problems = dist_problems("wheel", w, frozen, target) + tree_problems(w, source)
    s = sdist_info(sdist) if sdist else None
    if s:
        problems += dist_problems("sdist", s, frozen, target)
    if problems:
        raise SystemExit("no hand-over:\n  " + "\n  ".join(problems))
    lock = lock or os.path.join(ROOT, "uv.lock")
    return {"kind": KIND, "identity": {"sha256": sha(identity), "checkpoint": ident.get("checkpoint")},
            "from_version": code["version"], "to_version": target, "python": code["python"], "modules": frozen,
            "wheel": {"file": w["file"], "sha256": w["sha256"], "files": w["files"]},
            "sdist": {"file": s["file"], "sha256": s["sha256"]} if s else None,
            "uv_lock_sha256": sha(lock)}


def record_problems(record: Dict[str, Any], identity: str, wheel: Optional[str] = None, lock: Optional[str] = None) -> List[str]:
    """A hand-over record against the frozen IDENTITY.json (and the wheel that will be installed), before installing."""
    ident = json.load(open(identity))
    code = ident.get("code") or {}
    out = []
    if record.get("kind") != KIND:
        out.append("not a package hand-over record")
    if (record.get("identity") or {}).get("sha256") != sha(identity):
        out.append("the hand-over record is for another IDENTITY.json")
    if record.get("from_version") != code.get("version") or record.get("python") != code.get("python"):
        out.append("the hand-over's from-version / Python are not the frozen ones")
    if not code.get("version") or record.get("to_version") != final_of(code["version"]):
        out.append("the hand-over's target %r is not the final release of %r" % (record.get("to_version"), code.get("version")))
    if record.get("modules") != (frozen_modules(ident) if code.get("modules") else None):
        out.append("the hand-over's modules are not exactly the frozen modules")
    if sha(lock or os.path.join(ROOT, "uv.lock")) != record.get("uv_lock_sha256"):
        out.append("uv.lock differs from the hand-over's (other dependency pins)")
    if wheel is not None:
        if sha(wheel) != (record.get("wheel") or {}).get("sha256"):
            out.append("the wheel to install is not the hand-over's wheel")
        elif not out:
            w = wheel_info(wheel)
            out += dist_problems("wheel", w, record["modules"], record["to_version"])
            if w["files"] != record["wheel"].get("files"):
                out.append("the wheel's files are not the hand-over's")
    return out


def bridged_reference(frozen_code: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """The frozen code with only its version replaced by the hand-over's final version (used after record_problems)."""
    ref = copy.deepcopy(frozen_code)
    ref["version"] = record["to_version"]
    return ref


def installed_problems(record: Dict[str, Any], frozen_code: Dict[str, Any], got_code: Dict[str, Any]) -> List[str]:
    """The installed process's runtime_code: the recorded final version, the frozen Python, exactly the frozen
    modules' contents (the strict evaluate.code_problems against the bridged reference)."""
    from devision.train.evaluate import code_problems
    out = []
    if (got_code or {}).get("version") != record.get("to_version"):
        out.append("installed version %r is not the hand-over's %r" % ((got_code or {}).get("version"), record.get("to_version")))
    return out + code_problems(bridged_reference(frozen_code, record), got_code)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("make", "check"))
    ap.add_argument("--identity", required=True)
    ap.add_argument("--wheel")
    ap.add_argument("--sdist")
    ap.add_argument("--source", help="the frozen source tree (the worktree whose src/ the identity's modules are)")
    ap.add_argument("--bridge")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.command == "make":
        if not a.wheel or not a.out or not a.source:
            raise SystemExit("make needs --wheel, --source and --out")
        if os.path.exists(a.out):
            raise SystemExit("%s exists" % a.out)
        rec = make(a.identity, a.wheel, a.source, a.sdist)
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1)
        print("hand-over %s -> %s, %d modules, wheel %s" % (rec["from_version"], rec["to_version"], len(rec["modules"]), rec["wheel"]["sha256"]))
        return 0
    problems = record_problems(json.load(open(a.bridge)), a.identity, a.wheel)
    print("hand-over OK" if not problems else "hand-over refused:\n  " + "\n  ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
