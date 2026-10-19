"""The evaluated development package -> its final release package (critic V04-K-PACKAGE-PREP-01, PACKAGE-R1-01..03).
CPU only, loads no model, builds and uploads nothing.

The frozen combination was evaluated with devision 0.3.0.dev1 (IDENTITY.json "code": every module's name and content,
the package version, the Python version). The release ships the final 0.3.0 as a wheel + sdist pair.
`devision.__version__` comes only from the package metadata, so the final packages can hold exactly the evaluated code
-- but evaluate.code_problems, used everywhere an identity is compared, rightly refuses any version difference. Instead
of loosening that check, this script writes an explicit hand-over record that only the release install check
(install_check.py --bridge) consumes.

The approved source (APPROVED, by the frozen IDENTITY.json's sha256): the freeze commit and the digest of its complete
src/devision file list ("manifest": path under src/ -> sha256 of the content). The manifest is read from that commit
in any clone (`--repo`; the same content at another path is the same source), must hash to the pinned digest, and its
files must hold every frozen module with exactly its frozen content; the commit's README.md / LICENSE / version-less
pyproject.toml must hash to the pinned extra digest (both pins are checked again when a record is used). Both packages are checked against this one list:

  wheel  every devision/ file equal to the manifest, no file more or less; METADATA Name devision, Version the final
         release of the frozen development version (0.3.0.dev1 -> 0.3.0; never an rc, post, local or other version);
  sdist  every src/devision file equal to the manifest, no file more or less; README.md / LICENSE equal to the commit's,
         pyproject.toml equal to the commit's except the one `version = ...` line (which must name the target);
         a root .gitignore only if byte-equal to the approved commit's (pinned separately); PKG-INFO Name / Version as
         above; no other file.

  make   --identity <eval>/IDENTITY.json --repo <a clone holding the freeze commit> --wheel W --sdist S --out R
         (--wheel-only: a development record, never accepted by a release check)
  check  --identity ... --bridge R: record_problems with the wheel and sdist next to the record.

record_problems (before anything is installed): the record is an object with every field of the right type, for this
IDENTITY.json (sha256, from version, Python, module map), its source the approved one (commit, manifest digest), a
release record (sdist present), uv.lock unchanged, and the actual wheel and sdist files hash to the record and pass the
content checks again. installed_problems: the installed process reports the recorded final version, then the strict
evaluate.code_problems against the frozen code with only the version replaced. Ordinary evaluation keeps the strict
check; IDENTITY.json is never edited.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KIND = "devision package hand-over (evaluated development package -> final release)"
# V04-K-FREEZE-ACCEPT-01: runs/v04-k/eval/IDENTITY.json was frozen from devision-v04-freeze at 05c83b4; the 19 files of
# its src/devision (the nine frozen modules among them, content-checked) hash to the manifest digest below, and its
# README.md / LICENSE / version-less pyproject.toml to the extra digest.
APPROVED = {"c14863e21c0a4a1466d0ad7de0c8955486c6c6773e601f31a36ba5af23c8fee0":
            {"commit": "05c83b46be9a44a40df3a2a4ed502e404d4c4682",
             "manifest_sha256": "ccfa0d9f4561a786a1ab88237181f21233715f767d0e56856d0ba58abf7f798a",
             # README.md / LICENSE / pyproject.toml without its version value, at that commit (critic PACKAGE-R2-01)
             "extra_sha256": "26102aec2cd7dc543bff0b9634ac9023a8e030b7e7ad1ba36e06dcd121930b02",
             # that commit's root .gitignore, which the sdist build adds (critic GITIGNORE-01 / V04-K-PACKAGE-BUILD-02)
             "gitignore_sha256": "00d1f1486108a8a4163d41e6f4bec98960a4d3f446954d5cc0c122e99a6a37c4"}}
SDIST_EXTRA = ("README.md", "LICENSE", "pyproject.toml")
FIELDS = {"kind": str, "release": bool, "identity": dict, "from_version": str, "to_version": str, "python": str,
          "modules": dict, "source": dict, "wheel": dict, "sdist": (dict, type(None)), "uv_lock_sha256": str}


def sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def hb(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(files: Dict[str, str]) -> str:
    return hb(json.dumps(files, sort_keys=True).encode())


def module_path(name: str, files: Dict[str, str]) -> Optional[str]:
    """devision.train.evaluate -> devision/train/evaluate.py (or .../__init__.py for a package) within `files`."""
    base = "/".join(name.split("."))
    return next((p for p in (base + "/__init__.py", base + ".py") if p in files), None)


def final_of(dev: str) -> Optional[str]:
    """The only allowed target: the final release of a development version (0.3.0.dev1 -> 0.3.0); None otherwise."""
    from packaging.version import Version
    v = Version(dev)
    return v.base_version if v.is_devrelease and v.pre is None and v.post is None and v.local is None else None


def frozen_modules(ident: Dict[str, Any]) -> Dict[str, str]:
    return {n: m["sha256"] for n, m in ident["code"]["modules"].items()}


def _git(repo: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", repo] + list(args), capture_output=True, check=True).stdout


def source_files(repo: str, commit: str) -> Dict[str, bytes]:
    """The approved commit's src/devision files (keyed devision/...) and its README.md / LICENSE / pyproject.toml
    (keyed ':<name>'), by content."""
    names = _git(repo, "ls-tree", "-r", "--name-only", commit, "src/devision").decode().split()
    out = {n[len("src/"):]: _git(repo, "show", "%s:%s" % (commit, n)) for n in names}
    for f in SDIST_EXTRA:
        out[":" + f] = _git(repo, "show", "%s:%s" % (commit, f))
    return out


def manifest_problems(manifest: Dict[str, str], ident: Dict[str, Any], approved: Dict[str, str]) -> List[str]:
    out = []
    if digest(manifest) != approved["manifest_sha256"]:
        out.append("the source manifest is not the approved one (digest %s)" % digest(manifest)[:16])
    for n, h in frozen_modules(ident).items():
        p = module_path(n, manifest)
        if p is None or manifest[p] != h:
            out.append("frozen module %s: not in the source manifest with its frozen content" % n)
    return out


def _meta(text: str) -> Dict[str, str]:
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
        files = {n: hb(z.read(n)) for n in z.namelist() if n.startswith("devision/") and not n.endswith("/")}
        return {"file": os.path.basename(wheel), "sha256": sha(wheel), **_meta(z.read(meta).decode()), "files": files}


def sdist_info(sdist: str) -> Dict[str, Any]:
    data: Dict[str, bytes] = {}
    with tarfile.open(sdist) as t:
        members = [m for m in t.getmembers() if m.isfile()]
        top = members[0].name.split("/")[0] if members else ""
        for m in members:
            fh = t.extractfile(m)
            if fh is not None:
                data[m.name[len(top) + 1:]] = fh.read()
    if "PKG-INFO" not in data:
        raise SystemExit("%s: no PKG-INFO" % sdist)
    return {"file": os.path.basename(sdist), "sha256": sha(sdist), **_meta(data["PKG-INFO"].decode()),
            "files": {k[len("src/"):]: hb(v) for k, v in data.items() if k.startswith("src/")},
            "extra": {k: v for k, v in data.items() if not k.startswith("src/") and k != "PKG-INFO"}}


def _versionless(pyproject: bytes) -> bytes:
    return re.sub(rb'(?m)^version\s*=\s*"[^"]*"\s*$', b'version = ""', pyproject, count=1)


def package_problems(kind: str, info: Dict[str, Any], manifest: Dict[str, str], target: str) -> List[str]:
    out = []
    if info["name"] != "devision":
        out.append("%s %s is %r, not devision" % (kind, info["file"], info["name"]))
    if info["version"] != target:
        out.append("%s version %r is not the final release %r of the frozen development version" % (kind, info["version"], target))
    out += ["%s file %s differs from the approved source" % (kind, n) for n in sorted(info["files"]) if n in manifest and info["files"][n] != manifest[n]]
    out += ["%s file %s is not in the approved source" % (kind, n) for n in sorted(set(info["files"]) - set(manifest))]
    out += ["%s lacks the approved source file %s" % (kind, n) for n in sorted(set(manifest) - set(info["files"]))]
    return out


def sdist_extra_problems(info: Dict[str, Any], extra: Dict[str, str], target: str, gitignore: Optional[str] = None) -> List[str]:
    """README.md / LICENSE as the commit's, pyproject.toml as the commit's but for the version line; a root .gitignore
    only when the approved source pins one (`gitignore`, from APPROVED -- never from a record) and byte-equal to it;
    nothing else."""
    got = info["extra"]
    allowed = set(SDIST_EXTRA) | ({".gitignore"} if gitignore else set())
    out = ["sdist file %s is not part of the release source" % k for k in sorted(set(got) - allowed)]
    if gitignore and ".gitignore" in got and hb(got[".gitignore"]) != gitignore:
        out.append("sdist .gitignore differs from the approved source's")
    out += ["sdist lacks %s" % k for k in SDIST_EXTRA if k not in got]
    for k in ("README.md", "LICENSE"):
        if k in got and hb(got[k]) != extra[k]:
            out.append("sdist %s differs from the approved source" % k)
    if "pyproject.toml" in got:
        if hb(_versionless(got["pyproject.toml"])) != extra["pyproject.toml"]:
            out.append("sdist pyproject.toml differs from the approved source beyond its version line")
        if not re.search(rb'(?m)^version\s*=\s*"%s"\s*$' % re.escape(target.encode()), got["pyproject.toml"]):
            out.append("sdist pyproject.toml does not name version %s" % target)
    return out


def make(identity: str, repo: str, wheel: str, sdist: Optional[str], lock: Optional[str] = None,
         approved: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
    approved = APPROVED if approved is None else approved
    ident = json.load(open(identity))
    if not (ident.get("code") or {}).get("modules"):
        raise SystemExit("%s has no frozen code: nothing to hand over" % identity)
    ap = approved.get(sha(identity))
    if ap is None:
        raise SystemExit("%s is not an approved frozen identity (no pinned source)" % identity)
    code = ident["code"]
    target = final_of(code["version"])
    if not target:
        raise SystemExit("the frozen version %r is not a development release; no hand-over" % code["version"])
    src = source_files(repo, ap["commit"])
    manifest = {k: hb(v) for k, v in src.items() if not k.startswith(":")}
    extra = {k[1:]: hb(_versionless(v) if k == ":pyproject.toml" else v) for k, v in src.items() if k.startswith(":")}
    problems = manifest_problems(manifest, ident, ap)
    if digest(extra) != ap["extra_sha256"]:
        problems.append("the source's README.md / LICENSE / pyproject.toml are not the approved ones")
    if ap.get("gitignore_sha256") and hb(_git(repo, "show", "%s:.gitignore" % ap["commit"])) != ap["gitignore_sha256"]:
        problems.append("the source commit's .gitignore is not the pinned one")
    w = wheel_info(wheel)
    problems += package_problems("wheel", w, manifest, target)
    s = sdist_info(sdist) if sdist else None
    if s:
        problems += package_problems("sdist", s, manifest, target) + sdist_extra_problems(s, extra, target, ap.get("gitignore_sha256"))
    if problems:
        raise SystemExit("no hand-over:\n  " + "\n  ".join(problems))
    return {"kind": KIND, "release": s is not None, "identity": {"sha256": sha(identity), "checkpoint": ident.get("checkpoint")},
            "from_version": code["version"], "to_version": target, "python": code["python"], "modules": frozen_modules(ident),
            "source": {"commit": ap["commit"], "manifest_sha256": digest(manifest), "manifest": manifest, "extra": extra,
                       "extra_sha256": digest(extra), "gitignore_sha256": ap.get("gitignore_sha256")},
            "wheel": {"file": w["file"], "sha256": w["sha256"]},
            "sdist": {"file": s["file"], "sha256": s["sha256"]} if s else None,
            "uv_lock_sha256": sha(lock or os.path.join(ROOT, "uv.lock"))}


def structure_problems(record: Any) -> List[str]:
    if not isinstance(record, dict):
        return ["the hand-over record is %s, not an object" % type(record).__name__]
    out = ["the hand-over record lacks %s" % k for k in FIELDS if k not in record]
    out += ["the hand-over record's %s is %s" % (k, type(record[k]).__name__) for k, t in FIELDS.items() if k in record and not isinstance(record[k], t)]
    if not out:
        for part, keys in (("source", ("commit", "manifest_sha256", "manifest", "extra")), ("wheel", ("file", "sha256"))):
            out += ["the hand-over record's %s lacks %s" % (part, k) for k in keys if k not in record[part]]
    return out


def record_problems(record: Any, identity: str, folder: Optional[str], lock: Optional[str] = None,
                    approved: Optional[Dict[str, Dict[str, str]]] = None) -> List[str]:
    """A hand-over record against the frozen IDENTITY.json, the approved source and the actual wheel / sdist in
    `folder` (next to the record), before anything is installed."""
    out = structure_problems(record)
    if out:
        return out
    approved = APPROVED if approved is None else approved
    ident = json.load(open(identity))
    code = ident.get("code") or {}
    ap = approved.get(sha(identity))
    if record["kind"] != KIND:
        out.append("not a package hand-over record")
    if record["identity"].get("sha256") != sha(identity) or ap is None:
        out.append("the hand-over record is not for this approved IDENTITY.json")
    if record["from_version"] != code.get("version") or record["python"] != code.get("python"):
        out.append("the hand-over's from-version / Python are not the frozen ones")
    if not code.get("version") or record["to_version"] != final_of(code["version"]):
        out.append("the hand-over's target %r is not the final release of %r" % (record["to_version"], code.get("version")))
    if not code.get("modules") or record["modules"] != frozen_modules(ident):
        out.append("the hand-over's modules are not exactly the frozen modules")
    src = record["source"]
    if ap is not None and (src["commit"] != ap["commit"] or src["manifest_sha256"] != ap["manifest_sha256"]
                           or not isinstance(src["manifest"], dict) or digest(src["manifest"]) != ap["manifest_sha256"]
                           or not isinstance(src["extra"], dict) or digest(src["extra"]) != ap["extra_sha256"]
                           or src.get("extra_sha256") != ap["extra_sha256"]
                           or src.get("gitignore_sha256") != ap.get("gitignore_sha256")):
        out.append("the hand-over's source is not the approved source (commit / manifest / README, LICENSE, pyproject)")
    elif ap is not None:
        out += manifest_problems(src["manifest"], ident, ap)
    if not record["release"] or not record["sdist"]:
        out.append("a development (wheel-only) record: not a release hand-over (wheel + sdist)")
    if sha(lock or os.path.join(ROOT, "uv.lock")) != record["uv_lock_sha256"]:
        out.append("uv.lock differs from the hand-over's (other dependency pins)")
    if out or folder is None:
        return out
    for part in ("wheel", "sdist"):
        f = os.path.join(folder, record[part]["file"])
        if not os.path.exists(f):
            out.append("the hand-over's %s %s is not next to the record" % (part, record[part]["file"]))
        elif sha(f) != record[part]["sha256"]:
            out.append("the %s next to the record is not the hand-over's (sha256)" % part)
        else:
            info = wheel_info(f) if part == "wheel" else sdist_info(f)
            out += package_problems(part, info, src["manifest"], record["to_version"])
            if part == "sdist":
                out += sdist_extra_problems(info, src["extra"], record["to_version"], (ap or {}).get("gitignore_sha256"))
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
    ap.add_argument("--repo", help="a clone holding the approved freeze commit")
    ap.add_argument("--wheel")
    ap.add_argument("--sdist")
    ap.add_argument("--wheel-only", action="store_true", help="a development record without the sdist (never a release hand-over)")
    ap.add_argument("--bridge")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.command == "make":
        if not a.wheel or not a.out or not a.repo or (not a.sdist and not a.wheel_only):
            raise SystemExit("make needs --repo, --wheel, --sdist (or --wheel-only) and --out")
        if os.path.exists(a.out):
            raise SystemExit("%s exists" % a.out)
        rec = make(a.identity, a.repo, a.wheel, a.sdist)
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1)
        print("hand-over %s -> %s (%s), source %s, wheel %s" % (rec["from_version"], rec["to_version"],
              "release" if rec["release"] else "development, wheel only", rec["source"]["commit"][:12], rec["wheel"]["sha256"]))
        return 0
    if not a.bridge:
        raise SystemExit("check needs --bridge")
    problems = record_problems(json.load(open(a.bridge)), a.identity, os.path.dirname(os.path.abspath(a.bridge)))
    print("hand-over OK" if not problems else "hand-over refused:\n  " + "\n  ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
