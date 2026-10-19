"""scripts/release/package_bridge.py and install_check --bridge (critic V04-K-PACKAGE-PREP-01, PACKAGE-R1-01..03), CPU
only, on a tiny git repository standing in for the approved freeze commit, fake wheel / sdist files and a fake
IDENTITY.json -- nothing is built, installed or loaded:
  - the final release pair (wheel + sdist) of exactly the approved source gets a release hand-over record, also when
    the commit is read from a clone at another path; the installed code passes only under that record, while the strict
    evaluate.code_problems still refuses the version difference;
  - R1-03: a foreign tree changed together with the wheel (an unloaded file), an approved source that is not the frozen
    one, a record whose source commit / manifest was edited: refused;
  - R1-02: the sdist with a changed unloaded module or static file, a file added or missing under src/devision, another
    README / LICENSE, pyproject.toml changed beyond its version line or naming another version, an extra file; an sdist
    or wheel replaced after the record was written; a wheel-only (development) record in a release check: refused;
  - versions, names, identity, modules, uv.lock and installed-code refusals as before;
  - R1-01: install_check with --bridge pointing at null, a non-object, a record missing fields or unreadable JSON
    refuses before install and before any probe -- never falls back to building this tree; a valid record installs its
    own wheel and compares with only the version replaced; without --bridge the old path is unchanged."""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {"devision/__init__.py": "VERSION = 'metadata'\n", "devision/infer.py": "def decide(): pass\n",
         "devision/image.py": "# image\n", "devision/model.py": "# model\n", "devision/reading.py": "# reading\n",
         "devision/train/__init__.py": "", "devision/train/cli.py": "# cli\n", "devision/train/evaluate.py": "def code_problems(): pass\n",
         "devision/serve.py": "# never loaded by the evaluation\n", "devision/static/demo.html": "<html></html>\n"}
LOADED = {"devision": "devision/__init__.py", "devision.infer": "devision/infer.py", "devision.image": "devision/image.py",
          "devision.model": "devision/model.py", "devision.reading": "devision/reading.py", "devision.train": "devision/train/__init__.py",
          "devision.train.cli": "devision/train/cli.py", "devision.train.evaluate": "devision/train/evaluate.py"}
EXTRA = {"README.md": "# deVision\n", "LICENSE": "Apache-2.0\n",
         "pyproject.toml": '[project]\nname = "devision"\nversion = "0.3.0.dev1"\n\n[tool.hatch.build.targets.sdist]\nonly-include = ["src"]\n'}


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True, check=True).stdout.strip()


GITIGNORE = "runs/\ndata/\n"


def repo_with(path, files=FILES, extra=EXTRA):
    for k, v in files.items():
        (path / "src" / k).parent.mkdir(parents=True, exist_ok=True)
        (path / "src" / k).write_text(v)
    for k, v in extra.items():
        (path / k).write_text(v)
    (path / ".gitignore").write_text(GITIGNORE)
    git(path.parent, "init", "-q", str(path))
    git(path, "add", "-A")
    git(path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "freeze")
    return git(path, "rev-parse", "HEAD")


def wheel(path, version="0.3.0", name="devision", change=None):
    files = dict(FILES, **(change or {}))
    with zipfile.ZipFile(path, "w") as z:
        for k, v in files.items():
            if v is not None:
                z.writestr(k, v)
        z.writestr("devision-%s.dist-info/METADATA" % version, "Metadata-Version: 2.3\nName: %s\nVersion: %s\n\nbody\n" % (name, version))
    return str(path)


def sdist(path, version="0.3.0", change=None, extra_change=None):
    files = dict(FILES, **(change or {}))
    extra = dict(EXTRA, pyproject=None, **{"pyproject.toml": EXTRA["pyproject.toml"].replace("0.3.0.dev1", version)})
    extra.pop("pyproject")
    extra[".gitignore"] = GITIGNORE                        # as the real sdist build adds it
    extra.update(extra_change or {})
    top = "devision-%s" % version
    with tarfile.open(path, "w:gz") as t:
        entries = [("PKG-INFO", "Metadata-Version: 2.3\nName: devision\nVersion: %s\n\n" % version)]
        entries += [("src/" + k, v) for k, v in files.items() if v is not None] + [(k, v) for k, v in extra.items() if v is not None]
        for k, v in entries:
            data = v.encode()
            info = tarfile.TarInfo("%s/%s" % (top, k))
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return str(path)


@pytest.fixture
def ws(tmp_path):
    pb = load("package_bridge_t", "scripts/release/package_bridge.py")
    repo = tmp_path / "frozen"
    commit = repo_with(repo)
    code = {"modules": {n: {"path": str(repo / "src" / f), "sha256": h(FILES[f])} for n, f in LOADED.items()},
            "version": "0.3.0.dev1", "python": "3.12.13"}
    ident = tmp_path / "IDENTITY.json"
    ident.write_text(json.dumps({"checkpoint": "runs/v04-k/cand", "files": {}, "reading": {"x": 1}, "code": code}))
    manifest = {k: h(v) for k, v in FILES.items()}
    extra = {k: pb.hb(pb._versionless(v.encode()) if k == "pyproject.toml" else v.encode()) for k, v in EXTRA.items()}
    approved = {pb.sha(str(ident)): {"commit": commit, "manifest_sha256": pb.digest(manifest), "extra_sha256": pb.digest(extra),
                                     "gitignore_sha256": h(GITIGNORE)}}
    lock = tmp_path / "uv.lock"
    lock.write_text("pins\n")
    dist = tmp_path / "dist"
    dist.mkdir()
    return pb, str(ident), repo, str(lock), dist, code, approved


def pair(dist, **kw):
    return wheel(dist / "devision-0.3.0-py3-none-any.whl", **kw.get("w", {})), sdist(dist / "devision-0.3.0.tar.gz", **kw.get("s", {}))


def test_the_approved_final_pair_is_handed_over(ws, tmp_path):
    pb, ident, repo, lock, dist, code, approved = ws
    w, s = pair(dist)
    rec = pb.make(ident, str(repo), w, s, lock=lock, approved=approved)
    assert rec["release"] and rec["from_version"] == "0.3.0.dev1" and rec["to_version"] == "0.3.0"
    assert rec["source"]["commit"] == git(repo, "rev-parse", "HEAD") and rec["sdist"]["sha256"] == pb.sha(s)
    assert pb.record_problems(rec, ident, str(dist), lock=lock, approved=approved) == []
    moved = tmp_path / "elsewhere" / "clone"                                   # the same source content at another path
    moved.parent.mkdir()
    subprocess.run(["git", "clone", "-q", str(repo), str(moved)], check=True)
    assert pb.make(ident, str(moved), w, s, lock=lock, approved=approved)["source"] == rec["source"]
    got = {"modules": {n: dict(m, path="/venv/" + n) for n, m in code["modules"].items()}, "version": "0.3.0", "python": "3.12.13"}
    assert pb.installed_problems(rec, code, got) == []
    from devision.train.evaluate import code_problems
    assert code_problems(code, got) == ["version '0.3.0' differs from the frozen '0.3.0.dev1'"]       # the strict check is unchanged


def test_the_real_approved_source_is_pinned():
    pb = load("package_bridge_p", "scripts/release/package_bridge.py")
    assert pb.APPROVED == {"c14863e21c0a4a1466d0ad7de0c8955486c6c6773e601f31a36ba5af23c8fee0":
                           {"commit": "05c83b46be9a44a40df3a2a4ed502e404d4c4682",
                            "manifest_sha256": "ccfa0d9f4561a786a1ab88237181f21233715f767d0e56856d0ba58abf7f798a",
                            "extra_sha256": "26102aec2cd7dc543bff0b9634ac9023a8e030b7e7ad1ba36e06dcd121930b02",
                            "gitignore_sha256": "00d1f1486108a8a4163d41e6f4bec98960a4d3f446954d5cc0c122e99a6a37c4"}}


def test_a_foreign_tree_changed_with_the_wheel_is_refused(ws, tmp_path):
    """R1-03: a self-consistent tree + wheel both carrying another serve.py are not the approved source."""
    pb, ident, repo, lock, dist, _, approved = ws
    foreign = tmp_path / "foreign"
    commit = repo_with(foreign, dict(FILES, **{"devision/serve.py": "# changed\n"}))
    w, s = pair(dist, w={"change": {"devision/serve.py": "# changed\n"}}, s={"change": {"devision/serve.py": "# changed\n"}})
    with pytest.raises(subprocess.CalledProcessError):                      # the approved commit is not in that tree
        pb.make(ident, str(foreign), w, s, lock=lock, approved=approved)
    self_approved = {pb.sha(ident): dict(approved[pb.sha(ident)], commit=commit)}
    with pytest.raises(SystemExit, match="source manifest is not the approved one"):
        pb.make(ident, str(foreign), w, s, lock=lock, approved=self_approved)
    with pytest.raises(SystemExit, match="not an approved frozen identity"):
        pb.make(ident, str(repo), *pair(dist), lock=lock, approved={})


@pytest.mark.parametrize("bad,why", [
    ("w_module", "wheel file devision/infer.py differs"), ("w_unloaded", "wheel file devision/serve.py differs"),
    ("s_unloaded", "sdist file devision/serve.py differs"), ("s_static", "sdist file devision/static/demo.html differs"),
    ("s_added", "sdist file devision/extra.py is not in the approved source"), ("s_missing", "sdist lacks the approved source file devision/reading.py"),
    ("s_readme", "sdist README.md differs"), ("s_license", "sdist LICENSE differs"),
    ("s_pyproject", "pyproject.toml differs from the approved source beyond its version line"),
    ("s_pyversion", "does not name version 0.3.0"), ("s_extra_file", "sdist file setup.py is not part of the release source"),
    ("v031", "not the final release"), ("rc", "not the final release"), ("post", "not the final release"), ("name", "not devision")])
def test_make_refusals(ws, bad, why):
    pb, ident, repo, lock, dist, _, approved = ws
    pyproject = EXTRA["pyproject.toml"].replace("0.3.0.dev1", "0.3.0")
    kw = {"w_module": {"w": {"change": {"devision/infer.py": "x = 1\n"}}}, "w_unloaded": {"w": {"change": {"devision/serve.py": "# c\n"}}},
          "s_unloaded": {"s": {"change": {"devision/serve.py": "# c\n"}}}, "s_static": {"s": {"change": {"devision/static/demo.html": "<p>\n"}}},
          "s_added": {"s": {"change": {"devision/extra.py": "x = 1\n"}}}, "s_missing": {"s": {"change": {"devision/reading.py": None}}},
          "s_readme": {"s": {"extra_change": {"README.md": "# other\n"}}}, "s_license": {"s": {"extra_change": {"LICENSE": "MIT\n"}}},
          "s_pyproject": {"s": {"extra_change": {"pyproject.toml": pyproject + "\n[project.scripts]\nx = 'y'\n"}}},
          "s_pyversion": {"s": {"extra_change": {"pyproject.toml": pyproject.replace("0.3.0", "0.3.1")}}},
          "s_extra_file": {"s": {"extra_change": {"setup.py": "import os\n"}}},
          "v031": {"w": {"version": "0.3.1"}}, "rc": {"w": {"version": "0.3.0rc1"}}, "post": {"w": {"version": "0.3.0.post1"}},
          "name": {"w": {"name": "devision-x"}}}[bad]
    w, s = pair(dist, **kw)
    with pytest.raises(SystemExit, match=why):
        pb.make(ident, str(repo), w, s, lock=lock, approved=approved)


@pytest.mark.parametrize("bad,why", [
    ("sdist_replaced", "sdist next to the record is not the hand-over's"), ("wheel_replaced", "wheel next to the record is not the hand-over's"),
    ("sdist_gone", "sdist devision-0.3.0.tar.gz is not next to the record"), ("manifest", "not the approved source"),
    ("commit", "not the approved source"), ("wheel_only", "not a release hand-over"), ("identity", "not for this approved IDENTITY.json"),
    ("modules", "not exactly the frozen modules"), ("target", "not the final release"), ("lock", "uv.lock differs")])
def test_record_refusals(ws, bad, why):
    pb, ident, repo, lock, dist, _, approved = ws
    w, s = pair(dist)
    rec = pb.make(ident, str(repo), w, s, lock=lock, approved=approved)
    if bad == "sdist_replaced":
        sdist(dist / "devision-0.3.0.tar.gz", change={"devision/serve.py": "# swapped later\n"})
    elif bad == "wheel_replaced":
        wheel(dist / "devision-0.3.0-py3-none-any.whl", change={"devision/serve.py": "# swapped later\n"})
    elif bad == "sdist_gone":
        os.remove(s)
    elif bad == "manifest":
        rec["source"]["manifest"]["devision/serve.py"] = "0" * 64
    elif bad == "commit":
        rec["source"]["commit"] = "1" * 40
    elif bad == "wheel_only":
        rec = pb.make(ident, str(repo), w, None, lock=lock, approved=approved)
    elif bad == "identity":
        d = json.load(open(ident))
        d["checkpoint"] = "another"
        json.dump(d, open(ident, "w"))
    elif bad == "modules":
        rec["modules"]["devision.infer"] = "0" * 64
    elif bad == "target":
        rec["to_version"] = "0.3.1"
    else:
        open(lock, "a").write("other pin\n")
    assert any(why in x for x in pb.record_problems(rec, ident, str(dist), lock=lock, approved=approved))


@pytest.mark.parametrize("bad,why", [("version", "installed version '0.3.0.dev1'"), ("python", "python '3.12.0' differs"),
                                     ("content", "content differs"), ("extra", "not in the frozen reference")])
def test_installed_refusals(ws, bad, why):
    pb, ident, repo, lock, dist, code, approved = ws
    rec = pb.make(ident, str(repo), *pair(dist), lock=lock, approved=approved)
    got = {"modules": {n: dict(m) for n, m in code["modules"].items()}, "version": "0.3.0", "python": "3.12.13"}
    if bad == "version":
        got["version"] = "0.3.0.dev1"
    elif bad == "python":
        got["python"] = "3.12.0"
    elif bad == "content":
        got["modules"]["devision.infer"]["sha256"] = "1" * 64
    else:
        got["modules"]["devision.serve"] = {"path": "x", "sha256": h(FILES["devision/serve.py"])}
    assert any(why in x for x in pb.installed_problems(rec, code, got))


@pytest.fixture
def ic_ws(ws, monkeypatch, tmp_path):
    pb, ident, repo, lock, dist, code, approved = ws
    ic = load("install_check_pb", "scripts/release/install_check.py")
    real = ic._package_bridge

    def bound():                      # the test's approved source and uv.lock
        m = real()
        orig = m.record_problems
        monkeypatch.setattr(m, "record_problems", lambda r, i, f, lock_=None, approved_=None: orig(r, i, f, lock=lock, approved=approved))
        return m
    monkeypatch.setattr(ic, "_package_bridge", bound)
    monkeypatch.setattr(ic, "case_problems", lambda cases: [])
    monkeypatch.setattr(ic, "identity_problems", lambda *a, **k: [])
    seen = {"install": [], "probe": 0}

    def install(workdir, wheel_file=None):
        seen["install"].append(wheel_file)
        os.makedirs(os.path.join(workdir, "dist"), exist_ok=True)
        shutil.copy(wheel_file or wheel(tmp_path / "built.whl", version="0.3.0.dev1"), os.path.join(workdir, "dist"))
        return os.path.join(workdir, "venv", "bin", "python")
    got = {"modules": {n: dict(m) for n, m in code["modules"].items()}, "version": "0.3.0", "python": "3.12.13", "file": "x", "status": {}}

    def probe(py, ck, cases, cwd, isolated):
        seen["probe"] += 1
        return {"code": got if isolated else code}
    monkeypatch.setattr(ic, "install", install)
    monkeypatch.setattr(ic, "probe", probe)
    monkeypatch.setattr(ic, "compare", lambda ref, g, cases, reference, venv, b: seen.update(reference=reference) or [])
    return ic, pb, ident, repo, lock, dist, code, approved, seen


def test_install_check_installs_only_a_valid_hand_overs_wheel(ic_ws, tmp_path):
    ic, pb, ident, repo, lock, dist, code, approved, seen = ic_ws
    rec = pb.make(ident, str(repo), *pair(dist), lock=lock, approved=approved)
    bridge = dist / "PACKAGE_BRIDGE.json"
    bridge.write_text(json.dumps(rec))
    assert ic.install_check("ck", [], ident, workdir=str(tmp_path / "w1"), record=str(tmp_path / "r.json"), bridge=str(bridge)) == []
    assert seen["install"] == [str(dist / "devision-0.3.0-py3-none-any.whl")] and seen["reference"]["version"] == "0.3.0"
    assert seen["reference"]["modules"] == code["modules"] and json.load(open(tmp_path / "r.json"))["package_hand_over"]["sha256"]
    with pytest.raises(SystemExit, match="only in a release check"):
        ic.install_check("ck", [], None, bridge=str(bridge))


@pytest.mark.parametrize("content,why", [("null", "is NoneType, not an object"), ("[1, 2]", "is list, not an object"),
                                         ('{"kind": "x"}', "lacks release"), ("{not json", "cannot be read"),
                                         ("wheel_only", "not a release hand-over")])
def test_an_explicit_bad_bridge_is_refused_before_install_and_probe(ic_ws, tmp_path, content, why):
    """R1-01: whatever --bridge holds, it is checked; never the no-bridge path."""
    ic, pb, ident, repo, lock, dist, code, approved, seen = ic_ws
    bridge = dist / "PACKAGE_BRIDGE.json"
    if content == "wheel_only":
        content = json.dumps(pb.make(ident, str(repo), pair(dist)[0], None, lock=lock, approved=approved))
    bridge.write_text(content)
    problems = ic.install_check("ck", [], ident, workdir=str(tmp_path / "w2"), record=str(tmp_path / "r2.json"), bridge=str(bridge))
    assert any(why in x for x in problems), problems
    assert seen["install"] == [] and seen["probe"] == 0
    assert json.load(open(tmp_path / "r2.json"))["verdict"] == "refused"


def test_without_a_bridge_the_old_path_is_unchanged(ic_ws, tmp_path):
    ic, pb, ident, repo, lock, dist, code, approved, seen = ic_ws
    ic.install_check("ck", [], ident, workdir=str(tmp_path / "w3"), record=str(tmp_path / "r3.json"))
    assert seen["install"] == [None] and seen["probe"] == 2 and seen["reference"] == code       # builds this tree, strict reference


def test_a_record_whose_extra_and_sdist_were_changed_together_is_refused_before_install(ic_ws, tmp_path):
    """R2-01: the sdist's pyproject.toml changed beyond its version, the record's source.extra / extra_sha256 and
    sdist sha256 rewritten to match -- self-consistent, but not the approved source: refused, nothing installed."""
    ic, pb, ident, repo, lock, dist, code, approved, seen = ic_ws
    rec = pb.make(ident, str(repo), *pair(dist), lock=lock, approved=approved)
    changed = EXTRA["pyproject.toml"].replace("0.3.0.dev1", "0.3.0") + '\n[project.scripts]\nevil = "x:y"\n'
    s = sdist(dist / "devision-0.3.0.tar.gz", extra_change={"pyproject.toml": changed})
    rec["source"]["extra"]["pyproject.toml"] = pb.hb(pb._versionless(changed.encode()))
    rec["source"]["extra_sha256"] = pb.digest(rec["source"]["extra"])
    rec["sdist"]["sha256"] = pb.sha(s)
    bridge = dist / "PACKAGE_BRIDGE.json"
    bridge.write_text(json.dumps(rec))
    problems = ic.install_check("ck", [], ident, workdir=str(tmp_path / "w4"), record=str(tmp_path / "r4.json"), bridge=str(bridge))
    assert any("not the approved source" in x and "pyproject" in x for x in problems), problems
    assert seen["install"] == [] and seen["probe"] == 0
    with pytest.raises(SystemExit, match="README.md / LICENSE / pyproject.toml are not the approved ones"):
        pb.make(ident, str(repo), *pair(dist), lock=lock, approved={pb.sha(ident): dict(approved[pb.sha(ident)], extra_sha256="0" * 64)})


def test_the_pinned_gitignore_and_nothing_else(ic_ws, tmp_path):
    """GITIGNORE-01: the sdist's root .gitignore passes only byte-equal to the pin in APPROVED; a changed one is refused
    by make, and on use even with the record's sdist sha256 / gitignore hash rewritten; without a pin it is refused;
    other extra files stay refused (test_make_refusals[s_extra_file])."""
    ic, pb, ident, repo, lock, dist, code, approved, seen = ic_ws
    rec = pb.make(ident, str(repo), *pair(dist), lock=lock, approved=approved)
    assert rec["source"]["gitignore_sha256"] == h(GITIGNORE)
    assert pb.record_problems(rec, ident, str(dist), lock=lock, approved=approved) == []
    with pytest.raises(SystemExit, match="sdist .gitignore differs"):
        pb.make(ident, str(repo), *pair(dist, s={"extra_change": {".gitignore": "*\n"}}), lock=lock, approved=approved)
    w, s = pair(dist)
    rec = pb.make(ident, str(repo), w, s, lock=lock, approved=approved)
    s = sdist(dist / "devision-0.3.0.tar.gz", extra_change={".gitignore": "*\n"})        # swapped afterwards
    rec["sdist"]["sha256"] = pb.sha(s)
    rec["source"]["gitignore_sha256"] = h("*\n")
    bridge = dist / "PACKAGE_BRIDGE.json"
    bridge.write_text(json.dumps(rec))
    problems = ic.install_check("ck", [], ident, workdir=str(tmp_path / "w5"), record=str(tmp_path / "r5.json"), bridge=str(bridge))
    assert any("not the approved source" in x for x in problems) and seen["install"] == [] and seen["probe"] == 0
    rec["source"]["gitignore_sha256"] = h(GITIGNORE)                                   # only the sdist swapped: its content is checked
    assert any("sdist .gitignore differs" in x for x in pb.record_problems(rec, ident, str(dist), lock=lock, approved=approved))
    no_pin = {k: {kk: vv for kk, vv in v.items() if kk != "gitignore_sha256"} for k, v in approved.items()}
    with pytest.raises(SystemExit, match="sdist file .gitignore is not part of the release source"):
        pb.make(ident, str(repo), *pair(dist), lock=lock, approved=no_pin)
