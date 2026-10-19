"""scripts/release/package_bridge.py and install_check --bridge (critic V04-K-PACKAGE-PREP-01), CPU only, on a fake
source tree / wheel / sdist / IDENTITY.json -- nothing is installed or loaded:
  - the frozen development version's final release with exactly the frozen modules (and every other file equal to the
    frozen source tree) gets a hand-over record; the installed code passes only under that record, while the strict
    evaluate.code_problems still refuses the version difference;
  - refused: a changed frozen module, a changed file the evaluation never loaded, any other version step (0.3.1, an rc,
    a post release), another package name, a missing module, a different sdist module; a record for another identity,
    with edited modules or target, with another uv.lock or another wheel; an installed process with another version,
    Python, module content or an extra module;
  - install_check: a bad record is refused before anything is installed, the record's wheel is what gets installed and
    the comparison uses the frozen code with only the version replaced; a hand-over outside a release check is refused."""
import hashlib
import importlib.util
import io
import json
import os
import tarfile
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {"devision/__init__.py": "VERSION = 'metadata'\n", "devision/infer.py": "def decide(): pass\n",
         "devision/train/__init__.py": "", "devision/train/evaluate.py": "def code_problems(): pass\n",
         "devision/image.py": "# image\n", "devision/model.py": "# model\n", "devision/reading.py": "# reading\n",
         "devision/train/cli.py": "# cli\n", "devision/serve.py": "# never loaded by the evaluation\n", "devision/static/demo.html": "<html></html>\n"}
LOADED = {"devision": "devision/__init__.py", "devision.infer": "devision/infer.py", "devision.train": "devision/train/__init__.py",
          "devision.train.evaluate": "devision/train/evaluate.py", "devision.image": "devision/image.py", "devision.model": "devision/model.py",
          "devision.reading": "devision/reading.py", "devision.train.cli": "devision/train/cli.py"}


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def wheel(path, version="0.3.0", name="devision", change=None):
    files = dict(FILES, **(change or {}))
    with zipfile.ZipFile(path, "w") as z:
        for k, v in files.items():
            if v is not None:
                z.writestr(k, v)
        z.writestr("devision-%s.dist-info/METADATA" % version, "Metadata-Version: 2.3\nName: %s\nVersion: %s\n\nbody\n" % (name, version))
    return str(path)


def sdist(path, version="0.3.0", change=None):
    files = dict(FILES, **(change or {}))
    top = "devision-%s" % version
    with tarfile.open(path, "w:gz") as t:
        for k, v in [("PKG-INFO", "Metadata-Version: 2.3\nName: devision\nVersion: %s\n\n" % version)] + [("src/" + k, v) for k, v in files.items()]:
            data = v.encode()
            info = tarfile.TarInfo("%s/%s" % (top, k))
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return str(path)


@pytest.fixture
def ws(tmp_path):
    pb = load("package_bridge_t", "scripts/release/package_bridge.py")
    src = tmp_path / "frozen"
    for k, v in FILES.items():
        (src / "src" / k).parent.mkdir(parents=True, exist_ok=True)
        (src / "src" / k).write_text(v)
    code = {"modules": {n: {"path": str(src / "src" / f), "sha256": h(FILES[f])} for n, f in LOADED.items()},
            "version": "0.3.0.dev1", "python": "3.12.13"}
    ident = tmp_path / "IDENTITY.json"
    ident.write_text(json.dumps({"checkpoint": "runs/v04-k/cand", "files": {}, "reading": {"x": 1}, "code": code}))
    lock = tmp_path / "uv.lock"
    lock.write_text("pins\n")
    dist = tmp_path / "dist"
    dist.mkdir()
    return pb, str(ident), str(src), str(lock), dist, code


def test_the_final_release_of_the_frozen_modules_is_handed_over(ws):
    pb, ident, src, lock, dist, code = ws
    w = wheel(dist / "devision-0.3.0-py3-none-any.whl")
    rec = pb.make(ident, w, src, sdist(dist / "devision-0.3.0.tar.gz"), lock=lock)
    assert rec["from_version"] == "0.3.0.dev1" and rec["to_version"] == "0.3.0" and rec["python"] == "3.12.13"
    assert rec["modules"] == {n: m["sha256"] for n, m in code["modules"].items()} and rec["sdist"]["sha256"]
    assert pb.record_problems(rec, ident, w, lock=lock) == []
    got = {"modules": {n: dict(m, path="/venv/" + n) for n, m in code["modules"].items()}, "version": "0.3.0", "python": "3.12.13"}
    assert pb.installed_problems(rec, code, got) == []
    from devision.train.evaluate import code_problems
    assert code_problems(code, got) == ["version '0.3.0' differs from the frozen '0.3.0.dev1'"]       # the strict check is unchanged


@pytest.mark.parametrize("bad,why", [
    ("module", "module devision.infer: content differs"), ("unloaded_file", "devision/serve.py differs from the frozen source"),
    ("v031", "not the final release"), ("rc", "not the final release"), ("post", "not the final release"),
    ("name", "not devision"), ("missing", "module devision.infer: missing"), ("sdist", "sdist module devision.infer")])
def test_make_refusals(ws, bad, why):
    pb, ident, src, lock, dist, _ = ws
    kw = {"module": {"change": {"devision/infer.py": "def decide(): return 1\n"}}, "unloaded_file": {"change": {"devision/serve.py": "# changed\n"}},
          "v031": {"version": "0.3.1"}, "rc": {"version": "0.3.0rc1"}, "post": {"version": "0.3.0.post1"}, "name": {"name": "devision-x"},
          "missing": {"change": {"devision/infer.py": None}}}.get(bad, {})
    w = wheel(dist / "w.whl", **kw)
    sd = sdist(dist / "s.tar.gz", change={"devision/infer.py": "x = 1\n"}) if bad == "sdist" else None
    with pytest.raises(SystemExit, match=why):
        pb.make(ident, w, src, sd, lock=lock)


@pytest.mark.parametrize("bad,why", [("identity", "another IDENTITY.json"), ("modules", "not exactly the frozen modules"),
                                     ("target", "not the final release"), ("lock", "uv.lock differs"), ("wheel", "not the hand-over's wheel")])
def test_record_refusals(ws, bad, why):
    pb, ident, src, lock, dist, _ = ws
    w = wheel(dist / "devision-0.3.0-py3-none-any.whl")
    rec = pb.make(ident, w, src, lock=lock)
    if bad == "identity":
        d = json.load(open(ident))
        d["checkpoint"] = "another"
        json.dump(d, open(ident, "w"))
    elif bad == "modules":
        rec["modules"]["devision.infer"] = "0" * 64
    elif bad == "target":
        rec["to_version"] = "0.3.1"
    elif bad == "lock":
        open(lock, "a").write("other pin\n")
    else:
        w = wheel(dist / "devision-0.3.0-py3-none-any.whl", change={"devision/infer.py": "def decide(): return 2\n"})
    assert any(why in x for x in pb.record_problems(rec, ident, w, lock=lock))


@pytest.mark.parametrize("bad,why", [("version", "installed version '0.3.0.dev1'"), ("python", "python '3.12.0' differs"),
                                     ("content", "content differs"), ("extra", "not in the frozen reference")])
def test_installed_refusals(ws, bad, why):
    pb, ident, src, lock, dist, code = ws
    rec = pb.make(ident, wheel(dist / "devision-0.3.0-py3-none-any.whl"), src, lock=lock)
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


def test_install_check_consumes_only_a_valid_hand_over(ws, monkeypatch, tmp_path):
    pb, ident, src, lock, dist, code = ws
    ic = load("install_check_pb", "scripts/release/install_check.py")
    w = wheel(dist / "devision-0.3.0-py3-none-any.whl")
    rec = pb.make(ident, w, src, lock=lock)
    bridge = dist / "PACKAGE_BRIDGE.json"
    bridge.write_text(json.dumps(rec))
    real = ic._package_bridge

    def with_lock():
        m = real()
        orig = m.record_problems
        monkeypatch.setattr(m, "record_problems", lambda r, i, wh=None, lock_=None: orig(r, i, wh, lock=lock))
        return m
    monkeypatch.setattr(ic, "_package_bridge", with_lock)
    monkeypatch.setattr(ic, "case_problems", lambda cases: [])
    monkeypatch.setattr(ic, "identity_problems", lambda *a, **k: [])
    seen = {}
    monkeypatch.setattr(ic, "install", lambda workdir, wheel_file=None: seen.update(wheel=wheel_file) or os.path.join(workdir, "venv", "bin", "python"))
    got = {"modules": {n: dict(m) for n, m in code["modules"].items()}, "version": "0.3.0", "python": "3.12.13", "file": "x", "status": {}}
    monkeypatch.setattr(ic, "probe", lambda py, ck, cases, cwd, isolated: {"code": got if isolated else code})
    monkeypatch.setattr(ic, "compare", lambda ref, g, cases, reference, venv, bound: seen.update(reference=reference) or [])
    os.makedirs(tmp_path / "work" / "dist")
    open(tmp_path / "work" / "dist" / "devision-0.3.0-py3-none-any.whl", "w").write("")
    assert ic.install_check("ck", [], ident, workdir=str(tmp_path / "work"), record=str(tmp_path / "r.json"), bridge=str(bridge)) == []
    assert seen["wheel"] == str(dist / "devision-0.3.0-py3-none-any.whl") and seen["reference"]["version"] == "0.3.0"
    assert seen["reference"]["modules"] == code["modules"] and json.load(open(tmp_path / "r.json"))["package_hand_over"]["sha256"]
    rec["modules"]["devision.infer"] = "0" * 64                                   # an edited record: refused before installing
    bridge.write_text(json.dumps(rec))
    monkeypatch.setattr(ic, "install", lambda *a, **k: pytest.fail("installed despite a bad hand-over"))
    problems = ic.install_check("ck", [], ident, workdir=str(tmp_path / "work2"), record=str(tmp_path / "r2.json"), bridge=str(bridge))
    assert any("package hand-over" in x for x in problems)
    with pytest.raises(SystemExit, match="only in a release check"):
        ic.install_check("ck", [], None, bridge=str(bridge))
