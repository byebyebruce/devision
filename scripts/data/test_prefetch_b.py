"""prefetch_b.py plans the downloads at the paths the builders read (seam: the download helpers each builder
calls). Each builder's main() runs on an empty root with the helpers replaced by recorders (no network); every
file it asks for must be in the prefetch plan, and every planned file must be one the builder reads."""
import importlib
import os
import sys
import tarfile
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefetch_b  # noqa: E402

DATA = os.path.dirname(os.path.abspath(__file__))
MODULES = {s.name: s.builder for s in prefetch_b.SOURCES if s.builder.endswith(".py") and "exp14" not in s.builder
           and s.name != "gqa_bind"}
ARGV = {"clevr_math": ["--subset", "clevr_math"], "clevr": ["--subset", "clevr"], "dvqa": ["--subset", "dvqa"],
        "mapqa": ["--subset", "mapqa"]}


class _Held:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: set()

    def is_held(self, *a):
        return False

    def held(self, *a):
        return False


def record(monkeypatch, tmp_path, module):
    """Replace the download layer seen by `module` (and the shared helpers) with recorders; returns the list the
    calls are appended to as prefetch_b.Spec-like tuples (kind, a, b)."""
    calls = []
    fake = tmp_path / "fake.json"
    fake.write_text("{}")

    def hf_file(root, repo, filename, *a, **k):
        assert root == str(tmp_path / "root")
        calls.append(("hf", repo, filename))
        return str(fake)

    def hf_files(root, repo, pattern, *a, **k):
        calls.append(("hf_glob", repo, pattern))
        return []

    def download(url, path, *a, **k):
        calls.append(("url", url, path))
        return str(fake)

    def fetch_checked(url, path, md5, *a, **k):
        calls.append(("url_md5", url, path))
        return str(fake)

    def gdown(file_id, path):
        calls.append(("gdown", file_id, path))
        return str(fake)

    def ensure_file(url, path):
        calls.append(("shared", path))
        return str(fake)

    def no_network(*a, **k):
        raise OSError("network disabled in tests")

    import common
    import fetch
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(common, "hf_file", hf_file)
    monkeypatch.setattr(common, "hf_files", hf_files)
    monkeypatch.setattr(fetch, "ensure_file", ensure_file)
    stubs = {"hf_file": hf_file, "hf_files": hf_files, "download": download, "fetch_checked": fetch_checked,
             "gdown": gdown, "HeldOut": _Held, "Identity": _Held, "get_held": lambda root: _Held(),
             "coco2014_split": lambda root: {}}
    from huggingface_hub import HfApi
    monkeypatch.setattr(HfApi, "list_repo_files", no_network)
    ours = [mod for mod in list(sys.modules.values())
            if os.path.abspath(getattr(mod, "__file__", None) or "/").startswith(DATA)]
    for mod in set(ours) | {module}:      # the builder and the helper modules it imported names into
        for name, f in stubs.items():
            if hasattr(mod, name) and mod is not prefetch_b and mod.__name__ != __name__:
                monkeypatch.setattr(mod, name, f)
    return calls


def builder_module(name):
    rel = MODULES[name]
    import base  # noqa: F401  (puts scripts/data/r13 on the path)
    return importlib.import_module(os.path.splitext(os.path.basename(rel))[0])


def planned(root, name):
    return {(s.kind, s.a, s.b) if s.kind != "shared" else ("shared", s.b) for s in BY[name].specs(root)}


BY = prefetch_b.BY_NAME


@pytest.mark.parametrize("name", sorted(MODULES))
def test_builder_reads_only_planned_files(name, monkeypatch, tmp_path):
    m = builder_module(name)
    calls = record(monkeypatch, tmp_path, m)
    root = str(tmp_path / "root")
    try:
        m.main(["--root", root] + ARGV.get(name, []))
    except (Exception, SystemExit):     # the fake files end the run; what was asked for until then is recorded
        pass
    assert calls, "%s asked for no download" % name
    plan = planned(root, name)
    shared_ok = {c for c in calls if c[0] == "shared" and c[1].endswith(("sceneGraphs.zip", "annotations_trainval2014.zip",
                                                                           "image_data.json.zip"))}
    missing = [c for c in calls if c not in plan and c not in shared_ok]
    assert not missing, "%s reads files the prefetch does not plan: %s" % (name, missing)


@pytest.mark.parametrize("name", sorted(MODULES))
def test_every_planned_file_is_read_by_the_builder(name, monkeypatch, tmp_path):
    """A planned file the recorded run did not reach must still be named in the builder's own code."""
    m = builder_module(name)
    calls = record(monkeypatch, tmp_path, m)
    root = str(tmp_path / "root")
    try:
        m.main(["--root", root] + ARGV.get(name, []))
    except (Exception, SystemExit):
        pass
    src = open(os.path.join(DATA, MODULES[name])).read()
    reached = set(calls)
    for spec in BY[name].specs(root):
        key = (spec.kind, spec.a, spec.b) if spec.kind != "shared" else ("shared", spec.b)
        if key in reached or spec.kind == "shared":
            continue
        cauldron = spec.a == prefetch_b.CAULDRON and spec.b.split("/")[0] in (name, ) + tuple(
            c for c in ("clevr",) if 'cauldron_rows(a.root, "%s")' % c in src)
        assert spec.b in src or os.path.basename(spec.b) in src or cauldron, \
            "%s: planned %s is not something the builder reads" % (name, spec)


def test_objects365_patch_archives_and_pixmo_benchmark_splits(monkeypatch, tmp_path):
    """The two places a builder formats the file name at run time."""
    import objects365
    import pixmo_count
    calls = record(monkeypatch, tmp_path, objects365)
    empty = tmp_path / "empty.tar.gz"
    with tarfile.open(empty, "w:gz"):
        pass
    monkeypatch.setattr(objects365, "hf_file", lambda root, repo, n: calls.append(("hf", repo, n)) or str(empty))
    root = str(tmp_path / "root")
    objects365.extract(root, list(prefetch_b.OBJECTS365_PATCHES), {"x.jpg": "ext/objects365/images/x.jpg"})
    assert '--patches", default="0,1"' in open(objects365.__file__).read()
    record(monkeypatch, tmp_path, pixmo_count)
    seen = []
    monkeypatch.setattr(pixmo_count, "hf_files", lambda root, repo, pat: seen.append(("hf_glob", repo, pat)) or [])
    pixmo_count.benchmark_pictures(root)
    for name in ("objects365",):
        assert {c for c in calls if c[2].startswith("patch")} <= planned(root, name)
    for name in ("pixmo_count", "pixmo_points", "pixmo_two_kind"):
        assert set(seen) <= planned(root, name)


def test_exp14_globs_the_hub_cache_prefetch_fills():
    from huggingface_hub.file_download import repo_folder_name
    src = open(os.path.join(DATA, "..", "exp14", "build.py")).read()
    folder = repo_folder_name(repo_id=prefetch_b.PIXMO_POINTS, repo_type="dataset")
    assert "raw/hf/%s/snapshots/*/data/*.parquet" % folder in src
    assert ("hf_glob", prefetch_b.PIXMO_POINTS, "data/train-*.parquet") in planned("data", "exp14")


def test_picture_step_runs_the_builder_selection_and_stops_before_held_out(monkeypatch, tmp_path):
    """pixmo_count: the prefetch downloads exactly the rows the builder would keep, and stops before HeldOut."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pixmo_count
    root = str(tmp_path / "root")
    files = {}
    for split, rows in {"train": [("http://a/1.jpg", "s1", 3, "dog"), ("http://a/2.jpg", "s2", 12, "dog"),
                                  ("http://a/3.jpg", "s3", 2, "cat"), ("http://a/4.jpg", "s4", 1, "dog")],
                        "validation": [("http://a/3.jpg", "s3", 2, "cat")], "test": []}.items():
        path = str(tmp_path / ("%s.parquet" % split))
        pq.write_table(pa.table({"image_url": [r[0] for r in rows], "image_sha256": [r[1] for r in rows],
                                 "count": pa.array([r[2] for r in rows], pa.int64()),
                                 "label": [r[3] for r in rows],
                                 "points": [{"x": [], "y": []} for _ in rows]}), path)
        files[split] = path
    monkeypatch.setattr(pixmo_count, "hf_files", lambda r, repo, pat: [files[pat.split("/")[1].split("-")[0]]])
    asked = []
    monkeypatch.setattr(pixmo_count, "fetch_politely",
                        lambda root_, todo, *a, **k: (asked.extend(u for u, _ in todo), (set(), 0))[1])

    def held(*a, **k):
        raise AssertionError("the prefetch must stop before the held-out check")
    monkeypatch.setattr(pixmo_count, "HeldOut", held)
    summary, failures = prefetch_b.pictures(root, "pixmo_count", 0)
    assert sorted(asked) == ["http://a/1.jpg", "http://a/4.jpg"]     # count > 10 and the validation picture dropped
    assert failures == 0 and "of 2" in summary
    assert pixmo_count.get_pictures.__name__ == "get_pictures"         # restored after the run


def test_list_needs_no_network(monkeypatch, capsys):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("no network")))
    prefetch_b.main(["--list"])
    out = capsys.readouterr().out
    assert all(s.name in out for s in prefetch_b.SOURCES)
    with pytest.raises(SystemExit):
        prefetch_b.main(["--list", "--only", "nope"])

