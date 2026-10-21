"""Prefetch "data B": the raw downloads behind scripts/data/ext/ (build_ext.sh), scripts/data/r13/sources/
(build_sources.sh), scripts/data/r13/gqa_bind.py and scripts/exp14/build.py, into exactly the files / caches the
builders read, so that the builders (which need data-v9 / v11 first) later find them and download nothing.
Builds no JSONL.

    uv run python scripts/data/prefetch_b.py --root data [--only cogent,vismin ...] [--workers 4] [--no-pictures]
    uv run python scripts/data/prefetch_b.py --list

Phases (each source keeps going when another fails; re-running skips what is complete, interrupted downloads
resume):
  1. big files: Hugging Face files (ext/common.hf_file: hub cache <root>/raw/hf) and CLEVR-CoGenT's 24.7 GB zip
     (common.download to <root>/raw/r13/cogent/), --workers at a time, largest first;
  2. small files: GitHub raw (common.download; gh-proxy.com when GitHub fails), Zenodo (spatialsense.fetch_checked,
     MD5-checked), Google Drive (whatsup.gdown, `uvx gdown`);
  3. pictures fetched one by one (skip with --no-pictures): the builder's own main() is run up to its picture
     download and stopped right after it (before any held-out check), so the selection and target paths are the
     builder's by construction -- spatialsense (Flickr), pixmo_count (Flickr, rate-limited), pixmo_points and
     pixmo_two_kind (web), vision_flan (members of the remote 36 GB zip by HTTP Range; held-out filter and COCO
     pruning are disabled here, so this fetches a superset of what the builder keeps).

Not prefetched: pictures whose selection depends on the held-out sets / identity list that build_all.sh is still
producing (tallyqa_complex and vaw Visual Genome / COCO pictures, sugarcrepe's COCO val2014 pictures under
data/coco, gqa_bind's dev / test / train pictures), and files build_all.sh itself downloads (GQA scene graphs and
parquet, VG image_data, COCO 2014 annotations): those are only reported as present / missing ("shared").
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

DATA = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(DATA, "r13"), os.path.join(DATA, "r13", "sources"), DATA, os.path.join(DATA, "ext")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402  (scripts/data/ext/common.py)
import fetch  # noqa: E402

CAULDRON = common.CAULDRON


class Spec(NamedTuple):
    """One raw download. kind: "hf" (repo, filename), "hf_glob" (repo, pattern), "url" (url -> path),
    "url_md5" (url -> path, md5 in `extra`), "gdown" (Google Drive id -> path), "shared" (path build_all.sh owns)."""
    kind: str
    a: str          # repo / url / drive id / description
    b: str          # filename / pattern / local path
    extra: str = ""


class Source(NamedTuple):
    name: str
    builder: str                          # script that reads the files
    gb: float                             # approximate download size (files only)
    specs: Callable[[str], List[Spec]]    # root -> downloads
    pictures: Optional[str] = None        # picture step (see PICTURES) or None
    note: str = ""


def _raw_ext(root: str, source: str, name: str) -> str:
    return os.path.join(common.raw_dir(root, source), name)


def _raw_r13(root: str, source: str, name: str) -> str:
    import base  # pyright: ignore[reportMissingImports]
    return os.path.join(base.raw_dir(root, source), name)


def _cauldron(*configs: str) -> Callable[[str], List[Spec]]:
    return lambda root: [Spec("hf_glob", CAULDRON, "%s/train-*.parquet" % c) for c in configs]


def _shared_scene_graphs(root: str) -> Spec:
    return Spec("shared", "GQA scene graphs (build_all.sh: v2_gqa / v2_vg_relation)",
                os.path.join(root, "raw", "gqa", "sceneGraphs.zip"))


def _shared_common(root: str) -> List[Spec]:
    return [Spec("shared", "VG image_data (build_all.sh)", os.path.join(root, "raw", "vg", "image_data.json.zip")),
            Spec("shared", "COCO 2014 annotations (build_all.sh)",
                 os.path.join(root, "raw", "coco", "annotations_trainval2014.zip"))]


def _spatialsense(root: str) -> List[Spec]:
    import spatialsense as m  # pyright: ignore[reportMissingImports]
    return [Spec("url_md5", m.ZENODO % "annotations.json", _raw_ext(root, m.SOURCE, "annotations.json"),
                 m.MD5["annotations.json"])]


def _vision_flan(root: str) -> List[Spec]:
    import vision_flan as m  # pyright: ignore[reportMissingImports]
    return [Spec("hf", m.REPO, "annotation_191-task_1k.json")]


def _objects365(root: str) -> List[Spec]:
    import objects365 as m  # pyright: ignore[reportMissingImports]
    return [Spec("hf_glob", m.ANN_REPO, "data/train-*.parquet")] + \
        [Spec("hf", m.IMG_REPO, "patch%d.tar.gz" % p) for p in OBJECTS365_PATCHES]


OBJECTS365_PATCHES = (0, 1)      # objects365.py --patches default "0,1"


def _pixmo(repo_splits: Tuple[str, ...]) -> Callable[[str], List[Spec]]:
    def f(root: str) -> List[Spec]:
        out = []
        for s in repo_splits:
            repo, split = s.split(":")
            out.append(Spec("hf_glob", repo, "data/%s-*.parquet" % split))
        return out
    return f


def _sugarcrepe(root: str) -> List[Spec]:
    import sugarcrepe as m  # pyright: ignore[reportMissingImports]
    return [Spec("url", m.URL % sub, _raw_r13(root, "sugarcrepe", sub + ".json"))
            for subs in m.SETS.values() for sub in subs] + _shared_common(root)[1:]


def _whatsup(root: str) -> List[Spec]:
    import whatsup as m  # pyright: ignore[reportMissingImports]
    out = []
    for js, js_id, tar, tar_id in m.FILES.values():
        out += [Spec("gdown", js_id, _raw_r13(root, "whatsup", js)), Spec("gdown", tar_id, _raw_r13(root, "whatsup", tar))]
    return out


def _cogent(root: str) -> List[Spec]:
    import cogent as m  # pyright: ignore[reportMissingImports]
    return [Spec("url", m.URL, _raw_r13(root, "cogent", "CLEVR_CoGenT_v1.0.zip"))]


def _tallyqa_complex(root: str) -> List[Spec]:
    import tallyqa_complex as m  # pyright: ignore[reportMissingImports]
    return [Spec("url", m.URL, _raw_r13(root, m.SOURCE, "tallyqa.zip"))] + _shared_common(root)


def _vaw(root: str) -> List[Spec]:
    import vaw as m  # pyright: ignore[reportMissingImports]
    return [Spec("url", m.URL + part, _raw_r13(root, m.SOURCE, part)) for part in ("train_part1.json", "train_part2.json")] \
        + [_shared_scene_graphs(root)] + _shared_common(root)


def _vismin(root: str) -> List[Spec]:
    import vismin as m  # pyright: ignore[reportMissingImports]
    return [Spec("hf_glob", m.REPO, "data/train/*.arrow")]


def _countercurate(root: str) -> List[Spec]:
    import countercurate as m  # pyright: ignore[reportMissingImports]
    return [Spec("hf", "nlphuji/flickr30k", "flickr30k-images.zip"),
            Spec("hf", m.REPO, "clip_sample_data/train_left_right_.csv"),
            Spec("hf", m.REPO, "clip_sample_data/train_counterfactuals.csv"),
            Spec("hf", m.REPO, "train_data.tar.gz")]


def _countbench(root: str) -> List[Spec]:
    import countbench as m  # pyright: ignore[reportMissingImports]
    return [Spec("hf_glob", m.REPO, "data/test-*.parquet")]


def _gqa_bind(root: str) -> List[Spec]:
    from v2_gqa import TRAIN_SHARDS  # pyright: ignore[reportMissingImports]
    out = [_shared_scene_graphs(root)] + _shared_common(root)
    for s in ("train", "val"):
        out.append(Spec("shared", "GQA %s instructions (build_all.sh)" % s, "%s_balanced_instructions/%s-00000-of-00001.parquet"
                        % (s, s), fetch.GQA_REPO))
    out += [Spec("shared", "GQA train image shard (build_all.sh)", "train_balanced_images/" + sh, fetch.GQA_REPO)
            for sh in TRAIN_SHARDS]
    return out


PIXMO_POINTS = "allenai/pixmo-points"
PIXMO_COUNT = "allenai/pixmo-count"
PIXMO_BENCH = (PIXMO_COUNT + ":validation", PIXMO_COUNT + ":test")

SOURCES: List[Source] = [
    # ---- ext (build_ext.sh)
    Source("iconqa", "ext/iconqa.py", 0.33, _cauldron("iconqa")),
    Source("figureqa", "ext/figureqa.py", 2.22, _cauldron("figureqa")),
    Source("clevr", "ext/clevr.py", 13.27, _cauldron("clevr")),
    Source("clevr_math", "ext/clevr.py", 0.02, _cauldron("clevr_math", "clevr"),
           note="pictures come from the clevr subset"),
    Source("tallyqa", "ext/tallyqa.py", 4.66, _cauldron("tallyqa")),
    Source("dvqa", "ext/chartmap.py", 4.30, _cauldron("dvqa")),
    Source("mapqa", "ext/chartmap.py", 3.31, _cauldron("mapqa")),
    Source("superclevr", "ext/superclevr.py", 12.28,
           lambda root: [Spec("hf", "RyanWW/Super-CLEVR", "superCLEVR_questions_30k.json"),
                         Spec("hf", "RyanWW/Super-CLEVR", "images.zip")]),
    Source("visonlyqa", "ext/visonlyqa.py", 1.21,
           lambda root: [Spec("hf_glob", "ryokamoi/VisOnlyQA_Train", "data/syntheticgeometry__*.parquet")]),
    Source("spatialsense", "ext/spatialsense.py", 0.01, _spatialsense, "spatialsense",
           note="Flickr pictures (~6,800); NYU pictures only with --archive (1.8 GB Zenodo, not used by build_ext.sh)"),
    Source("snli_ve", "ext/snli_ve.py", 5.45, lambda root: [Spec("hf_glob", "J1mb0o/e-SNLI-VE", "data/train-*.parquet")]),
    Source("pixmo_count", "ext/pixmo_count.py", 0.01, _pixmo((PIXMO_COUNT + ":train",) + PIXMO_BENCH), "pixmo_count",
           note="Flickr pictures, HTTP 429 after a burst: rate-limited, re-run to get more"),
    Source("pixmo_points", "ext/pixmo_points.py", 0.20, _pixmo((PIXMO_POINTS + ":train",) + PIXMO_BENCH), "pixmo_points",
           note="60,000 sampled web pictures (SHA256 must match)"),
    Source("objects365", "ext/objects365.py", 9.26, _objects365),
    Source("vision_flan", "ext/vision_flan.py", 0.11, _vision_flan, "vision_flan",
           note="zip central directory + members by HTTP Range (superset: no held-out filter here)"),
    # ---- r13 (build_sources.sh, gqa_bind.py)
    Source("sugarcrepe", "r13/sources/sugarcrepe.py", 0.001, _sugarcrepe,
           note="its COCO val2014 pictures go to data/coco: not prefetched"),
    Source("whatsup", "r13/sources/whatsup.py", 0.1, _whatsup, note="Google Drive via `uvx gdown`"),
    Source("countbench", "r13/sources/countbench.py", 0.02, _countbench),
    Source("cogent", "r13/sources/cogent.py", 24.72, _cogent),
    Source("tallyqa_complex", "r13/sources/tallyqa_complex.py", 0.006, _tallyqa_complex,
           note="COCO / VG pictures depend on the identity list: not prefetched"),
    Source("vaw", "r13/sources/vaw.py", 0.19, _vaw, note="VG pictures depend on the identity list: not prefetched"),
    Source("pixmo_two_kind", "r13/sources/pixmo_two_kind.py", 0.20, _pixmo((PIXMO_POINTS + ":train",) + PIXMO_BENCH),
           "pixmo_two_kind", note="pictures into ext/pixmo_points/images (shared cache); run after pixmo_points"),
    Source("vismin", "r13/sources/vismin.py", 39.05, _vismin),
    Source("countercurate", "r13/sources/countercurate.py", 10.40, _countercurate),
    Source("gqa_bind", "r13/gqa_bind.py", 0.0, _gqa_bind,
           note="everything it reads comes from build_all.sh; pictures depend on the identity list"),
    Source("exp14", "../exp14/build.py", 0.20, _pixmo((PIXMO_POINTS + ":train",)),
           note="globs the pixmo-points hub cache; pictures from data/r13/pixmo_two_kind"),
]
BY_NAME = {s.name: s for s in SOURCES}


# ---------------------------------------------------------------- files

_lock = threading.Lock()


def _say(*a) -> None:
    with _lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def hf_names(repo: str, pattern: str) -> List[str]:
    """The files of a dataset repo matching `pattern`, as common.hf_files lists them."""
    import fnmatch
    from huggingface_hub import HfApi
    names = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern))
    if not names:
        raise SystemExit("no file of %s matches %s" % (repo, pattern))
    return names


def hf_sizes(repo: str) -> Dict[str, int]:
    from huggingface_hub import HfApi
    try:
        info = HfApi().dataset_info(repo, files_metadata=True)
        return {s.rfilename: s.size or 0 for s in info.siblings or []}
    except Exception:       # noqa: BLE001  sizes only order the work
        return {}


def hf_cached(root: str, repo: str, filename: str) -> Optional[str]:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(repo, filename, cache_dir=os.path.join(root, "raw", "hf"), repo_type="dataset")
    return p if isinstance(p, str) and os.path.exists(p) else None


def github_download(url: str, path: str) -> str:
    """common.download; when GitHub raw fails, the same file through gh-proxy.com (same target path)."""
    try:
        return common.download(url, path)
    except (SystemExit, OSError) as e:
        if "raw.githubusercontent.com" not in url:
            raise
        _say("github failed (%s), trying gh-proxy.com" % e)
        return common.download("https://gh-proxy.com/" + url, path)


class Job(NamedTuple):
    key: tuple          # dedup key
    kind: str
    size: int
    run: Callable[[], str]
    label: str


def expand(root: str, spec: Spec, sizes: Dict[str, Dict[str, int]]) -> List[Job]:
    """The download jobs of one spec (an hf_glob becomes one job per file)."""
    if spec.kind in ("hf", "hf_glob"):
        names = hf_names(spec.a, spec.b) if spec.kind == "hf_glob" else [spec.b]
        if spec.a not in sizes:
            sizes[spec.a] = hf_sizes(spec.a)
        return [Job(("hf", spec.a, n), "big", sizes[spec.a].get(n, 0),
                    (lambda n=n: hf_cached(root, spec.a, n) or common.hf_file(root, spec.a, n)), "%s/%s" % (spec.a, n))
                for n in names]
    if spec.kind == "url":
        big = "fbaipublicfiles" in spec.a
        dl = common.download if big else github_download
        return [Job(("file", spec.b), "big" if big else "small", 25 << 30 if big else 0,
                    lambda: dl(spec.a, spec.b), spec.a)]
    if spec.kind == "url_md5":
        import spatialsense  # pyright: ignore[reportMissingImports]
        return [Job(("file", spec.b), "small", 0, lambda: spatialsense.fetch_checked(spec.a, spec.b, spec.extra), spec.a)]
    if spec.kind == "gdown":
        import whatsup  # pyright: ignore[reportMissingImports]
        return [Job(("file", spec.b), "small", 0, lambda: whatsup.gdown(spec.a, spec.b), "gdrive:" + spec.a)]
    return []


def shared_present(root: str, spec: Spec) -> bool:
    if spec.extra:      # a hub file
        return hf_cached(root, spec.extra, spec.b) is not None
    return os.path.exists(spec.b)


# ---------------------------------------------------------------- pictures: the builder's own main, stopped early

class _Stop(Exception):
    pass


def run_until(module, attr: str, argv: List[str], patches: Optional[dict] = None):
    """Run module.main(argv) with module.<attr> wrapped: the real function runs, then main is stopped. Returns the
    real function's result, or None when main returned before reaching it (e.g. already built). `patches` replace
    other module attributes for the run."""
    real, got = getattr(module, attr), {}
    saved = {k: getattr(module, k) for k in (patches or {})}

    def wrapped(*a, **k):
        got["result"] = real(*a, **k)
        raise _Stop
    setattr(module, attr, wrapped)
    for k, v in (patches or {}).items():
        setattr(module, k, v)
    try:
        module.main(argv)
    except _Stop:
        return got.get("result")
    finally:
        setattr(module, attr, real)
        for k, v in saved.items():
            setattr(module, k, v)
    return None


class _NoneHeld:
    """Stands in for HeldOut while vision_flan selects its pictures (the held-out sets are not built yet)."""

    def __init__(self, root: str):
        pass

    def is_held(self, image_id: str) -> bool:
        return False


def pictures(root: str, step: str, minutes: float) -> Tuple[str, int]:
    """(summary, failures) of one picture step."""
    argv = ["--root", root]
    if step == "spatialsense":
        import spatialsense as m  # pyright: ignore[reportMissingImports]
        failed = run_until(m, "fetch_pictures", argv)
        return ("built already" if failed is None else "%d failed" % len(failed)), len(failed or ())
    if step == "pixmo_count":
        import pixmo_count as m  # pyright: ignore[reportMissingImports]
        r = run_until(m, "get_pictures", argv + ["--download-minutes", str(minutes)])
    elif step == "pixmo_points":
        import pixmo_points as m  # pyright: ignore[reportMissingImports]
        r = run_until(m, "get_pictures", argv)
    elif step == "pixmo_two_kind":
        import pixmo_two_kind as m  # pyright: ignore[reportMissingImports]
        r = run_until(m, "get_pictures", argv + ["--fetch-minutes", str(minutes)])
    elif step == "vision_flan":
        import vision_flan as m  # pyright: ignore[reportMissingImports]
        failed = run_until(m, "fetch_members", argv, {"HeldOut": _NoneHeld, "coco_objects": lambda root: {},
                                                      "prune_present": lambda r, present: (r, None)})
        return ("built already" if failed is None else "%d members failed" % len(failed)), len(failed or ())
    else:
        raise ValueError(step)
    if r is None:
        return "built already", 0
    ok, stats = r
    left = stats.get("not downloaded yet (rate limited / time limit)", 0)
    return "%d usable of %d, %d dead, %d left for a re-run" % (len(ok), stats["pictures"], stats.get("dead link", 0),
                                                              left), stats.get("dead link", 0)


def picture_dir(root: str, step: str) -> str:
    return os.path.join(root, "ext", "pixmo_points" if step == "pixmo_two_kind" else step, "images")


def dir_bytes(path: str) -> int:
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return total


# ---------------------------------------------------------------- main

def heartbeat(root: str, stop: threading.Event, every: float = 60) -> None:
    """Every `every` s: bytes in the hub cache (complete + .incomplete) and the rate since the last line."""
    hub = os.path.join(root, "raw", "hf")
    last, t0 = dir_bytes(hub), time.time()
    while not stop.wait(every):
        now = dir_bytes(hub)
        _say("  [hub cache %.1f GB, %.1f MB/s]" % (now / 1e9, (now - last) / 1e6 / (time.time() - t0)))
        last, t0 = now, time.time()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--only", default="", help="comma-separated source names (see --list)")
    p.add_argument("--list", action="store_true", help="print the sources and exit (no network)")
    p.add_argument("--workers", type=int, default=4, help="big files downloaded at the same time")
    p.add_argument("--no-pictures", dest="pictures", action="store_false", help="skip phase 3")
    p.add_argument("--minutes", type=float, default=0,
                   help="time limit of the rate-limited PixMo picture downloads (0: none; re-run for more)")
    a = p.parse_args(argv)
    names = [n for n in a.only.split(",") if n] or [s.name for s in SOURCES]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise SystemExit("unknown source(s) %s; known: %s" % (unknown, ", ".join(BY_NAME)))
    chosen = [BY_NAME[n] for n in names]
    if a.list:
        print("%-16s %8s  %-32s %s" % ("source", "~GB", "builder", "pictures / note"))
        for s in chosen:
            print("%-16s %8.2f  %-32s %s%s" % (s.name, s.gb, s.builder, "pictures; " if s.pictures else "", s.note))
        print("total ~%.0f GB of files (+ pictures)" % sum(s.gb for s in chosen))
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    t0 = time.time()
    report: Dict[str, dict] = {s.name: {"status": [], "bytes": 0, "files": 0, "failures": 0} for s in chosen}
    jobs: Dict[tuple, Job] = {}
    owners: Dict[tuple, List[str]] = {}
    sizes: Dict[str, Dict[str, int]] = {}
    for s in chosen:
        try:
            for spec in s.specs(a.root):
                if spec.kind == "shared":
                    have = shared_present(a.root, spec)
                    report[s.name]["status"].append("shared %s: %s" % (spec.a, "present" if have else "missing"))
                    continue
                for j in expand(a.root, spec, sizes):
                    jobs.setdefault(j.key, j)
                    owners.setdefault(j.key, []).append(s.name)
        except (Exception, SystemExit) as e:      # noqa: BLE001
            report[s.name]["status"].append("planning failed: %s" % e)
            report[s.name]["failures"] += 1
    done: Dict[tuple, Optional[str]] = {}

    def one(j: Job) -> None:
        start = time.time()
        _say("start %s%s" % (j.label, " (%.2f GB)" % (j.size / 1e9) if j.size else ""))
        try:
            path = j.run()
            done[j.key] = path
            _say("done  %s: %.2f GB in %.0f s" % (j.label, os.path.getsize(path) / 1e9, time.time() - start))
        except (Exception, SystemExit) as e:      # noqa: BLE001
            done[j.key] = None
            _say("FAILED %s: %s" % (j.label, e))

    big = sorted((j for j in jobs.values() if j.kind == "big"), key=lambda j: -j.size)
    small = [j for j in jobs.values() if j.kind == "small"]
    _say("phase 1: %d big files (~%.1f GB listed), %d at a time" % (len(big), sum(j.size for j in big) / 1e9, a.workers))
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.root, stop), daemon=True).start()
    with ThreadPoolExecutor(max(1, a.workers)) as pool:
        for f in as_completed([pool.submit(one, j) for j in big]):
            f.result()
    stop.set()
    _say("phase 2: %d small files" % len(small))
    for j in small:
        one(j)
    for key, path in done.items():
        for n in owners[key]:
            r = report[n]
            if path:
                r["files"] += 1
                r["bytes"] += os.path.getsize(path)
            else:
                r["failures"] += 1
    if a.pictures:
        _say("phase 3: pictures")
        for s in chosen:
            if not s.pictures:
                continue
            _say("pictures: %s" % s.name)
            try:
                summary, failures = pictures(a.root, s.pictures, a.minutes)
            except (Exception, SystemExit) as e:      # noqa: BLE001
                summary, failures = "pictures failed: %s" % e, 1
            r = report[s.name]
            r["status"].append("pictures: " + summary)
            r["failures"] += failures
            r["picture_bytes"] = dir_bytes(picture_dir(a.root, s.pictures))
    print("\n%-16s %6s %9s %9s %8s  %s" % ("source", "files", "files GB", "pics GB", "failures", "status"))
    for s in chosen:
        r = report[s.name]
        status = "; ".join(["ok" if not r["failures"] else "incomplete: re-run"] + r["status"])
        print("%-16s %6d %9.2f %9.2f %8d  %s" % (s.name, r["files"], r["bytes"] / 1e9, r.get("picture_bytes", 0) / 1e9,
                                                 r["failures"], status))
    unique = sum(os.path.getsize(x) for x in done.values() if x)
    print("total %.1f GB of files (each counted once), %d failed downloads, %.1f h"
          % (unique / 1e9, sum(x is None for x in done.values()), (time.time() - t0) / 3600))


if __name__ == "__main__":
    main()
