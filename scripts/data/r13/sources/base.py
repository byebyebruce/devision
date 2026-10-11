"""Shared pieces of the round-13 sources (scripts/data/r13/sources/; plan docs/experiments/2026-10-08-r13-binding-plan.md
1.4 and 1.5). Builds on the extension pack's helpers (scripts/data/ext/common.py: resumable downloads, sample
constructors, malformed-sample checks, question-only baselines).

Layout:
  <root>/raw/r13/<source>/          downloads (kept; interrupted ones resume with HTTP Range)
  <root>/r13/<source>/images/       pictures a source ships inside its files or that are fetched one by one
  <root>/r13/<source>/train.jsonl   training questions; MANIFEST.json (counts per step and rule); REPORT.md
  <root>/r13/bench/<name>.jsonl     evaluation-only sets (+ <name>.MANIFEST.json)
  runs/r13/spot/<source>_*.png      contact sheets for the spot checks (looked at by a person, results in REPORT.md)

Every training sample carries `group` = the canonical identity of its source picture (edited pictures inherit
their source's), so pairs stay together and the identity list (scripts/data/r13/identity.py) can check them.
Pictures held out anywhere in the project are dropped (`Identity.held`).
"""
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(os.path.dirname(HERE))            # scripts/data
sys.path.insert(0, os.path.join(DATA, "ext"))
sys.path.insert(0, DATA)
sys.path.insert(0, os.path.dirname(HERE))                # scripts/data/r13 (identity.py)

from common import (choice, download, hf_file, hf_files, noul, problems, question_only_baseline,  # noqa: E402,F401
                    save_bytes, sha256, fetch_pictures)

Sample = dict
REPO_ROOT = os.path.dirname(os.path.dirname(DATA))


def raw_dir(root: str, source: str) -> str:
    d = os.path.join(root, "raw", "r13", source)
    os.makedirs(d, exist_ok=True)
    return d


def out_dir(root: str, source: str) -> str:
    d = os.path.join(root, "r13", source)
    os.makedirs(d, exist_ok=True)
    return d


def spot_dir() -> str:
    d = os.path.join(REPO_ROOT, "runs", "r13", "spot")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- picture identity

class Identity:
    """Picture identity and held-out checks. Uses scripts/data/r13/identity.py (the project-wide identity list)
    when it is there; otherwise ext/common.HeldOut (every evaluation / monitoring / calibration set by COCO / VG
    identity) plus the rule that all of COCO val2014 is held out (our test sets and POPE sit on it, and COCO 2017
    train / val2017 contain val2014 pictures)."""

    def __init__(self, root: str):
        self.root = root
        from fetch import vg_to_coco  # pyright: ignore[reportMissingImports]
        self.v2c = vg_to_coco(root)
        self.coco_split = None
        self._ident = None
        try:
            import identity  # pyright: ignore[reportMissingImports]
            self._ident = identity.load(root) if _takes_root(identity.load) else identity.load()
            self.backend = "identity.py"
        except Exception as e:      # not written yet / no identity.json yet
            self.backend = "ext HeldOut (identity.py unavailable: %s)" % type(e).__name__
        from common import HeldOut
        self._held = HeldOut(root)

    def canonical(self, image_id: str) -> str:
        if self._ident is not None and hasattr(self._ident, "canonical"):
            return self._ident.canonical(image_id)
        if self._ident is not None and callable(getattr(sys.modules.get("identity"), "canonical", None)):
            return sys.modules["identity"].canonical(image_id)
        return self.v2c.get(image_id, image_id)

    def _coco_split(self) -> Dict[int, str]:
        if self.coco_split is None:
            self.coco_split = coco2014_split(self.root)
        return self.coco_split

    def held(self, image_id: str) -> bool:
        """True when the picture (any of its ids) is held out: in an evaluation / monitoring / calibration set,
        or a COCO val2014 picture."""
        c = self.canonical(image_id)
        if c.startswith("coco:") and self._coco_split().get(int(c[5:])) == "val2014":
            return True
        if self._held.is_held(image_id) or self._held.is_held(c):
            return True
        if c in self._r13_held():
            return True
        mod = sys.modules.get("identity")
        for name in ("held", "is_held"):
            f = getattr(self._ident, name, None) or getattr(mod, name, None)
            if callable(f):
                try:
                    if f(c):
                        return True
                except Exception:
                    pass
        return False

    def _r13_held(self) -> set:
        """Canonical ids of round 13's own dev / test sets and benches (data/r13/**/dev_*.jsonl, test_*.jsonl,
        data/r13/bench/*.jsonl): build those first."""
        if getattr(self, "_r13", None) is None:
            import glob
            ids = set()
            for f in sorted(glob.glob(os.path.join(self.root, "r13", "**", "*.jsonl"), recursive=True)):
                if os.path.basename(f).startswith(("dev_", "test_")) or os.sep + "bench" + os.sep in f:
                    for line in open(f):
                        if line.strip():
                            ids.add(self.canonical(json.loads(line)["image_id"]))
            self._r13 = ids
        return self._r13

    def seen(self, image_id: str) -> Optional[bool]:
        """Project-history "seen" flag when identity.py offers it, else None."""
        mod = sys.modules.get("identity")
        f = getattr(self._ident, "seen", None) or getattr(mod, "seen", None)
        return bool(f(self.canonical(image_id))) if callable(f) else None

    def near_held(self, rels: Iterable[str], cache_name: str) -> set:
        """Pictures among `rels` (paths under root) within dHash 2 of a held-out photo: the project's held-out
        sets (ext HeldOut) and round 13's benches / dev / test sets (photos only; CLEVR renders skipped)."""
        rels = set(rels)
        out = self._held.near_held(rels, "r13_" + cache_name)
        extra = self._r13_hashes()
        if extra:
            from v9_mix import near_held  # pyright: ignore[reportMissingImports]
            cache = os.path.join(self.root, "raw", "ext", "r13_" + cache_name + ".hashes.json")
            known = {k: int(v) for k, v in json.load(open(cache)).items()} if os.path.exists(cache) else {}
            out |= near_held({r: known[r] for r in rels if r in known}, extra)
        return out

    def _r13_hashes(self) -> Dict[str, int]:
        if getattr(self, "_r13h", None) is None:
            import glob
            import hashlib
            from v9_mix import hashes  # pyright: ignore[reportMissingImports]
            files = sorted(f for f in glob.glob(os.path.join(self.root, "r13", "**", "*.jsonl"), recursive=True)
                           if os.path.basename(f).startswith(("dev_", "test_")) or os.sep + "bench" + os.sep in f)
            rels = sorted({json.loads(l)["image"] for f in files for l in open(f) if l.strip()})
            rels = [r for r in rels if "cogent" not in r and "/synth" not in r]
            key = hashlib.sha256("\n".join(rels).encode()).hexdigest()
            cache = os.path.join(raw_dir(self.root, ""), "r13_held_hashes.json")
            if os.path.exists(cache) and json.load(open(cache)).get("key") == key:
                self._r13h = {k: int(v) for k, v in json.load(open(cache))["hashes"].items()}
            else:
                self._r13h = hashes(self.root, rels) if rels else {}
                json.dump({"key": key, "hashes": {k: str(v) for k, v in self._r13h.items()}}, open(cache, "w"))
        return self._r13h


def _takes_root(f) -> bool:
    import inspect
    try:
        return len(inspect.signature(f).parameters) >= 1
    except (TypeError, ValueError):
        return False


def coco2014_split(root: str) -> Dict[int, str]:
    """COCO image id -> "train2014" | "val2014" (from the 2014 instance annotations, cached as a small JSON)."""
    cache = os.path.join(raw_dir(root, "coco"), "split2014.json")
    if os.path.exists(cache):
        return {int(k): v for k, v in json.load(open(cache)).items()}
    out = {}
    for split, names in coco_instances(root).items():
        for i in names["images"]:
            out[i] = split
    json.dump({str(k): v for k, v in out.items()}, open(cache, "w"))
    return out


_INSTANCES: dict = {}


def coco_instances(root: str) -> Dict[str, dict]:
    """{"train2014"|"val2014": {"images": {id: (w, h)}, "counts": {id: Counter(category name)},
    "boxes": {id: [(name, [x, y, w, h])]}}} from data/raw/coco/annotations_trainval2014.zip (cached in memory)."""
    if _INSTANCES:
        return _INSTANCES
    import zipfile
    from fetch import coco_annotations  # pyright: ignore[reportMissingImports]
    with zipfile.ZipFile(coco_annotations(root)) as z:
        for split in ("train2014", "val2014"):
            d = json.load(z.open("annotations/instances_%s.json" % split))
            cats = {c["id"]: c["name"] for c in d["categories"]}
            counts: Dict[int, Counter] = {}
            boxes: Dict[int, list] = {}
            for a in d["annotations"]:
                counts.setdefault(a["image_id"], Counter())[cats[a["category_id"]]] += 1
                boxes.setdefault(a["image_id"], []).append((cats[a["category_id"]], a["bbox"], a.get("iscrowd", 0)))
            _INSTANCES[split] = {"images": {i["id"]: (i["width"], i["height"]) for i in d["images"]},
                                 "counts": counts, "boxes": boxes}
    return _INSTANCES


def coco_rel(split: str, image_id: int) -> str:
    return "coco/%s/COCO_%s_%012d.jpg" % (split, split, image_id)


# ---------------------------------------------------------------- Visual Genome pictures

VG_IMAGE_DATA = None


def vg_image_data(root: str) -> Dict[int, dict]:
    global VG_IMAGE_DATA
    if VG_IMAGE_DATA is None:
        import io
        import zipfile
        from fetch import VG_IMAGE_DATA_URL, _download_json_zip  # pyright: ignore[reportMissingImports]
        _download_json_zip(VG_IMAGE_DATA_URL, os.path.join(root, "raw", "vg"))
        with zipfile.ZipFile(os.path.join(root, "raw", "vg", "image_data.json.zip")) as z:
            (name,) = [n for n in z.namelist() if n.endswith(".json")]
            VG_IMAGE_DATA = {r["image_id"]: r for r in json.load(io.TextIOWrapper(z.open(name), encoding="utf-8"))}
    return VG_IMAGE_DATA


def vg_picture(root: str, vg_id: int) -> Optional[str]:
    """Path (under root) of a VG picture already on disk: GQA's copy (gqa/images/<id>.jpg), the COCO copy of
    the same picture, or a fetched one (r13/vg/<id>.jpg). None when none is there yet."""
    for rel in ("gqa/images/%d.jpg" % vg_id, "r13/vg/%d.jpg" % vg_id):
        if os.path.exists(os.path.join(root, rel)):
            return rel
    return None


def fetch_vg(root: str, vg_ids: Sequence[int], workers: int = 16) -> Dict[int, str]:
    """vg id -> picture path for every id, fetching from Visual Genome's server (image_data.json's url) the ones
    not on disk (to r13/vg/<id>.jpg). Ids that fail are left out."""
    data = vg_image_data(root)
    out, todo = {}, []
    for i in sorted(set(vg_ids)):
        rel = vg_picture(root, i)
        if rel:
            out[i] = rel
        elif i in data:
            todo.append(i)
    failed = fetch_pictures(root, todo, lambda i: data[i]["url"], lambda i: "r13/vg/%d.jpg" % i, "r13 vg", workers)
    for i in todo:
        rel = "r13/vg/%d.jpg" % i
        if rel not in failed and os.path.exists(os.path.join(root, rel)):
            out[i] = rel
    return out


def fetch_coco(root: str, items: Sequence[tuple], workers: int = 16) -> Dict[tuple, str]:
    """(split, id) -> path for COCO 2014 pictures, fetching the missing ones from images.cocodataset.org."""
    rel = {it: coco_rel(*it) for it in set(items)}
    todo = [it for it, r in rel.items() if not os.path.exists(os.path.join(root, r))]
    failed = fetch_pictures(root, todo, lambda it: "http://images.cocodataset.org/%s/COCO_%s_%012d.jpg"
                            % (it[0], it[0], it[1]), lambda it: rel[it], "r13 coco", workers)
    return {it: r for it, r in rel.items() if r not in failed and os.path.exists(os.path.join(root, r))}


def picture_ok(path: str) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- contact sheets

def contact_sheet(root: str, items: List[dict], path: str, cols: int = 5, cell: int = 300) -> str:
    """A grid of pictures with their text under each: items = [{"image": rel, "text": str, "boxes"?: [[x,y,w,h]]
    in pixels of the original picture}]. Written to `path` (PNG)."""
    from PIL import Image, ImageDraw, ImageFont
    rows = (len(items) + cols - 1) // cols
    text_h = 110
    sheet = Image.new("RGB", (cols * cell, rows * (cell + text_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for k, it in enumerate(items):
        x0, y0 = (k % cols) * cell, (k // cols) * (cell + text_h)
        try:
            with Image.open(os.path.join(root, it["image"])) as im:
                im = im.convert("RGB")
                w, h = im.size
                s = min(cell / w, cell / h)
                im2 = im.resize((max(1, int(w * s)), max(1, int(h * s))))
                d2 = ImageDraw.Draw(im2)
                for b in it.get("boxes") or []:
                    d2.rectangle([b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s], outline="red", width=3)
                sheet.paste(im2, (x0, y0))
        except OSError:
            draw.text((x0 + 5, y0 + 5), "missing picture", fill="red", font=font)
        draw.text((x0 + 3, y0 + cell + 2), "%d. " % (k + 1) + wrap(it["text"], 44), fill="black", font=font)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)
    return path


def wrap(text: str, width: int, lines: int = 7) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width)[:lines])


def describe(r: Sample) -> str:
    """'question [options] -> gold' for a contact sheet."""
    q = r["questions"]["q"]
    opts = list(q.get("criteria") or {}) if q["type"] == "choice" else ["no", "yes"]
    return "%s %s -> %s" % (q["instructions"], "/".join(opts), r["answer_key"])


# ---------------------------------------------------------------- writing

def check(rows: List[Sample], root: str, label: str) -> None:
    bad = [(r["id"], p) for r in rows for p in [problems(r, root)] if p]
    if bad:
        raise SystemExit("%s: %d malformed samples, e.g. %s" % (label, len(bad), bad[:3]))
    dup = [i for i, n in Counter(r["id"] for r in rows).items() if n > 1]
    if dup:
        raise SystemExit("%s: %d duplicate ids, e.g. %s" % (label, len(dup), dup[:3]))


def write_jsonl(path: str, rows: List[Sample]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def summary(rows: List[Sample], path: str) -> dict:
    return {
        "questions": len(rows), "pictures": len({r["image"] for r in rows}),
        "groups": len({r.get("group") for r in rows}),
        "types": dict(Counter(r["questions"]["q"]["type"] for r in rows)),
        "kinds": dict(Counter(r["kind"] for r in rows).most_common()),
        "answers_noul": dict(Counter(r["answer_key"] for r in rows if r["questions"]["q"]["type"] == "noul")),
        "options": dict(sorted(Counter(len(r["gold"]["q"]["probabilities"]) for r in rows
                                       if r["questions"]["q"]["type"] == "choice").items())),
        "with_boxes": sum(1 for r in rows if r.get("boxes")),
        "question_only_baseline": [question_only_baseline(rows, "kind"), question_only_baseline(rows, "wording")]
        if rows else [],
        "sha256": sha256(path)}


def finish(root: str, source: str, rows: List[Sample], steps: dict, dropped: dict, extra: Optional[dict] = None,
           report: str = "") -> None:
    """Check, write <root>/r13/<source>/train.jsonl, MANIFEST.json and REPORT.md."""
    check(rows, root, source)
    d = out_dir(root, source)
    path = os.path.join(d, "train.jsonl")
    write_jsonl(path, rows)
    manifest = {"source": source, "steps": steps, "dropped": dropped, **summary(rows, path), **(extra or {})}
    json.dump(manifest, open(os.path.join(d, "MANIFEST.json"), "w"), indent=1)
    if report:
        open(os.path.join(d, "REPORT.md"), "w").write(report)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=1)[:4000])


def finish_bench(root: str, name: str, rows: List[Sample], extra: Optional[dict] = None) -> None:
    check(rows, root, name)
    path = os.path.join(root, "r13", "bench", name + ".jsonl")
    write_jsonl(path, rows)
    manifest = {"set": name, "use": "evaluation only (never training)", **summary(rows, path), **(extra or {})}
    json.dump(manifest, open(os.path.join(root, "r13", "bench", name + ".MANIFEST.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=1)[:3000])


def already_built(root: str, rel: str, force: bool) -> bool:
    p = os.path.join(root, rel)
    if not force and os.path.exists(p):
        print("%s: already built; --force to rebuild" % p)
        return True
    return False


def base_args(doc: Optional[str]):
    import argparse
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--force", action="store_true", help="rebuild the output even if it exists (downloads are kept)")
    p.add_argument("--seed", type=int, default=0)
    return p


def sample_spot(rows: List[Sample], n: int, seed: int = 0, key=None) -> List[Sample]:
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    return pool[:n]


# ---------------------------------------------------------------- members of a remote zip (HTTP Range)

class RangeFile:
    """A read-only, seekable file over an HTTP URL that supports Range requests, so `zipfile.ZipFile` can list
    a remote archive and extract single members without downloading the whole archive."""

    def __init__(self, url: str, block: int = 1 << 20):
        import urllib.request
        self.url, self.pos, self.block = url, 0, block
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache: Dict[int, bytes] = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        import time
        import urllib.request
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (start, end - 1)})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) != end - start:
                    raise OSError("short read %d of %d" % (len(data), end - start))
                return data
            except OSError:
                time.sleep(2 * (attempt + 1))
        raise OSError("could not read %s bytes %d-%d" % (self.url, start, end))

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n > self.block:                      # big reads go straight through
            data = self._get(self.pos, self.pos + n)
        else:
            out = b""
            while len(out) < n:
                b = (self.pos + len(out)) // self.block
                if b not in self._cache:
                    if len(self._cache) > 64:
                        self._cache.clear()
                    self._cache[b] = self._get(b * self.block, min(self.size, (b + 1) * self.block))
                chunk = self._cache[b]
                off = (self.pos + len(out)) - b * self.block
                out += chunk[off: off + n - len(out)]
            data = out
        self.pos += len(data)
        return data

    def close(self):
        pass


def remote_zip(url: str, index_cache: Optional[str] = None):
    """zipfile.ZipFile over a remote archive (only the central directory is read up front)."""
    import zipfile
    return zipfile.ZipFile(RangeFile(url))


def zip_index(url: str, cache: str) -> Dict[str, list]:
    """name -> [header_offset, compress_size, file_size, compress_type] of a remote zip, cached as JSON."""
    if os.path.exists(cache):
        return json.load(open(cache))
    z = remote_zip(url)
    idx = {i.filename: [i.header_offset, i.compress_size, i.file_size, i.compress_type] for i in z.infolist()}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(idx, open(cache + ".tmp", "w"))
    os.replace(cache + ".tmp", cache)
    return idx


def zip_member(url: str, entry: list) -> bytes:
    """The bytes of one member of a remote zip (`entry` from zip_index), with one or two Range requests."""
    import struct
    import zlib
    off, csize, size, method = entry
    f = RangeFile.__new__(RangeFile)
    f.url = url
    head = f._get(off, off + 30)
    if head[:4] != b"PK\x03\x04":
        raise OSError("not a local file header at %d" % off)
    n, m = struct.unpack("<HH", head[26:30])
    data = f._get(off + 30 + n + m, off + 30 + n + m + csize) if csize else b""
    if method == 0:
        out = data
    elif method == 8:
        out = zlib.decompress(data, -15)
    else:
        raise OSError("compression %d not supported" % method)
    if len(out) != size:
        raise OSError("member size %d, expected %d" % (len(out), size))
    return out


def zip_members(root: str, url: str, idx: Dict[str, list], items: Sequence[tuple], label: str,
                workers: int = 16) -> set:
    """Extract members [(name in the zip, path under root)] that are not on disk yet; returns the failed paths."""
    from fetch import run_with_progress  # pyright: ignore[reportMissingImports]

    def one(it) -> str:
        name, rel = it
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return ""
        try:
            save_bytes(path, zip_member(url, idx[name]))
            return ""
        except Exception:
            return rel
    todo = [it for it in items if not os.path.exists(os.path.join(root, it[1]))]
    return set(run_with_progress(label, todo, one, workers)) if todo else set()
