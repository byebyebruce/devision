"""Real first-person (egocentric) everyday-task decision data for v0.5: one EPIC-KITCHENS frame + a question about
what the person is doing / which object the hand is on, with distractors taken from the same video and frame.

    uv run --no-sync python scripts/data/v05_ego.py fetch --root data      # resumable; log in data/v05/ego/fetch.log
    uv run --no-sync python scripts/data/v05_ego.py build --root data      # uses whatever frames are downloaded

Sources (survey and choice: docs/research/egocentric-data.md). hf-mirror.com answered every request with a 308 to
huggingface.co on 2026-10-27, so nothing here uses it; everything comes from ModelScope (domestic) and GitHub raw:

  * EPIC-KITCHENS VISOR (data.bris DOI 2v6cgv1x04ol22qp9rm9x2j6a7, CC BY-NC 4.0), re-hosted as ONE 30.6 GB zip in
    ModelScope dataset qiup0307/EK100-VISOR. We never download the whole zip: the central directory is read with
    HTTP range requests and only the needed members (all stored, not deflated) are fetched byte for byte by range,
    each checked against its CRC32: the sparse-annotation JSONs (train / val, 0.87 GB), frame_mapping.json, and the
    per-video frame zips GroundTruth-SparseAnnotations/rgb_frames/{val,train}/<P>/<video>.zip (3.0 + 12.2 GB;
    the VISOR test frames have no public annotations and are not fetched).
  * EPIC-KITCHENS-100 action narrations: github.com/epic-kitchens/epic-kitchens-100-annotations at commit
    ea8b4045 (EPIC_100_train.csv, EPIC_100_validation.csv, noun / verb classes).
  * EgoPlan-Bench questions (lmms-lab/EgoPlan on ModelScope, the 0.6 MB parquet only, not the 120 GB of clips):
    used to keep its EPIC-KITCHENS question windows out of our data (below), not converted (no frames without
    decoding the clips; no ffmpeg / PyAV on this machine).
  * EgoThink (lmms-lab/Egothink on ModelScope, 12 parquet files, 0.9 GB, 700 Ego4D pictures; eval-only benchmark
    with open answers) -> bench_egothink.jsonl.

Questions (one sample per question; at most 4 per frame; image_id "visor:<frame name>"):

  action    "What is the person doing right now?" Gold = the EPIC-KITCHENS-100 narration of the action segment that
            contains the frame (VISOR frame number -> EPIC rgb frame number by frame_mapping.json). Dropped when no
            segment contains the frame, when overlapping segments name different nouns, or when the gold noun is not
            among the frame's segmented objects (by noun class, or the noun word in an object name). Distractors =
            narrations of OTHER segments of the same video whose nouns share no class with the gold and none of
            whose nouns is an object a hand is touching in this frame; ones whose noun is visible in the frame come
            first (same scene, plausible), then the rest, in a fixed random order. 3..5 distractors; fewer -> drop.
  contact   "What is the person's left / right hand touching?" (or "What are the person's hands touching?" when
            both hands touch the same object). Gold = VISOR's in_contact_object of that hand (gloves count for the
            hand they are on). Distractors = other objects segmented in the same frame: not a hand / glove, not
            the gold's class, name not equal to / contained in the gold's (and vice versa), and of a class that
            hands touch in >= 15% of the frames it appears in (over all VISOR frames), so "hand-held item vs fixed
            surface" alone does not give the answer away. The other hand's object
            may be a distractor of a one-hand question. Dropped: hand not in contact / touching an unsegmented
            object / inconclusive, gold too small at 256 (bounding box under 6 px on both sides after letterbox),
            fewer than 3 distractors. 3..5 distractors.
  touching  noul "Is the person touching the <object>?" -- true for a contact gold, false for one of its
            distractors (chosen by a hash of the frame, so about half are true), at most one per frame.

Object names: VISOR's open-vocabulary names, "noun:modifier" turned into "modifier noun" ("lid:pan" -> "pan lid").

Splits by participant (= kitchen), so dev / test kitchens are never seen in training: participants ordered by
sha256("v05-ego:" + participant); taken into test until it holds >= 10% of the annotated frames, then into dev
until >= 10%; the rest is train. Decided from the annotation JSONs alone, so a partial download keeps the split.

EgoPlan-Bench overlap: 134 of VISOR's 158 videos are EgoPlan-Bench source videos. Frames inside any EgoPlan-Bench
EPIC question window (first task-progress start frame .. current_observation_frame + 300 frames) are dropped from
every split ("egoplan_window"), so EgoPlan-Bench stays free of our exact frames; video-level overlap remains and is
counted in MANIFEST.json.

Pictures: the frame (1920x1080) re-encoded at 640x360 JPEG q90 to data/v05/ego/images/visor/<video>/<name>.jpg
(the model letterboxes to 256 anyway). Options are shuffled by random.Random(sha256 of the question id); the
manifest reports gold-position balance and text-only baselines (diagnostics, never used to drop questions).
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import os
import random
import re
import struct
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

OUT = os.path.join("v05", "ego")
MS = "https://www.modelscope.cn/api/v1/datasets/%s/repo?Revision=master&FilePath=%s"
VISOR_REPO, VISOR_ZIP = "qiup0307/EK100-VISOR", "2v6cgv1x04ol22qp9rm9x2j6a7.zip"
VISOR_TOP = "2v6cgv1x04ol22qp9rm9x2j6a7/"
EK_COMMIT = "ea8b40457a400c3fffa1c7f406ef3dc169cc2522"
EK_RAW = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/%s/%%s" % EK_COMMIT
EK_FILES = ["EPIC_100_train.csv", "EPIC_100_validation.csv", "EPIC_100_noun_classes_v2.csv",
            "EPIC_100_verb_classes.csv", "README.md", "license.txt"]
EGOPLAN = ("lmms-lab/EgoPlan", "egoplan/validation-00000-of-00001.parquet")
EGOTHINK_REPO = "lmms-lab/Egothink"
EGOTHINK_CATS = ["Activity", "Forecasting", "Localization_location", "Localization_spatial", "Object_affordance",
                 "Object_attribute", "Object_existence", "Planning_assistance", "Planning_navigation",
                 "Reasoning_comparing", "Reasoning_counting", "Reasoning_situated"]
SPLIT_SALT = "v05-ego:"
TEST_FRAC = DEV_FRAC = 0.10
MIN_DISTRACTORS, MAX_DISTRACTORS = 3, 5
MIN_BOX_PX = 6.0          # gold object bbox at 256 letterbox: dropped when both sides are below this
EGOPLAN_AFTER = 300       # frames after current_observation_frame still counted as the EgoPlan question window
SAVE_SIZE = (640, 360)
GRASP_RATE = 0.05         # contact distractors: classes touched in >= 5% of the frames they appear in,
                          # the most often touched first
HAND_NAMES = {"left hand", "right hand", "hand", "left glove", "right glove", "glove"}
HAND_CLASSES = {300, 301, 302, 303}
NO_CONTACT = {"hand-not-in-contact": "hand_not_in_contact", "none-of-the-above": "contact_unsegmented",
              "inconclusive": "contact_inconclusive"}

QUESTION_ACTION = "What is the person doing right now?"
QUESTION_CONTACT = {"left": "What is the person's left hand touching?",
                    "right": "What is the person's right hand touching?",
                    "both": "What are the person's hands touching?"}
QUESTION_TOUCHING = "Is the person touching the %s?"


# ============================================================================ helpers

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rng_for(key: str) -> random.Random:
    return random.Random(int(sha(key)[:16], 16))


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def object_name(name: str) -> str:
    """VISOR / EPIC 'noun:modifier' -> 'modifier noun' ('lid:pan' -> 'pan lid'); plain names unchanged."""
    parts = [p.strip() for p in name.split(":") if p.strip()]
    return " ".join(reversed(parts)) if len(parts) > 1 else (parts[0] if parts else "")


def related(a: str, b: str) -> bool:
    """Equal names, or one contained in the other as whole words ('pan' / 'pan lid'): too close to be a distractor."""
    a, b = norm(a), norm(b)
    return a == b or (" %s " % a) in (" %s " % b) or (" %s " % b) in (" %s " % a)


def log(path: Optional[str], msg: str) -> None:
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


# ============================================================================ fetch (ModelScope range reads)

class HttpFile(io.RawIOBase):
    """A read-only seekable file over HTTP range requests (enough for zipfile to read a central directory)."""

    def __init__(self, url: str):
        super().__init__()
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
            self.url = r.geturl()
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        data = range_bytes(self.url, self.pos, min(self.size, self.pos + n) - 1)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def range_bytes(url: str, start: int, end: int, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end)})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


def visor_index(raw: str) -> List[dict]:
    """Every member of the VISOR zip with the absolute offset of its data (cached in raw/visor/index.json)."""
    path = os.path.join(raw, "visor", "index.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    f = HttpFile(MS % (VISOR_REPO, VISOR_ZIP))
    z = zipfile.ZipFile(f)
    members = []
    for info in z.infolist():
        if info.is_dir():
            continue
        members.append({"name": info.filename[len(VISOR_TOP):] if info.filename.startswith(VISOR_TOP)
                        else info.filename, "header_offset": info.header_offset, "size": info.file_size,
                        "compress_size": info.compress_size, "compress_type": info.compress_type, "crc": info.CRC})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as out:
        json.dump({"zip_size": f.size, "members": members}, out)
    return visor_index(raw)


def fetch_member(url_of, m: dict, dest: str, progress: Optional["Progress"] = None) -> str:
    """Fetch one stored / deflated zip member by range into `dest` (resumable via dest + '.part'); returns status."""
    if os.path.exists(dest) and os.path.getsize(dest) == m["size"]:
        if progress:
            progress.add(m["compress_size"], done=True)
        return "exists"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = url_of()
    head = range_bytes(url, m["header_offset"], m["header_offset"] + 29)
    sig, = struct.unpack("<I", head[:4])
    if sig != 0x04034B50:
        raise RuntimeError("bad local header for %s" % m["name"])
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = m["header_offset"] + 30 + name_len + extra_len
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > m["compress_size"]:
        os.remove(part)
        have = 0
    if progress:
        progress.add(have)
    chunk = 32 << 20
    with open(part, "ab") as out:
        while have < m["compress_size"]:
            end = min(m["compress_size"], have + chunk) - 1
            try:
                data = range_bytes(url, start + have, start + end)
            except Exception:
                url = url_of()          # the CDN link carries an expiring auth key: resolve again
                data = range_bytes(url, start + have, start + end)
            out.write(data)
            have += len(data)
            if progress:
                progress.add(len(data))
    with open(part, "rb") as f:
        payload = f.read()
    if m["compress_type"] == zipfile.ZIP_DEFLATED:
        payload = zlib.decompressobj(-15).decompress(payload)
    elif m["compress_type"] != zipfile.ZIP_STORED:
        raise RuntimeError("unsupported compression %d for %s" % (m["compress_type"], m["name"]))
    if len(payload) != m["size"] or (zlib.crc32(payload) & 0xFFFFFFFF) != m["crc"]:
        os.remove(part)
        raise RuntimeError("CRC / size mismatch for %s (part removed, run fetch again)" % m["name"])
    with open(dest, "wb") as out:
        out.write(payload)
    os.remove(part)
    if progress:
        progress.add(0, done=True)
    return "fetched"


class Progress:
    def __init__(self, total_bytes: int, total_files: int, logfile: str, every: float = 60.0):
        self.total, self.files, self.logfile, self.every = total_bytes, total_files, logfile, every
        self.bytes = self.done = 0
        self.t0 = self.last = time.time()
        self.lock = threading.Lock()

    def add(self, n: int, done: bool = False) -> None:
        with self.lock:
            self.bytes += n
            self.done += int(done)
            now = time.time()
            if now - self.last >= self.every or self.done == self.files:
                self.last = now
                rate = self.bytes / max(1.0, now - self.t0) / 1e6
                log(self.logfile, "progress %d/%d files, %.2f/%.2f GB, %.1f MB/s" % (
                    self.done, self.files, self.bytes / 1e9, self.total / 1e9, rate))


def download(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "exists"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(dest + ".part", "wb") as out:
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    out.write(b)
            os.replace(dest + ".part", dest)
            return "fetched"
        except Exception:
            if k == 4:
                raise
            time.sleep(5 * (k + 1))
    return "failed"


def cmd_fetch(args) -> None:
    root = args.root
    raw = os.path.join(root, OUT, "raw")
    logfile = os.path.join(root, OUT, "fetch.log")
    os.makedirs(raw, exist_ok=True)
    log(logfile, "fetch start (ModelScope + GitHub raw; hf-mirror not used: it redirects to huggingface.co)")
    for name in EK_FILES:
        log(logfile, "ek100 %s: %s" % (name, download(EK_RAW % name, os.path.join(raw, "ek100", name))))
    log(logfile, "egoplan questions: %s" % download(MS % EGOPLAN, os.path.join(raw, "egoplan", "egoplan_bench.parquet")))
    if not args.skip_egothink:
        for c in EGOTHINK_CATS:
            st = download(MS % (EGOTHINK_REPO, "%s/test-00000-of-00001.parquet" % c),
                          os.path.join(raw, "egothink", "%s.parquet" % c))
            log(logfile, "egothink %s: %s" % (c, st))
    index = visor_index(raw)["members"]
    url_cache = {"url": None, "t": 0.0}
    url_lock = threading.Lock()

    def url_of() -> str:
        with url_lock:
            if url_cache["url"] is None or time.time() - url_cache["t"] > 1800:
                url_cache["url"] = HttpFile(MS % (VISOR_REPO, VISOR_ZIP)).url
                url_cache["t"] = time.time()
            return url_cache["url"]

    def wanted(m: dict) -> Optional[int]:
        n = m["name"]
        if n in ("README.txt", "frame_mapping.json", "EPIC_100_noun_classes_v2.csv"):
            return 0
        if re.match(r"GroundTruth-SparseAnnotations/annotations/(train|val)/P\d+_\d+\.json$", n):
            return 1
        if not args.annotations_only and re.match(r"GroundTruth-SparseAnnotations/rgb_frames/val/P\d+/P\d+_\d+\.zip$", n):
            return 2
        if not args.annotations_only and re.match(r"GroundTruth-SparseAnnotations/rgb_frames/train/P\d+/P\d+_\d+\.zip$", n):
            return 3
        return None

    todo = sorted((m for m in index if wanted(m) is not None), key=lambda m: (wanted(m), m["compress_size"]))
    if args.max_videos:
        keep, frames = [], 0
        for m in todo:
            if wanted(m) >= 2:
                if frames >= args.max_videos:
                    continue
                frames += 1
            keep.append(m)
        todo = keep
    prog = Progress(sum(m["compress_size"] for m in todo), len(todo), logfile)
    log(logfile, "visor: %d members, %.2f GB to check / fetch" % (len(todo), prog.total / 1e9))
    errors = []

    def one(m: dict) -> None:
        try:
            fetch_member(url_of, m, os.path.join(raw, "visor", m["name"]), prog)
        except Exception as e:  # keep going; the next run resumes
            errors.append((m["name"], repr(e)))
            log(logfile, "ERROR %s: %r" % (m["name"], e))

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo))
    log(logfile, "fetch done, %d errors%s" % (len(errors), " (run fetch again to resume)" if errors else ""))


# ============================================================================ annotations -> questions

def read_segments(paths: Iterable[str]) -> Dict[str, List[dict]]:
    """EPIC-KITCHENS-100 action segments per video (train + validation CSVs)."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                out[r["video_id"]].append({
                    "id": r["narration_id"], "start": int(r["start_frame"]), "stop": int(r["stop_frame"]),
                    "narration": r["narration"].strip(), "verb_class": int(r["verb_class"]),
                    "noun": r["noun"], "noun_class": int(r["noun_class"]),
                    "noun_classes": [int(x) for x in ast.literal_eval(r["all_noun_classes"])],
                    "nouns": [str(x) for x in ast.literal_eval(r["all_nouns"])]})
    return out


def frame_number(name: str) -> int:
    m = re.search(r"frame_(\d+)\.jpg$", name)
    if not m:
        raise ValueError("no frame number in %r" % name)
    return int(m.group(1))


def bbox(segments) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for poly in segments or []:
        for pt in poly:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def box_px_256(box, width: int, height: int) -> Tuple[float, float]:
    scale = 256.0 / max(width, height)
    return (box[2] - box[0]) * scale, (box[3] - box[1]) * scale


def is_hand(a: dict) -> bool:
    return a.get("class_id") in HAND_CLASSES or norm(a.get("name", "")) in HAND_NAMES


def hand_side(a: dict) -> Optional[str]:
    n = norm(a.get("name", ""))
    if n in ("left hand", "right hand"):
        return n.split()[0]
    if n in ("left glove", "right glove"):
        return n.split()[0]
    side = str(a.get("on_which_hand") or "")
    return side.split()[0] if side.split() and side.split()[0] in ("left", "right") else None


def contact_targets(frame: dict, width: int, height: int, stats: Counter) -> List[Tuple[str, dict]]:
    """(side, gold annotation) per hand with a usable contact; 'both' when the two hands touch the same object."""
    anns = frame["annotations"]
    by_id = {a["id"]: a for a in anns}
    per_side: Dict[str, Optional[dict]] = {}
    for a in anns:
        if not is_hand(a) or "in_contact_object" not in a:
            continue
        side = hand_side(a)
        if side is None or side in per_side and per_side[side] is not None:
            continue
        c = a.get("in_contact_object")
        if c in NO_CONTACT:
            stats["contact:" + NO_CONTACT[c]] += 1
            per_side.setdefault(side, None)
            continue
        gold = by_id.get(c)
        if gold is None:
            stats["contact:contact_id_missing"] += 1
        elif is_hand(gold):
            stats["contact:touching_a_hand"] += 1
        else:
            box = bbox(gold.get("segments"))
            if box is None:
                stats["contact:no_polygon"] += 1
            elif max(box_px_256(box, width, height)) < MIN_BOX_PX:
                stats["contact:gold_below_6px"] += 1
            else:
                per_side[side] = gold
                continue
        per_side.setdefault(side, None)
    golds = {s: g for s, g in per_side.items() if g is not None}
    if len(golds) == 2 and golds["left"]["id"] == golds["right"]["id"]:
        return [("both", golds["left"])]
    return sorted(golds.items())


def contact_distractors(frame: dict, gold: dict, graspable: Optional[Dict[int, float]] = None) -> List[str]:
    """Other objects of the frame (sorted by name); with `graspable` (class -> touch rate), only those classes
    (so 'hand-held vs fixed surface' alone does not give the answer away)."""
    names: Dict[str, str] = {}
    gold_name = object_name(gold["name"])
    for a in frame["annotations"]:
        if a["id"] == gold["id"] or is_hand(a) or a.get("class_id") == gold.get("class_id"):
            continue
        if graspable is not None and a.get("class_id") not in graspable:
            continue
        n = object_name(a["name"])
        if not n or related(n, gold_name) or any(related(n, o) for o in names.values()):
            continue
        names[norm(n)] = n
    return [names[k] for k in sorted(names)]


def visible_classes(frame: dict) -> Tuple[set, List[str]]:
    anns = [a for a in frame["annotations"] if not is_hand(a)]
    return {a.get("class_id") for a in anns}, [object_name(a["name"]) for a in anns]


def touched_classes(frame: dict) -> set:
    by_id = {a["id"]: a for a in frame["annotations"]}
    out = set()
    for a in frame["annotations"]:
        g = by_id.get(a.get("in_contact_object")) if is_hand(a) else None
        if g is not None and not is_hand(g):
            out.add(g.get("class_id"))
    return out


def noun_visible(seg: dict, classes: set, names: Sequence[str]) -> bool:
    if set(seg["noun_classes"]) & classes:
        return True
    words = {object_name(n).split()[-1] for n in seg["nouns"] if object_name(n)}
    return any(w and any(w in norm(n).split() for n in names) for w in words)


def action_question(frame: dict, ek_frame: int, segments: List[dict], stats: Counter) -> Optional[Tuple[dict, List[str]]]:
    """(gold segment, ordered distractor narrations) or None (reason counted in stats)."""
    here = [s for s in segments if s["start"] <= ek_frame <= s["stop"]]
    if not here:
        stats["action:no_segment_at_frame"] += 1
        return None
    if len({s["noun_class"] for s in here}) > 1:
        stats["action:overlapping_actions"] += 1
        return None
    gold = here[0]
    classes, names = visible_classes(frame)
    if not noun_visible(gold, classes, names):
        stats["action:gold_object_not_visible"] += 1
        return None
    touched = touched_classes(frame)
    gold_nouns = set(gold["noun_classes"])
    near, far, seen = [], [], {norm(gold["narration"])}
    for s in segments:
        key = norm(s["narration"])
        if key in seen or set(s["noun_classes"]) & gold_nouns or set(s["noun_classes"]) & touched:
            continue
        seen.add(key)
        (near if noun_visible(s, classes, names) else far).append(s["narration"])
    rng = rng_for("action-d:" + frame["image"]["name"])
    rng.shuffle(near)
    rng.shuffle(far)
    ds = (near + far)[:MAX_DISTRACTORS]
    if len(ds) < MIN_DISTRACTORS:
        stats["action:fewer_than_3_distractors"] += 1
        return None
    return gold, ds


def choice_sample(qid: str, kind: str, question: str, gold: str, distractors: Sequence[str], image: str,
                  image_id: str, extra: dict) -> dict:
    opts = [gold] + list(distractors)
    assert len({norm(o) for o in opts}) == len(opts), opts
    rng_for(qid).shuffle(opts)
    return dict({"id": qid, "source": "epic-visor", "kind": kind, "image_id": image_id, "image": image,
                 "questions": {"q": {"type": "choice", "instructions": question, "criteria": dict.fromkeys(opts)}},
                 "gold": {"q": {"probabilities": {o: float(o == gold) for o in opts}}},
                 "answer": gold, "answer_position": opts.index(gold)}, **extra)


def frame_questions(frame: dict, ek_frame: Optional[int], segments: List[dict], image: str,
                    width: int, height: int, stats: Counter,
                    graspable: Optional[Dict[int, float]] = None) -> List[dict]:
    name = frame["image"]["name"].rsplit(".", 1)[0]
    video = frame["image"]["video"]
    image_id = "visor:" + name
    base = {"video": video, "participant": video.split("_")[0], "frame": name}
    out = []
    if ek_frame is None:
        stats["action:no_frame_mapping"] += 1
    else:
        got = action_question(frame, ek_frame, segments, stats)
        if got:
            gold, ds = got
            out.append(choice_sample("ego:action:%s" % name, "action", QUESTION_ACTION, gold["narration"], ds,
                                     image, image_id, dict(base, narration_id=gold["id"])))
    touching_done = False
    for side, gold in contact_targets(frame, width, height, stats):
        ds = contact_distractors(frame, gold, graspable)
        if len(ds) < MIN_DISTRACTORS:
            stats["contact:fewer_than_3_distractors"] += 1
            continue
        ds = sorted(ds)
        rng_for("contact-d:%s:%s" % (name, side)).shuffle(ds)
        if graspable:   # most often touched classes first (stable sort keeps the random order among equals)
            rate = {object_name(a["name"]): graspable.get(a.get("class_id"), 0.0) for a in frame["annotations"]}
            ds.sort(key=lambda d: -rate.get(d, 0.0))
        g = object_name(gold["name"])
        out.append(choice_sample("ego:contact:%s:%s" % (name, side), "contact", QUESTION_CONTACT[side], g,
                                 ds[:MAX_DISTRACTORS], image, image_id, dict(base, hand=side)))
        if not touching_done:
            touching_done = True
            # both hands' objects count as touched; the false object must be touched by neither hand
            touched = {object_name(a["name"]) for _, a in contact_targets(frame, width, height, Counter())}
            free = [d for d in ds if not any(related(d, t) for t in touched)]
            truth = int(sha("touch:" + name)[:8], 16) % 2 == 0 or not free
            obj = g if truth else free[0]
            out.append({"id": "ego:touching:%s" % name, "source": "epic-visor", "kind": "touching",
                        "image_id": image_id, "image": image,
                        "questions": {"q": {"type": "noul", "instructions": QUESTION_TOUCHING % obj}},
                        "gold": {"q": {"probabilities": {"true": float(truth), "false": float(not truth)}}},
                        "answer": "true" if truth else "false", **base})
    return out


# ============================================================================ splits and baselines

def graspable_classes(frames: Iterable[dict], min_rate: float = GRASP_RATE, min_frames: int = 20) -> Dict[int, float]:
    """Object classes a hand touches in >= min_rate of the frames they appear in (over all annotated frames),
    with that rate."""
    seen: Counter = Counter()
    touched: Counter = Counter()
    for fr in frames:
        seen.update({a.get("class_id") for a in fr["annotations"] if not is_hand(a)})
        touched.update(touched_classes(fr))
    return {c: touched[c] / n for c, n in seen.items() if n >= min_frames and touched[c] / n >= min_rate}


def split_participants(frames_per_participant: Dict[str, int]) -> Dict[str, str]:
    total = sum(frames_per_participant.values())
    order = sorted(frames_per_participant, key=lambda p: sha(SPLIT_SALT + p))
    out, test, dev = {}, 0, 0
    for p in order:
        if test < TEST_FRAC * total:
            out[p] = "test"
            test += frames_per_participant[p]
        elif dev < DEV_FRAC * total:
            out[p] = "dev"
            dev += frames_per_participant[p]
        else:
            out[p] = "train"
    return out


def egoplan_windows(path: str) -> Dict[str, List[Tuple[int, int]]]:
    import pyarrow.parquet as pq
    out: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for r in pq.read_table(path).to_pylist():
        if r["video_source"] != "EpicKitchens":
            continue
        obs = int(r["current_observation_frame"])
        meta = r["task_progress_metadata"]
        if isinstance(meta, str):
            meta = ast.literal_eval(meta)
        start = min([int(m["start_frame"]) for m in meta or []] + [obs])
        out[r["video_id"]].append((start, obs + EGOPLAN_AFTER))
    return out


def text_baselines(train: List[dict], evaluate: List[dict]) -> dict:
    """Choice questions only, per kind: longest / shortest option, and the option most often right relative to how
    often it was offered in train (answer prior, add-one smoothed); noul: train majority. Fitted on train only."""
    res: dict = {}
    for kind in sorted({r["kind"] for r in evaluate}):
        tr = [r for r in train if r["kind"] == kind]
        ev = [r for r in evaluate if r["kind"] == kind]
        if not ev:
            continue
        if ev[0]["questions"]["q"]["type"] == "noul":
            maj = Counter(r["answer"] for r in tr).most_common(1)
            guess = maj[0][0] if maj else "true"
            res[kind] = {"n": len(ev), "train_majority": round(sum(r["answer"] == guess for r in ev) / len(ev), 4),
                         "chance": 0.5}
            continue
        gold_n: Counter = Counter()
        offered: Counter = Counter()
        for r in tr:
            for o in r["gold"]["q"]["probabilities"]:
                offered[norm(o)] += 1
            gold_n[norm(r["answer"])] += 1
        hits = Counter()
        chance = 0.0
        for r in ev:
            opts = list(r["gold"]["q"]["probabilities"])
            chance += 1 / len(opts)
            hits["longest"] += max(opts, key=len) == r["answer"]
            hits["shortest"] += min(opts, key=len) == r["answer"]
            prior = max(opts, key=lambda o: ((gold_n[norm(o)] + 1) / (offered[norm(o)] + 2), o))
            hits["train_answer_prior"] += prior == r["answer"]
        res[kind] = dict({k: round(v / len(ev), 4) for k, v in sorted(hits.items())},
                         n=len(ev), chance=round(chance / len(ev), 4))
    return res


def position_balance(rows: List[dict]) -> dict:
    out: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if "answer_position" in r:
            n = len(r["gold"]["q"]["probabilities"])
            out["%s/%d_options" % (r["kind"], n)][r["answer_position"]] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


# ============================================================================ EgoThink (eval-only benchmark)

STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "is", "it", "my", "your", "with", "for", "at", "i", "you"}


def egothink_rows(items: List[dict], max_words: int = 6, n_opts: int = 4) -> Tuple[List[dict], Counter]:
    """items: {id, category, question, answer, image}. Short answers only; distractors are answers to OTHER
    questions of the same category that share no content word with the gold (fixed random order by id)."""
    stats: Counter = Counter()
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    rows = []
    for cat, its in sorted(by_cat.items()):
        pool = {}
        for it in its:
            a = it["answer"].strip().rstrip(".")
            if 0 < len(a.split()) <= max_words:
                pool.setdefault(norm(a), a)
        for it in its:
            gold = it["answer"].strip().rstrip(".")
            if not gold or len(gold.split()) > max_words:
                stats["long_answer"] += 1
                continue
            words = set(norm(gold).split()) - STOP
            cands = [v for k, v in sorted(pool.items()) if k != norm(gold) and not (set(k.split()) - STOP) & words
                     and not related(v, gold)]
            if len(cands) < n_opts - 1:
                stats["fewer_than_3_distractors"] += 1
                continue
            rng = rng_for("egothink-d:" + it["id"])
            ds = rng.sample(cands, n_opts - 1)
            qid = "egothink:%s" % it["id"]
            row = choice_sample(qid, "egothink/" + cat, it["question"], gold, ds, it["image"],
                                "egothink:" + it["id"], {"category": cat})
            row["source"] = "egothink"
            rows.append(row)
            stats["kept"] += 1
    return rows, stats


def build_egothink(root: str, raw: str) -> Tuple[List[dict], dict]:
    import pyarrow.parquet as pq
    from PIL import Image
    items = []
    img_dir = os.path.join(root, OUT, "images", "egothink")
    os.makedirs(img_dir, exist_ok=True)
    for c in EGOTHINK_CATS:
        p = os.path.join(raw, "egothink", "%s.parquet" % c)
        if not os.path.exists(p):
            continue
        for k, r in enumerate(pq.read_table(p).to_pylist()):
            iid = "%s_%s" % (c, r.get("id") or k)
            rel = os.path.join(OUT, "images", "egothink", re.sub(r"[^A-Za-z0-9_.-]", "_", iid) + ".jpg")
            dest = os.path.join(root, rel)
            if not os.path.exists(dest):
                with Image.open(io.BytesIO(r["image"]["bytes"])) as im:
                    im = im.convert("RGB")
                    im.thumbnail((1024, 1024))
                    im.save(dest, quality=90)
            items.append({"id": iid, "category": c, "question": r["question"], "answer": str(r["answer"]),
                          "image": rel})
    rows, stats = egothink_rows(items)
    return rows, {"questions_in_source": len(items), **dict(stats)}


# ============================================================================ build

def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(path + ".tmp", path)
    return n


def cmd_build(args) -> None:
    from PIL import Image
    root = args.root
    raw = os.path.join(root, OUT, "raw")
    vroot = os.path.join(raw, "visor")
    segments = read_segments([os.path.join(raw, "ek100", n) for n in ("EPIC_100_train.csv", "EPIC_100_validation.csv")])
    with open(os.path.join(vroot, "frame_mapping.json")) as f:
        mapping = json.load(f)
    windows = egoplan_windows(os.path.join(raw, "egoplan", "egoplan_bench.parquet"))
    ann_files = []
    for vsplit in ("train", "val"):
        d = os.path.join(vroot, "GroundTruth-SparseAnnotations", "annotations", vsplit)
        ann_files += [(vsplit, os.path.join(d, n)) for n in sorted(os.listdir(d)) if n.endswith(".json")]
    # pass 1: frames per participant (split decided from annotations only)
    per_part: Counter = Counter()
    videos = {}
    all_frames: List[dict] = []
    for vsplit, p in ann_files:
        with open(p) as f:
            frames = json.load(f)["video_annotations"]
        video = os.path.basename(p)[:-5]
        videos[video] = (vsplit, p, len(frames))
        all_frames += [{"annotations": [{k: a.get(k) for k in ("id", "name", "class_id", "in_contact_object")
                                         if k in a} for a in fr["annotations"]]} for fr in frames]
        per_part[video.split("_")[0]] += len(frames)
    split_of = split_participants(per_part)
    graspable = graspable_classes(all_frames)
    del all_frames
    stats: Counter = Counter()
    rows: Dict[str, List[dict]] = defaultdict(list)
    img_root = os.path.join(root, OUT, "images", "visor")
    videos_used, videos_missing = [], []
    egoplan_videos_used = set()
    for video, (vsplit, p, nframes) in sorted(videos.items()):
        zpath = os.path.join(vroot, "GroundTruth-SparseAnnotations", "rgb_frames", vsplit, video.split("_")[0],
                             video + ".zip")
        if not os.path.exists(zpath):
            videos_missing.append(video)
            stats["frames_not_downloaded_yet"] += nframes
            continue
        videos_used.append(video)
        with open(p) as f:
            frames = json.load(f)["video_annotations"]
        vmap = mapping.get(video, {})
        segs = sorted(segments.get(video, []), key=lambda s: s["start"])
        split = split_of[video.split("_")[0]]
        with zipfile.ZipFile(zpath) as z:
            members = {os.path.basename(n): n for n in z.namelist()}
            for fr in frames:
                stats["frames"] += 1
                name = fr["image"]["name"]
                if name not in members:
                    stats["frame:missing_in_zip"] += 1
                    continue
                ek = vmap.get(name)
                ek_frame = frame_number(ek) if ek else None
                fnum = ek_frame if ek_frame is not None else frame_number(name)
                if any(a <= fnum <= b for a, b in windows.get(video, [])):
                    stats["frame:egoplan_window"] += 1
                    continue
                if video in windows:
                    egoplan_videos_used.add(video)
                rel = os.path.join(OUT, "images", "visor", video, name)
                qs = frame_questions(fr, ek_frame, segs, rel, 1920, 1080, stats, graspable)
                if not qs:
                    stats["frame:no_question"] += 1
                    continue
                dest = os.path.join(root, rel)
                if not os.path.exists(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with Image.open(io.BytesIO(z.read(members[name]))) as im:
                        if im.size != (1920, 1080):
                            stats["frame:not_1920x1080"] += 1
                        im.convert("RGB").resize(SAVE_SIZE, Image.BICUBIC).save(dest, quality=90)
                rows[split] += qs
    out = os.path.join(root, OUT)
    counts = {}
    for split in ("train", "dev", "test"):
        rs = sorted(rows[split], key=lambda r: r["id"])
        counts[split] = {"questions": write_jsonl(os.path.join(out, split + ".jsonl"), rs),
                         "frames": len({r["image_id"] for r in rs}),
                         "videos": len({r["video"] for r in rs}),
                         "participants": sorted({r["participant"] for r in rs}),
                         "by_kind": dict(Counter(r["kind"] for r in rs)),
                         "noul_true_rate": round(sum(r["answer"] == "true" for r in rs if r["kind"] == "touching")
                                                 / max(1, sum(r["kind"] == "touching" for r in rs)), 4)}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        pa = {r["participant"] for r in rows[a]}
        pb = {r["participant"] for r in rows[b]}
        assert not pa & pb, (a, b, pa & pb)
    manifest = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/data/v05_ego.py",
        "sources": {"visor": "ModelScope %s (%s), CC BY-NC 4.0" % (VISOR_REPO, VISOR_ZIP),
                    "ek100_annotations": "github epic-kitchens/epic-kitchens-100-annotations@%s" % EK_COMMIT,
                    "egoplan_bench": "ModelScope %s (questions only, for exclusion)" % EGOPLAN[0],
                    "egothink": "ModelScope %s (eval-only)" % EGOTHINK_REPO},
        "complete": not videos_missing,
        "videos_used": len(videos_used), "videos_not_downloaded": videos_missing,
        "participant_split": split_of,
        "graspable_classes": {"rate": GRASP_RATE, "n": len(graspable)},
        "counts": counts,
        "drops": dict(sorted(stats.items())),
        "egoplan_overlap": {"egoplan_epic_videos": len(windows),
                            "our_videos_also_in_egoplan": len(egoplan_videos_used),
                            "note": "frames inside EgoPlan-Bench question windows dropped; same videos remain"},
        "answer_position": {s: position_balance(rows[s]) for s in ("train", "dev", "test")},
        "text_only_baselines": {"dev": text_baselines(rows["train"], rows["dev"]),
                                "test": text_baselines(rows["train"], rows["test"])},
    }
    if not args.skip_egothink and os.path.isdir(os.path.join(raw, "egothink")):
        et, et_stats = build_egothink(root, raw)
        manifest["bench_egothink"] = {"questions": write_jsonl(os.path.join(out, "bench_egothink.jsonl"), et),
                                      "drops": et_stats, "by_category": dict(Counter(r["category"] for r in et)),
                                      "answer_position": position_balance(et),
                                      "text_only_baselines": {k: {m: v for m, v in b.items()
                                                                  if m != "train_answer_prior"}
                                                              for k, b in text_baselines([], et).items()},
                                      "note": "eval-only; Ego4D pictures, no overlap with the EPIC-KITCHENS frames"}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: manifest[k] for k in ("complete", "videos_used", "counts", "text_only_baselines")}, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--root", default="data")
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--annotations-only", action="store_true")
    f.add_argument("--max-videos", type=int, default=0, help="only the N smallest frame zips (val first)")
    f.add_argument("--skip-egothink", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--root", default="data")
    b.add_argument("--skip-egothink", action="store_true")
    args = ap.parse_args(argv)
    {"fetch": cmd_fetch, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
