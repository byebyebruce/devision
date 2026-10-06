"""Extension pack, Vision-Flan (191 human-labelled vision tasks, 1,000 items each; we keep the tasks that become
clean choice / yes-no questions). Licence: per task, from each underlying dataset (see TASKS and
https://vision-flan.github.io/tasks.html; not verified one by one).

    uv run python scripts/data/ext/vision_flan.py --root data [--force]

Source: Vision-Flan/vision-flan_191-task_1k (annotation_191-task_1k.json, one instruction + answer per item; the
pictures are in one 36 GB zip). The zip is not downloaded: its central directory is read once over HTTP Range
(cached as data/raw/ext/vision_flan/zip_index.json) and only the members we use are fetched, neighbouring members
in one request, CRC-checked, and saved once to data/ext/vision_flan/images/.

Only the tasks in TASKS are used (every other task is listed in the MANIFEST with the reason it is left out).
Each kept task has one conversion:
  - options: "Options: (a) X (b) Y ..." in the instruction and "(b) Y" as the answer -> choice over the options
    (a fixed short question per task replaces the long task description); a No / Yes option pair -> noul;
  - labels:  the instruction names the label set (e.g. "daytime, nighttime, twilight") -> choice over it;
  - yesno:   a yes / no answer -> noul, the question taken out of the instruction (other answers dropped);
  - count:   a number word up to ten -> choice with nearby numbers (common.count_options);
  - itm:     caption-matching items -> noul 'Does this caption describe the picture? "..."'.
Pictures: tasks on COCO train2014 (file names carry the COCO id) get image_id "coco:<id>" and pass HeldOut.is_held;
every picture passes HeldOut.near_held. Balance per task: yes = no for noul, choice answers flattened (none above
twice the task's mean per answer).
"""
import io
import json
import os
import random
import re
import sys
import threading
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HeldOut, already_built, base_args, choice, count_options, finish, flatten_answers,  # noqa: E402
                    hf_file, noul, out_dir, raw_dir, save_bytes)

SOURCE = "vision_flan"
REPO = "Vision-Flan/vision-flan_191-task_1k"
ZIP = "image_191-task_1k.zip"
ZIP_URL = "https://huggingface.co/datasets/%s/resolve/main/%s" % (REPO, ZIP)
ZIP_DIR = "images_191task_1k/"

PACS_STYLES = ["Art painting", "Cartoon", "Photograph", "Sketch"]
PACS_Q = "What kind of picture is this: a photograph, an art painting, a cartoon or a sketch?"

# task -> (conversion, kind, question or None, extra); extra: "labels" (label set), "strip" (option prefix regex),
# "licence" (of the underlying dataset, as far as known).
TASKS: Dict[str, Tuple[str, str, Optional[str], dict]] = {
    "Caltech101+Living_Thing_classification": ("options", "living", "Is the main object in the picture a living thing?",
                                               {"licence": "Caltech101: CC BY 4.0"}),
    "NUS-WIDE+Animal_classification": ("options", "presence", "Is there an animal in the picture?",
                                       {"licence": "NUS-WIDE: Flickr pictures, research use"}),
    "GTSRB+image_classification": ("options", "traffic-sign", "Which traffic sign is this?",
                                   {"licence": "GTSRB: CC0"}),
    "LSUN+Image_Classification": ("options", "scene", "What kind of place is this?",
                                  {"licence": "LSUN: research use"}),
    "STL-10+Image_Classification": ("options", "object", "What is shown in the picture?",
                                    {"licence": "STL-10 (ImageNet pictures): research use"}),
    "Places205+Image_env_classification": ("options", "indoor-outdoor", "Is this place indoors or outdoors?",
                                           {"licence": "Places205: research use"}),
    "VisDA-2017+object_classification_train": ("options", "object", "What object is shown in this rendering?",
                                               {"licence": "VisDA-2017 synthetic renders: research use"}),
    "image_text_selection": ("options", "caption-select", "Which caption describes the picture?",
                             {"licence": "COCO captions: CC BY 4.0, Flickr picture terms"}),
    "Dark-Zurich+time_classification": ("labels", "time-of-day", "What time of day is it in this picture?",
                                        {"labels": ["daytime", "nighttime", "twilight"],
                                         "licence": "Dark Zurich: CC BY-NC 4.0"}),
    "300w+human_portrait_classification": ("labels", "indoor-outdoor", "Was this portrait taken indoors or outdoors?",
                                           {"labels": ["Indoor", "Outdoor"], "licence": "300-W: research use"}),
    "SKETCH+living_organism_detection": ("labels", "living", "Does this sketch show a living organism?",
                                         {"labels": ["Non-Living", "Living"], "licence": "TU-Berlin sketches: CC BY 4.0"}),
    "PACS+person_image_category_classification": ("labels", "style", PACS_Q,
                                                  {"labels": PACS_STYLES, "licence": "PACS: research use"}),
    "ITM": ("itm", "caption-match", None, {"licence": "COCO captions: CC BY 4.0, Flickr picture terms"}),
    "VQA_object_presence": ("yesno", "presence", None, {"licence": "TDIUC (COCO): CC BY 4.0"}),
    "VQA_sentiment_understanding": ("yesno", "sentiment", None, {"licence": "TDIUC (COCO): CC BY 4.0"}),
    "VQA_scene_recognition": ("yesno", "scene", None, {"licence": "TDIUC (COCO): CC BY 4.0"}),
    "VQA_utility_affordance": ("yesno", "affordance", None, {"licence": "TDIUC (COCO): CC BY 4.0"}),
    "VQA_counting": ("count", "count", None, {"licence": "TDIUC (COCO): CC BY 4.0"}),
}
for _o in ["dog", "elephant", "giraffe", "guitar", "horse", "house"]:
    TASKS["PACS+%s_image_category_classification" % _o] = ("options", "style", PACS_Q, {"licence": "PACS: research use"})
_COCO_Q = {"animal": "Which animal is in the picture?", "appliance": "Which appliance is in the picture?",
           "furniture": "Which piece of furniture is in the picture?", "kitchen": "Which kitchen item is in the picture?",
           "sports": "Which sports item is in the picture?", "vehicle": "Which vehicle is in the picture?"}
for _c, _q in _COCO_Q.items():
    TASKS["coco+image_classification_%s" % _c] = ("options", "object", _q,
                                                  {"strip": r"^This image contains an? ", "licence": "COCO: CC BY 4.0"})

# Every other task, grouped by why it is left out (checked against the task list in the MANIFEST).
EXCLUDED_PATTERNS: List[Tuple[str, str]] = [
    (r"caption|description|CONCADIA|WIT\+|REDCAPS|LOC_NARRATIVES|NOCAPS|VIZWIZ\+image|FLICKR30K|PICKAPIC|VQG|"
     r"SentiCap|MemeCap|semart\+image_description|FFHQ|LAD\+|HICO\+human|rationales|DeepFashion|WIKIART|"
     r"spot-the-diff|CHART2TEXT", "long free-text answer (caption / description / rationale)"),
    (r"DOCVQA|infographicvqa|STVQA|Total-Text|SCUT|FUNSD|MEMOTION|CoVA|textcaps|FoodLogo|FlickrLogos|"
     r"DVQA|PlotQA|GEOMETRY3K|AI2D|MNIST", "OCR-heavy or reads small text / labels (unreadable at 256 px)"),
    (r"RAVEN|recipe-qa|wikihow|Question_Answer_Matching|Multiple_Question", "several pictures in one / several questions in one item / text ordering"),
    (r"ImageNet-A|ImageNet-C|ImageNet-R|Winoground|ObjectNet|model-vs-human|ayahoo_test|MVTecAD|"
     r"VisDA-2017\+object_classification_validation|Set5|Road-Anomaly",
     "benchmark test / validation split (or COCO crops that cannot be checked against held-out pictures)"),
    (r"^GQA$|^VQAv2$|^VQA$|VQA-E|visualgenome_vqa|CLEVR|Clevr|iconqa|A-OKVQA|ok_vqa|vizwiz",
     "source already in our training data or evaluation sets (GQA, VQAv2, Visual7W, CLEVR, IconQA, A-OKVQA, VizWiz)"),
    (r"fairface|LFW|expw|KVQA", "people's identity / demographic attributes or knowledge about named people"),
    (r"VQARAD|multimodal_factual_checking|question_image_match|image_quality|Office_31\+Image_Classification_Category",
     "expert / subjective / meta labels (radiology, fact checking, answerability, multi-label quality flaws, camera domain)"),
    (r"cinic-10", "32x32 pictures (CINIC-10)"),
    (r"visdial", "dialogue: the question refers to earlier turns given as long context"),
]
EXCLUDED_DEFAULT = "open-ended answer with no option list in the instruction (label set not given to the model)"

NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


# ---------------------------------------------------------------- pure conversion

def clean(text: str) -> str:
    return " ".join(text.replace("<image>", " ").split())


def parse_options(prompt: str) -> Optional[List[Tuple[str, str]]]:
    """'... Options: (a) X (b) Y |' -> [("a", "X"), ("b", "Y")]."""
    m = re.search(r"Options:\s*(.*?)(?:\n|\||$)", prompt.replace("<image>", ""), re.S)
    if not m:
        return None
    parts = re.split(r"\(([a-j])\)\s*", m.group(1))
    opts = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
    return opts if len(opts) >= 2 else None


def answer_letter(answer: str) -> Optional[str]:
    m = re.match(r"\s*\(([a-j])\)", answer)
    return m.group(1) if m else None


def extract_question(prompt: str) -> Optional[str]:
    """The actual question inside a MultiInstruct (TDIUC) instruction: 'Here is the question "X".', 'The question
    is: X', 'Question: X', or the sentence ending in '?'."""
    text = clean(prompt).replace("|", " ")
    m = re.search(r'question "([^"]+\?)"', text)
    if m:
        return " ".join(m.group(1).split())
    cands = [c for c in re.findall(r'[A-Z][^?.!"]*\?', text)
             if not re.match(r"(?:What is the answer|What is your answer|Can you|Could you)\b", c)]
    if not cands:
        return None
    q = re.sub(r"^(?:The question is:?|Question:|Here is the question:?)\s*", "", cands[0]).strip()
    q = " ".join(q.split())
    q = re.sub(r"\s+\?$", "?", q)
    return q[0].upper() + q[1:] if q else None


def itm_caption(prompt: str) -> Optional[str]:
    m = re.search(r'"([^"]{8,})"', prompt)
    return " ".join(m.group(1).split()) if m else None


def itm_match(option: str) -> Optional[bool]:
    o = option.lower().strip()
    if o.startswith(("no", "not", "false", "the text is not")):
        return False
    if o.startswith(("yes", "match", "true", "the description matches")):
        return True
    return None


def label_answer(answer: str, labels: Sequence[str]) -> Optional[str]:
    """The label an answer sentence ends with ('The time of the day is nighttime.' -> 'nighttime'); longest
    label first so 'Non-Living' wins over 'Living'."""
    a = answer.strip().rstrip(".").strip()
    for lab in sorted(labels, key=len, reverse=True):
        if a.lower() == lab.lower() or a.lower().endswith(" " + lab.lower()):
            return lab
    return None


def coco_id(image: str) -> Optional[int]:
    m = re.search(r"COCO_train2014_(\d{12})", image) or re.search(r"^coco\+.*_(\d{12})\.jpg$", image)
    return int(m.group(1)) if m else None


COCO_ALIAS = {"ski": "skis"}


def prune_present(sample: dict, present: Set[str]) -> Tuple[Optional[dict], Optional[str]]:
    """A coco+image_classification question is about a whole COCO photo with several objects: drop every
    distractor option that COCO's instance annotations mark as present, and drop the question if its answer is not
    marked present or fewer than two options are left."""
    has = lambda o: COCO_ALIAS.get(o, o) in present  # noqa: E731
    ans = sample["answer_key"]
    if not has(ans):
        return None, "answer not in the COCO annotations"
    opts = [o for o in sample["questions"]["q"]["criteria"] if o == ans or not has(o)]
    if len(opts) < 2:
        return None, "fewer than 2 options after removing objects present"
    sample["questions"]["q"]["criteria"] = {o: None for o in opts}
    sample["gold"]["q"]["probabilities"] = {o: float(o == ans) for o in opts}
    return sample, None


def coco_objects(root: str) -> Dict[int, Set[str]]:
    """COCO 2014 image id -> names of the object categories annotated in it (train + val), cached."""
    path = os.path.join(raw_dir(root, SOURCE), "coco_objects.json")
    if not os.path.exists(path):
        import zipfile
        from fetch import coco_annotations  # pyright: ignore[reportMissingImports]
        out: Dict[int, Set[str]] = defaultdict(set)
        with zipfile.ZipFile(coco_annotations(root)) as z:
            for split in ("train2014", "val2014"):
                inst = json.load(z.open("annotations/instances_%s.json" % split))
                cat = {c["id"]: c["name"] for c in inst["categories"]}
                for a_ in inst["annotations"]:
                    out[a_["image_id"]].add(cat[a_["category_id"]])
        json.dump({str(k): sorted(v) for k, v in out.items()}, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    return {int(k): set(v) for k, v in json.load(open(path)).items()}


def excluded_reason(task: str) -> str:
    for pat, why in EXCLUDED_PATTERNS:
        if re.search(pat, task):
            return why
    return EXCLUDED_DEFAULT


def convert(item: dict, image: str, image_id: str, rng: random.Random):
    """One annotation item of a kept task -> (sample, None) or (None, reason)."""
    task = item["task_name"]
    how, kind, fixed, extra = TASKS[task]
    question = fixed or ""
    prompt, answer = item["conversations"][0]["value"], item["conversations"][1]["value"].strip()
    # the annotation's "id" is not unique (many items of a task share one); the picture file name is
    sid = "ext-vision_flan:%s:%s" % (task, os.path.splitext(os.path.basename(item["image"]))[0])
    common: Dict[str, Any] = dict(source=SOURCE, kind=kind, image_id=image_id, image=image, group=task)
    if how == "options":
        opts = parse_options(prompt)
        letter = answer_letter(answer)
        if not opts or letter is None or letter not in dict(opts):
            return None, "options not parsed"
        strip = extra.get("strip")
        by = {k: re.sub(strip, "", v) if strip else v for k, v in opts}
        by = {k: v.strip().rstrip(".").strip() if task != "image_text_selection" else v.strip() for k, v in by.items()}
        values = list(by.values())
        if len(set(v.lower() for v in values)) != len(values) or not 2 <= len(values) <= 10:
            return None, "options not distinct / 2-10"
        if {v.lower() for v in values} == {"no", "yes"}:
            return noul(sid, question=question, answer=by[letter].lower() == "yes", **common), None
        return choice(sid, question=question, options=values, answer=by[letter], rng=rng, **common), None
    if how == "labels":
        lab = label_answer(answer, extra["labels"])
        if lab is None:
            return None, "answer not a label"
        if extra["labels"] == ["Non-Living", "Living"]:
            return noul(sid, question=question, answer=lab == "Living", **common), None
        return choice(sid, question=question, options=extra["labels"], answer=lab, rng=rng, **common), None
    if how == "itm":
        opts, letter, cap = parse_options(prompt), answer_letter(answer), itm_caption(prompt)
        if not opts or letter not in dict(opts or []) or not cap:
            return None, "caption / options not parsed"
        match = itm_match(dict(opts)[letter])
        if match is None:
            return None, "options not parsed"
        return noul(sid, question='Does this caption describe the picture? "%s"' % cap, answer=match, **common), None
    q = extract_question(prompt)
    if not q:
        return None, "no question found"
    if how == "yesno":
        a = answer.lower().rstrip(".")
        if a not in ("yes", "no") or " or " in q:
            return None, "not a yes / no answer"
        return noul(sid, question=q, answer=a == "yes", **common), None
    if how == "count":
        n = NUMBERS.get(answer.lower().rstrip("."), int(answer) if answer.isdigit() else None)
        if n is None or n > 10 or not q.lower().startswith("how many"):
            return None, "count > 10 or not a count question"
        return choice(sid, question=q, options=count_options(n, rng), answer=str(n), rng=rng, **common), None
    raise ValueError(how)


# ---------------------------------------------------------------- pictures from the remote zip

class _Remote:
    """The zip's CDN address (re-resolved when a signed address expires)."""

    def __init__(self):
        self.lock, self.url, self.size = threading.Lock(), None, 0

    def resolve(self, stale: Optional[str] = None) -> str:
        with self.lock:
            if self.url is None or self.url == stale:
                req = urllib.request.Request(ZIP_URL, headers={"Range": "bytes=0-0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    self.url = r.url
                    self.size = int(r.headers["Content-Range"].split("/")[1])
            return self.url

    def get(self, start: int, end: int, retries: int = 5) -> bytes:
        import time
        url = self.resolve()
        end = min(end, self.size - 1)
        for attempt in range(retries):
            url = self.resolve()
            try:
                req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end)})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == end - start + 1:
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 410):
                    self.resolve(stale=url)
            except OSError:
                pass
            time.sleep(3 * (attempt + 1))
        raise OSError("range %d-%d failed" % (start, end))


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of the remote zip, enough for zipfile to read the central directory."""

    def __init__(self, remote: _Remote):
        self.remote, self.pos = remote, 0
        remote.resolve()
        self.size = remote.size

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = self.remote.get(self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


def zip_index(root: str, remote: _Remote) -> Dict[str, list]:
    """{member name: [local header offset, compressed size, size, method, crc]}, read once from the remote zip's
    central directory and cached."""
    path = os.path.join(raw_dir(root, SOURCE), "zip_index.json")
    if os.path.exists(path):
        return json.load(open(path))
    import zipfile
    print("reading the central directory of %s (no download of the zip)" % ZIP, flush=True)
    z = zipfile.ZipFile(_RangeFile(remote))
    index = {i.filename: [i.header_offset, i.compress_size, i.file_size, i.compress_type, i.CRC]
             for i in z.infolist() if not i.filename.endswith("/")}
    json.dump(index, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return index


def plan_ranges(members: Sequence[Tuple[str, list]], gap: int = 256 << 10, span: int = 16 << 20,
                slack: int = 4096) -> List[Tuple[int, int, List[Tuple[str, list]]]]:
    """Members (name, index entry) sorted by offset, grouped so neighbours within `gap` bytes share one request of
    at most `span` bytes; each request ends `slack` bytes past its last member's data (local header extra field)."""
    out: List[Tuple[int, int, List[Tuple[str, list]]]] = []
    for name, e in sorted(members, key=lambda m: m[1][0]):
        end = e[0] + 30 + len(name.encode()) + e[1] + slack
        if out and e[0] - out[-1][1] <= gap and end - out[-1][0] <= span:
            out[-1] = (out[-1][0], max(out[-1][1], end), out[-1][2] + [(name, e)])
        else:
            out.append((e[0], end, [(name, e)]))
    return out


def member_bytes(buf: bytes, base: int, name: str, e: list) -> Optional[bytes]:
    """A member's content out of a fetched range starting at file offset `base`; None if the buffer is short or
    the CRC does not match."""
    off = e[0] - base
    if buf[off:off + 4] != b"PK\x03\x04":
        return None
    n, x = int.from_bytes(buf[off + 26:off + 28], "little"), int.from_bytes(buf[off + 28:off + 30], "little")
    start = off + 30 + n + x
    raw = buf[start:start + e[1]]
    if len(raw) != e[1]:
        return None
    data = raw if e[3] == 0 else zlib.decompress(raw, -15) if e[3] == 8 else None
    if data is None or len(data) != e[2] or zlib.crc32(data) != e[4]:
        return None
    return data


def fetch_members(root: str, remote: _Remote, wanted: Dict[str, str], index: Dict[str, list]) -> set:
    """Save each wanted zip member (name -> path relative to root) not on disk yet; returns the names that failed."""
    from fetch import run_with_progress  # pyright: ignore[reportMissingImports]
    todo = [(n, index[n]) for n, rel in wanted.items() if not os.path.exists(os.path.join(root, rel))]
    if not todo:
        return set()
    ranges = plan_ranges(todo)
    print("fetching %d pictures (%.2f GB) in %d range requests" % (
        len(todo), sum(e[1] for _, e in todo) / 1e9, len(ranges)), flush=True)

    def one(rng_) -> str:
        start, end, ms = rng_
        try:
            buf = remote.get(start, end - 1)
        except OSError:
            return "\n".join(n for n, _ in ms)
        bad = []
        for name, e in ms:
            data = member_bytes(buf, start, name, e)
            if data is None:   # rare: a longer extra field than the slack; fetch the member alone
                try:
                    data = member_bytes(remote.get(e[0], e[0] + 30 + len(name.encode()) + e[1] + 65535), e[0], name, e)
                except OSError:
                    data = None
            if data is None:
                bad.append(name)
            else:
                save_bytes(os.path.join(root, wanted[name]), data)
        return "\n".join(bad)
    failed = run_with_progress("vision_flan pictures", ranges, one, workers=16)
    return {n for f in failed for n in f.split("\n") if n}


# ---------------------------------------------------------------- build

def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--limit", type=int, default=40000, help="at most this many questions in total")
    a = p.parse_args(argv)
    if already_built(a.root, SOURCE, a.force):
        return
    rng = random.Random(a.seed)
    items = json.load(open(hf_file(a.root, REPO, "annotation_191-task_1k.json")))
    tasks = Counter(x["task_name"] for x in items)
    steps, dropped = Counter(items=len(items)), Counter()
    kept = [x for x in items if x["task_name"] in TASKS]
    missing = sorted(set(TASKS) - set(tasks))
    if missing:
        raise SystemExit("tasks not in the annotation file: %s" % missing)
    steps["items_in_kept_tasks"] = len(kept)

    held = HeldOut(a.root)
    objects = coco_objects(a.root)
    rows, wanted = [], {}
    for x in kept:
        cid = coco_id(x["image"])
        image_id = "coco:%d" % cid if cid is not None else "vision_flan:%s" % x["image"]
        if cid is not None and held.is_held(image_id):
            dropped["held-out COCO picture"] += 1
            continue
        rel = os.path.join("ext", SOURCE, "images", x["image"])
        r, why = convert(x, rel, image_id, rng)
        if r is not None and x["task_name"].startswith("coco+"):
            r, why = prune_present(r, objects.get(cid or -1, set()))
        if r is None:
            dropped[why] += 1
            continue
        rows.append(r)
        wanted[ZIP_DIR + x["image"]] = rel
    steps["converted"] = len(rows)

    remote = _Remote()
    index = zip_index(a.root, remote)
    absent = {n for n in wanted if n not in index}
    failed = fetch_members(a.root, remote, {n: r for n, r in wanted.items() if n not in absent}, index) | absent
    bad_rels = {wanted[n] for n in failed}
    if bad_rels:
        dropped["picture missing from the zip / not fetched"] += sum(r["image"] in bad_rels for r in rows)
        rows = [r for r in rows if r["image"] not in bad_rels]
    near = held.near_held({r["image"] for r in rows}, SOURCE)
    dropped["near a held-out photo (dHash)"] += sum(r["image"] in near for r in rows)
    rows = [r for r in rows if r["image"] not in near]
    steps["after_held_out"] = len(rows)

    # balance per task: yes = no (noul), answers flattened (choice)
    rng.shuffle(rows)
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["group"]].append(r)
    out = []
    for task in sorted(by_task):
        rs = by_task[task]
        nouls = [r for r in rs if r["questions"]["q"]["type"] == "noul"]
        k = min(Counter(r["answer_key"] for r in nouls).values()) if len({r["answer_key"] for r in nouls}) == 2 else 0
        seen: Counter = Counter()
        for r in nouls:
            if seen[r["answer_key"]] < k:
                seen[r["answer_key"]] += 1
                out.append(r)
        out += flatten_answers([r for r in rs if r["questions"]["q"]["type"] == "choice"])
    steps["after_balance"] = len(out)
    if len(out) > a.limit:
        rng.shuffle(out)
        out = out[:a.limit]
    seen: Counter = Counter()
    for r in sorted(out, key=lambda r: (r["id"], r["questions"]["q"]["instructions"])):
        seen[r["id"]] += 1
        if seen[r["id"]] > 1:          # the same picture asked twice within a task
            r["id"] = "%s#%d" % (r["id"], seen[r["id"]])
    out.sort(key=lambda r: r["id"])
    steps["kept"] = len(out)

    per_task = Counter(r["group"] for r in out)
    included = {t: {"kind": TASKS[t][1], "conversion": TASKS[t][0], "questions": per_task.get(t, 0),
                    "licence": TASKS[t][3].get("licence", "")} for t in sorted(TASKS)}
    excluded = {t: excluded_reason(t) for t in sorted(tasks) if t not in TASKS}
    finish(a.root, SOURCE, out, dict(steps), {
        "dropped": dict(dropped), "tasks_included": included, "tasks_excluded": excluded,
        "tasks_excluded_by_reason": dict(Counter(excluded.values())),
        "licence": "per task (underlying datasets; see tasks_included[*].licence and "
                   "https://vision-flan.github.io/tasks.html), not verified one by one"})


if __name__ == "__main__":
    main()
