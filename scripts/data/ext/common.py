"""Shared pieces of the extension data pack (scripts/data/ext/; sources in docs/research/data-pack-sources-2026-10-06.md).

Every source script (scripts/data/ext/<source>.py) follows the same layout and rules:

  <root>/raw/ext/<source>/        downloads (archives, JSON, parquet), kept; a finished file is never fetched again,
                                  an interrupted one resumes from its .part file (HTTP Range)
  <root>/raw/hf/                  Hugging Face downloads (the hub cache shared with the other builders; resumable)
  <root>/ext/<source>/images/     pictures the source ships inside its files, extracted once (existing files skipped)
  <root>/ext/<source>/train.jsonl the converted questions (format: src/devision/train/samples.py)
  <root>/ext/<source>/MANIFEST.json counts per step, kinds, answers, option counts, question-only baselines, SHA256

Re-running a script downloads nothing that is already there and redoes only the cheap conversion; `--force`
rebuilds train.jsonl even if it exists. Steps that are slow to recompute (picture hashes) are cached as files.

Held-out pictures (`HeldOut`): every picture of the project's evaluation / monitoring / calibration sets
(v9_mix.held_files_all, data/v11 included), by picture identity (COCO and VG ids of one picture are one), and for
photos also any picture within NEAR_BITS of a held-out photo's dHash. Sources on COCO / VG pictures must pass
their ids through `is_held`; every source passes its photos through `near_held` (diagrams: identical pixels).
"""
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # scripts/data: convert, fetch, v8_mix, v9_mix ...

Sample = Dict[str, Any]
NOUL = ("false", "true")


# ---------------------------------------------------------------- paths and downloads

def raw_dir(root: str, source: str) -> str:
    d = os.path.join(root, "raw", "ext", source)
    os.makedirs(d, exist_ok=True)
    return d


def out_dir(root: str, source: str) -> str:
    d = os.path.join(root, "ext", source)
    os.makedirs(d, exist_ok=True)
    return d


def download(url: str, path: str, retries: int = 5) -> str:
    """`path`, fetched from `url` unless it is already there. Resumes an interrupted download from `path`.part
    with an HTTP Range request; prints progress about every 10%."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": "bytes=%d-" % have} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:      # server ignored the range: start over
                    have = 0
                total = int(r.headers.get("Content-Length") or 0) + have
                print("downloading %s%s" % (url, " (resuming at %.0f MB)" % (have / 1e6) if have else ""), flush=True)
                start, shown, got = time.time(), -1, have
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        step = got * 10 // total if total else got // (500 << 20)
                        if step > shown:
                            shown = step
                            print("  %s: %.0f / %s MB, %.0f s" % (os.path.basename(path), got / 1e6,
                                  "%.0f" % (total / 1e6) if total else "?", time.time() - start), flush=True)
            if total and got < total:            # the connection closed early: resume on the next attempt
                raise OSError("short read: %d of %d bytes" % (got, total))
            os.replace(part, path)
            return path
        except OSError as e:
            print("  download error (%s), retry %d" % (e, attempt + 1), flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit("could not download %s" % url)


def hf_file(root: str, repo: str, filename: str, repo_type: str = "dataset", revision: Optional[str] = None) -> str:
    """A file of a Hugging Face repo, cached under <root>/raw/hf (resumes, never fetched twice)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type=repo_type, revision=revision,
                           cache_dir=os.path.join(root, "raw", "hf"))


def hf_files(root: str, repo: str, pattern: str, repo_type: str = "dataset") -> List[str]:
    """All files of a Hugging Face repo matching the glob `pattern`, cached as `hf_file`."""
    import fnmatch
    from huggingface_hub import HfApi
    names = sorted(f for f in HfApi().list_repo_files(repo, repo_type=repo_type) if fnmatch.fnmatch(f, pattern))
    if not names:
        raise SystemExit("no file of %s matches %s" % (repo, pattern))
    return [hf_file(root, repo, n, repo_type) for n in names]


def save_bytes(path: str, data: bytes) -> None:
    """Write a picture once (an existing file is kept), through a temporary name."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)


def fetch_pictures(root: str, items: Sequence[Any], url_of: Callable[[Any], str], rel_of: Callable[[Any], str],
                   label: str, workers: int = 16, retry_dead: bool = False) -> set:
    """Download each item's picture (`url_of`) to <root>/`rel_of`, skipping files already there; returns the
    items' rel paths that failed. Progress about every 5% and every 60 s. Failed links are remembered in
    <root>/raw/ext/<label>.dead.json and not tried again on a re-run (unless `retry_dead`)."""
    from fetch import run_with_progress  # pyright: ignore[reportMissingImports]
    dead_path = os.path.join(raw_dir(root, ""), label.replace(" ", "_") + ".dead.json")
    dead = set() if retry_dead or not os.path.exists(dead_path) else set(json.load(open(dead_path)))

    def one(it) -> str:
        rel = rel_of(it)
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return ""
        if rel in dead:
            return rel
        try:
            with urllib.request.urlopen(urllib.request.Request(url_of(it), headers={"User-Agent": "Mozilla/5.0"}),
                                        timeout=30) as r:
                save_bytes(path, r.read())
            return ""
        except Exception:     # dead links are expected for web pictures
            return rel
    failed = set(run_with_progress(label, list(items), one, workers))
    json.dump(sorted(dead | failed), open(dead_path, "w"))
    return failed


# ---------------------------------------------------------------- held-out pictures

class HeldOut:
    """Pictures training must not use (see the module doc). Photo hashes are cached in
    <root>/raw/ext/held_hashes.json, keyed by the held files' SHA256s, so they are computed once."""

    def __init__(self, root: str):
        from convert import picture  # pyright: ignore[reportMissingImports]
        from fetch import vg_to_coco  # pyright: ignore[reportMissingImports]
        from v8_mix import held_pictures  # pyright: ignore[reportMissingImports]
        self.root = root
        self.v2c = vg_to_coco(root)
        self._picture = picture
        self.ids = held_pictures(root, self.v2c)
        self._hashes: Optional[Dict[str, int]] = None

    def is_held(self, image_id: str) -> bool:
        return self._picture(image_id, self.v2c) in self.ids

    def photo_hashes(self) -> Dict[str, int]:
        if self._hashes is None:
            from v9_mix import hashes, held_files_all, is_photo, original_image  # pyright: ignore[reportMissingImports]
            files = held_files_all(self.root)
            key = hashlib.sha256("".join(sha256(f) for f in files).encode()).hexdigest()
            cache = os.path.join(raw_dir(self.root, ""), "held_hashes.json")
            if os.path.exists(cache):
                c = json.load(open(cache))
                if c.get("key") == key:
                    self._hashes = {k: int(v) for k, v in c["hashes"].items()}
                    return self._hashes
            photos = {original_image(r["image"]) for f in files for r in map(json.loads, open(f))
                      if r.get("image") and is_photo(r["image"])}
            self._hashes = hashes(self.root, photos)
            json.dump({"key": key, "hashes": {k: str(v) for k, v in self._hashes.items()}}, open(cache, "w"))
        return self._hashes

    def diagram_pixels(self) -> set:
        """Pixel hashes (v3_build.pixel_hash) of every held-out diagram (non-photo) picture, cached like the photo
        hashes. Diagrams match only when identical: ScienceQA templates differ in one marked region."""
        if getattr(self, "_pixels", None) is None:
            from v3_build import pixel_hash  # pyright: ignore[reportMissingImports]
            from v9_mix import held_files_all, is_photo  # pyright: ignore[reportMissingImports]
            from PIL import Image
            files = held_files_all(self.root)
            key = hashlib.sha256("".join(sha256(f) for f in files).encode()).hexdigest()
            cache = os.path.join(raw_dir(self.root, ""), "held_diagram_pixels.json")
            if os.path.exists(cache) and json.load(open(cache)).get("key") == key:
                self._pixels = set(json.load(open(cache))["pixels"])
            else:
                rels = {r["image"] for f in files for r in map(json.loads, open(f))
                        if r.get("image") and not is_photo(r["image"])}
                out = set()
                for rel in sorted(rels):
                    try:
                        with Image.open(os.path.join(self.root, rel)) as im:
                            out.add(pixel_hash(im))
                    except OSError:
                        pass
                self._pixels = out
                json.dump({"key": key, "pixels": sorted(out)}, open(cache, "w"))
        return self._pixels

    def same_diagram(self, rel: str) -> bool:
        """True when the picture at `rel` has exactly the pixels of a held-out diagram."""
        from v3_build import pixel_hash  # pyright: ignore[reportMissingImports]
        from PIL import Image
        with Image.open(os.path.join(self.root, rel)) as im:
            return pixel_hash(im) in self.diagram_pixels()

    def near_held(self, rels: Iterable[str], cache_name: str) -> set:
        """The pictures among `rels` within NEAR_BITS of a held-out photo. The hashes of `rels` are cached in
        <root>/raw/ext/<cache_name>.hashes.json (added to, never recomputed)."""
        from v9_mix import hashes, near_held  # pyright: ignore[reportMissingImports]
        cache = os.path.join(raw_dir(self.root, ""), cache_name + ".hashes.json")
        known = {k: int(v) for k, v in json.load(open(cache)).items()} if os.path.exists(cache) else {}
        todo = sorted(set(rels) - set(known))
        if todo:
            known.update(hashes(self.root, todo))
            json.dump({k: str(v) for k, v in known.items()}, open(cache, "w"))
        return near_held({r: known[r] for r in set(rels) if r in known}, self.photo_hashes())


# ---------------------------------------------------------------- samples

def noul(sid: str, source: str, kind: str, image_id: str, image: str, question: str, answer: bool,
         group: str = "", **extra) -> Sample:
    """A yes / no question; gold puts all mass on the answer."""
    key = "true" if answer else "false"
    return {"id": sid, "source": source, "kind": kind, "image_id": image_id, "image": image,
            "questions": {"q": {"type": "noul", "instructions": question}},
            "gold": {"q": {"probabilities": {"false": float(not answer), "true": float(answer)}}},
            "answer_key": key, "group": group or kind, **extra}


def choice(sid: str, source: str, kind: str, image_id: str, image: str, question: str, options: Sequence[str],
           answer: str, rng: random.Random, group: str = "", shuffle: bool = True, **extra) -> Sample:
    """A choice question over `options` (2-10, distinct, containing `answer`), in shuffled order."""
    opts = list(dict.fromkeys(o.strip() for o in options))
    if answer not in opts or not 2 <= len(opts) <= 10:
        raise ValueError("bad options %s for answer %r" % (opts, answer))
    if shuffle:
        rng.shuffle(opts)
    return {"id": sid, "source": source, "kind": kind, "image_id": image_id, "image": image,
            "questions": {"q": {"type": "choice", "instructions": question, "criteria": {o: None for o in opts}}},
            "gold": {"q": {"probabilities": {o: float(o == answer) for o in opts}}},
            "answer_key": answer, "group": group or kind, **extra}


def count_options(n: int, rng: random.Random, n_options: Optional[int] = None, top: int = 10) -> List[str]:
    """`n` plus distractors within 3 of it, 0..`top` (data-v2's counting rule); 2-5 options."""
    k = n_options or rng.choice([2, 3, 3, 4, 4, 5])
    pool = [m for m in range(max(0, n - 3), min(top, n + 3) + 1) if m != n]
    rng.shuffle(pool)
    return [str(n)] + [str(m) for m in pool[:k - 1]]


def problems(r: Sample, root: Optional[str] = None) -> List[str]:
    """Why a sample is malformed; [] when fine."""
    out = []
    q, g = r["questions"]["q"], r["gold"]["q"]["probabilities"]
    if not q.get("instructions", "").strip():
        out.append("empty question")
    if q["type"] == "noul":
        if set(g) != set(NOUL):
            out.append("noul gold %s" % list(g))
    elif q["type"] == "choice":
        if list(q["criteria"]) != list(g) or not 2 <= len(g) <= 10:
            out.append("choice options %s / gold %s" % (list(q["criteria"]), list(g)))
    else:
        out.append("type %s" % q["type"])
    if abs(sum(g.values()) - 1) > 1e-6:
        out.append("gold sums to %s" % sum(g.values()))
    if root and not os.path.exists(os.path.join(root, r["image"])):
        out.append("missing picture %s" % r["image"])
    return out


def question_only_baseline(rows: List[Sample], field: str = "group", seed: int = 0) -> dict:
    """Accuracy of answering from the question alone: the most common answer of the question's `field` value
    ("wording" = the normalised question text), fitted on a random half of the rows and measured on the other
    half; unseen values count as chance. Compare with "chance" (mean 1 / options)."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    fit, test = rows[: len(rows) // 2], rows[len(rows) // 2:]
    counts: Dict[str, Counter] = defaultdict(Counter)
    val = (lambda r: " ".join(r["questions"]["q"]["instructions"].lower().split())) if field == "wording" \
        else (lambda r: r.get(field, ""))
    for r in fit:
        counts[val(r)][r["answer_key"]] += 1
    # a question whose wording / group was never seen in the fitting half is answered at chance
    hit = sum(1 if counts.get(val(r)) and counts[val(r)].most_common(1)[0][0] == r["answer_key"]
              else (0 if counts.get(val(r)) else 1 / len(r["gold"]["q"]["probabilities"])) for r in test)
    chance = sum(1 / len(r["gold"]["q"]["probabilities"]) for r in test) / max(1, len(test))
    return {"n": len(test), "by": field, "accuracy": round(hit / max(1, len(test)), 4), "chance": round(chance, 4)}


def cap_per_picture(rows: List[Sample], cap: int, rng: random.Random) -> List[Sample]:
    by: Dict[str, List[Sample]] = defaultdict(list)
    for r in rows:
        by[r["image_id"]].append(r)
    out = []
    for k in sorted(by):
        v = by[k]
        rng.shuffle(v)
        out += v[:cap]
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finish(root: str, source: str, rows: List[Sample], steps: Dict[str, int], extra: Optional[dict] = None) -> None:
    """Check, write <root>/ext/<source>/train.jsonl and MANIFEST.json, print the summary. Stops on any
    malformed sample or duplicate id."""
    bad = [(r["id"], p) for r in rows for p in [problems(r, root)] if p]
    if bad:
        raise SystemExit("%s: %d malformed samples, e.g. %s" % (source, len(bad), bad[:3]))
    dup = [i for i, n in Counter(r["id"] for r in rows).items() if n > 1]
    if dup:
        raise SystemExit("%s: %d duplicate ids, e.g. %s" % (source, len(dup), dup[:3]))
    d = out_dir(root, source)
    path = os.path.join(d, "train.jsonl")
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    manifest = {
        "source": source, "questions": len(rows), "pictures": len({r["image_id"] for r in rows}),
        "steps": steps,
        "types": dict(Counter(r["questions"]["q"]["type"] for r in rows)),
        "kinds": dict(Counter(r["kind"] for r in rows).most_common()),
        "answers_noul": dict(Counter(r["answer_key"] for r in rows if r["questions"]["q"]["type"] == "noul")),
        "options": dict(sorted(Counter(len(r["gold"]["q"]["probabilities"]) for r in rows
                                       if r["questions"]["q"]["type"] == "choice").items())),
        "question_only_baseline": [question_only_baseline(rows), question_only_baseline(rows, "wording")],
        "sha256": sha256(path), **(extra or {})}
    with open(os.path.join(d, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=1))


def already_built(root: str, source: str, force: bool) -> bool:
    """True (and says so) when train.jsonl and MANIFEST.json exist and --force was not given."""
    d = os.path.join(root, "ext", source)
    if not force and os.path.exists(os.path.join(d, "train.jsonl")) and os.path.exists(os.path.join(d, "MANIFEST.json")):
        print("%s: already built (%s); --force to rebuild" % (source, os.path.join(d, "train.jsonl")))
        return True
    return False


def base_args(doc: Optional[str]):
    import argparse
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--force", action="store_true", help="rebuild train.jsonl even if it exists (downloads are kept)")
    p.add_argument("--seed", type=int, default=0)
    return p


# ---------------------------------------------------------------- HuggingFaceM4/the_cauldron

CAULDRON = "HuggingFaceM4/the_cauldron"


def cauldron_rows(root: str, config: str, batch_size: int = 64):
    """(key, texts, images) of every row of a Cauldron subset, shard by shard (downloaded once into the hub
    cache). key = "<config>/<shard>/<row>" is stable; texts = [{"user", "assistant", ...}]; images = [{"bytes"}]."""
    import pyarrow.parquet as pq
    for s, path in enumerate(hf_files(root, CAULDRON, "%s/train-*.parquet" % config)):
        i = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["images", "texts"]):
            for images, texts in zip(batch.column("images").to_pylist(), batch.column("texts").to_pylist()):
                yield "%s/%d/%d" % (config, s, i), texts, images
                i += 1


def picture_bytes(images: list) -> Optional[bytes]:
    """The single picture of a row (None when a row has no picture or several)."""
    if not images or len(images) != 1 or not images[0] or not images[0].get("bytes"):
        return None
    return images[0]["bytes"]


def extract_picture(root: str, source: str, key: str, data: bytes) -> str:
    """Save a row's picture once under <root>/ext/<source>/images/; returns the path relative to root."""
    ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    rel = os.path.join("ext", source, "images", key.replace("/", "_") + ext)
    save_bytes(os.path.join(root, rel), data)
    return rel


_PROMPT_TAILS = ("answer", "provide", "give", "be ", "quick", "short", "keep", "offer", "ensure", "respond",
                 "your answer", "your response", "please", "only", "just", "the answer", "a short", "concise", "brief", "write", "reply", "use ", "make", "state")


def question_text(user: str) -> str:
    """The question of a Cauldron turn without its answer-format instruction line(s) ("Answer yes or no.",
    "Be succinct." ...)."""
    lines = [l.strip() for l in user.strip().split("\n") if l.strip()]
    while len(lines) > 1 and not lines[-1].endswith("?") and lines[-1].lower().startswith(_PROMPT_TAILS):
        lines.pop()
    text = " ".join(lines).strip()
    # the instruction may also follow the question on the same line: "...what shape? Your response must be concise."
    head, mark, tail = text.rpartition("?")
    if mark and tail.strip() and tail.strip().lower().startswith(_PROMPT_TAILS):
        text = head + "?"
    return text


def short_answer(assistant: str) -> str:
    """'Answer: 2.' / '2.' / 'Yes.' -> '2' / 'Yes'."""
    a = assistant.strip()
    if a.lower().startswith("answer:"):
        a = a[7:].strip()
    return a.rstrip(".").strip()


def yes_no(assistant: str) -> Optional[bool]:
    a = short_answer(assistant).lower()
    return True if a == "yes" else False if a == "no" else None


def multiple_choice(user: str, assistant: str):
    """'Question: ...\\nChoices:\\nA. x\\nB. y\\nAnswer with the letter.' + 'Answer: B' -> (question, options,
    answer) or None."""
    import re
    m = re.match(r"\s*(?:Question:\s*)?(.*?)\nChoices:\n(.*?)(?:\nAnswer with.*)?$", user.strip(), re.S)
    if not m:
        return None
    opts = re.findall(r"^([A-J])\.\s*(.+)$", m.group(2), re.M)
    letter = short_answer(assistant).strip()[:1].upper()
    by = {k: v.strip() for k, v in opts}
    if letter not in by or len(by) < 2 or len(set(by.values())) != len(by):
        return None
    return m.group(1).strip(), [by[k] for k, _ in opts], by[letter]


def short_turn(turn: dict):
    """A Cauldron short-answer turn -> ("yesno", question, bool) | ("number", question, int) |
    ("word", question, answer) | None."""
    q = question_text(turn["user"])
    a = short_answer(turn["assistant"])
    yn = yes_no(turn["assistant"])
    if yn is not None:
        return "yesno", q, yn
    if a.isdigit():
        return "number", q, int(a)
    if a and len(a.split()) <= 3:
        return "word", q, a
    return None


def keep_row(key: str, share: float, salt: str = "") -> bool:
    """Deterministic subsample of rows (by key), so a re-run keeps the same rows."""
    h = int(hashlib.sha1((salt + key).encode()).hexdigest()[:8], 16)
    return h < share * 0x100000000


def flatten_answers(rows: List[Sample], field: str = "answer_key", factor: float = 2.0) -> List[Sample]:
    """Cap every answer at `factor` x the mean count per answer (data-v2's rule against a dominant answer)."""
    by: Dict[str, List[Sample]] = defaultdict(list)
    for r in rows:
        by[str(r[field])].append(r)
    if not by:
        return rows
    cap = max(1, int(factor * len(rows) / len(by)))
    keep = {id(r) for v in by.values() for r in v[:cap]}
    return [r for r in rows if id(r) in keep]


def balance_yes_no(rows: List[Sample]) -> List[Sample]:
    """Per group, as many yes as no among the noul rows (choice rows untouched)."""
    from v2_common import equalize  # pyright: ignore[reportMissingImports]
    nouls = [r for r in rows if r["questions"]["q"]["type"] == "noul"]
    kept = {id(r) for r in equalize(nouls)}
    return [r for r in rows if r["questions"]["q"]["type"] != "noul" or id(r) in kept]
