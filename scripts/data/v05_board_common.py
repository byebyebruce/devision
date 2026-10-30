"""Shared pieces of the Decision Index board-gap training data (scripts/data/v05_board_*.py).

The board items (/home/bruce/ext/decision-index-data/vision) are EVAL ONLY. Every builder:
  * reads only train splits of its upstream sources (the board's upstream dev / test splits are listed in
    data/v05/board_exclusion.json under "upstream_splits" and never read);
  * passes every image through `BoardIndex.hit` (exact sha256 of the bytes, exact sha256 of the decoded pixels, or a
    near-duplicate by BOTH a 64-bit dHash and a 64-bit pHash of the grey image) and through the source-id list;
  * writes choice samples in the samples.py format with the gold position spread evenly (`place_gold`), splits
    train / dev by image (`dev_by_image`) and records counts, drops, exclusion hits, gold positions and text-only
    baselines in MANIFEST.json.
"""
import hashlib
import io
import json
import os
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LETTERS = "ABCDEFGHIJ"
DHASH_BITS = 6     # near-duplicate = dHash within DHASH_BITS AND pHash within PHASH_BITS
PHASH_BITS = 8
DEV_FRAC = 0.04


# ----------------------------------------------------------------------------- hashes

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _grey(im, size: Tuple[int, int]):
    from PIL import Image
    return im.convert("L").resize(size, Image.Resampling.BILINEAR)


def dhash(im, size: int = 8) -> int:
    """Difference hash (same bit order as lv_bench.dhash)."""
    import numpy as np
    g = np.asarray(_grey(im, (size + 1, size)), dtype=np.int16)
    bits = (g[:, :-1] < g[:, 1:]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _dct_matrix(n: int):
    import numpy as np
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


def phash(im, size: int = 32, keep: int = 8) -> int:
    """DCT perceptual hash: 32x32 grey -> 2-D DCT -> top-left 8x8 (minus DC) above their median."""
    import numpy as np
    g = np.asarray(_grey(im, (size, size)), dtype=np.float64)
    d = _dct_matrix(size)
    c = (d @ g @ d.T)[:keep, :keep].flatten()
    med = np.median(c[1:])
    bits = c > med
    return int("".join("1" if b else "0" for b in bits), 2)


def pixel_sha(im) -> str:
    im = im.convert("RGB")
    return hashlib.sha256(("%dx%d" % im.size).encode() + im.tobytes()).hexdigest()


def image_hashes(data: bytes) -> Dict[str, Any]:
    """sha256 of the bytes + of the decoded RGB pixels, dHash and pHash (hex) of one encoded image."""
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data))
    im.load()
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return {"sha256": sha256_bytes(data), "pixel_sha256": pixel_sha(im),
            "dhash": "%016x" % dhash(im), "phash": "%016x" % phash(im), "size": list(im.size)}


# ----------------------------------------------------------------------------- the board index

class BoardIndex:
    """Membership test against every board image (and source id)."""

    def __init__(self, index: Dict[str, Any], dhash_bits: int = DHASH_BITS, phash_bits: int = PHASH_BITS):
        import numpy as np
        imgs = index["images"]
        self.sha = {i["sha256"]: i["path"] for i in imgs}
        self.pix = {i["pixel_sha256"]: i["path"] for i in imgs}
        self.paths = [i["path"] for i in imgs]
        self.d = np.array([int(i["dhash"], 16) for i in imgs], dtype=np.uint64)
        self.p = np.array([int(i["phash"], 16) for i in imgs], dtype=np.uint64)
        self.ids = set(index.get("source_ids", []))
        self.dhash_bits, self.phash_bits = dhash_bits, phash_bits

    @classmethod
    def load(cls, path: str, **kw) -> "BoardIndex":
        with open(path) as f:
            return cls(json.load(f), **kw)

    def hit_hashes(self, h: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """(kind, board image) when the hashed image is a board image, else None."""
        import numpy as np
        if h["sha256"] in self.sha:
            return "sha256", self.sha[h["sha256"]]
        if h["pixel_sha256"] in self.pix:
            return "pixels", self.pix[h["pixel_sha256"]]
        if len(self.d):
            dd = np.bitwise_count(self.d ^ np.uint64(int(h["dhash"], 16)))
            pp = np.bitwise_count(self.p ^ np.uint64(int(h["phash"], 16)))
            near = np.nonzero((dd <= self.dhash_bits) & (pp <= self.phash_bits))[0]
            if len(near):
                return "near", self.paths[int(near[0])]
        return None

    def hit(self, data: bytes) -> Optional[Tuple[str, str]]:
        return self.hit_hashes(image_hashes(data))

    def hit_id(self, source_id: str) -> bool:
        return source_id in self.ids


# ----------------------------------------------------------------------------- samples

def seeded(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def place_gold(gold: str, distractors: Sequence[str], slot: int) -> List[str]:
    """Options with the gold at index `slot` (callers rotate `slot` so positions stay balanced)."""
    opts = list(distractors)
    opts.insert(slot % (len(opts) + 1), gold)
    return opts


def choice_sample(sid: str, source: str, image_id: str, image: str, question: str, options: Sequence[str],
                  gold: str, state_text: str = "", kind: str = "") -> Dict[str, Any]:
    assert options.count(gold) == 1, (sid, gold, options)
    assert len({o.casefold().strip() for o in options}) == len(options), (sid, options)
    crit = {LETTERS[i]: o for i, o in enumerate(options)}
    g = LETTERS[list(options).index(gold)]
    s = {"id": sid, "source": source, "image_id": image_id, "image": image,
         "questions": {"q": {"type": "choice", "instructions": question, "criteria": crit}},
         "gold": {"q": {"probabilities": {k: (1.0 if k == g else 0.0) for k in crit}}}}
    if state_text:
        s["state_text"] = state_text
    if kind:
        s["kind"] = kind
    return s


def gold_letter(s: Dict[str, Any]) -> str:
    return next(k for k, p in s["gold"]["q"]["probabilities"].items() if p == 1.0)


def gold_text(s: Dict[str, Any]) -> str:
    return s["questions"]["q"]["criteria"][gold_letter(s)]


def dev_by_image(image_id: str, frac: float = DEV_FRAC) -> bool:
    return int(hashlib.sha256(("dev:" + image_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def template(question: str) -> str:
    """The question with quoted spans and numbers blanked ("What is the price of "x"?" -> "What is the price of _?")."""
    import re
    return re.sub(r"\d+([.,]\d+)*", "#", re.sub(r'"[^"]*"', "_", question))


def text_only(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Image-blind baselines: first / longest / shortest option, and the option text most often gold for this question
    template in the OTHER rows (leave-one-out, ties -> first option), plus the gold position histogram."""
    if not rows:
        return {}
    n, hits, pos, chance = len(rows), Counter(), Counter(), 0.0
    prior: Dict[str, Counter] = {}
    for r in rows:
        prior.setdefault(template(r["questions"]["q"]["instructions"]), Counter())[gold_text(r).casefold()] += 1
    for r in rows:
        crit = r["questions"]["q"]["criteria"]
        opts, g = [crit[k] for k in crit], gold_text(r)
        pos[gold_letter(r)] += 1
        chance += 1 / len(opts)
        hits["first"] += opts[0] == g
        hits["longest"] += max(opts, key=len) == g
        hits["shortest"] += min(opts, key=len) == g
        p = prior[template(r["questions"]["q"]["instructions"])]
        hits["question_prior"] += max(opts, key=lambda o: p.get(o.casefold(), 0) - (o == g)) == g
    return {"n": n, "chance": round(chance / n, 4), **{k: round(v / n, 4) for k, v in sorted(hits.items())},
            "gold_position": {k: pos[k] for k in sorted(pos)}}


def write_split(out_dir: str, rows: Iterable[Dict[str, Any]], dev_frac: float = DEV_FRAC) -> Dict[str, int]:
    """data/v05/board/<family>/{train,dev}.jsonl (dev = held-out images of the same train sources)."""
    from devision.train.samples import sample_errors
    os.makedirs(out_dir, exist_ok=True)
    counts: Counter = Counter()
    with open(os.path.join(out_dir, "train.jsonl"), "w") as ft, open(os.path.join(out_dir, "dev.jsonl"), "w") as fd:
        for r in rows:
            errs = sample_errors(r)
            assert not errs, (r["id"], errs)
            split = "dev" if dev_by_image(r["image_id"], dev_frac) else "train"
            (fd if split == "dev" else ft).write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[split] += 1
    return dict(counts)


def manifest(out_dir: str, meta: Dict[str, Any]) -> None:
    rows = {sp: [json.loads(l) for l in open(os.path.join(out_dir, sp + ".jsonl"))] for sp in ("train", "dev")}
    meta = dict(meta)
    meta["counts"] = {sp: len(rs) for sp, rs in rows.items()}
    meta["images"] = {sp: len({r["image_id"] for r in rs}) for sp, rs in rows.items()}
    meta["by_kind"] = {sp: dict(Counter(r.get("kind", "") for r in rs).most_common()) for sp, rs in rows.items()}
    meta["text_only"] = {sp: text_only(rs) for sp, rs in rows.items()}
    meta["train_dev_image_overlap"] = len({r["image_id"] for r in rows["train"]} & {r["image_id"] for r in rows["dev"]})
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
