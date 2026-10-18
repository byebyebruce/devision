"""Optional reading mode (exp56; docs/spec/reading-mode.md): a frozen automatic number reader in front of ordinary decide.

From the request's own picture: the full picture and its 12 fixed local views (the decision checkpoint's own
local-view path, one encoding per request) -> a region selector (picture + natural question -> one box) -> a character
head on the box's tokens -> strict decoding -> a fact table with thousands separators -> ordinary decide on the text
state plus the table (image=None, as validated in exp55). Nothing is trained here.

Supported questions -- a fixed, limited set, matched on the natural question text only (never on ids, file names,
dataset fields, gold, options or precomputed boxes):
  - six reading questions: the mass of each particle / the average particle speed in sample A or B; the distance
    between the magnets in Pair 1 or 2;
  - two comparison questions: which sample has the higher temperature (kinetic energies of the particles), and the
    magnetic force between the magnets in each pair.
A request with any supported question reads every value of its domain (temperature: 4, magnets: 2) -- the same set
whatever the question ids or option order -- and the supported questions are decided on the text with the table;
any other question in the request goes through ordinary decide with the picture, unchanged. A request with no supported
question is exactly ordinary decide. The magnet table holds distances only: magnet size / pole facts are not read.

The components are frozen copies of exp53's selector, exp51's character head (scripts/exp49/probe.Head), exp51's
region rule, exp53's box rule and exp55's separator format; their behaviour is checked against the originals by the
CPU fixtures. A reading bundle (selector.pt, head.pt, reading.json) is bound to the decision checkpoint's weights /
config / tokenizer by sha256; any mismatch is refused.
"""
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .image import patch_geometry, view_rects

CHARS, EOS, SLOTS, CLASSES = "0123456789.", 11, 5, 12
VALID = re.compile(r"^[0-9]+(\.[0-9]+)?$")
UNREADABLE = "unreadable"
GRID = {"n": 12, "overlap": 0.25, "size": 256, "side": 16}
POLICY = "templates-v1"
MAX_Q = 48
HEADER = "Values read from the picture (numbers only; units as in the question):"
# the six reading templates (natural question -> object, quantity); their order is the table's row order
TEMPLATES = {
    "What is the mass of each particle in sample A?": ("Sample A", "mass of each particle"),
    "What is the mass of each particle in sample B?": ("Sample B", "mass of each particle"),
    "What is the average particle speed in sample A?": ("Sample A", "average particle speed"),
    "What is the average particle speed in sample B?": ("Sample B", "average particle speed"),
    "What is the distance between the magnets in Pair 1?": ("Pair 1", "distance between the magnets"),
    "What is the distance between the magnets in Pair 2?": ("Pair 2", "distance between the magnets"),
}
DOMAINS = {"temperature": list(TEMPLATES)[:4], "magnets": list(TEMPLATES)[4:]}


# ------------------------------------------------------------------ routing (question text only)

def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


_READ = {_norm(t): t for t in TEMPLATES}


def domain_of(instructions: str) -> Optional[str]:
    """The reading domain a natural question belongs to, or None (unsupported)."""
    n = _norm(instructions)
    if n in _READ:
        return "temperature" if _READ[n] in DOMAINS["temperature"] else "magnets"
    if "higher temperature" in n and "kinetic energ" in n and "sample" in n:
        return "temperature"
    if "magnetic force" in n and "pair" in n:
        return "magnets"
    return None


def reads_for(questions: Dict[str, Dict]) -> List[str]:
    """The internal reading questions of a request: every template of every domain its questions belong to, in the
    fixed order -- independent of question ids and options."""
    doms = {domain_of(q.get("instructions", "")) for q in questions.values()} - {None}
    return [t for t in TEMPLATES if any(t in DOMAINS[d] for d in doms if d is not None)]


# ------------------------------------------------------------------ frozen components (copies, see the fixtures)

class Selector(nn.Module):
    """exp53's region selector (scripts/exp53/selector.Selector), unchanged."""

    def __init__(self, vision_dim: int = 768, text_dim: int = 1024, width: int = 256, heads: int = 4):
        super().__init__()
        self.patch = nn.Linear(vision_dim, width)
        self.geom = nn.Linear(4, width)
        self.word = nn.Linear(text_dim, width)
        self.pos = nn.Embedding(MAX_Q, width)
        self.context = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.ask = nn.MultiheadAttention(width, heads, batch_first=True)
        self.look1 = nn.MultiheadAttention(width, heads, batch_first=True)
        self.look2 = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.box = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))

    def forward(self, patches, geom, words, word_mask):
        pos = self.pos(torch.arange(words.size(1), device=words.device))[None]
        q = self.context(self.word(words.float()) + pos, src_key_padding_mask=~word_mask)
        img = self.patch(patches.float()) + self.geom(geom.float())
        s = self.query[None].expand(words.size(0), -1, -1)
        s = s + self.ask(s, q, q, key_padding_mask=~word_mask, need_weights=False)[0]
        s = s + self.look1(s, img, img, need_weights=False)[0]
        s = s + self.look2(self.norm(s), img, img, need_weights=False)[0]
        return torch.sigmoid(self.box(s[:, 0]))


class CharHead(nn.Module):
    """exp49 / exp51's character head (scripts/exp49/probe.Head), unchanged."""

    def __init__(self, d_in: int = 768, width: int = 256, heads: int = 4, max_rel: int = 16):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.row, self.col = nn.Embedding(max_rel, width), nn.Embedding(max_rel, width)
        self.geom = nn.Linear(4, width)
        self.queries = nn.Parameter(torch.randn(SLOTS, width) * 0.02)
        self.att = nn.MultiheadAttention(width, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, CLASSES))

    def forward(self, x, mask, rel, geom):
        kv = self.inp(x.float()) + self.row(rel[..., 0]) + self.col(rel[..., 1]) + self.geom(geom.float())
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        o = self.att(q, kv, kv, key_padding_mask=~mask, need_weights=False)[0]
        return self.out(o)


def region(box, w: int, h: int, n: int = 12, overlap: float = 0.25, size: int = 256, side: int = 16) -> Dict:
    """exp51's region rule (scripts/exp51/regions.region), unchanged."""
    bx, by, bw, bh = box
    views = [(0, 0, w, h)] + view_rects(w, h, n, overlap)
    out: Dict[str, List] = {"index": [], "view": [], "rel": [], "geom": []}
    for k, (x0, y0, x1, y1) in enumerate(views):
        g = patch_geometry((x0, y0, x1, y1), w, h, size, side).tolist()
        cells = []
        for i, (cx, cy, pw, ph) in enumerate(g):
            px0, px1 = cx * w - pw * w / 2, cx * w + pw * w / 2
            py0, py1 = cy * h - ph * h / 2, cy * h + ph * h / 2
            ix = min(px1, x1, bx + bw) - max(px0, x0, bx)
            iy = min(py1, y1, by + bh) - max(py0, y0, by)
            if ix > 1e-9 and iy > 1e-9:
                cells.append(i)
        if not cells:
            continue
        r0, c0 = min(i // side for i in cells), min(i % side for i in cells)
        for i in cells:
            cx, cy, pw, ph = g[i]
            out["index"].append(k * side * side + i)
            out["view"].append(k)
            out["rel"].append([i // side - r0, i % side - c0])
            out["geom"].append([(cx * w - bx) / bw, (cy * h - by) / bh, pw * w / bw, ph * h / bh])
    return out


def pixel_box(frac: Sequence[float], w: int, h: int) -> Tuple[List[float], bool]:
    """exp53's box rule (scripts/exp53/selector.pixel_box), unchanged."""
    cx, cy, bw, bh = (float(v) for v in frac)
    x0, x1 = max(0.0, (cx - bw / 2) * w), min(float(w), (cx + bw / 2) * w)
    y0, y1 = max(0.0, (cy - bh / 2) * h), min(float(h), (cy + bh / 2) * h)
    bad = False
    if x1 - x0 < 1:
        c = min(max((x0 + x1) / 2, 0.5), w - 0.5)
        x0, x1, bad = c - 0.5, c + 0.5, True
    if y1 - y0 < 1:
        c = min(max((y0 + y1) / 2, 0.5), h - 0.5)
        y0, y1, bad = c - 0.5, c + 0.5, True
    return [x0, y0, x1 - x0, y1 - y0], bad


def decode(ids: List[int]) -> Dict:
    """exp49's strict decoding: the characters before the first EOS; valid only with an EOS and a strict number."""
    if EOS not in ids:
        return {"text": "".join(CHARS[i] for i in ids if i < len(CHARS)), "valid": False}
    text = "".join(CHARS[i] for i in ids[:ids.index(EOS)])
    return {"text": text, "valid": bool(VALID.match(text))}


def fmt(value: str) -> str:
    """exp55's thousands separators (integer part only; UNREADABLE unchanged; anything else refused)."""
    if value == UNREADABLE:
        return value
    if not VALID.match(value):
        raise ValueError("not a strict number: %r" % value)
    whole, dot, frac = value.partition(".")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    return ",".join([whole] + groups) + dot + frac


def serialise(state_text: str, table: List[List[str]]) -> str:
    """exp54's table serialisation (header and one row per value)."""
    lines = ["%s: %s = %s" % (o, q, v) for o, q, v in table]
    return (state_text + "\n" if state_text else "") + HEADER + "\n" + "\n".join(lines)


# ------------------------------------------------------------------ the pipeline

def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(path: str) -> str:
    """sha256 of a directory: its files' relative paths and sha256, sorted."""
    items = []
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            items.append([os.path.relpath(p, path), sha(p)])
    return hashlib.sha256(json.dumps(sorted(items)).encode()).hexdigest()


class Reading:
    """The frozen reader bound to one decision checkpoint (a `Decider` with local views 12 / 0.25 / T)."""

    def __init__(self, decider, selector: Selector, head: CharHead, identity: Dict):
        cfg = decider.cfg
        if (cfg.local_views, cfg.local_overlap, cfg.local_source) != (GRID["n"], GRID["overlap"], "T"):
            raise ValueError("reading mode needs the decision checkpoint's local views 12 / 0.25 / T")
        self.decider, self.selector, self.head, self.identity = decider, selector.eval(), head.eval(), identity
        for p in list(selector.parameters()) + list(head.parameters()):
            p.requires_grad_(False)

    @classmethod
    def load(cls, decision_dir: str, bundle_dir: str, device: str = "auto") -> "Reading":
        """The verified entry: checks every sha256 in the bundle's reading.json -- its two component files and the
        decision checkpoint's weights / config / tokenizer -- and only then loads the Decider from that same
        `decision_dir` itself (no caller-supplied model), then the components."""
        from .infer import Decider, best_device
        ident = json.load(open(os.path.join(bundle_dir, "reading.json")))
        checks = {"selector.pt": sha(os.path.join(bundle_dir, "selector.pt")), "head.pt": sha(os.path.join(bundle_dir, "head.pt")),
                  "model.safetensors": sha(os.path.join(decision_dir, "model.safetensors")),
                  "config.json": sha(os.path.join(decision_dir, "config.json")),
                  "tokenizer": tree_sha(os.path.join(decision_dir, "tokenizer"))}
        bad = [k for k, v in checks.items() if ident["sha256"].get(k) != v]
        if bad or ident.get("policy") != POLICY or ident.get("grid") != GRID:
            raise ValueError("reading bundle does not bind this checkpoint / its components: %s" % (bad or "policy / grid"))
        decider = Decider.load(decision_dir, device=best_device() if device == "auto" else device)
        return cls._attach(decider, bundle_dir, ident)

    @classmethod
    def _attach(cls, decider, bundle_dir: str, ident: Dict) -> "Reading":
        """The components on an already loaded Decider WITHOUT checking that the Decider is the bound checkpoint --
        for `load` and CPU fixtures only; not a verified public entry."""
        dev = next(decider.model.parameters()).device
        sel = Selector(*ident["dims"]["selector"])
        sel.load_state_dict(torch.load(os.path.join(bundle_dir, "selector.pt"), map_location="cpu"), strict=True)
        head = CharHead(ident["dims"]["head"])
        head.load_state_dict(torch.load(os.path.join(bundle_dir, "head.pt"), map_location="cpu"), strict=True)
        return cls(decider, sel.to(dev), head.to(dev), ident)

    def _words(self, texts: List[str]):
        from .model import text_encoder
        emb: Any = text_encoder(self.decider.model.decision).get_input_embeddings()
        dev = emb.weight.device
        ids = [self.decider.tok(t, add_special_tokens=False)["input_ids"][:MAX_Q] for t in texts]
        n = max(map(len, ids))
        words = torch.zeros(len(texts), n, emb.weight.shape[1], dtype=torch.float16, device=dev)
        mask = torch.zeros(len(texts), n, dtype=torch.bool, device=dev)
        for i, x in enumerate(ids):
            words[i, :len(x)] = emb(torch.tensor(x, device=dev)).half()      # exp53's float16 boundary
            mask[i, :len(x)] = True
        return words, mask

    @torch.no_grad()
    def read(self, image, templates: List[str]) -> Dict:
        """{"table", "reads": [{"question", "box", "degenerate", "decoded", "valid", "value"}], "seconds"} for the given
        reading templates on one picture (one visual encoding)."""
        import time

        from .infer import _image_arg, image_tensor
        t0 = time.perf_counter()
        img = _image_arg(image)
        w, h = img.size
        m = self.decider.model
        dev = next(m.parameters()).device
        kv = m.visual_inputs(image_tensor(img, self.decider.cfg)[None].to(dev))[0][0]          # [3328, 768]
        patches = kv[:256].half()[None].expand(len(templates), -1, -1)                         # exp53's float16 boundary
        geom = patch_geometry((0, 0, w, h), w, h, GRID["size"], GRID["side"]).to(dev)[None].expand(len(templates), -1, -1)
        words, mask = self._words(templates)
        frac = self.selector(patches, geom, words, mask).float().cpu().tolist()
        reads, regs = [], []
        for t, f in zip(templates, frac):
            box, bad = pixel_box(f, w, h)
            regs.append(region(box, w, h))
            reads.append({"question": t, "box": box, "degenerate": bad})
        n = max(len(r["index"]) for r in regs)
        x = torch.zeros(len(regs), n, kv.shape[1], device=dev)
        msk = torch.zeros(len(regs), n, dtype=torch.bool, device=dev)
        rel = torch.zeros(len(regs), n, 2, dtype=torch.long, device=dev)
        gm = torch.zeros(len(regs), n, 4, device=dev)
        for i, r in enumerate(regs):
            k = len(r["index"])
            x[i, :k] = kv[torch.tensor(r["index"], device=dev)].half().float()                 # exp51's float16 boundary
            msk[i, :k] = True
            rel[i, :k] = torch.tensor(r["rel"], device=dev)
            gm[i, :k] = torch.tensor(r["geom"], dtype=torch.float32, device=dev)
        ids = self.head(x, msk, rel, gm).argmax(-1).cpu().tolist()
        table = []
        for rd, t, c in zip(reads, templates, ids):
            d = decode(c)
            rd.update(decoded=d["text"], valid=d["valid"], value=fmt(d["text"]) if d["valid"] else UNREADABLE)
            table.append([*TEMPLATES[t], rd["value"]])
        return {"table": table, "reads": reads, "seconds": time.perf_counter() - t0}

    def decide(self, state: Any, questions: Dict[str, Dict], image: Any = None, diagnostics: Optional[Dict] = None) -> Dict:
        """Ordinary decide's response (Jev structure unchanged). Supported questions are decided on the text state plus
        the table read from `image`; the others, and every request without an image or without a supported question,
        go through ordinary decide unchanged. `diagnostics` (optional dict) receives the reads and timings.
        Encoding: a request whose questions are all supported encodes the picture once; a mixed request encodes it
        twice (once for the reading, once in ordinary decide for the other questions)."""
        from .infer import _split_state
        templates = reads_for(questions)
        if image is None or not templates:
            return self.decider.decide(state, questions, image=image)
        supported = {k: q for k, q in questions.items() if domain_of(q.get("instructions", "")) is not None}
        others = {k: q for k, q in questions.items() if k not in supported}
        text_state, state_image = _split_state(state)
        if state_image is not None:
            raise ValueError("give the image as image=, not inside state, in reading mode")
        r = self.read(image, templates)
        text = text_state if isinstance(text_state, str) else (json.dumps(text_state) if text_state else "")
        res = self.decider.decide(serialise(text, r["table"]), supported, image=None)
        if others:
            rest = self.decider.decide(state, others, image=image)
            res["answers"].update(rest["answers"])
            res["usage"] = {k: res["usage"].get(k, 0) + rest["usage"].get(k, 0) for k in res["usage"]}
        if diagnostics is not None:
            diagnostics.update(table=r["table"], reads=r["reads"], read_seconds=r["seconds"], state=serialise(text, r["table"]))
        return res


def write_bundle(out_dir: str, decision_dir: str, selector_pt: str, head_pt: str, sources: Dict, dims: Optional[Dict] = None) -> Dict:
    """A reading bundle: the two frozen component files copied as they are, and reading.json binding them and the
    decision checkpoint (weights / config / tokenizer) by sha256 (no tensor is loaded)."""
    import shutil
    os.makedirs(out_dir, exist_ok=False)
    shutil.copyfile(selector_pt, os.path.join(out_dir, "selector.pt"))
    shutil.copyfile(head_pt, os.path.join(out_dir, "head.pt"))
    ident = {"policy": POLICY, "grid": GRID, "templates": TEMPLATES, "domains": DOMAINS, "dims": dims or {"selector": [768, 1024], "head": 768},
             "sha256": {"selector.pt": sha(os.path.join(out_dir, "selector.pt")), "head.pt": sha(os.path.join(out_dir, "head.pt")),
                        "model.safetensors": sha(os.path.join(decision_dir, "model.safetensors")),
                        "config.json": sha(os.path.join(decision_dir, "config.json")),
                        "tokenizer": tree_sha(os.path.join(decision_dir, "tokenizer"))},
             "sources": sources, "float16": "selector patches and words, head region features (as in exp53 / exp51)"}
    with open(os.path.join(out_dir, "reading.json"), "w") as fh:
        json.dump(ident, fh, indent=1)
    return ident


__all__ = ["Reading", "write_bundle", "reads_for", "domain_of", "TEMPLATES", "UNREADABLE", "fmt", "decode"]
