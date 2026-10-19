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
TABLE_FORMAT = "exp54-v1"              # serialise(): the state text, HEADER, one "object: quantity = value" row per value


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


# LIVE-K-PREP-01 (docs/experiments/2026-10-19-live-k-plan.md): the optional K rule, EXP70's fixed K block computed from
# the table actually read -- off unless the bundle declares it (reading.json "k_rule", part of its rules hash)
K_RULE = {"name": "temperature-k-v1", "header": "Computed from the values above (mass times speed squared, no unit):",
          "quantity": "mass times speed squared", "objects": ["Sample A", "Sample B"],
          "inputs": ["mass of each particle", "average particle speed"], "formula": "mass * speed ** 2, exact integers",
          "defined": "all four values readable and integral; otherwise no K block", "applies_to": "temperature questions only"}


def k_block(table: List[List[str]]) -> Optional[str]:
    """EXP70's K block (scripts/exp70/build.py k_block) from a read table: K = mass x speed^2 per sample, A then B, exact
    integers with thousands separators -- or None when a value is missing, unreadable or not an integer (the request is
    then decided without K). Nothing but the table's own values is used."""
    vals = {(o, q): v for o, q, v in table}
    ks = []
    for obj in K_RULE["objects"]:
        pair = [vals.get((obj, q)) for q in K_RULE["inputs"]]
        if any(v is None or v == UNREADABLE for v in pair):
            return None
        try:
            m, v = (float(str(x).replace(",", "")) for x in pair)
        except ValueError:
            return None
        if m != int(m) or v != int(v):
            return None
        ks.append(int(m) * int(v) * int(v))
    return "%s\n%s: %s = %s\n%s: %s = %s" % (K_RULE["header"], "Sample A", K_RULE["quantity"], fmt(str(ks[0])),
                                            "Sample B", K_RULE["quantity"], fmt(str(ks[1])))


def serialise(state_text: str, table: List[List[str]]) -> str:
    """exp54's table serialisation (header and one row per value)."""
    lines = ["%s: %s = %s" % (o, q, v) for o, q, v in table]
    return (state_text + "\n" if state_text else "") + HEADER + "\n" + "\n".join(lines)


# ------------------------------------------------------------------ the optional speed-region corrector (exp69)

SPEED = ("What is the average particle speed in sample A?", "What is the average particle speed in sample B?")
CORRECTOR = {"templates": list(SPEED), "grow_y_box_heights": 0.5, "grow_x_px": 8.0, "views": "local (view > 0)"}


def corrector_window(box: Sequence[float], w: int, h: int) -> List[float]:
    """exp69's window around the selector's box [x, y, w, h]: grown by 0.5 of its height above / below and 8 px left /
    right, clipped to the canvas, as xyxy (scripts/exp69 window_of / build.window, unchanged)."""
    x, y, bw, bh = box
    b = [x, y, x + bw, y + bh]
    hh = b[3] - b[1]
    gy, gx = CORRECTOR["grow_y_box_heights"], CORRECTOR["grow_x_px"]
    return [max(0.0, b[0] - gx), max(0.0, b[1] - gy * hh), min(float(w), b[2] + gx), min(float(h), b[3] + gy * hh)]


def corrector_tokens(win: Sequence[float], w: int, h: int) -> Dict[str, List]:
    """region() of the window, local views only: token indices and their geom against the window (exp69, unchanged)."""
    r = region([win[0], win[1], win[2] - win[0], win[3] - win[1]], w, h)
    keep = [i for i, v in enumerate(r["view"]) if v > 0]
    return {"index": [r["index"][i] for i in keep], "geom": [r["geom"][i] for i in keep]}


def corrector_box(frac: Sequence[float], win: Sequence[float], w: int, h: int) -> List[float]:
    """A window-fraction (cx, cy, w, h) to a pixel box [x, y, w, h]: clipped to the window, at least 1 px each side,
    inside the canvas (exp69 to_pixels, unchanged)."""
    cx, cy, bw, bh = frac
    ww, wh = win[2] - win[0], win[3] - win[1]
    x0, x1 = max(win[0], win[0] + (cx - bw / 2) * ww), min(win[2], win[0] + (cx + bw / 2) * ww)
    y0, y1 = max(win[1], win[1] + (cy - bh / 2) * wh), min(win[3], win[1] + (cy + bh / 2) * wh)
    if x1 - x0 < 1:
        c = min(max((x0 + x1) / 2, win[0] + 0.5), win[2] - 0.5)
        x0, x1 = c - 0.5, c + 0.5
    if y1 - y0 < 1:
        c = min(max((y0 + y1) / 2, win[1] + 0.5), win[3] - 0.5)
        y0, y1 = c - 0.5, c + 0.5
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    return [x0, y0, x1 - x0, y1 - y0]


def corrector_forward(m: "Selector", patches, geom, pmask, words, wmask):
    """Selector.forward with a key padding mask on the picture tokens (exp69's corrector, unchanged)."""
    pos = m.pos(torch.arange(words.size(1), device=words.device))[None]
    q = m.context(m.word(words.float()) + pos, src_key_padding_mask=~wmask)
    img = m.patch(patches.float()) + m.geom(geom.float())
    s = m.query[None].expand(words.size(0), -1, -1)
    s = s + m.ask(s, q, q, key_padding_mask=~wmask, need_weights=False)[0]
    s = s + m.look1(s, img, img, key_padding_mask=~pmask, need_weights=False)[0]
    s = s + m.look2(m.norm(s), img, img, key_padding_mask=~pmask, need_weights=False)[0]
    return torch.sigmoid(m.box(s[:, 0]))


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


def bundle_state(checkpoint: str) -> str:
    """The one check of a checkpoint's reading bundle, shared by devision.load, check_eval freeze and the release
    build: "none" (no reading/ at all: a plain decision checkpoint), "complete" (reading.json and every component it
    lists are there; their hashes are checked by Reading.load) or "incomplete" (reading/ exists but its manifest or a
    listed component is missing -- refused everywhere except an explicit reading="off" control)."""
    d = os.path.join(checkpoint, "reading")
    if not os.path.lexists(d):
        return "none"
    try:
        ident = json.load(open(os.path.join(d, "reading.json")))
        listed = [c for c in ("selector.pt", "head.pt", "corrector.pt") if c in ident["sha256"]]
    except (OSError, ValueError, KeyError, TypeError):
        return "incomplete"
    ok = {"selector.pt", "head.pt"} <= set(listed) and all(os.path.isfile(os.path.join(d, c)) for c in listed)
    return "complete" if ok else "incomplete"


def rules() -> Dict[str, Any]:
    """The reading rules in this code, as a bundle's reading.json records them (JSON round trip: tuples -> lists)."""
    return json.loads(json.dumps({"policy": POLICY, "grid": GRID, "templates": TEMPLATES, "domains": DOMAINS, "header": HEADER,
                                  "table_format": TABLE_FORMAT, "corrector": CORRECTOR}))


def rules_sha(r: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()


def check_rules(ident: Dict[str, Any], full: bool = False) -> List[str]:
    """The rules a bundle declares that differ from this code's (empty: all agree). Older bundles (exp66-exp71) carry
    no header / table_format / rules_sha256: they are compared on the rest, unless `full` (the release entry), which
    requires all three."""
    code = rules()
    bad = [k for k in ("policy", "grid", "templates", "domains") if ident.get(k) != code[k]]
    if "corrector.pt" in ident.get("sha256", {}) and ident.get("corrector") != code["corrector"]:
        bad.append("corrector")
    for k in ("header", "table_format"):
        if (k in ident or full) and ident.get(k) != code[k]:
            bad.append(k)
    if "k_rule" in ident and ident["k_rule"] != json.loads(json.dumps(K_RULE)):
        bad.append("k_rule")
    if ("rules_sha256" in ident or full) and ident.get("rules_sha256") != rules_sha(declared_rules(ident)):
        bad.append("rules_sha256")
    return bad


def declared_rules(ident: Dict[str, Any]) -> Dict[str, Any]:
    """The rule entries of a reading.json (the corrector rule only when the bundle has a corrector)."""
    keys = ["policy", "grid", "templates", "domains", "header", "table_format"] + (["corrector"] if "corrector.pt" in ident.get("sha256", {}) else []) \
        + (["k_rule"] if "k_rule" in ident else [])
    return {k: ident.get(k) for k in keys}


class Reading:
    """The frozen reader bound to one decision checkpoint (a `Decider` with local views 12 / 0.25 / T)."""

    def __init__(self, decider, selector: Selector, head: CharHead, identity: Dict, corrector: Optional[Selector] = None):
        cfg = decider.cfg
        if (cfg.local_views, cfg.local_overlap, cfg.local_source) != (GRID["n"], GRID["overlap"], "T"):
            raise ValueError("reading mode needs the decision checkpoint's local views 12 / 0.25 / T")
        self.decider, self.selector, self.head, self.identity = decider, selector.eval(), head.eval(), identity
        self.corrector = corrector.eval() if corrector is not None else None     # optional (exp69): speed templates only
        self.reading_status: Dict[str, Any] = {"mode": "on", "reading_json_sha256": None, "rules_sha256": identity.get("rules_sha256"),
                                               "corrector": corrector is not None, "k_rule": "k_rule" in identity}
        self.k_rule = "k_rule" in identity                                   # LIVE-K: off unless the bundle declares it
        for p in list(selector.parameters()) + list(head.parameters()) + (list(corrector.parameters()) if corrector is not None else []):
            p.requires_grad_(False)

    # the ordinary decider's interface, on the same objects (not copies): eval, serve and callers use these
    @property
    def cfg(self):
        return self.decider.cfg

    @property
    def model(self):
        return self.decider.model

    @property
    def tok(self):
        return self.decider.tok

    def use_raw_temperatures(self) -> None:
        """Every temperature this reader applies set to 1: the supported questions are decided by the same Decider,
        so this covers both paths."""
        self.decider.use_raw_temperatures()

    def predict(self, state: Any, questions: Any, image: Any = None) -> Dict:
        """Same as `decide`, under Laya's name."""
        return self.decide(state, questions, image=image)

    @classmethod
    def load(cls, decision_dir: str, bundle_dir: str, device: str = "auto", full_rules: bool = False) -> "Reading":
        """The verified entry: checks every sha256 in the bundle's reading.json -- its component files and the
        decision checkpoint's weights / config / tokenizer -- and its declared rules against this code (`check_rules`;
        `full_rules`: header / table format / rules hash required, as the release entry does), and only then loads
        the Decider from that same `decision_dir` itself (no caller-supplied model), then the components."""
        from .infer import Decider, best_device
        ident = json.load(open(os.path.join(bundle_dir, "reading.json")))
        checks = {"selector.pt": sha(os.path.join(bundle_dir, "selector.pt")), "head.pt": sha(os.path.join(bundle_dir, "head.pt")),
                  **({"corrector.pt": sha(os.path.join(bundle_dir, "corrector.pt"))} if "corrector.pt" in ident["sha256"] else {}),
                  "model.safetensors": sha(os.path.join(decision_dir, "model.safetensors")),
                  "config.json": sha(os.path.join(decision_dir, "config.json")),
                  "tokenizer": tree_sha(os.path.join(decision_dir, "tokenizer"))}
        bad = [k for k, v in checks.items() if ident["sha256"].get(k) != v]
        if bad:
            raise ValueError("reading bundle does not bind this checkpoint / its components: %s" % bad)
        rule_bad = check_rules(ident, full=full_rules)
        if rule_bad:
            raise ValueError("reading bundle's rules differ from this code: %s" % rule_bad)
        decider = Decider.load(decision_dir, device=best_device() if device == "auto" else device)
        r = cls._attach(decider, bundle_dir, ident)
        r.reading_status["reading_json_sha256"] = sha(os.path.join(bundle_dir, "reading.json"))
        return r

    @classmethod
    def _attach(cls, decider, bundle_dir: str, ident: Dict) -> "Reading":
        """The components on an already loaded Decider WITHOUT checking that the Decider is the bound checkpoint --
        for `load` and CPU fixtures only; not a verified public entry."""
        dev = next(decider.model.parameters()).device
        sel = Selector(*ident["dims"]["selector"])
        sel.load_state_dict(torch.load(os.path.join(bundle_dir, "selector.pt"), map_location="cpu"), strict=True)
        head = CharHead(ident["dims"]["head"])
        head.load_state_dict(torch.load(os.path.join(bundle_dir, "head.pt"), map_location="cpu"), strict=True)
        corr = None
        if "corrector.pt" in ident["sha256"]:
            corr = Selector(*ident["dims"]["selector"])
            corr.load_state_dict(torch.load(os.path.join(bundle_dir, "corrector.pt"), map_location="cpu"), strict=True)
            corr = corr.to(dev)
        return cls(decider, sel.to(dev), head.to(dev), ident, corr)

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
            rd: Dict[str, Any] = {"question": t, "box": box, "degenerate": bad}
            if self.corrector is not None and t in SPEED:                       # exp69: the speed box from local detail
                win = corrector_window(box, w, h)
                ct = corrector_tokens(win, w, h)
                cw, cm = self._words([t])
                c = corrector_forward(self.corrector, kv[torch.tensor(ct["index"], device=dev)].half().float()[None],
                                      torch.tensor(ct["geom"], dtype=torch.float32, device=dev)[None],
                                      torch.ones(1, len(ct["index"]), dtype=torch.bool, device=dev), cw, cm)[0].float().cpu().tolist()
                rd = {"question": t, "box": corrector_box(c, win, w, h), "degenerate": False, "selector_box": box}
            regs.append(region(rd["box"], w, h))
            reads.append(rd)
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
        """Ordinary decide's response (Jev structure unchanged). The request is first resolved exactly as ordinary
        decide resolves it (`infer.resolve_request`: question checks, the picture from `image=` or an image part in
        `state`, both refused); then supported questions are decided on the text state plus the table read from that
        picture, and the others -- and every request without a picture or without a supported question -- go through
        ordinary decide unchanged. `diagnostics` (optional dict) receives each question's route ("read" /
        "ordinary"), the picture's pixel identity, and for a read request the reads, table, state and timing.
        Encoding: a request whose questions are all supported encodes the picture once; a mixed request encodes it
        twice (once for the reading, once in ordinary decide for the other questions)."""
        from .infer import picture_identity, resolve_request
        text_state, picture = resolve_request(state, questions, image)
        templates = reads_for(questions) if picture is not None else []
        supported = {k: q for k, q in questions.items() if templates and domain_of(q.get("instructions", "")) is not None}
        if diagnostics is not None:
            diagnostics.update(route={k: ("read" if k in supported else "ordinary") for k in questions}, picture=picture_identity(picture))
        if not supported:
            return self.decider._decide(text_state, questions, picture)
        others = {k: q for k, q in questions.items() if k not in supported}
        r = self.read(picture, templates)
        text = text_state if isinstance(text_state, str) else (json.dumps(text_state) if text_state else "")
        plain = serialise(text, r["table"])
        kb = k_block(r["table"]) if self.k_rule else None
        hot = {k: q for k, q in supported.items() if kb and domain_of(q.get("instructions", "")) == "temperature"}
        cold = {k: q for k, q in supported.items() if k not in hot}
        res = self.decider._decide(plain + "\n" + str(kb), hot, None) if hot else {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        if cold:                                     # magnets (and every question without a defined K): the old contract
            part = self.decider._decide(plain, cold, None)
            res["answers"].update(part["answers"])
            res["usage"] = {k: res["usage"].get(k, 0) + part["usage"].get(k, 0) for k in part["usage"]}
            res.setdefault("model", part.get("model"))
        if others:
            rest = self.decider._decide(text_state, others, picture)
            res["answers"].update(rest["answers"])
            res["usage"] = {k: res["usage"].get(k, 0) + rest["usage"].get(k, 0) for k in res["usage"]}
        res["answers"] = {k: res["answers"][k] for k in questions}       # the request's question order
        if diagnostics is not None:
            diagnostics.update(table=r["table"], reads=r["reads"], read_seconds=r["seconds"], state=plain + ("\n" + str(kb) if hot else ""),
                               k_block=kb if hot else None)
        return res


def write_bundle(out_dir: str, decision_dir: str, selector_pt: str, head_pt: str, sources: Dict, dims: Optional[Dict] = None,
                 corrector_pt: Optional[str] = None, k_rule: bool = False) -> Dict:
    """A reading bundle: the frozen component files copied as they are (the optional exp69 speed corrector too), and
    reading.json binding them and the decision checkpoint (weights / config / tokenizer) by sha256 (no tensor is
    loaded). Without corrector_pt the bundle is exactly the earlier two-component one."""
    import shutil
    os.makedirs(out_dir, exist_ok=False)
    shutil.copyfile(selector_pt, os.path.join(out_dir, "selector.pt"))
    shutil.copyfile(head_pt, os.path.join(out_dir, "head.pt"))
    if corrector_pt is not None:
        shutil.copyfile(corrector_pt, os.path.join(out_dir, "corrector.pt"))
    ident = {"policy": POLICY, "grid": GRID, "templates": TEMPLATES, "domains": DOMAINS, "header": HEADER, "table_format": TABLE_FORMAT,
             "dims": dims or {"selector": [768, 1024], "head": 768},
             "sha256": {"selector.pt": sha(os.path.join(out_dir, "selector.pt")), "head.pt": sha(os.path.join(out_dir, "head.pt")),
                        "model.safetensors": sha(os.path.join(decision_dir, "model.safetensors")),
                        "config.json": sha(os.path.join(decision_dir, "config.json")),
                        "tokenizer": tree_sha(os.path.join(decision_dir, "tokenizer"))},
             "sources": sources, "float16": "selector patches and words, head region features (as in exp53 / exp51)"}
    if corrector_pt is not None:
        ident["sha256"]["corrector.pt"] = sha(os.path.join(out_dir, "corrector.pt"))
        ident["corrector"] = CORRECTOR
    if k_rule:                                       # LIVE-K: a different combination, its own reading.json / rules hash
        ident["k_rule"] = K_RULE
    ident = json.loads(json.dumps(ident))
    ident["rules_sha256"] = rules_sha(declared_rules(ident))
    with open(os.path.join(out_dir, "reading.json"), "w") as fh:
        json.dump(ident, fh, indent=1)
    return ident


__all__ = ["Reading", "write_bundle", "bundle_state", "k_block", "K_RULE", "rules", "check_rules", "reads_for", "domain_of", "TEMPLATES", "UNREADABLE", "fmt", "decode"]
