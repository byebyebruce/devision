"""Real phone / web GUI "next action" choice questions from AITW (Android in the Wild) and Multimodal-Mind2Web.

    uv run --no-sync python scripts/data/v05_gui_fetch.py --root data     # download (ModelScope, resume-safe)
    uv run --no-sync python scripts/data/v05_gui_action.py --root data   # build data/v05/gui/{train,dev,test}.jsonl

One question per recorded step: the screenshot of that step + the user goal (in the instructions) + the previous
actions of the episode (state_text) -> choose the action the demonstrator took among 4-6 short English action
phrases. Rules (every drop is counted by reason in MANIFEST.json):

AITW (russwest/AitW on ModelScope = the AITW TFRecords as parquet: goal_info, screenshot, OCR / icon detector
annotations `ui_annotations_{positions,text,ui_types}` with positions (y, x, h, w) normalised, touch / lift point
normalised, action_type 4 dual-point / 3 type / 5 back / 6 home / 7 enter / 10 complete / 11 impossible):
  * dual point with |touch - lift| <= TAP_DIST is a tap, otherwise a scroll named after the content movement
    (finger moves up -> "scroll down").
  * tap gold: the annotated elements whose box contains the touch point; none -> drop "no_element"; containing
    elements with different labels -> drop "ambiguous_element"; label unreadable (no letter / digit, longer than
    MAX_LABEL) -> drop "unreadable_label"; another element elsewhere on the screen with the same label -> drop
    "duplicate_label". Text elements are named by their text (`tap "Settings"`), icons by the detector class
    (`tap the magnifying glass icon`).
  * status steps (complete / impossible) -> drop "status_action"; type with empty text -> drop "empty_type".
Mind2Web (osunlp/Multimodal-Mind2Web: task, full-page screenshot, cleaned HTML, target + negative candidate
elements with bounding boxes, the recorded action list):
  * the gold element label is the one in `target_action_reprs` ("[button]  Search -> CLICK"); empty -> drop
    "unreadable_label". The screenshot is cropped to a CROP_W x CROP_H window (full page width) that contains the
    target, at a seeded random vertical offset (so the target is not always centred); target outside the
    screenshot or taller than the window -> drop "target_outside".
  * other candidates: label = own text (from the cleaned HTML) or aria label / placeholder / title / alt; a
    candidate nested in / containing / overlapping the target box is the same widget and ignored; a different
    candidate inside the crop with the gold's label -> drop "duplicate_label".
Options (both sources): gold + up to N_SCREEN distractors that are actions on OTHER elements present in the same
screen (tap / click their label; for a type / select gold, typing the same value into another input of the screen
comes first) + generic actions (scroll down / up, go back, go to home screen, press enter, and typing a text the
same episode types at another step) until N_OPTIONS options. Fewer than MIN_SCREEN same-screen distractors -> drop
"few_screen_elements". Back / home never appear together as gold + distractor (both can be valid); option strings
are distinct case-insensitively. Order: random.Random(sha256(id)) shuffle.

Scroll golds are capped at SCROLL_MAX of each source (lowest sha256(id) kept, the rest counted as "scroll_cap").
Splits (by group, so dev / test goals or websites are never seen in train): AITW general by normalised goal text;
AITW web_shopping by the website named in the goal (SITES; the 7 big sites and goals naming no site are train,
the small sites in SHOP_HELD are dev / test); Mind2Web: the official test_website / test_domain websites
(unseen in the official train) are dev / test by a hash of the website, train = official train + test_task.
Dev / test are capped (DEV_CAP / TEST_CAP per source; groups in hash order, at most ~PER_GROUP questions = whole
episodes per group); held-out rows past the caps are not used anywhere. Every image is checked against the existing UI eval sets (EVAL_SETS) by perceptual hash.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

OUT = os.path.join("v05", "gui")
TAP_DIST = 0.04
MAX_LABEL = 40
N_OPTIONS = 5
N_SCREEN = 3
MIN_SCREEN = 2
CROP_W, CROP_H = 1280, 1000
SAVE_MAX = 1024
DEV_FRAC, TEST_FRAC = 0.05, 0.10
DEV_CAP = {"aitw-general": 350, "aitw-shopping": 350, "mind2web": 400}   # whole groups -> a bit above
TEST_CAP = {"aitw-general": 700, "aitw-shopping": 700, "mind2web": 800}
EVAL_SETS = ("v05/exp82v2/ui_dev.jsonl", "v05/exp85/ui_dev.jsonl", "v05/exp85/ui_dev_rotated.jsonl")
NEAR_BITS = 6
SCROLL_MAX = 0.2
PER_GROUP = 30
PER_SITE = {"aitw-shopping": 150}   # its held-out groups are a few whole websites
STD_MIN = 8.0
LAYER_ROLE = re.compile(r"^(menu|menuitem|menubar|listbox|option|dialog|tooltip|tabpanel|alertdialog)$")
LAYER_CLASS = re.compile(r"dropdown|drop-down|flyout|submenu|sub-menu|megamenu|mega-menu|popover|popup|modal|"
                         r"tooltip|hidden|offscreen|sr-only|visually", re.I)
GENERIC = ("scroll down", "scroll up", "go back", "go to home screen", "press enter")
NAV = "ICON_NAV_BAR"
GROW = 1.4
BACK_ICONS = {"arrow backward"}
SIBLING = {"go back": "go to home screen", "go to home screen": "go back"}
CLICKABLE = {"a", "button", "input", "select", "textarea", "label", "span", "li", "img", "svg", "option", "div",
             "p", "h1", "h2", "h3", "h4", "td", "th"}
INPUTS = {"input", "textarea", "select"}
VOID = {"input", "img", "br", "meta", "link", "hr", "source", "area", "base", "col", "embed", "wbr"}

Box = Tuple[float, float, float, float]   # x0, y0, x1, y1


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.casefold()).split())


def clean(s: str) -> str:
    return " ".join(str(s).replace('"', "'").split())


def readable(label: str) -> bool:
    return 2 <= len(label) <= MAX_LABEL and bool(re.search(r"[A-Za-z0-9]", label))


def contains(b: Box, x: float, y: float) -> bool:
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def inter(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def area(a: Box) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def same_widget(a: Box, b: Box) -> bool:
    """Nested or heavily overlapping boxes are treated as one widget."""
    i = inter(a, b)
    return i > 0 and (i >= 0.8 * min(area(a), area(b)) or i / max(1e-9, area(a) + area(b) - i) >= 0.3)


# ----------------------------------------------------------------------------- options

def tap(label: str, icon: bool = False, verb: str = "tap") -> str:
    return "%s the %s icon" % (verb, label) if icon else '%s "%s"' % (verb, label)


def make_options(qid: str, gold: str, screen: Sequence[str], generic: Sequence[str]) -> Optional[Tuple[List[str], int]]:
    """Options (shuffled) and how many came from the screen; None when fewer than MIN_SCREEN screen distractors."""
    seen = {gold.casefold()}
    rng = random.Random(sha("d:" + qid))
    pick: List[str] = []
    for o in screen:
        if o.casefold() not in seen and len(pick) < N_SCREEN:
            seen.add(o.casefold())
            pick.append(o)
    if len(pick) < MIN_SCREEN:
        return None
    n_screen = len(pick)
    gen = [g for g in generic if g.casefold() not in seen and SIBLING.get(gold) != g]
    rng.shuffle(gen)
    for g in gen:
        if len(pick) >= N_OPTIONS - 1:
            break
        seen.add(g.casefold())
        pick.append(g)
    options = [gold] + pick
    random.Random(sha(qid)).shuffle(options)
    return options, n_screen


ASK = "Which action should be done next on this screen?"


def tidy(row: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse whitespace in the goal of a (possibly cached) row: "Goal: <goal>\n<ASK>"."""
    q = row["questions"]["q"]
    goal = q["instructions"][len("Goal: "):-len("\n" + ASK)]
    q["instructions"] = "Goal: %s\n%s" % (" ".join(goal.split()), ASK)
    return row


def sample(qid: str, source: str, image: str, image_id: str, goal: str, history: Sequence[str], options: List[str],
           gold: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    prev = "; ".join(history[-4:]) if history else "none"
    if len(history) > 4:
        prev = "... " + prev
    row = {"id": qid, "source": source, "kind": "gui-next-action", "image": image, "image_id": image_id,
           "state_text": "Previous actions: %s." % prev,
           "questions": {"q": {"type": "choice",
                               "instructions": "Goal: %s\n%s" % (" ".join(goal.split()), ASK),
                               "criteria": {o: None for o in options}}},
           "gold": {"q": {"probabilities": {o: 1.0 if o == gold else 0.0 for o in options}}}}
    row.update(extra)
    return row


# ----------------------------------------------------------------------------- AITW

def icon_name(ui_type: str) -> str:
    return ui_type[len("ICON_"):].lower().replace("_", " ")


def aitw_elements(positions: Sequence[float], texts: Sequence[str], types: Sequence[str]) -> List[Dict[str, Any]]:
    """Annotated elements: label, icon flag and box (x0, y0, x1, y1), from AITW's flat (y, x, h, w) positions."""
    out = []
    for i, (t, u) in enumerate(zip(texts, types)):
        y, x, h, w = positions[4 * i: 4 * i + 4]
        icon = u.startswith("ICON_")
        label = icon_name(u) if icon else clean(t)
        out.append({"label": label, "icon": icon, "box": (x, y, x + w, y + h), "nav": u.startswith(NAV)})
    return out


def grow(b: Box, f: float = GROW) -> Box:
    cx, cy, hw, hh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * f / 2, (b[3] - b[1]) * f / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


def aitw_action(row: Dict[str, Any]) -> Dict[str, Any]:
    t = row["results/action_type"]
    if t == 4:
        tx, ty, lx, ly = row["touch_x"], row["touch_y"], row["lift_x"], row["lift_y"]
        if ((tx - lx) ** 2 + (ty - ly) ** 2) ** 0.5 <= TAP_DIST:
            return {"kind": "tap", "x": tx, "y": ty}
        if abs(ly - ty) >= abs(lx - tx):
            return {"kind": "scroll", "phrase": "scroll down" if ly < ty else "scroll up"}
        return {"kind": "scroll", "phrase": "scroll right" if lx < tx else "scroll left"}
    if t == 3:
        return {"kind": "type", "text": clean(row.get("results/type_action") or "")}
    return {"kind": {5: "back", 6: "home", 7: "enter", 10: "status", 11: "status"}.get(t, "other")}


def tap_gold(elements: List[Dict[str, Any]], x: float, y: float) -> Tuple[Optional[Dict[str, Any]], str]:
    """The tapped element: boxes containing the point, else (AITW's own matching tolerance) boxes grown by GROW."""
    hit = [e for e in elements if contains(e["box"], x, y)] or \
        [e for e in elements if contains(grow(e["box"]), x, y)]
    if not hit:
        return None, "no_element"
    if any(e.get("nav") for e in hit):
        return None, "nav_bar_tap"   # system home / back / recents buttons: same as the generic actions
    if len({(e["icon"], e["label"].casefold()) for e in hit}) > 1:
        return None, "ambiguous_element"
    g = min(hit, key=lambda e: area(e["box"]))
    if not readable(g["label"]):
        return None, "unreadable_label"
    for e in elements:
        if e not in hit and (e["icon"], e["label"].casefold()) == (g["icon"], g["label"].casefold()) \
                and not same_widget(e["box"], g["box"]):
            return None, "duplicate_label"
    return g, ""


def aitw_phrase(action: Dict[str, Any], elements: List[Dict[str, Any]]) -> Optional[str]:
    """The action as an option phrase (for history too); None when it cannot be named."""
    k = action["kind"]
    if k == "tap":
        g, _ = tap_gold(elements, action["x"], action["y"])
        return tap(g["label"], g["icon"]) if g else None
    if k == "scroll":
        return action["phrase"]
    if k == "type":
        return 'type "%s"' % action["text"] if action["text"] else None
    return {"back": "go back", "home": "go to home screen", "enter": "press enter"}.get(k)


def aitw_question(qid: str, goal: str, action: Dict[str, Any], elements: List[Dict[str, Any]],
                  episode_types: Sequence[str]) -> Tuple[Optional[Tuple[str, List[str], int]], str]:
    """(gold, options, n_screen) or the drop reason."""
    k = action["kind"]
    if k in ("status", "other"):
        return None, "status_action" if k == "status" else "other_action"
    if k == "type" and not action["text"]:
        return None, "empty_type"
    gold_el = None
    if k == "tap":
        gold_el, why = tap_gold(elements, action["x"], action["y"])
        if gold_el is None:
            return None, why
    gold = aitw_phrase(action, elements)
    assert gold is not None
    counts = Counter((e["icon"], e["label"].casefold()) for e in elements)
    rng = random.Random(sha("s:" + qid))
    pool = [e for e in elements if readable(e["label"]) and counts[(e["icon"], e["label"].casefold())] == 1
            and not e.get("nav") and not (k == "back" and e["icon"] and e["label"] in BACK_ICONS)
            and (gold_el is None or not same_widget(e["box"], gold_el["box"]))
            and not (k == "tap" and contains(e["box"], action["x"], action["y"]))]
    rng.shuffle(pool)
    screen = [tap(e["label"], e["icon"]) for e in pool]
    generic = list(GENERIC) + ['type "%s"' % t for t in episode_types if t and 'type "%s"' % t != gold]
    if gold_el is not None and gold_el["icon"] and gold_el["label"] in BACK_ICONS:
        generic.remove("go back")
    made = make_options(qid, gold, screen, generic)
    if made is None:
        return None, "few_screen_elements"
    return (gold, made[0], made[1]), ""


SITES = {"amazon": r"amazon", "walmart": r"walmart", "target": r"\btarget\b", "bestbuy": r"best ?buy",
         "ebay": r"\bebay\b", "newegg": r"newegg", "costco": r"costco", "nike": r"\bnike\b", "etsy": r"\betsy\b",
         "goodreads": r"goodreads", "ikea": r"\bikea\b", "lowes": r"\blowe'?s\b", "homedepot": r"home ?depot",
         "aliexpress": r"ali ?express", "article": r"\barticle(?:\.com)?\s*[.,]?\s*$|on article\b",
         "crateandbarrel": r"crate ?(?:and|&)? ?barrel|\bon crate\b"}
# web_shopping split by website: the 7 big sites (and goals naming none) train; small sites held out
SHOP_HELD = {"ikea": "dev", "homedepot": "dev", "aliexpress": "dev",
             "nike": "test", "etsy": "test", "lowes": "test", "goodreads": "test", "article": "test",
             "crateandbarrel": "test"}


def site_of(goal: str) -> Optional[str]:
    """The shopping website a goal names (the one named last when several are), or None."""
    g = goal.casefold()
    hits = [(m.start(), k) for k, pat in SITES.items() for m in re.finditer(pat, g)]
    return max(hits)[1] if hits else None


def aitw_group(source: str, goal: str) -> str:
    if source == "aitw-shopping":
        site = site_of(goal)
        return "site:" + site if site else "goal:" + norm(goal)
    return "goal:" + norm(goal)


def aitw_part(source: str, group: str) -> str:
    if source == "aitw-shopping":
        return SHOP_HELD.get(group[len("site:"):], "train") if group.startswith("site:") else "train"
    return held_part(group)


# ----------------------------------------------------------------------------- Mind2Web

class _Texts(HTMLParser):
    """backend_node_id -> (tag, collapsed own + descendant text, attributes) from Mind2Web's cleaned HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: List[Tuple[str, Optional[str], bool]] = []
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def _open(self, tag, attrs, void):
        a = dict(attrs)
        nid = a.get("backend_node_id")
        layer = bool(LAYER_ROLE.search(a.get("role") or "") or LAYER_CLASS.search(a.get("class") or "")) or \
            any(f for _, _, f in self.stack)
        if nid:
            self.nodes[nid] = {"tag": tag, "text": [], "attrs": a, "layer": layer}
        if not void:
            self.stack.append((tag, nid, layer))

    def handle_starttag(self, tag, attrs):
        self._open(tag, attrs, tag in VOID)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs, True)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if data.strip():
            for _, nid, _ in self.stack:
                if nid:
                    self.nodes[nid]["text"].append(data)


def html_labels(html: str) -> Dict[str, Tuple[str, str]]:
    return parse_html(html)[0]


def parse_html(html: str) -> Tuple[Dict[str, Tuple[str, str]], set]:
    """(backend_node_id -> (tag, label), ids inside a menu / listbox / dialog / dropdown-like layer)."""
    p = _Texts()
    p.feed(html)
    layered = {nid for nid, n in p.nodes.items() if n["layer"]}
    out = {}
    for nid, n in p.nodes.items():
        text = clean(" ".join(n["text"]))
        if not readable(text):
            a = n["attrs"]
            text = next((clean(a[k]) for k in ("aria_label", "placeholder", "title", "alt", "label")
                         if a.get(k) and readable(clean(a[k]))), "")
        out[nid] = (n["tag"], text)
    return out, layered


REPR = re.compile(r"^\[(?P<tag>[^\]]*)\]\s*(?P<label>.*?)\s*->\s*(?P<op>[A-Z]+)(?::\s*(?P<value>.*))?$")


def parse_repr(r: str) -> Optional[Dict[str, str]]:
    m = REPR.match(r.strip())
    if not m:
        return None
    return {"tag": m.group("tag"), "label": clean(m.group("label")), "op": m.group("op"),
            "value": clean(m.group("value") or "")}


def m2w_phrase(rep: Dict[str, str]) -> Optional[str]:
    if not readable(rep["label"]):
        return None
    if rep["op"] == "TYPE":
        return 'type "%s" into "%s"' % (rep["value"], rep["label"]) if rep["value"] else None
    if rep["op"] == "SELECT":
        return 'select "%s" from "%s"' % (rep["value"], rep["label"]) if rep["value"] else None
    return tap(rep["label"], verb="click")


def candidate(c: str) -> Optional[Dict[str, Any]]:
    d = json.loads(c)
    a = json.loads(d["attributes"]) if isinstance(d.get("attributes"), str) else (d.get("attributes") or {})
    r = a.get("bounding_box_rect")
    if not r:
        return None
    try:
        x, y, w, h = (float(v) for v in r.split(","))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return {"id": str(d.get("backend_node_id")), "tag": d.get("tag", ""), "box": (x, y, x + w, y + h)}


def crop_window(qid: str, box: Box, page_w: int, page_h: int) -> Optional[Tuple[int, int, int, int]]:
    h = min(CROP_H, page_h)
    if box[0] < 0 or box[1] < 0 or box[2] > page_w + 1 or box[3] > page_h + 1 or box[3] - box[1] > h:
        return None
    lo, hi = max(0, int(box[3]) - h + 1), min(int(box[1]), page_h - h)
    if hi < lo:
        return None
    y0 = random.Random(sha("c:" + qid)).randint(lo, hi)
    return 0, y0, min(CROP_W, page_w), y0 + h


def blank(pixels, box: Box) -> bool:
    """The box shows (almost) nothing on the screenshot: grey-level std under STD_MIN (hidden element)."""
    if pixels is None:
        return False
    p = pixels[max(0, int(box[1])):int(box[3]), max(0, int(box[0])):int(box[2])]
    return p.size == 0 or float(p.std()) < STD_MIN


def crossing(a: Box, b: Box) -> bool:
    """Boxes that overlap without one being nested in the other: on a rendered page two visible labelled elements
    do not do that, so one of them is a hidden layer (closed menu, off-screen carousel item)."""
    i = inter(a, b)
    return i > 0.1 * min(area(a), area(b)) and i < 0.8 * min(area(a), area(b))


def m2w_question(qid: str, rep: Dict[str, str], target: Dict[str, Any], others: List[Dict[str, Any]],
                 labels: Dict[str, Tuple[str, str]], window: Tuple[int, int, int, int],
                 task_values: Sequence[str], pixels=None, layered: frozenset = frozenset()
                 ) -> Tuple[Optional[Tuple[str, List[str], int]], str]:
    gold = m2w_phrase(rep)
    target_layer = target["id"] in layered
    if gold is None:
        return None, "unreadable_label"
    inside = []
    for c in others:
        b = c["box"]
        if not (b[0] >= window[0] and b[1] >= window[1] and b[2] <= window[2] and b[3] <= window[3]):
            continue
        tag, label = labels.get(c["id"], (c["tag"], ""))
        if same_widget(b, target["box"]):
            if label.casefold() != rep["label"].casefold() and crossing(b, target["box"]):
                inside.append((c, "", label))   # crosses the target: kept only to mark it hidden below
            continue
        if tag not in CLICKABLE or not readable(label):
            continue
        inside.append((c, tag, label))
    if any(tag and label.casefold() == rep["label"].casefold() for _, tag, label in inside):
        return None, "duplicate_label"
    boxes = [c["box"] for c, _, _ in inside] + [target["box"]]
    labs = [l.casefold() for _, _, l in inside] + [rep["label"].casefold()]
    visible = []
    for k, (c, tag, label) in enumerate(inside):
        if not tag or blank(pixels, c["box"]) or inter(c["box"], target["box"]) > 0:
            continue
        if c["id"] in layered and not target_layer:   # a closed menu / dropdown is usually not on screen
            continue
        if any(j != k and labs[j] != labs[k] and crossing(c["box"], boxes[j]) for j in range(len(boxes))):
            continue
        visible.append((c, tag, label))
    inside = visible
    rng = random.Random(sha("s:" + qid))
    rng.shuffle(inside)
    # keep one candidate per label, and only labels that occur once (a nested pair with one label is one widget)
    counts = Counter(label.casefold() for _, _, label in inside)
    screen, inputs = [], []
    for c, tag, label in inside:
        if counts[label.casefold()] > 1 and any(s.casefold() == tap(label, verb="click").casefold() for s in screen):
            continue
        screen.append(tap(label, verb="click"))
        if tag in INPUTS:
            inputs.append(label)
    if rep["op"] in ("TYPE", "SELECT"):
        fmt = 'type "%s" into "%s"' if rep["op"] == "TYPE" else 'select "%s" from "%s"'
        screen = [fmt % (rep["value"], l) for l in inputs] + screen
    generic = list(GENERIC)
    if inputs:
        generic += ['type "%s" into "%s"' % (v, inputs[0]) for v in task_values if v and v != rep["value"]][:2]
    made = make_options(qid, gold, screen, generic)
    if made is None:
        return None, "few_screen_elements"
    return (gold, made[0], made[1]), ""


# ----------------------------------------------------------------------------- splits, baselines

def held_part(group: str, dev_frac: float = DEV_FRAC, test_frac: float = TEST_FRAC) -> str:
    u = int(sha("split:" + group)[:12], 16) / float(1 << 48)
    return "dev" if u < dev_frac else "test" if u < dev_frac + test_frac else "train"


def cap_groups(rows: List[Dict[str, Any]], cap: int, per_group: int = PER_GROUP
               ) -> Tuple[List[Dict[str, Any]], int]:
    """Groups in hash order until `cap` rows; from each group whole episodes (hash order) until `per_group` rows,
    so many goals / websites are represented. Returns (kept, unused)."""
    by: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["group"]][r.get("episode", r["id"])].append(r)
    kept: List[Dict[str, Any]] = []
    for g in sorted(by, key=lambda g: sha("cap:" + g)):
        if len(kept) >= cap:
            break
        n = 0
        for ep in sorted(by[g], key=lambda e: sha("ep:" + e)):
            if n >= per_group:
                break
            kept += by[g][ep]
            n += len(by[g][ep])
    return kept, len(rows) - len(kept)


def cap_scroll(rows: List[Dict[str, Any]], drops: Counter) -> List[Dict[str, Any]]:
    """Keep scroll-gold questions to at most SCROLL_MAX of a source (lowest sha256(id) first); AITW records long
    scroll runs that would otherwise make "scroll" a third of the golds."""
    scroll = [r for r in rows if gold_of(r).startswith("scroll ")]
    other = [r for r in rows if not gold_of(r).startswith("scroll ")]
    allowed = int(SCROLL_MAX * len(other) / (1 - SCROLL_MAX))
    keep = sorted(scroll, key=lambda r: sha("scroll:" + r["id"]))[:allowed]
    drops["scroll_cap"] += len(scroll) - len(keep)
    ids = {r["id"] for r in keep}
    return [r for r in rows if r["id"] in ids or not gold_of(r).startswith("scroll ")]


def words(s: str) -> set:
    return set(norm(s).split()) - {"the", "a", "an", "to", "on", "in", "of", "for", "and", "go", "tap", "click",
                                    "type", "into", "icon", "select", "from"}


def text_only(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy of text-only heuristics (no image): first / longest / shortest option, most word overlap with the
    goal + previous actions, the most common gold action verb, plus the gold position histogram and chance."""
    if not rows:
        return {}
    hits: Counter = Counter()
    pos: Counter = Counter()
    verb = Counter(gold_of(r).split()[0] for r in rows)
    top_verb = verb.most_common(1)[0][0]
    chance = 0.0
    for r in rows:
        opts = list(r["questions"]["q"]["criteria"])
        g = gold_of(r)
        pos[opts.index(g)] += 1
        chance += 1 / len(opts)
        ctx = words(r["questions"]["q"]["instructions"] + " " + r.get("state_text", ""))
        hits["first"] += opts[0] == g
        hits["longest"] += max(opts, key=len) == g
        hits["shortest"] += min(opts, key=len) == g
        hits["goal_overlap"] += max(opts, key=lambda o: len(words(o) & ctx)) == g
        same = [o for o in opts if o.split()[0] == top_verb]
        hits["top_verb"] += bool(same) and same[0] == g
    n = len(rows)
    return {"n": n, "chance": round(chance / n, 4), **{k: round(v / n, 4) for k, v in sorted(hits.items())},
            "top_verb": top_verb, "gold_position": {str(k): pos[k] for k in sorted(pos)},
            "gold_verb": dict(verb.most_common())}


def gold_of(r: Dict[str, Any]) -> str:
    return next(o for o, p in r["gold"]["q"]["probabilities"].items() if p == 1.0)


# ----------------------------------------------------------------------------- build

def save_image(im, path: str, crop: Optional[Tuple[int, int, int, int]] = None) -> None:
    from PIL import Image
    if crop:
        im = im.crop(crop)
    im = im.convert("RGB")
    s = SAVE_MAX / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, "JPEG", quality=90)


def complete_files(raw: str, name: str) -> List[str]:
    meta = os.path.join(raw, name + ".files.json")
    if not os.path.exists(meta):
        return []
    files = json.load(open(meta))["files"]
    return [os.path.join(raw, name, p) for p, s in files
            if os.path.exists(os.path.join(raw, name, p)) and os.path.getsize(os.path.join(raw, name, p)) == s]


def build_aitw(root: str, source: str, files: List[str], drops: Counter, log) -> Iterable[Dict[str, Any]]:
    import pyarrow.parquet as pq
    from PIL import Image
    meta_cols = ["episode_id", "step_id", "goal_info", "results/action_type", "results/type_action", "touch_x",
                 "touch_y", "lift_x", "lift_y", "ui_annotations_positions", "ui_annotations_text",
                 "ui_annotations_ui_types", "current_activity", "device_type"]
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            meta = pf.read_row_group(rg, columns=meta_cols).to_pylist()
            episodes = defaultdict(list)
            for i, m in enumerate(meta):
                episodes[m["episode_id"]].append(i)
            need, plans = set(), {}
            for ep, idx in episodes.items():
                idx.sort(key=lambda i: meta[i]["step_id"])
                types = sorted({clean(meta[i]["results/type_action"] or "") for i in idx
                                if meta[i]["results/action_type"] == 3} - {""})
                history: List[str] = []
                for i in idx:
                    m = meta[i]
                    drops["steps"] += 1
                    els = aitw_elements(m["ui_annotations_positions"] or [], m["ui_annotations_text"] or [],
                                        m["ui_annotations_ui_types"] or [])
                    act = aitw_action(m)
                    qid = "%s:%s:%d" % (source, ep, m["step_id"])
                    got, why = aitw_question(qid, m["goal_info"], act, els, types)
                    if got:
                        plans[i] = (qid, got, list(history))
                        need.add(i)
                    else:
                        drops[why] += 1
                    ph = aitw_phrase(act, els)
                    history.append(ph or {"tap": "tap an unlabeled element"}.get(act["kind"], act["kind"]))
            if not need:
                continue
            imgs = pf.read_row_group(rg, columns=["image/encoded"]).column(0)
            for i in sorted(need):
                qid, (gold, options, n_screen), history = plans[i]
                m = meta[i]
                rel = os.path.join(OUT, "images", source, "%s_%d.jpg" % (m["episode_id"], m["step_id"]))
                if not os.path.exists(os.path.join(root, rel)):
                    with Image.open(io.BytesIO(imgs[i].as_py())) as im:
                        save_image(im, os.path.join(root, rel))
                yield {**sample(qid, source, rel, "aitw:%s:%d" % (m["episode_id"], m["step_id"]), m["goal_info"],
                                history, options, gold,
                                {"episode": "aitw:%s" % m["episode_id"], "n_screen_distractors": n_screen}),
                       "group": aitw_group(source, m["goal_info"])}
            log("%s: %s row group %d done" % (source, os.path.basename(path), rg))


def build_m2w(root: str, files: List[str], drops: Counter, log) -> Iterable[Dict[str, Any]]:
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    cols = ["action_uid", "cleaned_html", "operation", "pos_candidates", "neg_candidates", "website",
            "annotation_id", "confirmed_task", "screenshot", "action_reprs", "target_action_index",
            "target_action_reprs"]
    for path in files:
        split = os.path.basename(path).split("-")[0]
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=16, columns=cols):
            for r in batch.to_pylist():
                drops["steps"] += 1
                qid = "mind2web:%s" % r["action_uid"]
                rep = parse_repr(r["target_action_reprs"] or "")
                pos = [c for c in (candidate(x) for x in r["pos_candidates"] or []) if c]
                if rep is None or not pos:
                    drops["no_target"] += 1
                    continue
                target = pos[0]
                shot = r["screenshot"]
                if not shot or not shot.get("bytes"):
                    drops["no_screenshot"] += 1
                    continue
                with Image.open(io.BytesIO(shot["bytes"])) as im:
                    window = crop_window(qid, target["box"], im.width, im.height)
                    if window is None:
                        drops["target_outside"] += 1
                        continue
                    k = int(r["target_action_index"])
                    reps = [parse_repr(x) for x in r["action_reprs"]]
                    values = sorted({p["value"] for p in reps if p and p["op"] == "TYPE" and p["value"]})
                    others = [c for c in (candidate(x) for x in r["neg_candidates"] or []) if c]
                    labels, layered = parse_html(r["cleaned_html"] or "")
                    got, why = m2w_question(qid, rep, target, others, labels, window, values,
                                            np.asarray(im.convert("L"), dtype=np.float32), frozenset(layered))
                    if not got:
                        drops[why] += 1
                        continue
                    rel = os.path.join(OUT, "images", "mind2web", "%s.jpg" % r["action_uid"])
                    if not os.path.exists(os.path.join(root, rel)):
                        save_image(im, os.path.join(root, rel), window)
                history = [(m2w_phrase(p) if p else None) or "an unlabeled action" for p in reps[:k]]
                gold, options, n_screen = got
                part = "train" if split in ("train", "test_task") else held_part("site:" + r["website"], 0.33, 0.67)
                yield {**sample(qid, "mind2web", rel, "mind2web:%s" % r["action_uid"], r["confirmed_task"], history,
                                options, gold, {"episode": "mind2web:%s" % r["annotation_id"],
                                                "website": r["website"], "official_split": split,
                                                "n_screen_distractors": n_screen}),
                       "group": "site:" + r["website"], "part": part}
        log("mind2web: %s done" % os.path.basename(path))


def eval_hashes(root: str):
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from v04_confirm import hash_file
    paths = set()
    for f in EVAL_SETS:
        p = os.path.join(root, f)
        if os.path.exists(p):
            paths |= {json.loads(l)["image"] for l in open(p) if l.strip()}
    hs = [h for h in (hash_file(os.path.join(root, p)) for p in sorted(paths)) if not isinstance(h, str)]
    return (np.array([h[0] for h in hs], dtype=np.uint64), np.array([h[2] for h in hs], dtype=np.uint64),
            len(paths))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--sources", nargs="*", default=["aitw-general", "mind2web", "aitw-shopping"])
    ap.add_argument("--max-files", type=int, default=0, help="per source, for a quick partial build")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the per-source questions in v05/gui/work/ (only re-split and re-write)")
    a = ap.parse_args(argv)
    root, raw = a.root, os.path.join(a.root, OUT, "raw")
    out = os.path.join(root, OUT)
    logf = open(os.path.join(out, "build.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from v04_confirm import hash_file, is_near, nearest
    ref_d, ref_p, n_ref = eval_hashes(root)
    log("eval-set reference images: %d" % n_ref)
    parts: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    manifest: Dict[str, Any] = {"sources": {}, "rules": __doc__}
    for source in a.sources:
        drops: Counter = Counter()
        name = source
        files = complete_files(raw, name)
        meta = json.load(open(os.path.join(raw, name + ".files.json"))) if os.path.exists(
            os.path.join(raw, name + ".files.json")) else {"files": []}
        if a.max_files:
            files = files[:a.max_files]
        if not files:
            log("%s: no complete files, skipped" % source)
            continue
        cache = os.path.join(out, "work", source + ".jsonl")
        if a.reuse and os.path.exists(cache):
            with open(cache) as f:
                head = json.loads(f.readline())
                rows = [json.loads(l) for l in f]
            drops.update(head["drops"])
            files = [os.path.join(raw, p) for p in head["files_used"]]
            log("%s: reused %d rows from %s" % (source, len(rows), cache))
        else:
            gen = build_m2w(root, files, drops, log) if source == "mind2web" else build_aitw(root, source, files,
                                                                                             drops, log)
            rows = []
            for r in gen:
                near = nearest(hash_file(os.path.join(root, r["image"])), ref_d, ref_p)  # type: ignore[arg-type]
                if is_near(near, NEAR_BITS):
                    drops["near_eval_image"] += 1
                    continue
                rows.append(r)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w") as f:
                f.write(json.dumps({"drops": dict(drops), "files_used": [os.path.relpath(x, raw) for x in files]})
                        + "\n")
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        rows = cap_scroll(rows, drops)
        by_part = defaultdict(list)
        for r in rows:
            by_part[r.pop("part", None) or aitw_part(source, r["group"])].append(r)
        per = PER_SITE.get(source, PER_GROUP)
        dev, unused_dev = cap_groups(by_part["dev"], DEV_CAP.get(source, 400), per)
        test, unused_test = cap_groups(by_part["test"], TEST_CAP.get(source, 800), per)
        for part, rs in (("train", by_part["train"]), ("dev", dev), ("test", test)):
            parts[part] += rs
        groups = {p: len({r["group"] for r in rs}) for p, rs in (("train", by_part["train"]), ("dev", dev),
                                                                 ("test", test))}
        overlap = {r["group"] for r in by_part["train"]} & {r["group"] for r in dev + test}
        assert not overlap, overlap
        manifest["sources"][source] = {
            "repo": meta.get("repo"), "files_used": [os.path.relpath(f, raw) for f in files],
            "files_listed": len(meta["files"]), "questions": len(rows),
            "train": len(by_part["train"]), "dev": len(dev), "test": len(test),
            "held_out_unused": unused_dev + unused_test, "groups": groups,
            "drops": dict(sorted(drops.items()))}
        log("%s: %s" % (source, json.dumps(manifest["sources"][source]["drops"])))
    for part, rs in parts.items():
        rs.sort(key=lambda r: r["id"])
        for r in rs:
            tidy(r)
        with open(os.path.join(out, part + ".jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["counts"] = {p: dict(Counter(r["source"] for r in rs), total=len(rs)) for p, rs in parts.items()}
    manifest["text_only"] = {p: {s: text_only([r for r in rs if r["source"] == s])
                                 for s in sorted({r["source"] for r in rs})} for p, rs in parts.items()}
    tr_img = {r["image_id"] for r in parts["train"]}
    ho_img = {r["image_id"] for r in parts["dev"] + parts["test"]}
    tr_ep = {r["episode"] for r in parts["train"]}
    manifest["isolation"] = {"image_ids_train_and_heldout": len(tr_img & ho_img),
                             "episodes_train_and_heldout": len(tr_ep & {r["episode"] for r in parts["dev"] +
                                                                         parts["test"]}),
                             "eval_reference_images": n_ref, "near_bits": NEAR_BITS}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    log(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    sys.exit(main())
