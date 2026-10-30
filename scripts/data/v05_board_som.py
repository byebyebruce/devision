"""Set-of-marks web-action questions (the board's Mind2Web format) from Multimodal-Mind2Web TRAIN -> data/v05/board/som/.

    uv run --no-sync python scripts/data/v05_board_som.py --root data [--font /mnt/c/Windows/Fonts/arialbd.ttf]

Source: osunlp/Multimodal-Mind2Web `train` shards already fetched for the GUI data (data/v05/gui/raw/mind2web/data/,
v05_gui_fetch.py). Only `train` is read: the board uses test_website, and test_task / test_domain are left out too.
Format = the board kit's public converter (MIT): a viewport-high (VH px) window of the full-page screenshot around the
target at a seeded vertical offset, the target + 3 non-overlapping negative candidates inside the window (each
8 px .. 60% of the width / 40% of the window height) outlined in 5 colours with letter tags A-D (Arial Bold when
--font exists, as the board's build), and the question
    "Task: <task>\\nPrevious actions:\\n- <last <= 5 action reprs or (none)>\\nWhich marked element should the next
     action (<CLICK|TYPE|SELECT>) be performed on?"   options "box A" .. "box D".
The gold letter is rotated evenly (the board shuffles). Every window is checked against data/v05/board_exclusion.json
(board Mind2Web pictures are marked crops of test_website pages) and annotation ids against the board's. dev = ~4% of
tasks (annotation id; all steps of a task on one side).
"""
import argparse
import glob
import io
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v05_board_common import BoardIndex, LETTERS, choice_sample, image_hashes, manifest, seeded, write_split  # noqa: E402

OUT = os.path.join("v05", "board", "som")
SHARDS = "v05/gui/raw/mind2web/data/train-*.parquet"
VH = 1080
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
COLS = ["action_uid", "operation", "pos_candidates", "neg_candidates", "annotation_id", "confirmed_task", "screenshot",
        "action_reprs", "target_action_index"]


def box(cand: str) -> Optional[List[float]]:
    a = json.loads(json.loads(cand)["attributes"])
    r = a.get("bounding_box_rect")
    if not r:
        return None
    x, y, w, h = [float(v) for v in r.split(",")]
    return [x, y, x + w, y + h]


def overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def layout(g: List[float], negs: List[List[float]], W: int, H: int, rng: random.Random, slot: int):
    """(window top, 4 boxes in window coordinates with the gold at `slot`) or None."""
    vh = min(H, VH)
    cy = (g[1] + g[3]) / 2
    top = max(0, int(min(max(0, cy - vh / 2 + rng.uniform(-200, 200)), H - vh)))
    if not (g[1] >= top and g[3] <= top + vh and g[2] <= W):
        return None
    keep: List[List[float]] = []
    for b in negs:
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 8 or h < 8 or w > 0.6 * W or h > 0.4 * vh or b[1] < top or b[3] > top + vh or b[2] > W:
            continue
        if overlaps(b, g) or any(overlaps(b, x) for x in keep):
            continue
        keep.append(b)
    if len(keep) < 3:
        return None
    boxes = rng.sample(keep, 3)
    boxes.insert(slot, g)
    return top, vh, [[b[0], b[1] - top, b[2], b[3] - top] for b in boxes]


def question(task: str, reprs: List[str], k: int, op: str) -> str:
    hist = "\n".join("- %s" % x for x in reprs[:k][-5:]) or "- (none)"
    return "Task: %s\nPrevious actions:\n%s\nWhich marked element should the next action (%s) be performed on?" % (
        task, hist, op)


def draw_marks(im, boxes, font_path: Optional[str]):
    from PIL import ImageDraw, ImageFont
    im = im.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(2, int(min(W, H) / 300))
    fs = max(14, int(min(W, H) / 40))
    try:
        f = ImageFont.truetype(font_path, fs) if font_path and os.path.exists(font_path) else ImageFont.load_default(fs)
    except Exception:
        f = ImageFont.load_default()
    for i, b in enumerate(boxes):
        c = COLORS[i % len(COLORS)]
        d.rectangle(b, outline=c, width=lw)
        tx, ty = b[0], max(0, b[1] - fs - 4)
        d.rectangle([tx, ty, tx + fs + 6, ty + fs + 4], fill=c)
        d.text((tx + 3, ty + 1), LETTERS[i], fill="white", font=f)
    return im


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--font", default="/mnt/c/Windows/Fonts/arialbd.ttf")
    a = ap.parse_args(argv)
    t0 = time.time()
    import pyarrow.parquet as pq
    from PIL import Image
    root = a.root
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    shards = sorted(glob.glob(os.path.join(root, SHARDS)))
    if not shards:
        sys.exit("no Mind2Web train shards at %s (v05_gui_fetch.py --only mind2web)" % SHARDS)
    drops: Counter = Counter()
    hits: List[Dict] = []
    rows = []
    slot_n = 0
    for sh in shards:
        pf = pq.ParquetFile(sh)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=COLS).to_pylist():
                sid = r["action_uid"]
                if board.hit_id("m2w:" + r["annotation_id"]):
                    drops["board_annotation_id"] += 1
                    continue
                if not r["pos_candidates"]:
                    drops["no_target"] += 1
                    continue
                g = box(r["pos_candidates"][0])
                if not g or g[2] - g[0] < 4 or g[3] - g[1] < 4:
                    drops["no_target_box"] += 1
                    continue
                im = Image.open(io.BytesIO(r["screenshot"]["bytes"])).convert("RGB")
                W, H = im.size
                negs = [b for b in (box(c) for c in r["neg_candidates"]) if b]
                lay = layout(g, negs, W, H, seeded("som:" + sid), slot_n % 4)
                if not lay:
                    drops["target_outside_or_few_negatives"] += 1
                    continue
                top, vh, px = lay
                crop = im.crop((0, top, W, top + vh))
                buf = io.BytesIO()
                crop.save(buf, "PNG")
                got = board.hit(buf.getvalue())
                if got:
                    hits.append({"action_uid": sid, "kind": got[0], "board": got[1]})
                    drops["board_image_" + got[0]] += 1
                    continue
                rel = os.path.join(OUT, "images", sid.replace("/", "_") + "_marked.jpg")
                if not os.path.exists(os.path.join(root, rel)):
                    os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                    draw_marks(crop, px, a.font).save(os.path.join(root, rel), quality=92)
                op = json.loads(r["operation"])["op"]
                q = question(r["confirmed_task"], r["action_reprs"], int(r["target_action_index"]), op)
                opts = ["box %s" % LETTERS[i] for i in range(4)]
                rows.append(choice_sample("board-som:" + sid, "board-som-mind2web", "m2w:" + r["annotation_id"], rel,
                                          q, opts, opts[slot_n % 4], kind=op.lower()))
                slot_n += 1
        print("%s %s: %d questions, drops %s" % (time.strftime("%H:%M:%S"), os.path.basename(sh), len(rows),
                                                dict(drops)), flush=True)
    out = os.path.join(root, OUT)
    counts = write_split(out, rows)
    manifest(out, {"family": "som", "board": "Mind2Web (set-of-marks)",
                   "source": "osunlp/Multimodal-Mind2Web train (ModelScope mirror, data/v05/gui/raw/mind2web)",
                   "font": a.font if os.path.exists(a.font) else "PIL default", "drops": dict(drops),
                   "board_exclusion_hits": hits, "split_counts": counts, "seconds": round(time.time() - t0)})
    print("som: %s drops %s, %.0f s" % (counts, dict(drops), time.time() - t0))


if __name__ == "__main__":
    main()
