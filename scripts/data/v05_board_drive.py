"""Driving front-camera questions (like the board's RealWorldQA) from DriveLM-nuScenes TRAIN -> data/v05/board/drive/.

    uv run --no-sync python scripts/data/v05_board_drive.py --root data

Source: DriveLM v1.1 train (v1_1_train_nus.json + drivelm_nus_imgs_train.zip = nuScenes train keyframes, fetched
from ModelScope by v05_board_fetch.py --only drivelm). Only the CAM_FRONT picture of a keyframe is used (one image,
the driver's view, as in RealWorldQA). Questions:
  describe   a vehicle / pedestrian / cycle key object of the front camera in a red box: "Which description matches
             the object in the red box?" options = its DriveLM visual description (must name a colour) + 3
             descriptions of key objects of the SAME category from other frames that share NO colour word with it
             ("White truck." vs "Black SUV.", never vs "White commercial vehicle.").
  light      a front key object described as a red / green / yellow light, in a red box: "What colour is the traffic
             light in the red box?" [red, green, yellow] (gold rotated).
  element    "Is there any traffic element in the front view?" Yes / No from DriveLM's perception answer, balanced.
Boxes cover < MAX_BOX of the picture and > MIN_BOX px on each side. Every picture is checked against
data/v05/board_exclusion.json (CV-Bench uses Omni3D nuScenes test pictures; RealWorldQA has its own photos). dev =
~4% of scenes (all keyframes of a scene on one side).
"""
import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v05_board_common import BoardIndex, choice_sample, manifest, place_gold, seeded, write_split  # noqa: E402

OUT = os.path.join("v05", "board", "drive")
RAW = os.path.join("raw", "board", "drivelm")
MAX_BOX, MIN_BOX = 0.4, 24
SAVE_MAX = 960
DESCRIBE = {"Vehicle", "Pedestrian", "Cycle"}
LIGHTS = {"red light.": "red", "green light.": "green", "yellow light.": "yellow"}


def norm_desc(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def front_objects(frame: Dict) -> List[Tuple[str, Dict]]:
    return [(k, o) for k, o in frame["key_object_infos"].items() if ",CAM_FRONT," in k and o.get("2d_bbox")]


def box_ok(b: List[float], w: int = 1600, h: int = 900) -> bool:
    bw, bh = b[2] - b[0], b[3] - b[1]
    return bw >= MIN_BOX and bh >= MIN_BOX and bw * bh <= MAX_BOX * w * h


COLOURS = {"white", "black", "red", "blue", "green", "yellow", "orange", "grey", "gray", "silver", "brown", "pink",
           "purple", "beige", "gold", "dark", "light"}


def colours(s: str) -> set:
    return {w for w in norm_desc(s).split() if w in COLOURS}


def describe_question(obj: Dict, pool: Dict[str, List[str]], key: str, slot: int) -> Optional[Tuple[List[str], str]]:
    """Distractors share the category but name other colours only, so one look at the box separates them
    ("White truck." vs "Black SUV."; never "White truck." vs "White commercial vehicle.")."""
    if obj["Category"] not in DESCRIBE:
        return None
    gold = obj["Visual_description"].strip()
    g = norm_desc(gold)
    gc = colours(gold)
    if not g or len(gold) > 80 or not gc:
        return None
    rng = seeded("desc:" + key)
    cands = [d for d in pool.get(obj["Category"], []) if norm_desc(d) != g and colours(d) and not colours(d) & gc]
    rng.shuffle(cands)
    ds: List[str] = []
    for d in cands:
        if norm_desc(d) not in {norm_desc(x) for x in ds}:
            ds.append(d)
        if len(ds) == 3:
            break
    if len(ds) < 3:
        return None
    return place_gold(gold, ds, slot), gold


def element_answer(a: str) -> Optional[str]:
    a = a.strip().lower()
    return "Yes" if a.startswith("yes") else "No" if a.startswith("no") else None


def draw_box(im, b: List[float]):
    from PIL import ImageDraw
    im = im.copy()
    ImageDraw.Draw(im).rectangle(b, outline=(230, 25, 75), width=max(3, round(min(im.size) / 150)))
    return im


def save(im, root: str, rel: str) -> None:
    from PIL import Image
    s = SAVE_MAX / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
    im.save(os.path.join(root, rel), quality=90)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args(argv)
    t0 = time.time()
    from PIL import Image
    root = a.root
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    data = json.load(open(os.path.join(root, RAW, "v1_1_train_nus.json")))
    zf = zipfile.ZipFile(os.path.join(root, RAW, "drivelm_nus_imgs_train.zip"))
    names = set(zf.namelist())
    pool: Dict[str, List[str]] = defaultdict(list)
    for scene in data.values():
        for fr in scene["key_frames"].values():
            for _, o in front_objects(fr):
                if o.get("Visual_description"):
                    pool[o["Category"]].append(o["Visual_description"].strip())
    drops: Counter = Counter()
    hits = []
    rows = []
    slots: Counter = Counter()
    elements: List[Tuple[str, str, str, str]] = []
    for sid, scene in sorted(data.items()):
        for fid, fr in sorted(scene["key_frames"].items()):
            path = fr["image_paths"]["CAM_FRONT"].replace("../", "")
            if path not in names:
                drops["image_missing"] += 1
                continue
            raw = zf.read(path)
            got = board.hit(raw)
            if got:
                hits.append({"frame": fid, "kind": got[0], "board": got[1]})
                drops["board_image_" + got[0]] += 1
                continue
            im = None
            image_id = "nuscenes-scene:" + sid
            for k, o in front_objects(fr):
                b = o["2d_bbox"]
                if not box_ok(b):
                    drops["box_size"] += 1
                    continue
                key = "%s:%s" % (fid, k)
                desc = (o.get("Visual_description") or "").strip()
                light = LIGHTS.get(desc.lower())
                if light:
                    slot = slots["light"] % 3
                    q = ("What colour is the traffic light in the red box?",
                         place_gold(light, [c for c in ("red", "green", "yellow") if c != light], slot), light, "light")
                else:
                    got_q = describe_question(o, pool, key, slots["describe"] % 4)
                    if not got_q:
                        drops["describe_skipped"] += 1
                        continue
                    q = ("Which description matches the object in the red box?", got_q[0], got_q[1], "describe")
                slots[q[3]] += 1
                if im is None:
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                rel = os.path.join(OUT, "images", "%s_%s.jpg" % (fid, re.sub(r"[^A-Za-z0-9]", "", k.split(",")[0])))
                if not os.path.exists(os.path.join(root, rel)):
                    save(draw_box(im, b), root, rel)
                rows.append(choice_sample("board-drive:%s" % key, "board-drivelm", image_id, rel, q[0], q[1], q[2],
                                          kind=q[3]))
            for x in fr["QA"]["perception"]:
                if x["Q"] == "Is there any traffic element in the front view?":
                    ans = element_answer(x["A"])
                    if ans:
                        rel = os.path.join(OUT, "images", "%s_front.jpg" % fid)
                        if not os.path.exists(os.path.join(root, rel)):
                            if im is None:
                                im = Image.open(io.BytesIO(raw)).convert("RGB")
                            save(im, root, rel)
                        elements.append((fid, image_id, rel, ans))
    # balance the yes / no element questions
    by = defaultdict(list)
    for e in elements:
        by[e[3]].append(e)
    n = min(len(by["Yes"]), len(by["No"]))
    for lab in ("Yes", "No"):
        keep = sorted(by[lab], key=lambda e: seeded("el:" + e[0]).random())[:n]
        drops["element_balance"] += len(by[lab]) - n
        for fid, image_id, rel, ans in keep:
            rows.append(choice_sample("board-drive:%s:element" % fid, "board-drivelm", image_id, rel,
                                      "Is there any traffic element in the front view?", ["Yes", "No"], ans,
                                      kind="element"))
    out = os.path.join(root, OUT)
    counts = write_split(out, rows)
    manifest(out, {"family": "drive", "board": "RealWorldQA (driving / street scenes)",
                   "source": "DriveLM v1.1 nuScenes train (ModelScope OpenDriveLab/DriveLM), CAM_FRONT only",
                   "drops": dict(drops), "board_exclusion_hits": hits, "split_counts": counts,
                   "seconds": round(time.time() - t0)})
    print("drive: %s drops %s, %.0f s" % (counts, dict(drops), time.time() - t0))


if __name__ == "__main__":
    main()
