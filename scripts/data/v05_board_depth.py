"""3D / depth questions (like CV-Bench Distance and BLINK Relative_Depth) on indoor COCO train2014 photos with
pseudo depth from a monocular metric-depth teacher -> data/v05/board/depth/.

    CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/data/v05_board_depth.py infer --root data [--limit 12000]
    uv run --no-sync python scripts/data/v05_board_depth.py build --root data

infer: picks indoor COCO train2014 images (>= 2 indoor categories present exactly once, no outdoor category; minus
project held-out pictures, board source ids and board image hits), runs Depth-Anything-V2-Metric-Indoor-Small (HF
depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf, CPU only, preprocessing as its DPT processor: shorter side
518, multiple of 14, ImageNet normalisation) and caches per image a 1/8-resolution depth map (metres, float16) and per
unique object its median depth inside the polygon mask and the mask centroid. Resumable: cached images are skipped.

build: three kinds of 2-way questions, answers balanced A / B, marks drawn on a copy of the photo:
  distance  CV-Bench style: "Estimate the real-world distances between objects in this image. Which object is closer
            to the Z (highlighted by a red box), the X (highlighted by a blue box) or the Y (highlighted by a green
            box)?" options [X, Y]. 3-D points = mask centroid back-projected at its median depth with a 60 degree
            horizontal field of view; kept only when the far distance >= RATIO x the near one and they differ by
            >= MIN_GAP_M metres.
  points    BLINK style: two circled points labelled A and B, "Which point is closer to the camera?"
            ["A is closer", "B is closer"]; points on flat depth (local spread < 10%), depth ratio >= POINT_RATIO,
            >= 20% of the image diagonal apart.
  camera    "Which object is closer to the camera taking this photo, the X (highlighted by a red box) or the Y
            (highlighted by a blue box)?" [X, Y]; median depth ratio >= CAM_RATIO.
Objects whose box covers > MAX_BOX of the photo (a table spanning near to far) are not asked about. These are teacher
pseudo-labels (no depth sensor); margins keep only clear cases. dev = ~4% of images.
"""
import argparse
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v05_board_common import BoardIndex, choice_sample, image_hashes, manifest, seeded, write_split  # noqa: E402

OUT = os.path.join("v05", "board", "depth")
MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
MODEL_REV = "8078d68a9c75a972131914f6afd0c1723be0da7f"
INDOOR = {"chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "refrigerator", "oven", "sink",
          "microwave", "potted plant", "bottle", "cup", "bowl", "book", "clock", "vase", "cell phone", "keyboard",
          "mouse", "remote", "teddy bear", "suitcase", "backpack", "handbag", "toaster", "wine glass", "person", "cat",
          "dog", "umbrella", "scissors", "hair drier", "toothbrush", "tie", "cake", "pizza", "donut", "sandwich"}
OUTDOOR = {"car", "bus", "truck", "train", "airplane", "boat", "traffic light", "fire hydrant", "stop sign",
           "parking meter", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "kite", "surfboard",
           "skis", "snowboard", "skateboard", "motorcycle", "bicycle", "bird"}
NAME = {"dining table": "table", "tv": "television", "cell phone": "phone", "potted plant": "potted plant",
        "hair drier": "hair dryer"}
MIN_AREA = 0.01
MAX_BOX = 0.35
RATIO, MIN_GAP_M = 1.5, 0.4
POINT_RATIO = 1.4
CAM_RATIO = 1.3
HFOV = 60.0
SAVE_MAX = 640
DS = 8


# ----------------------------------------------------------------------------- geometry (pure, tested)

def backproject(cx: float, cy: float, depth: float, w: int, h: int, hfov: float = HFOV) -> Tuple[float, float, float]:
    f = 0.5 * w / math.tan(math.radians(hfov) / 2)
    return ((cx - w / 2) * depth / f, (cy - h / 2) * depth / f, depth)


def dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def distance_question(objs: List[Dict], w: int, h: int, key: str, slot: int) -> Optional[Tuple[Dict, Dict, Dict, str]]:
    """(anchor, blue, green, gold name) for a clear 'which is closer to the anchor' triple, or None."""
    pts = {o["cat"]: backproject(o["cx"], o["cy"], o["depth"], w, h) for o in objs}
    rng = seeded("dist:" + key)
    order = list(objs)
    rng.shuffle(order)
    for z in order:
        rest = [o for o in order if o is not z]
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                a, b = rest[i], rest[j]
                da, db = dist(pts[a["cat"]], pts[z["cat"]]), dist(pts[b["cat"]], pts[z["cat"]])
                near, far = min(da, db), max(da, db)
                if near <= 0 or far / near < RATIO or far - near < MIN_GAP_M:
                    continue
                closer, other = (a, b) if da < db else (b, a)
                blue, green = (closer, other) if slot == 0 else (other, closer)
                return z, blue, green, closer["cat"]
    return None


def camera_question(objs: List[Dict], key: str, slot: int) -> Optional[Tuple[Dict, Dict, str]]:
    rng = seeded("cam:" + key)
    order = list(objs)
    rng.shuffle(order)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            near, far = (a, b) if a["depth"] < b["depth"] else (b, a)
            if far["depth"] / near["depth"] >= CAM_RATIO:
                red, blue = (near, far) if slot == 0 else (far, near)
                return red, blue, near["cat"]
    return None


def point_question(dm, key: str, slot: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    """Two points on flat depth with a clear ratio, in depth-map cells; ((xA, yA), (xB, yB), gold option)."""
    import numpy as np
    hh, ww = dm.shape
    rng = seeded("pt:" + key)
    diag = math.hypot(ww, hh)
    cands = []
    for _ in range(200):
        x, y = rng.randrange(2, ww - 2), rng.randrange(2, hh - 2)
        patch = dm[y - 1:y + 2, x - 1:x + 2].astype(np.float32)
        d = float(np.median(patch))
        if d > 0 and float(patch.max() - patch.min()) < 0.1 * d:
            cands.append((x, y, d))
        if len(cands) >= 40:
            break
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            (x1, y1, d1), (x2, y2, d2) = cands[i], cands[j]
            if math.hypot(x1 - x2, y1 - y2) < 0.2 * diag:
                continue
            if max(d1, d2) / min(d1, d2) < POINT_RATIO:
                continue
            near, far = ((x1, y1), (x2, y2)) if d1 < d2 else ((x2, y2), (x1, y1))
            a, b = (near, far) if slot == 0 else (far, near)
            return a, b, "A is closer" if slot == 0 else "B is closer"
    return None


# ----------------------------------------------------------------------------- infer

def coco(root: str):
    z = zipfile.ZipFile(os.path.join(root, "raw", "coco", "annotations_trainval2014.zip"))
    inst = json.loads(z.read("annotations/instances_train2014.json"))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    anns: Dict[int, List[Dict]] = defaultdict(list)
    for a in inst["annotations"]:
        anns[a["image_id"]].append(a)
    sizes = {im["id"]: (im["width"], im["height"]) for im in inst["images"]}
    return cats, anns, sizes


def unique_objects(anns: List[Dict], cats: Dict[int, str], w: int, h: int) -> Optional[List[Dict]]:
    names = Counter(cats[a["category_id"]] for a in anns)
    if any(n in OUTDOOR for n in names):
        return None
    out = [a for a in anns if names[cats[a["category_id"]]] == 1 and not a.get("iscrowd")
           and cats[a["category_id"]] in INDOOR and a["area"] >= MIN_AREA * w * h and isinstance(a["segmentation"], list)]
    return out if len(out) >= 2 else None


def mask(seg: List[List[float]], w: int, h: int, s: float):
    import numpy as np
    from PIL import Image, ImageDraw
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in seg:
        if len(poly) >= 6:
            d.polygon([(poly[k] * s, poly[k + 1] * s) for k in range(0, len(poly), 2)], fill=1)
    return np.asarray(m, dtype=bool)


def infer(a) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForDepthEstimation
    from huggingface_hub import snapshot_download
    torch.set_num_threads(a.threads)
    root = a.root
    cache = os.path.join(root, OUT, "cache")
    os.makedirs(cache, exist_ok=True)
    index_path = os.path.join(cache, "objects.jsonl")
    done = set()
    if os.path.exists(index_path):
        done = {json.loads(l)["image_id"] for l in open(index_path)}
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    sys.path[:0] = [os.path.join(HERE, "ext"), os.path.join(HERE, "pub")]
    from held import PubHeld  # pyright: ignore[reportMissingImports]
    held = PubHeld(root)
    cats, anns, sizes = coco(root)
    pick = []
    for iid in sorted(anns, key=lambda i: seeded("depth-img:%d" % i).random()):
        w, h = sizes[iid]
        objs = unique_objects(anns[iid], cats, w, h)
        if objs and not held.is_held("coco:%d" % iid) and not board.hit_id("coco:%d" % iid):
            pick.append((iid, objs))
    pick = pick[:a.limit]
    print("indoor candidates: %d, cached %d" % (len(pick), len(done)), flush=True)
    try:   # cached after the first run; the first download needs HF_ENDPOINT=https://hf-mirror.com
        path = snapshot_download(MODEL, revision=MODEL_REV, cache_dir=os.path.join(root, "raw", "hf"), local_files_only=True)
    except Exception:
        if "hf-mirror" not in os.environ.get("HF_ENDPOINT", ""):
            sys.exit("depth model not cached: set HF_ENDPOINT=https://hf-mirror.com for the first download")
        path = snapshot_download(MODEL, revision=MODEL_REV, cache_dir=os.path.join(root, "raw", "hf"))
    model = AutoModelForDepthEstimation.from_pretrained(path).eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    t0, n = time.time(), 0
    with open(index_path, "a") as out:
        for iid, objs in pick:
            key = "coco:%d" % iid
            if key in done:
                continue
            rel = "coco/train2014/COCO_train2014_%012d.jpg" % iid
            data = open(os.path.join(root, rel), "rb").read()
            got = board.hit_hashes(image_hashes(data))
            if got:
                out.write(json.dumps({"image_id": key, "board_hit": list(got)}) + "\n")
                continue
            im = Image.open(os.path.join(root, rel)).convert("RGB")
            W, H = im.size
            s = 518 / min(W, H)
            w, h = int(round(W * s / 14)) * 14, int(round(H * s / 14)) * 14
            x = (np.asarray(im.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255 - mean) / std
            with torch.no_grad():
                d = model(pixel_values=torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)[None]).predicted_depth[0]
            dm = torch.nn.functional.interpolate(d[None, None], size=(H // DS, W // DS), mode="bilinear")[0, 0].numpy()
            np.save(os.path.join(cache, "%d.npy" % iid), dm.astype(np.float16))
            rec = []
            for o in objs:
                m = mask(o["segmentation"], W // DS, H // DS, 1 / DS)
                if m.sum() < 4:
                    continue
                ys, xs = np.nonzero(m)
                rec.append({"cat": cats[o["category_id"]], "bbox": o["bbox"], "depth": float(np.median(dm[m])),
                            "cx": float(xs.mean() * DS), "cy": float(ys.mean() * DS)})
            out.write(json.dumps({"image_id": key, "image": rel, "w": W, "h": H, "objects": rec}) + "\n")
            out.flush()
            n += 1
            if n % 200 == 0:
                el = time.time() - t0
                print("%s %d/%d images, %.2f s/img, ETA %.1f h" % (time.strftime("%H:%M:%S"), n, len(pick) - len(done),
                      el / n, (len(pick) - len(done) - n) * el / n / 3600), flush=True)


# ----------------------------------------------------------------------------- build

COLORS = {"red": (230, 25, 75), "blue": (40, 90, 230), "green": (40, 180, 75)}


def draw_boxes(im, boxes: List[Tuple[Any, str]]):
    from PIL import ImageDraw
    im = im.copy()
    d = ImageDraw.Draw(im)
    lw = max(2, round(min(im.size) / 160))
    for (x, y, w, h), c in boxes:
        d.rectangle([x, y, x + w, y + h], outline=COLORS[c], width=lw)
    return im


def draw_points(im, pts: List[Tuple[Any, str]]):
    from PIL import ImageDraw, ImageFont
    im = im.copy()
    d = ImageDraw.Draw(im)
    r = max(6, round(min(im.size) / 45))
    try:
        font = ImageFont.load_default(size=int(r * 1.8))
    except TypeError:   # pragma: no cover - old Pillow
        font = ImageFont.load_default()
    for (x, y), lab in pts:
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=max(2, r // 3))
        d.text((x + r + 2, y - r - 2), lab, fill=(255, 0, 0), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    return im


def save(im, root: str, rel: str) -> None:
    s = SAVE_MAX / max(im.size)
    if s < 1:
        from PIL import Image
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
    im.save(os.path.join(root, rel), quality=90)


def name(cat: str) -> str:
    return NAME.get(cat, cat)


def render(kind: str, objs: List[Dict], W: int, H: int, iid: str, slot: int, cache: str):
    """(question, options, gold, marks to draw) or a drop reason."""
    import numpy as np
    if kind == "distance":
        dq = distance_question(objs, W, H, iid, slot) if len(objs) >= 3 else None
        if dq is None:
            return "no_clear_distance"
        z, blue, green, gold = dq
        if len({name(z["cat"]), name(blue["cat"]), name(green["cat"])}) != 3:
            return "name_clash"
        text = ("Estimate the real-world distances between objects in this image. Which object is closer to the %s "
                "(highlighted by a red box), the %s (highlighted by a blue box) or the %s (highlighted by a green "
                "box)?" % (name(z["cat"]), name(blue["cat"]), name(green["cat"])))
        return text, [name(blue["cat"]), name(green["cat"])], name(gold), \
            [(z["bbox"], "red"), (blue["bbox"], "blue"), (green["bbox"], "green")]
    if kind == "camera":
        cq = camera_question(objs, iid, slot)
        if cq is None:
            return "no_clear_camera"
        red, blue, gold = cq
        if name(red["cat"]) == name(blue["cat"]):
            return "name_clash"
        text = ("Which object is closer to the camera taking this photo, the %s (highlighted by a red box) or the %s "
                "(highlighted by a blue box)?" % (name(red["cat"]), name(blue["cat"])))
        return text, [name(red["cat"]), name(blue["cat"])], name(gold), [(red["bbox"], "red"), (blue["bbox"], "blue")]
    pq = point_question(np.load(os.path.join(cache, "%s.npy" % iid.split(":")[1])).astype(np.float32), iid, slot)
    if pq is None:
        return "no_clear_points"
    pa, pb, gold = pq
    text = "Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?"
    return text, ["A is closer", "B is closer"], gold, [((pa[0] * DS + DS / 2, pa[1] * DS + DS / 2), "A"),
                                                        ((pb[0] * DS + DS / 2, pb[1] * DS + DS / 2), "B")]


def build(a) -> None:
    from PIL import Image
    root = a.root
    cache = os.path.join(root, OUT, "cache")
    recs = [json.loads(l) for l in open(os.path.join(cache, "objects.jsonl"))]
    drops: Counter = Counter()
    hits = [r for r in recs if "board_hit" in r]
    drops["board_image_hit"] = len(hits)
    slots: Counter = Counter()
    rows = []
    for r in recs:
        if "board_hit" in r:
            continue
        iid, W, H = r["image_id"], r["w"], r["h"]
        objs = [o for o in r["objects"] if o["bbox"][2] * o["bbox"][3] <= MAX_BOX * W * H]   # huge boxes: depth spread
        img = None
        for kind in ("distance", "camera", "points"):
            got = render(kind, objs, W, H, iid, slots[kind] % 2, cache)
            if isinstance(got, str):
                drops[got] += 1
                continue
            text, opts, gold_t, marks = got
            if img is None:
                img = Image.open(os.path.join(root, r["image"])).convert("RGB")
            rel = os.path.join(OUT, "images", "%s_%s.jpg" % (iid.replace(":", "_"), kind))
            if not os.path.exists(os.path.join(root, rel)):
                save(draw_points(img, marks) if kind == "points" else draw_boxes(img, marks), root, rel)
            slots[kind] += 1
            rows.append(choice_sample("board-depth:%s:%s" % (iid, kind), "board-depth", iid, rel, text, opts, gold_t,
                                      kind=kind))
    out = os.path.join(root, OUT)
    counts = write_split(out, rows)
    manifest(out, {"family": "depth", "board": "CV-Bench Distance, BLINK Relative_Depth",
                   "source": "COCO train2014 photos + instances_train2014; pseudo depth %s@%s (CPU)" % (MODEL, MODEL_REV),
                   "images_cached": len(recs), "drops": dict(drops),
                   "board_exclusion_hits": [{"image": r["image_id"], "kind": r["board_hit"][0],
                                             "board": r["board_hit"][1]} for r in hits],
                   "margins": {"distance_ratio": RATIO, "distance_gap_m": MIN_GAP_M, "point_ratio": POINT_RATIO,
                               "camera_ratio": CAM_RATIO, "hfov_deg": HFOV},
                   "split_counts": counts})
    print("depth: %s drops %s" % (counts, dict(drops)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["infer", "build"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--limit", type=int, default=12000)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    if a.stage == "infer":
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            sys.exit("run with CUDA_VISIBLE_DEVICES= (CPU only; the GPU is for training)")
        infer(a)
    else:
        build(a)


if __name__ == "__main__":
    main()
