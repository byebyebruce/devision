"""Zero-cost text-height readability check for native-resolution tiles (docs/research/v05-training-directions.md,
direction 2). For each domain: text height in pixels after the global letterbox to SIZE (256) and in the 2x2 tile view
(letterbox to N * SIZE = 512, only when the picture's shorter side >= MIN_SIDE; otherwise the global view is all the
model gets), and the fraction of pictures that would be tiled. Plus image size / tiled fraction of eval sets.

Text heights are in the coordinates of the picture the model is given (the saved training image); a ratio
h * SIZE / max(w, h) gives the height after letterbox, which does not depend on a uniform resize before it.

Sources (measured unless marked):
  funsd     word boxes (annotations), box height                                   -- measured
  aitw-*    AITW OCR boxes (ui_annotations type TEXT), box height                  -- measured
  mind2web  candidate element boxes with own text, texty tag, single line (h <= 48 raw px); text height =
            min(box height, 24) (24 px ~ line box of 16 px body text); only elements inside the 1280x1000 crop
            window v05_gui_action uses                                             -- approximate (element boxes)
  screenqa  RICO-ScreenQA answer ui_elements bounds; text height = min(box height, 66) (66 px ~ line box of 14 sp
            text at RICO's 3.5x density)                                           -- approximate (element boxes)
  chartqa   no boxes; image sizes measured, tick-label height assumed --chart-text px at native -- estimate
"""
import argparse
import glob
import io
import json
import os
import random
import sys
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))

Image.MAX_IMAGE_PIXELS = None
TEXTY = {"a", "span", "button", "label", "li", "option", "td", "th", "p", "h1", "h2", "h3", "h4", "h5", "h6", "b",
         "strong", "em", "input", "select", "textarea"}
EVAL_SETS = ["v05/exp116/gui_test.jsonl", "v05/exp118/gui_test_state.jsonl", "v05/exp119/gui_shop_test_state.jsonl",
             "v05/exp116/ego_test.jsonl", "v05/chartqa_test/test_chartqa.jsonl", "v2/bench_pope.jsonl",
             "n2/bench/mmbench.jsonl", "lv_bench/scienceqa.jsonl"]


def views(h, w, H, a):
    """(height at SIZE, height in the best view, tiled) of text height h in a w x H picture."""
    g = h * a.size / max(w, H)
    tiled = a.n >= 2 and min(w, H) >= a.min_side
    return g, (g * a.n if tiled else g), tiled


def saved_dims(w, h, cap):
    s = cap / max(w, h)
    return (round(w * s), round(h * s)) if s < 1 else (w, h)


def summarise(name, rows, images, a, kind):
    """rows: [(text height at native, w, h)] (w, h = dims of the picture the model gets); images: [(w, h)]."""
    out = {"kind": kind, "n_images": len(images), "n_text_boxes": len(rows)}
    if images:
        out["pct_images_tiled"] = round(100 * np.mean([min(w, h) >= a.min_side for w, h in images]), 1)
        ws, hs = np.array([w for w, _ in images]), np.array([h for _, h in images])
        out["image_w_median"], out["image_h_median"] = float(np.median(ws)), float(np.median(hs))
        out["shorter_side_p10_p50_p90"] = [float(np.percentile(np.minimum(ws, hs), q)) for q in (10, 50, 90)]
    if rows:
        g, b, t = zip(*(views(h, w, H, a) for h, w, H in rows))
        g, b, t = np.array(g), np.array(b), np.array(t)
        nat = np.array([h for h, _, _ in rows])
        q = lambda x: [round(float(np.percentile(x, p)), 2) for p in (10, 50, 90)]
        out.update({
            "native_p10_p50_p90": q(nat),
            "at256_p10_p50_p90": q(g),
            "best_view_p10_p50_p90": q(b),   # tiles when tiled, else the 256 view
            "tiled_only_p10_p50_p90": q(b[t]) if t.any() else None,
            "pct_lt6_at256": round(100 * float(np.mean(g < 6)), 1),
            "pct_lt6_best_view": round(100 * float(np.mean(b < 6)), 1),
            "pct_lt8_at256": round(100 * float(np.mean(g < 8)), 1),
            "pct_lt8_best_view": round(100 * float(np.mean(b < 8)), 1),
            "pct_text_boxes_on_tiled_images": round(100 * float(np.mean(t)), 1),
        })
    print(name, json.dumps(out), flush=True)
    return out


def funsd(a, rng):
    z = zipfile.ZipFile(os.path.join(a.root, "v05/funsd/raw/dataset.zip"))
    anns = sorted(x for x in z.namelist() if x.endswith(".json") and not x.startswith("__MACOSX"))
    rng.shuffle(anns)
    rows, images = [], []
    for p in anns[:a.max_images]:
        img = p.replace("/annotations/", "/images/").replace(".json", ".png")
        w, h = Image.open(io.BytesIO(z.read(img))).size
        images.append((w, h))
        for f in json.loads(z.read(p))["form"]:
            for wd in f.get("words", []):
                x0, y0, x1, y1 = wd["box"]
                if y1 > y0 and wd.get("text", "").strip():
                    rows.append((y1 - y0, w, h))
    return rows, images


def aitw(a, rng, source):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(a.root, "v05/gui/raw", source, "data/*/*.parquet")))
    rows, images = [], []
    cols = ["image/height", "image/width", "ui_annotations_positions", "ui_annotations_ui_types"]
    per = max(1, a.max_images // len(files))
    for f in files:
        pf = pq.ParquetFile(f)
        rg = rng.randrange(pf.num_row_groups)
        recs = pf.read_row_group(rg, columns=cols).to_pylist()
        for r in rng.sample(recs, min(per, len(recs))):
            W, H = saved_dims(r["image/width"], r["image/height"], a.gui_save_max)
            images.append((W, H))
            pos = r["ui_annotations_positions"] or []
            for i, t in enumerate(r["ui_annotations_ui_types"] or []):
                if t == "TEXT":
                    hn = pos[4 * i + 2]
                    if hn > 0:
                        rows.append((hn * H, W, H))
    return rows, images


def mind2web(a, rng):
    import pyarrow.parquet as pq
    import v05_gui_action as g
    files = sorted(glob.glob(os.path.join(a.root, "v05/gui/raw/mind2web/data/*.parquet")))
    rng.shuffle(files)
    rows, images = [], []
    cols = ["action_uid", "cleaned_html", "pos_candidates", "neg_candidates", "screenshot"]
    per = max(1, a.max_images // 10)
    for f in files[:10]:
        pf = pq.ParquetFile(f)
        recs = pf.read_row_group(rng.randrange(pf.num_row_groups), columns=cols).to_pylist()
        for r in rng.sample(recs, min(per, len(recs))):
            pos = [c for c in (g.candidate(x) for x in r["pos_candidates"] or []) if c]
            if not pos or not (r["screenshot"] or {}).get("bytes"):
                continue
            pw, ph = Image.open(io.BytesIO(r["screenshot"]["bytes"])).size
            win = g.crop_window("mind2web:%s" % r["action_uid"], pos[0]["box"], pw, ph)
            if win is None:
                continue
            cw, ch = win[2] - win[0], win[3] - win[1]
            s = min(1.0, a.gui_save_max / max(cw, ch))
            W, H = round(cw * s), round(ch * s)
            images.append((W, H))
            labels = g.html_labels(r["cleaned_html"] or "")
            seen = set()
            for c in (g.candidate(x) for x in (r["pos_candidates"] or []) + (r["neg_candidates"] or [])):
                if not c or c["id"] in seen:
                    continue
                seen.add(c["id"])
                tag, label = labels.get(c["id"], (c["tag"], ""))
                x0, y0, x1, y1 = c["box"]
                hb = y1 - y0
                if (c["tag"] in TEXTY and label and 6 <= hb <= 48 and x1 - x0 >= hb and x0 >= win[0] and y0 >= win[1]
                        and x1 <= win[2] and y1 <= win[3]):
                    rows.append((min(hb, a.web_line) * s, W, H))
    return rows, images


def screenqa(a, rng):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(a.root, "raw/hf/datasets--bevaya--RICO-ScreenQA/snapshots/*/data/*.parquet")))
    rng.shuffle(files)
    rows, images, seen = [], [], set()
    per = max(1, a.max_images // 5)
    for f in files[:5]:
        pf = pq.ParquetFile(f)
        recs = pf.read_row_group(rng.randrange(pf.num_row_groups), columns=["screen_id", "ground_truth", "image"]).to_pylist()
        rng.shuffle(recs)
        k = 0
        for r in recs:
            if r["screen_id"] in seen or not (r["image"] or {}).get("bytes"):
                continue
            seen.add(r["screen_id"])
            W, H = Image.open(io.BytesIO(r["image"]["bytes"])).size
            images.append((W, H))
            boxes = {tuple(e["bounds"]) for gt in r["ground_truth"] or [] for e in gt["ui_elements"] or []
                     if e.get("bounds") and (e.get("text") or "").strip()}
            for x0, y0, x1, y1 in boxes:
                if y1 > y0:
                    rows.append((min(y1 - y0, a.rico_line), W, H))
            k += 1
            if k >= per:
                break
    return rows, images


def read_jsonl_images(a, rel, rng):
    paths = []
    with open(os.path.join(a.root, rel)) as fh:
        for line in fh:
            im = json.loads(line).get("image")
            if isinstance(im, str):
                paths.append(im)
    uniq = sorted(set(paths))
    rng.shuffle(uniq)
    out = []
    for p in uniq[:a.max_images]:
        try:
            out.append(Image.open(os.path.join(a.root, p)).size)
        except (OSError, FileNotFoundError):
            pass
    return out, len(uniq)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/v05-monitor/tile_readability.json")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--gui-save-max", type=int, default=1024, help="v05_gui_action.SAVE_MAX")
    ap.add_argument("--web-line", type=float, default=24.0, help="Mind2Web text height cap, raw page px")
    ap.add_argument("--rico-line", type=float, default=66.0, help="RICO text height cap, native px")
    ap.add_argument("--chart-text", type=float, default=11.5, help="assumed ChartQA tick-label height, native px")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    res = {"settings": {k: v for k, v in vars(a).items()}, "domains": {}, "eval_sets": {}}
    res["domains"]["funsd"] = summarise("funsd", *funsd(a, rng), a, "measured: word boxes")
    for s in ("aitw-general", "aitw-shopping"):
        res["domains"][s] = summarise(s, *aitw(a, rng, s), a, "measured: OCR TEXT boxes")
    res["domains"]["mind2web"] = summarise("mind2web", *mind2web(a, rng), a,
                                           "approx: element boxes, min(h, %g raw px)" % a.web_line)
    res["domains"]["screenqa"] = summarise("screenqa", *screenqa(a, rng), a,
                                           "approx: answer element boxes, min(h, %g px)" % a.rico_line)
    for name, rel in (("chartqa_test", "v05/chartqa_test/test_chartqa.jsonl"), ("chart_dev", "v05/exp81/chart_dev.jsonl")):
        ims, _ = read_jsonl_images(a, rel, rng)
        res["domains"][name] = summarise(name, [(a.chart_text, w, h) for w, h in ims], ims, a,
                                         "estimate: sizes measured, text = %g px assumed" % a.chart_text)
    for rel in EVAL_SETS:
        ims, n = read_jsonl_images(a, rel, rng)
        d = summarise(rel, [], ims, a, "image sizes")
        d["n_unique_images"] = n
        res["eval_sets"][rel] = d
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
