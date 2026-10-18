"""exp50 CPU geometry (critic EXP50-PREP-02): for candidate fixed view rules (devision.image.view_rects: n views, overlap),
on the real-layout pictures of the experiment (the training rows' pictures and the readout sets'), the number boxes of
both objects and all fields -- original pictures from the generator's own drawing (the number's text box inside its
row or label, read-only replay of real_layout.families()), L pictures from exp37's / exp40's number boxes -- measure:
digit ink height at 256 in the full picture and in the best view that holds the whole number (glyph-box ink of the
drawn size x scale); the share of numbers wholly inside at least one view, cut by every view's edge, or missing; per field
median and >= 12 / >= 16 px shares. Boxes are used only here, for the audit; no rule depends on them. The rule is chosen
by a criterion written before looking: among candidates with n <= 16 that keep >= 99.5% of the original numbers whole in
some view, the largest share of original numbers with >= 16 px ink in their best view; ties -> fewer views. That rule
picked n16 / 0.5 ("rule_choice"); the critic's round-1 review (EXP50-R1-01, before any run) fixed n12 / 0.25 by hand
instead ("chosen", "chosen_by"): every mass / speed number whole and 72% >= 16 px against 7%, at the cost of 15
original distance numbers (0.55%) that no single view holds whole. Those numbers' questions are written as a
report-only slice ("cut_slice": set, id, qid); the main sets and their denominators keep them. No other grid searched.
Writes data/exp50/geometry.json and runs/exp50-preview/{sheet.png, notes.json} (the chosen grid).

    uv run python scripts/exp50/geometry.py --root data
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import real_layout as rl  # type: ignore  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from devision.image import letterbox, local_views, view_rects  # noqa: E402

_s = importlib.util.spec_from_file_location("exp45_build", os.path.join(HERE, "..", "exp45", "build.py"))
assert _s is not None and _s.loader is not None
b45 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(b45)
CANDIDATES = [(8, 0.25), (12, 0.25), (12, 0.5), (16, 0.25), (16, 0.5)]
CHOSEN = (12, 0.25)          # critic EXP50-R1-01 (round-1 review, before any run)


def digit_ink(size: int) -> int:
    b = ImageFont.truetype(rl.FONT, size).getbbox("0123456789")
    return int(b[3] - b[1])


def original_numbers(form: str, s: Dict) -> List[Dict]:
    """[{field, object, box (x0, y0, x1, y1), ink (px in the original)}] of the drawn numbers of an original picture."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    out = []
    if form == "temp":
        rows = b45.temp_rows(s)
        for obj, r in rows.items():
            font = ImageFont.truetype(rl.FONT, r["size"])
            mass_line, speed_line = rl.LINES[s.get("wording", 0)]
            for field in ("mass", "speed"):
                num = str(s[obj]["mass"]) if field == "mass" else rl.fmt(s[obj]["speed"])
                t = mass_line % s[obj]["mass"] if field == "mass" else speed_line % rl.fmt(s[obj]["speed"])
                full = probe.textbbox((r["cx"], r["y"][field]), t, font=font, anchor="mm")
                start = full[0] + probe.textlength(t[:t.index(num)], font=font)
                out.append({"field": field, "object": obj, "box": [start, full[1], start + probe.textlength(num, font=font), full[3]],
                            "ink": digit_ink(r["size"])})
    else:
        g = rl.force_geometry(s)
        font = ImageFont.truetype(rl.FONT, s["font"])
        for p in range(2):
            q = g["pairs"][p]
            ax0, ax1 = q["start"] + q["widths"][0], q["start"] + q["widths"][0] + q["gap"]
            num = rl.fmt(s["d"][p])
            label = "%s %s" % (num, s["unit"])
            full = probe.textbbox(((ax0 + ax1) / 2, g["h"] - 14), label, font=font, anchor="mm")
            out.append({"field": "distance", "object": str(p + 1), "box": [full[0], full[1], full[0] + probe.textlength(num, font=font), full[3]],
                        "ink": digit_ink(s["font"])})
    return out


def measure(pictures: List[Dict], n: int, overlap: float) -> Dict:
    stats: Dict = {}
    for layout in ("original", "L"):
        rows = []
        for p in (x for x in pictures if x["layout"] == layout):
            w, h = p["size"]
            rects = view_rects(w, h, n, overlap)
            g = 256 / max(w, h)
            for num in p["numbers"]:
                x0, y0, x1, y1 = num["box"]
                whole = [r for r in rects if r[0] <= x0 and r[1] <= y0 and r[2] >= x1 and r[3] >= y1]
                touch = [r for r in rects if min(r[2], x1) > max(r[0], x0) and min(r[3], y1) > max(r[1], y0)]
                best = max((256 / max(r[2] - r[0], r[3] - r[1]) for r in whole), default=None)
                rows.append({"field": num["field"], "global": num["ink"] * g, "local": num["ink"] * best if best else None,
                             "state": "whole" if whole else "cut" if touch else "missing",
                             "picture": p["picture"], "object": num["object"]})
        res: Dict = {"numbers": len(rows), "whole": sum(r["state"] == "whole" for r in rows) / max(1, len(rows)),
                     "cut": sum(r["state"] == "cut" for r in rows), "missing": sum(r["state"] == "missing" for r in rows)}
        for f in ("mass", "speed", "distance", "all"):
            sel = [r for r in rows if f == "all" or r["field"] == f]
            loc = [r["local"] for r in sel if r["local"] is not None]
            glo = [r["global"] for r in sel]
            if sel:
                res[f] = {"n": len(sel), "global_median": statistics.median(glo), "global_ge12": sum(x >= 12 for x in glo) / len(glo),
                          "global_ge16": sum(x >= 16 for x in glo) / len(glo),
                          "local_median": statistics.median(loc) if loc else None,
                          "local_ge12": sum(x >= 12 for x in loc) / len(sel), "local_ge16": sum(x >= 16 for x in loc) / len(sel)}
        res["not_whole"] = [{"picture": r["picture"], "object": r["object"], "field": r["field"], "state": r["state"]}
                            for r in rows if r["state"] != "whole"]
        stats[layout] = res
    return stats


def cut_slice(root: str, not_whole: List[Dict]) -> List[Dict]:
    """The readout questions (exp49's map: set, id, qid) asking an original number no view holds whole."""
    want = {(x["picture"], x["object"], x["field"]) for x in not_whole}
    qs = [json.loads(x) for x in open(os.path.join(root, "exp49", "questions.jsonl"))]
    return sorted(({"set": q["set"], "id": q["id"], "qid": q["qid"], "family": q["family"], "field": q["field"]}
                   for q in qs if q["layout"] == "original" and (q["picture"], q["object"], q["field"]) in want),
                  key=lambda x: (x["set"], x["id"], x["qid"]))


def pictures_of(root: str) -> List[Dict]:
    qs = [json.loads(x) for x in open(os.path.join(root, "exp49", "questions.jsonl"))]
    pics, rows, checked = {}, {}, [0]
    for q in qs:
        pics.setdefault(q["picture"], {"picture": q["picture"], "layout": q["layout"], "size": [q["width"], q["height"]],
                                       "family": q["family"], "view": q["view"], "form": "force" if q["field"] == "distance" else "temp"})
        rows[(q["picture"], q["object"], q["field"])] = q["box"]          # EXP45's checked row box, [x, y, w, h]
    sc = b45.b43.scenes({p["family"] for p in pics.values()})
    meta = json.load(open(os.path.join(root, "exp37", "pictures.json")))
    meta.update(json.load(open(os.path.join(root, "exp40", "pictures_redrawn.json"))))
    for p in pics.values():
        if p["layout"] == "original":
            p["numbers"] = original_numbers(*sc[(p["family"], p["view"])])
            for n in p["numbers"]:                       # the replayed number box must lie in the row EXP45 checked
                if (p["picture"], n["object"], n["field"]) not in rows:      # not asked about this picture
                    continue
                checked[0] += 1
                bx, by, bw, bh = rows[(p["picture"], n["object"], n["field"])]
                if not (bx - 1 <= n["box"][0] and by - 1 <= n["box"][1] and n["box"][2] <= bx + bw + 1 and n["box"][3] <= by + bh + 1):
                    raise SystemExit("replayed number box outside its row: %s %s" % (p["picture"], n))
        else:
            # exp37 / exp40 boxes are the padded slot the number is centred in (widest number + 6 px wide, ink + descender
            # room + 6 px tall); the glyph box is approximated as the slot less its padding: width - 6, height = digit ink
            m = meta["%s_%s" % (p["family"], p["view"])]
            scale = 256 / max(p["size"])
            p["numbers"] = []
            for x in m["numbers"]:
                ink = (x.get("ink_256_L") or 16) / scale
                x0, y0, x1, y1 = x["box"]
                cy = (y0 + y1) / 2
                p["numbers"].append({"field": x["field"], "object": x["object"].split()[-1],
                                     "box": [x0 + 3, cy - ink / 2, x1 - 3, cy + ink / 2], "ink": ink})
    print("replayed original number boxes inside EXP45's checked rows: %d" % checked[0])
    return list(pics.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args(argv)
    pics = pictures_of(a.root)
    cands: Dict[str, Dict] = {"n%d_o%.2f" % c: dict(measure(pics, *c), n=c[0], overlap=c[1]) for c in CANDIDATES}
    ok = {k: v for k, v in cands.items() if v["n"] <= 16 and v["original"]["whole"] >= 0.995}
    rule_choice = max(ok, key=lambda k: (ok[k]["original"]["all"]["local_ge16"], -ok[k]["n"])) if ok else None
    chosen = "n%d_o%.2f" % CHOSEN
    sl = cut_slice(a.root, cands[chosen]["original"]["not_whole"])
    out = {"pictures": {"original": sum(p["layout"] == "original" for p in pics), "L": sum(p["layout"] == "L" for p in pics)},
           "criterion": "n <= 16 and >= 99.5% original numbers whole in a view; max original share >= 16 px; ties fewer views",
           "candidates": cands, "rule_choice": rule_choice, "chosen": chosen,
           "chosen_by": "critic EXP50-R1-01 (round-1 review, before any run): n12 / 0.25 by hand, not by the criterion above",
           "cut_slice": sl}
    os.makedirs(os.path.join(a.root, "exp50"), exist_ok=True)
    json.dump(out, open(os.path.join(a.root, "exp50", "geometry.json"), "w"), indent=1)
    if chosen:
        preview(a.root, pics, cands[chosen]["n"], cands[chosen]["overlap"])
    for k, v in cands.items():
        o, l_ = v["original"], v["L"]
        print("%-10s O whole %.4f cut %d miss %d | ink global med %.1f >=16 %.2f | local med %.1f >=12 %.2f >=16 %.2f | %s | L whole %.4f local>=16 %.2f" % (
            k, o["whole"], o["cut"], o["missing"], o["all"]["global_median"], o["all"]["global_ge16"], o["all"]["local_median"],
            o["all"]["local_ge12"], o["all"]["local_ge16"],
            {f: (round(o[f]["local_median"], 1), round(o[f]["local_ge16"], 2)) for f in ("mass", "speed", "distance")},
            l_["whole"], l_["all"]["local_ge16"]))
    print("rule choice:", rule_choice, "| chosen (critic EXP50-R1-01):", chosen, "| cut slice:", len(sl), "questions",
          sorted({x["family"] for x in sl}), {x["set"] for x in sl}, {x["field"] for x in sl})
    return 0


def preview(root: str, pics: List[Dict], n: int, overlap: float) -> None:
    """A fixed choice of pictures (first original temperature / force, first L temperature / force, by name): the full
    picture with the view rectangles, then for two views each the T and C crops as the model gets them."""
    pick = []
    for layout in ("original", "L"):
        for form in ("temp", "force"):
            pick.append(sorted((p for p in pics if p["layout"] == layout and p["form"] == form), key=lambda p: p["picture"])[0])
    tile, pad = 256, 8
    sheet = Image.new("RGB", (3 * (tile + pad), len(pick) * (tile + 24 + pad)), "white")
    d = ImageDraw.Draw(sheet)
    notes = []
    for i, p in enumerate(pick):
        img = Image.open(os.path.join(root, p["picture"])).convert("RGB")
        w, h = img.size
        rects = view_rects(w, h, n, overlap)
        full = img.copy()
        dd = ImageDraw.Draw(full)
        for r in rects:
            dd.rectangle(r, outline=(255, 0, 0), width=2)
        y = i * (tile + 24 + pad)
        sheet.paste(letterbox(full, tile), (0, y))
        num = p["numbers"][0]
        cx, cy = (num["box"][0] + num["box"][2]) / 2, (num["box"][1] + num["box"][3]) / 2
        k = min(range(len(rects)), key=lambda j: abs((rects[j][0] + rects[j][2]) / 2 - cx) + abs((rects[j][1] + rects[j][3]) / 2 - cy))
        for col, src in ((1, "T"), (2, "C")):
            px, _ = local_views(img, tile, n, overlap, src)
            arr = ((px[k].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8")
            sheet.paste(Image.fromarray(arr), (col * (tile + pad), y))
        d.text((0, y + tile + 4), "%s  views %d  view %d (T | C)" % (p["picture"], len(rects), k), fill="black")
        notes.append({"picture": p["picture"], "size": [w, h], "rects": rects, "shown_view": k})
    os.makedirs("runs/exp50-preview", exist_ok=True)
    sheet.save("runs/exp50-preview/sheet.png")
    json.dump(notes, open("runs/exp50-preview/notes.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
