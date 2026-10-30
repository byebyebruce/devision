"""Receipt / form key-information questions (KIE, like the board's CORD + FUNSD) from TRAIN splits only
-> data/v05/board/kie/.

    uv run --no-sync python scripts/data/v05_board_kie.py --root data [--no-cord]

Sources: CORD v2 train (naver-clova-ix/cord-v2 data/train-*.parquet, fetched by v05_board_fetch.py; the board uses
CORD validation + test) and FUNSD training_data (the official zip at data/v05/funsd/raw/dataset.zip; the board uses
testing_data). The question design follows the board kit's public converter (MIT) so the training task has the
board's format, but every document here is a train-split document:

  value    CORD: "What is the total amount on this receipt?" (total / subtotal / tax / change / an item's price);
           options = the gold amount + 3 other amounts printed on the SAME receipt, the number of distractors above
           vs below the gold rotated so the largest / smallest option is not a cue.
           FUNSD: "On this form, what is filled in for the field "<key>"?"; options = the linked answer + 3 other
           answers of the SAME form of the same type (date / number / text).
  present  CORD: "Does this receipt show <a tax amount | a discount | ...>?" Yes / No from the parse;
           FUNSD: "On this form, is the field "<key>" filled in?" Yes = a linked answer, No = no linked answer and no
           answer box right of / below the key. Yes and No are balanced per field by dropping the majority.
Yes / No questions keep the board's option order (A = Yes, B = No); value questions rotate the gold letter evenly.
Every image is checked against data/v05/board_exclusion.json (bytes, pixels, dHash + pHash) and FUNSD ids against
the board's test documents. Images are saved at most SAVE_MAX px on the long side. dev = ~10% of documents.
"""
import argparse
import glob
import io
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v05_board_common import BoardIndex, choice_sample, image_hashes, manifest, seeded, write_split  # noqa: E402

OUT = os.path.join("v05", "board", "kie")
SAVE_MAX = 1280
CORD_GLOB = "raw/board/cord/data/train-*.parquet"
FUNSD_ZIP = "v05/funsd/raw/dataset.zip"

CORD_VALUE_Q = {"total_price": "What is the total amount on this receipt?",
                "changeprice": "How much change was given back?",
                "tax_price": "What is the tax amount on this receipt?",
                "subtotal_price": "What is the subtotal on this receipt?",
                "item_price": "What is the price listed for the item \"{nm}\"?"}
CORD_PRESENT = {"tax_price": "a tax amount", "service_price": "a service charge", "discount_price": "a discount",
                "changeprice": "an amount of change given back", "creditcardprice": "a credit card payment",
                "cashprice": "a cash payment amount", "subtotal_price": "a subtotal"}


# ----------------------------------------------------------------------------- amounts (same rules as the board kit)

def parse_amount(s) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    t = re.sub(r"(?i)^(rp\.?|idr|rm|@|x)\s*", "", t)
    t = re.sub(r"(?i)\s*(rp|sr|zr|z|s)$", "", t).strip()
    neg = t.startswith("-") or t.startswith("(")
    t = t.strip("-() ")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", t):
        v: float = int(re.sub(r"\D", "", t))
    elif re.fullmatch(r"\d+", t):
        v = int(t)
    elif re.fullmatch(r"\d{1,3}(,\d{3})*\.\d{1,2}", t):
        v = float(t.replace(",", ""))
    elif re.fullmatch(r"\d+,\d{1,2}", t):
        v = float(t.replace(",", "."))
    else:
        return None
    return None if neg or v <= 0 else v


def as_list(x) -> list:
    return [] if x is None else (x if isinstance(x, list) else [x])


def first(v):
    return v[0] if isinstance(v, list) and v else v


def cord_fields(gp: Dict[str, Any]) -> Tuple[Dict[str, Any], List[float], List[Tuple[str, float]]]:
    """(summary fields, every amount printed on the receipt, [(item name, price)])."""
    def merged(x):
        return {k: v for d in x for k, v in d.items()} if isinstance(x, list) else (x or {})
    sub, tot = merged(gp.get("sub_total")), merged(gp.get("total"))
    f = {k: sub[k] for k in ("subtotal_price", "tax_price", "service_price", "discount_price") if sub.get(k) is not None}
    f.update({k: tot[k] for k in ("total_price", "cashprice", "changeprice", "creditcardprice", "emoneyprice")
              if tot.get(k) is not None})
    amounts = [a for a in (parse_amount(first(v)) for v in f.values()) if a]
    amounts += [a for a in (parse_amount(v) for v in as_list(sub.get("etc"))) if a]
    items = []
    for m in as_list(gp.get("menu")):
        if not isinstance(m, dict):
            continue
        for key in ("price", "unitprice", "itemsubtotal"):
            amounts += [a for a in (parse_amount(v) for v in as_list(m.get(key))) if a]
        nm = first(m.get("nm"))
        pr = parse_amount(first(m.get("price")))
        if isinstance(nm, str) and nm.strip() and pr:
            items.append((nm.strip(), pr))
        for s in as_list(m.get("sub")):
            if isinstance(s, dict):
                amounts += [a for a in (parse_amount(v) for v in as_list(s.get("price"))) if a]
    return f, amounts, items


def fmt(v: float, dec: bool) -> str:
    return "{:,.2f}".format(v) if dec else "{:,}".format(int(round(v)))


def by_rank(gold: float, pool: List[float], k: int, rng: random.Random, counts: Counter) -> Optional[List[float]]:
    """k distractors with the number above the gold rotated (least used feasible count first)."""
    above, below = [p for p in pool if p > gold], [p for p in pool if p < gold]
    if len(above) + len(below) < k:
        return None
    feas = list(range(max(0, k - len(below)), min(k, len(above)) + 1))
    lo = min(counts[n] for n in feas)
    na = rng.choice([n for n in feas if counts[n] == lo])
    counts[na] += 1
    return rng.sample(above, na) + rng.sample(below, k - na)


def cord_questions(key: str, gp: Dict[str, Any], rank_counts: Counter, slot_counts: Counter) -> List[Tuple]:
    """[(qid, kind, field, question, options, gold)] for one CORD receipt."""
    f, amounts, menu = cord_fields(gp)
    vals = sorted(set(amounts))
    out = []
    rng = seeded("cord:" + key)
    if len(vals) >= 4:
        plan = [(fld, None, parse_amount(first(f.get(fld)))) for fld in ("total_price", "changeprice", "tax_price",
                                                                          "subtotal_price")]
        names = Counter(n.lower() for n, _ in menu)
        plan += [("item_price", n, pr) for n, pr in menu
                 if names[n.lower()] == 1 and 2 <= len(n) <= 40 and re.search(r"[A-Za-z]{2}", n)]
        for fld, nm, g in plan:
            if not g:
                continue
            ds = by_rank(g, [v for v in vals if abs(v - g) > 1e-9], 3, rng, rank_counts)
            if ds is None:
                continue
            dec = any(v != int(v) for v in ds + [g])
            slot = min(range(4), key=lambda s: (slot_counts[s], rng.random()))
            slot_counts[slot] += 1
            rng.shuffle(ds)
            opts = [fmt(v, dec) for v in ds[:slot] + [g] + ds[slot:]]
            if len(set(opts)) != 4:
                continue
            q = CORD_VALUE_Q[fld].format(nm=nm) if nm else CORD_VALUE_Q[fld]
            out.append(("value:%s%s" % (fld, ":" + nm if nm else ""), "value", fld, q, opts, fmt(g, dec)))
    for fld, phrase in CORD_PRESENT.items():
        has = fld in f and parse_amount(first(f[fld])) is not None
        if fld in f and not has:
            continue   # printed but unparseable: neither a clean yes nor a clean no
        out.append(("present:" + fld, "present", fld, "Does this receipt show %s?" % phrase, ["Yes", "No"],
                    "Yes" if has else "No"))
    return out


# ----------------------------------------------------------------------------- FUNSD

def funsd_type(t: str) -> str:
    if re.search(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d", t):
        return "date"
    d = sum(c.isdigit() for c in t)
    return "number" if d and d >= 0.5 * sum(c.isalnum() for c in t) else "text"


def funsd_questions(fid: str, ents: List[Dict[str, Any]], slot_counts: Counter) -> List[Tuple]:
    byid = {e["id"]: e for e in ents}
    answers = [e for e in ents if e["label"] == "answer" and 1 <= len(e["text"].strip()) <= 60]
    qtexts = Counter(e["text"].strip().lower() for e in ents if e["label"] == "question")
    rng = seeded("funsd:" + fid)
    out = []
    for e in ents:
        if e["label"] != "question":
            continue
        qt = e["text"].strip()
        if qtexts[qt.lower()] != 1 or not re.search(r"[A-Za-z]{2}", qt) or len(qt) > 50:
            continue
        name = qt.strip(" :\"'").strip()
        linked = [byid[b] for a, b in e["linking"] if a == e["id"] and b in byid and byid[b]["label"] == "answer"]
        linked = [x for x in linked if x["text"].strip()]
        if len(linked) == 1 and 1 <= len(linked[0]["text"].strip()) <= 60:
            gold = linked[0]["text"].strip()
            gt = funsd_type(gold)
            same = sorted({a["text"].strip() for a in answers if a["id"] != linked[0]["id"]
                           and a["text"].strip().casefold() != gold.casefold() and funsd_type(a["text"].strip()) == gt})
            if len(same) >= 3:
                ds = rng.sample(same, 3)
                if len({d.casefold() for d in ds}) == 3:
                    slot = min(range(4), key=lambda s: (slot_counts[s], rng.random()))
                    slot_counts[slot] += 1
                    out.append(("value:%d" % e["id"], "value", gt, 'On this form, what is filled in for the field "%s"?'
                                % name, ds[:slot] + [gold] + ds[slot:], gold))
            out.append(("present:%d" % e["id"], "present", "form_field", 'On this form, is the field "%s" filled in?'
                        % name, ["Yes", "No"], "Yes"))
        elif not linked and not any(byid.get(b, {}).get("label") == "answer" for _, b in e["linking"]):
            x0, y0, x1, y1 = e["box"]
            near = [a for a in answers if (abs((a["box"][1] + a["box"][3]) / 2 - (y0 + y1) / 2) < 12
                                           and 0 <= a["box"][0] - x1 < 250)
                    or (0 <= a["box"][1] - y1 < 25 and abs(a["box"][0] - x0) < 60)]
            if not near:
                out.append(("present:%d" % e["id"], "present", "form_field", 'On this form, is the field "%s" filled in?'
                            % name, ["Yes", "No"], "No"))
    return out


def balance_present(qs: List[Tuple[str, str, str]]) -> Tuple[set, Counter]:
    """qs = [(question id, field key, "Yes" | "No")]. Keep min(#Yes, #No) of each per field (seeded by id);
    returns (kept ids, dropped counts per field)."""
    groups: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for qid, fld, gold in qs:
        groups[fld][gold].append(qid)
    keep, dropped = set(), Counter()
    for fld, by in sorted(groups.items()):
        n = min(len(by["Yes"]), len(by["No"]))
        for lab in ("Yes", "No"):
            ranked = sorted(by[lab], key=lambda q: seeded("bal:" + q).random())
            keep.update(ranked[:n])
            dropped["present_unbalanced:" + fld] += len(ranked) - n
    return keep, dropped


# ----------------------------------------------------------------------------- build

def save(data: bytes, path: str) -> None:
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    s = SAVE_MAX / max(im.size)
    if s < 1:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, quality=92)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--no-cord", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    root = a.root
    board = BoardIndex.load(os.path.join(root, "v05", "board_exclusion.json"))
    drops: Counter = Counter()
    hits: List[Dict[str, Any]] = []
    qs: List[Tuple] = []   # (sid, source, kind, field, question, options, gold, image_id, image)
    docs: Counter = Counter()

    def admit(source: str, key: str, data: bytes, sid_ok: bool = True) -> Optional[str]:
        if not sid_ok:
            drops[source + ":board_source_id"] += 1
            return None
        got = board.hit(data)
        if got:
            hits.append({"source": source, "doc": key, "kind": got[0], "board": got[1]})
            drops[source + ":board_image_" + got[0]] += 1
            return None
        rel = os.path.join(OUT, "images", source, re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".jpg")
        if not os.path.exists(os.path.join(root, rel)):
            save(data, os.path.join(root, rel))
        docs[source] += 1
        return rel

    slot_counts: Counter = Counter()
    zf = zipfile.ZipFile(os.path.join(root, FUNSD_ZIP))
    base = "dataset/training_data"
    forms = sorted(os.path.splitext(os.path.basename(n))[0] for n in zf.namelist()
                   if n.startswith(base + "/annotations/") and n.endswith(".json"))
    for fid in forms:
        rel = admit("funsd", fid, zf.read("%s/images/%s.png" % (base, fid)), not board.hit_id("funsd:" + fid))
        if not rel:
            continue
        for qid, kind, fld, q, opts, gold in funsd_questions(fid, json.loads(zf.read(
                "%s/annotations/%s.json" % (base, fid)))["form"], slot_counts):
            qs.append(("board-kie:funsd:%s:%s" % (fid, qid), "board-kie-funsd", kind, q, opts, gold,
                       "funsd:" + fid, rel, fld))
    shards = sorted(glob.glob(os.path.join(root, CORD_GLOB)))
    if not a.no_cord and not shards:
        sys.exit("no CORD train shards under %s (run v05_board_fetch.py --only cord, or pass --no-cord)" % CORD_GLOB)
    rank_counts: Counter = Counter()
    for sh in ([] if a.no_cord else shards):
        import pyarrow.parquet as pq
        t = pq.read_table(sh)
        name = os.path.basename(sh).split("-of-")[0]
        for i in range(t.num_rows):
            r = t.slice(i, 1).to_pylist()[0]
            key = "%s-%d" % (name, i)
            rel = admit("cord", key, r["image"]["bytes"])
            if not rel:
                continue
            gp = json.loads(r["ground_truth"])["gt_parse"]
            for qid, kind, fld, q, opts, gold in cord_questions(key, gp, rank_counts, slot_counts):
                qs.append(("board-kie:cord:%s:%s" % (key, qid), "board-kie-cord", kind, q, opts, gold, "cord:" + key,
                           rel, fld))
    kept_pres, bal_drops = balance_present([(q[0], q[1] + ":" + q[8], q[5]) for q in qs if q[2] == "present"])
    drops.update(bal_drops)
    keep_ids = kept_pres | {q[0] for q in qs if q[2] != "present"}

    def rows():
        for sid, source, kind, q, opts, gold, iid, rel, fld in qs:
            if sid in keep_ids:
                yield choice_sample(sid, source, iid, rel, q, opts, gold, kind="%s:%s" % (kind, source.split("-")[-1]))

    out = os.path.join(root, OUT)
    counts = write_split(out, rows(), dev_frac=0.1)   # few documents: a larger dev share
    manifest(out, {"family": "kie", "board": "KIE (CORD+FUNSD)",
                   "source": {"cord": "naver-clova-ix/cord-v2@7f0115a4 train (4 parquet shards)",
                              "funsd": "FUNSD dataset.zip training_data (sha256 c3173564...)"},
                   "documents": dict(docs), "drops": dict(drops), "board_exclusion_hits": hits,
                   "split_counts": counts, "cord_shards": [os.path.basename(s) for s in shards],
                   "seconds": round(time.time() - t0)})
    print("kie: %s docs %s drops %s, %.0f s" % (counts, dict(docs), dict(drops), time.time() - t0))


if __name__ == "__main__":
    main()
