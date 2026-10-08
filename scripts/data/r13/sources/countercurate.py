"""Round 13 source: CounterCurate (Flickr30k photos with counterfactual partners). Licence: the authors' release
(research); Flickr30k images under the Flickr30k terms (research only). Plan 1.4.

    uv run python scripts/data/r13/sources/countercurate.py --root data [--limit 15000] [--force]

Source: HanSolo9682/Flickr30k-Counterfactuals (Hugging Face): clip_sample_data/train_left_right_.csv (caption /
negative caption, the negative picture is the mirrored original) and train_counterfactuals.csv (original caption
and picture; noun / adjective / reverse counterfactual captions with DALL-E 3 pictures in train_data.tar.gz).
The original Flickr30k pictures are not in that release: they come from nlphuji/flickr30k (flickr30k-images.zip).

Position pairs (kept; reliable by construction): the original and its left-right mirror (made here with PIL, as
the authors' prepare_training_data.py does). Both pictures get the same choice "Which caption describes the
image?" between the caption ("A is to the left of B") and its swap; the original's answer is the caption, the
mirror's the swap. Captions mentioning left / right more than once ("his left hand") are dropped, since
mirroring changes which hand is meant; one caption per picture.
Attribute pairs (noun / adjective counterfactuals with regenerated pictures; the reverse ones are word-order
nonsense and not used): kept only if a 100-pair spot check (runs/r13/spot/countercurate_attr_*.png) finds >= 90%
consistent (the regenerated picture fits its caption and not the original caption); see SPOT_ATTR. Missing pictures
and "nonexistent" placeholders are dropped.
Identity: "flickr30k:<id>" for the original; derived pictures are "flickr30k:<id>:mirror" / ":cc-noun" /
":cc-adjective" with group "flickr30k:<id>" (they inherit the original's identity).
--limit counts questions (two per pair).
"""
import csv
import glob
import hashlib
import os
import random
import re
import sys
import tarfile
import zipfile
from collections import Counter
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import (Identity, already_built, base_args, choice, contact_sheet, finish, hf_file, save_bytes,  # noqa: E402
                  spot_dir)

SOURCE = "countercurate"
REPO = "HanSolo9682/Flickr30k-Counterfactuals"
QUESTION = "Which caption describes the image?"
# 100-pair spot check of the attribute pairs: (looked at, consistent); filled in after looking
# Looked at sheets 1-4 (40 pairs): 28 consistent, 12 not (the original caption also fits the regenerated picture:
# "children" / "people", "tourists" / "crowd", "cyclists" / "people in helmets"; the regenerated picture does not
# show the change: "small basket", "fourth dog", "cloudy sky", "tired children", "different vehicle", "talks on a
# camera"; age / gender words that the picture leaves open: "adults" / "teenagers", "woman" / "girl"). 12 wrong
# already exceeds the 10 a 100-pair check may have at >= 90%, so the remaining 60 were not needed: not used.
SPOT_ATTR: Optional[tuple] = (40, 28)
MIN_CONSISTENT = 0.90


def lr_words(text: str) -> int:
    return len(re.findall(r"\b(left|right)\b", text.lower()))


def clean(text: str) -> str:
    t = " ".join(text.split()).strip()
    return re.sub(r"\s+([.,])", r"\1", t)


def flickr_id(path: str) -> str:
    return os.path.basename(path).split("_")[0].split(".")[0]


def pair_rows(kind: str, fid: str, a_rel: str, a_cap: str, b_rel: str, b_cap: str, b_role: str,
              rng: random.Random, group: str) -> List[dict]:
    """Two choice questions: the same two captions on picture a (answer a_cap) and picture b (answer b_cap)."""
    out = []
    for role, rel, iid, gold in (("original", a_rel, "flickr30k:%s" % fid, a_cap),
                                 (b_role, b_rel, "flickr30k:%s:%s" % (fid, b_role), b_cap)):
        r = choice("r13-cc:%s:%s:%s" % (kind, fid, role), SOURCE, "cc_%s" % kind, iid, rel, QUESTION,
                   [a_cap, b_cap], gold, rng, group=group)
        r["picture_role"] = role
        out.append(r)
    return out


def main(argv=None):
    p = base_args(__doc__)
    p.add_argument("--limit", type=int, default=15000)
    a = p.parse_args(argv)
    if already_built(a.root, "r13/%s/train.jsonl" % SOURCE, a.force):
        return
    from PIL import Image
    rng = random.Random(a.seed)
    steps, dropped = Counter(), Counter()
    ident = Identity(a.root)
    fz = zipfile.ZipFile(hf_file(a.root, "nlphuji/flickr30k", "flickr30k-images.zip"))
    have = {flickr_id(n) for n in fz.namelist() if n.startswith("flickr30k-images/") and n.endswith(".jpg")}
    img_dir = "r13/countercurate/images"

    def original(fid: str) -> Optional[str]:
        rel = "%s/flickr/%s.jpg" % (img_dir, fid)
        if not os.path.exists(os.path.join(a.root, rel)):
            if fid not in have:
                return None
            save_bytes(os.path.join(a.root, rel), fz.read("flickr30k-images/%s.jpg" % fid))
        return rel

    # ---- positions (mirror)
    lr = list(csv.DictReader(open(hf_file(a.root, REPO, "clip_sample_data/train_left_right_.csv")), delimiter="\t"))
    steps["position_rows_read"] = len(lr)
    by_img: Dict[str, list] = {}
    for r in lr:
        fid = flickr_id(r["images"])
        if lr_words(r["captions"]) != 1 or lr_words(r["neg_captions"]) != 1:
            dropped["position: caption mentions left / right more than once"] += 1
            continue
        if fid not in have:
            dropped["position: Flickr30k picture missing"] += 1
            continue
        by_img.setdefault(fid, []).append(r)
    pos_rows = []
    for fid in sorted(by_img, key=lambda f: _h(f)):
        if ident.held("flickr30k:%s" % fid):
            dropped["position: held-out picture"] += 1
            continue
        r = random.Random(_h(fid)).choice(by_img[fid])
        rel = original(fid)
        mrel = "%s/mirror/%s.jpg" % (img_dir, fid)
        if not os.path.exists(os.path.join(a.root, mrel)):
            os.makedirs(os.path.dirname(os.path.join(a.root, mrel)), exist_ok=True)
            with Image.open(os.path.join(a.root, rel)) as im:
                im.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(a.root, mrel), quality=95)
        pos_rows += pair_rows("position_mirror", fid, rel, clean(r["captions"]), mrel, clean(r["neg_captions"]),
                              "mirror", rng, "flickr30k:%s" % fid)
    steps["position_pairs"] = len(pos_rows) // 2

    # ---- attributes (regenerated pictures)
    attr_rows: List[dict] = []
    cf = list(csv.DictReader(open(hf_file(a.root, REPO, "clip_sample_data/train_counterfactuals.csv")),
                             delimiter="\t"))
    steps["attribute_rows_read"] = len(cf)
    cands = []
    for r in cf:
        fid = flickr_id(r["images"])
        for kind in ("noun", "adj"):
            img, cap = r["%s_images" % kind], r["%s_captions" % kind]
            if img == "nonexistent" or cap == "nonexistent" or not cap.strip():
                dropped["attribute: placeholder (nonexistent)"] += 1
                continue
            if clean(cap).lower() == clean(r["captions"]).lower():
                dropped["attribute: captions identical"] += 1
                continue
            cands.append((fid, kind, clean(r["captions"]), os.path.splitext(os.path.basename(img))[0], clean(cap)))
    steps["attribute_candidate_pairs"] = len(cands)
    tar_path = None
    try:
        tar_path = hf_file(a.root, REPO, "train_data.tar.gz")
    except Exception as e:      # noqa: BLE001
        print("CounterCurate train_data.tar.gz not available (%s): attribute pairs skipped" % e)
    if tar_path:
        gen = extract_generated(a.root, tar_path, img_dir)
        cands = [c for c in cands if c[3] in gen and c[0] in have and not ident.held("flickr30k:%s" % c[0])]
        steps["attribute_pairs_with_pictures"] = len(cands)
        sample = random.Random(1).sample(cands, min(100, len(cands)))
        sheets = spot_attr(a.root, sample, gen, original)
        steps["attribute_spot_sheets"] = len(sheets)
        # used only when a full 100-pair check passed (>= 90 consistent)
        if SPOT_ATTR and SPOT_ATTR[0] >= 100 and SPOT_ATTR[1] / SPOT_ATTR[0] >= MIN_CONSISTENT:
            per = Counter()
            for fid, kind, cap, name, ncap in sorted(cands, key=lambda c: _h(c[3])):
                if per[fid] >= 1:
                    continue
                per[fid] += 1
                attr_rows += pair_rows("attribute_" + kind, fid, original(fid), cap, gen[name], ncap,
                                       "cc-" + kind, rng, "flickr30k:%s" % fid)
        else:
            dropped["attribute pairs: spot check not passed / not done (%s)" % (SPOT_ATTR,)] = len(cands)
    # mix: attributes (if any) and positions, then the limit
    rows = attr_rows[: a.limit // 2] + pos_rows
    rows = rows[: a.limit - a.limit % 2]
    near = ident.near_held({r["image"] for r in rows if r["picture_role"] == "original"}, SOURCE)
    bad = {r["group"] for r in rows if r["image"] in near}
    dropped["pair whose original is within dHash 2 of a held-out photo"] = len(bad)
    rows = [r for r in rows if r["group"] not in bad]
    steps["questions"] = len(rows)
    pos_sheet = spot_pos(a.root, [r for r in rows if r["kind"] == "cc_position_mirror"])
    finish(a.root, SOURCE, rows, dict(steps), dict(dropped), {
        "licence": "CounterCurate release (research); Flickr30k terms (research only)",
        "identity_backend": ident.backend, "spot_attr": SPOT_ATTR, "position_sheet": pos_sheet,
        "limit_questions": a.limit},
        report=REPORT.format(steps=dict(steps), dropped=dict(dropped), spot=SPOT_ATTR, pos=SPOT_POS))


REPORT = """# CounterCurate (round 13)

Built by `scripts/data/r13/sources/countercurate.py`; counts in MANIFEST.json.

- Steps: {steps}
- Dropped per rule: {dropped}
- Attribute pairs, spot check (looked at, consistent): {spot}. 12 inconsistent among the first 40 already exceed
  the 10 that 100 pairs may have at >= 90% consistency, so the attribute pairs are not used (the authors' own check
  was 84.67%).
- Position (mirror) pairs, spot check: {pos}
"""
SPOT_POS = ("15 pairs looked at (countercurate_position_1.png): the mirror always flips the answer, as constructed; the "
            "caption itself is clearly right in 8 / 15, unclear at sheet size or for overlapping / vertically "
            "stacked things in 6 ('a man is to the right of a microphone' for a singer holding one, 'water is to "
            "the right of a harness'), meaningless in 1 ('a bite is to the left of glasses'). Mirroring does not "
            "make such a caption wrong on one side only, so these pairs carry no false label, only weak ones.")


def extract_generated(root: str, tar_path: str, img_dir: str) -> Dict[str, str]:
    """file stem -> path of every regenerated (DALL-E) picture, extracted once from train_data.tar.gz."""
    out_dir = os.path.join(root, img_dir, "generated")
    done = os.path.join(out_dir, ".extracted")
    if not os.path.exists(done):
        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(tar_path) as t:
            for m in t:
                if m.isfile() and "dalle_gen_counterfactuals/" in m.name and m.name.endswith((".jpg", ".png")):
                    dest = os.path.join(out_dir, os.path.basename(m.name))
                    if not os.path.exists(dest):
                        save_bytes(dest, t.extractfile(m).read())
        open(done, "w").write("ok")
    return {os.path.splitext(n)[0]: "%s/generated/%s" % (img_dir, n) for n in os.listdir(out_dir)
            if not n.startswith(".")}


def spot_attr(root, sample, gen, original) -> List[str]:
    paths = []
    for k in range(0, len(sample), 10):
        items = []
        for fid, kind, cap, name, ncap in sample[k:k + 10]:
            items.append({"image": original(fid), "text": "ORIG: %s" % cap})
            items.append({"image": gen[name], "text": "%s: %s" % (kind.upper(), ncap)})
        path = os.path.join(spot_dir(), "%s_attr_%d.png" % (SOURCE, k // 10 + 1))
        contact_sheet(root, items, path, cols=4, cell=280)
        paths.append(path)
    return paths


def spot_pos(root, rows) -> str:
    origs = [r for r in rows if r["picture_role"] == "original"]
    pick = random.Random(1).sample(origs, min(15, len(origs)))
    items = []
    for r in pick:
        items.append({"image": r["image"], "text": "ORIG -> %s" % r["answer_key"]})
        m = next(x for x in rows if x["group"] == r["group"] and x["picture_role"] == "mirror")
        items.append({"image": m["image"], "text": "MIRROR -> %s" % m["answer_key"]})
    path = os.path.join(spot_dir(), "%s_position_1.png" % SOURCE)
    contact_sheet(root, items, path, cols=6, cell=260)
    return path


def _h(x) -> str:
    return hashlib.sha1(("%s%s" % (SOURCE, x)).encode()).hexdigest()


if __name__ == "__main__":
    main()
