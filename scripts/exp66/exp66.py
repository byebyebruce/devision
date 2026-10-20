"""exp66: the selector with column / margin layout variants, EXP64's head frozen (critic EXP66-PREP-01 / -R1-ACCEPT-01;
docs/experiments/2026-10-19-exp66-layout-plan.md). Fixed: EXP64's head 0e275bdf, EXP57's vision / decision (962503fb /
09a5b447), grid n12 / 0.25, the region rule. The only trained part: a new selector from EXP53's end (8f10f1fd), EXP62's
recipe item by item (L1 + GIoU, AdamW 1e-3 / 0.01, warm-up 100, cosine, seed 0, 2,000 updates, last step, CPU), the
batches from data/exp66/plan.jsonl (EXP62's generator; the pair pool = EXP62's 1,200 + exp66's 1,200 pairs). The control
is the same head with EXP62's selector c92b7b7e; its reads must reproduce EXP64's registered ones (else stop / invalid).
candidate_eligible is always false.

  start   model-free: EXP64's run (START / registry / report / head) and registered reads, EXP62 (run, selector,
          parents / plan), EXP61's registered features, EXP53's selector, EXP60, EXP58 bundle + EXP57 end, data/exp61,
          data/exp66 (MANIFEST, questions, pictures, plan), exp49, code.
  global  MPS: EXP57's frozen full-picture patches of the 800 exp66 training pictures (train_C + train_C_parent); the
          question words are those of an EXP61 question with the same text (the selector embeds the text only).
  train   CPU: the plan; the new selector; its boxes on every training record; a bundle (new selector + EXP64 head).
  reads   MPS: per dev picture ONE visual encoding (its first 256 keys / values are the selector's patches -- tune_C's
          patches are taken here, not in global), the EXP64 head on the tokens of each box source: rl_tune, tune_V and
          dev24 with the new selector and c92 (c92 reproduces EXP64), tune_C with the new selector, c92 and its label
          boxes (descriptive); dev24's 24 comparison requests with the new selector's reads through ordinary decide
          (report only).
  report  model-free: EXP61's statistics; new - c92 per slice (case clusters); the pre-written gates and branch.

    uv run python scripts/exp66/exp66.py start --deadline EPOCH | global | train | reads | report [--finalize CODES]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, E64, E62 = "runs/exp66", "runs/exp64", "runs/exp62"
E64_SHA = {"START.json": "5ac4da33924376e19e0e450d157cccdf204cc60680f9969df7fae300e8b6d8ad",
           "registry.json": "d743f3e290db25d45091cd27fde3335974d3123e7cdf0411ad7ae31674f5dff3",
           "report.json": "ae61798133c40deef6c728790e4dc9f451dcf1fc3224aa86f807e6163bda3bdf",
           "head.pt": "0e275bdf3b47bff9b8786160b090b79ae903734d784c828d5de65e61c6a2d7c3"}
HEAD_SHA = E64_SHA["head.pt"]
D66 = "data/exp66"
Q66, PLAN, PLAN_JSON, M66 = D66 + "/questions.jsonl", D66 + "/plan.jsonl", D66 + "/PLAN.json", D66 + "/MANIFEST.json"
M66_SHA = "c8f0962706c245b3a73b2194b48de1d84ee93822c0fb8487d4689671dc5cec38"
FILES = ["scripts/exp66/exp66.py", "scripts/exp66/build.py", "scripts/exp66/plan.py", "scripts/exp66/run.sh", "scripts/exp64/exp64.py",
         "scripts/exp62/exp62.py", "scripts/exp62/build.py", "scripts/exp61/exp61.py", "scripts/exp53/selector.py", "scripts/exp60/exp60.py",
         "scripts/exp59/exp59.py", "scripts/exp58/live.py", "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py",
         "src/devision/image.py", "scripts/exp53/watch.py"]
# the c92 control's reference: EXP64's registered reads with c92 + this head (group -> EXP64 file, picture key)
REF = {"rl_tune": ("rl_tune_new", lambda x: x["id"]), "tune_V": ("tune_V_auto_new", lambda x: x["picture"]),
       "dev24": ("dev24_auto_new", lambda x: x["id"])}
GROUPS = {"rl_tune": ["rl_tune_new", "rl_tune_c92"], "tune_V": ["tune_V_new", "tune_V_c92"],
          "tune_C": ["tune_C_new", "tune_C_c92", "tune_C_label"], "dev24": ["dev24_new", "dev24_c92"]}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e64 = _load("exp64", "../exp64/exp64.py")
e62 = _load("exp62", "../exp62/exp62.py")
p66 = _load("exp66_plan", "plan.py")
e61, e60, e59, e58, e54 = e64.e61, e64.e60, e64.e59, e64.e58, e64.e54
sel = e61.sel
sha, write_json, register, jsonl = e61.sha, e61.write_json, e61.register, e61.jsonl
C92 = e64.E62_SHA["selector.pt"]


# ------------------------------------------------------------------ identities

def bound_e64() -> Dict[str, str]:
    bad = [f for f, h in E64_SHA.items() if sha(os.path.join(E64, f)) != h]
    if bad:
        raise SystemExit("EXP64's %s changed" % bad)
    reg = json.load(open(os.path.join(E64, "registry.json")))
    for files in reg.values():
        for f, h in files.items():
            if sha(os.path.join(E64, f)) != h:
                raise SystemExit("EXP64's registered %s changed" % f)
    if reg["head"]["head.pt"] != HEAD_SHA:
        raise SystemExit("EXP64's registered head is not 0e275bdf")
    return dict(E64_SHA)


def pictures_digest(pics: List[str], root: str) -> str:
    return hashlib.sha256(json.dumps({p: sha(os.path.join(root, p)) for p in sorted(set(pics))}, sort_keys=True).encode()).hexdigest()


def bound_d66(root: str = "data") -> Dict[str, str]:
    """data/exp66 as accepted (MANIFEST c8f09627) and its plan (PLAN.json); with root the pictures' bytes too (the variants
    against the MANIFEST, the parents -- the generator's own files -- by their digest)."""
    if sha(M66) != M66_SHA:
        raise SystemExit("data/exp66/MANIFEST.json is not the accepted one")
    m = json.load(open(M66))
    if sha(Q66) != m["questions_sha256"]:
        raise SystemExit("data/exp66 questions differ from the MANIFEST")
    pm = json.load(open(PLAN_JSON))
    if sha(PLAN) != pm["plan_sha256"] or pm["sources"]["exp66_questions"] != m["questions_sha256"] or pm["sources"]["exp66_manifest"] != M66_SHA \
            or pm["sources"]["exp62_parents"] != sha(e62.PARENTS) or pm["sources"]["exp61_questions"] != sha(e61.V_QUESTIONS) \
            or pm["sources"]["exp49_questions"] != sha(e61.QUESTIONS):
        raise SystemExit("data/exp66 plan differs from PLAN.json or was built on other sources")
    out = {"manifest": M66_SHA, "questions": m["questions_sha256"], "plan": pm["plan_sha256"], "plan_json": sha(PLAN_JSON)}
    if root:
        qs = jsonl(Q66)
        if pictures_digest([q["picture"] for q in qs if q["set"] != "train_C_parent"], root) != m["pictures_sha256"]:
            raise SystemExit("data/exp66 pictures differ from the MANIFEST")
        out["parent_pictures"] = pictures_digest([q["picture"] for q in qs if q["set"] == "train_C_parent"], root)
    return out


def identities(data_root: str) -> Dict:
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)
    if st58["bundle"]["selector.pt"] != e61.SELECTOR_SHA:
        raise SystemExit("EXP58's bundle does not hold EXP53's selector")
    return {"exp64": bound_e64(), "exp62": e64.bound_e62(), "exp61": e62.bound_e61(), "exp62_data": e62.check_parents(data_root),
            "exp53": e61.bound_e53(), "exp60": e61.bound_e60(), "variants": e64.bound_variants(data_root), "data": bound_d66(data_root),
            "exp49_questions": sha(e61.QUESTIONS), "model": st58["model"], "bundle": st58["bundle"]}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    ident = identities(data_root)
    pm = json.load(open(PLAN_JSON))
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    qs = jsonl(Q66)
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, **ident,
         "recipe": {"init": "runs/exp53/selector.pt " + e61.SELECTOR_SHA, "steps": len(jsonl(PLAN)), "lr": sel.LR, "weight_decay": sel.WD,
                    "warmup": sel.WARMUP, "seed": sel.SEED, "loss": "L1(cx, cy, w, h) + (1 - GIoU)", "device": "cpu",
                    "batch": "24 (parent, crop) pairs + 16 fit_O / fit_L replay (data/exp66/plan.jsonl)", "exposures": pm["stats"]},
         "frozen": {"head": HEAD_SHA, "control_selector": C92},
         "requests": {"global_pictures": len({q["picture"] for q in qs if q["set"] in ("train_C", "train_C_parent")}),
                      "dev_pictures": {"rl_tune": 600, "tune_V": 600, "tune_C": len({q["picture"] for q in qs if q["set"] == "tune_C"}), "dev24": 24},
                      "dev24_decide": 24},
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline,
         "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    now = {"exp64": bound_e64(), "exp62": e64.bound_e62(), "exp61": e62.bound_e61(), "exp53": e61.bound_e53(), "exp60": e61.bound_e60(),
           "exp49_questions": sha(e61.QUESTIONS), "exp62_data": e62.check_parents(data_root), "variants": e64.bound_variants(data_root)}
    for k, v in now.items():
        if v != st[k]:
            raise SystemExit("%s differs from START" % k)
    d = bound_d66(data_root)
    if any(st["data"][k] != v for k, v in d.items()):
        raise SystemExit("data/exp66 differs from START")
    return st


# ------------------------------------------------------------------ global / train

def train_pictures() -> List[str]:
    return sorted({q["picture"] for q in jsonl(Q66) if q["set"] in ("train_C", "train_C_parent")})


def stage_global(out: str = OUT, data_root: str = "data") -> Dict:
    import numpy as np
    import torch
    from PIL import Image

    from devision.image import patch_geometry, to_pixel_values
    from devision.infer import Decider, best_device
    st = check_start(out, data_root)
    if e58.fz.weights(e58.NEW_RUN) != st["model"]:
        raise SystemExit("EXP57's weights are not START's: refused before loading")
    pics = train_pictures()
    dev = best_device()
    m = Decider.load(os.path.join(e58.NEW_RUN, "stage2"), device=dev).model.eval()
    patches = np.zeros((len(pics), 256, 768), np.float16)
    geom = np.zeros((len(pics), 256, 4), np.float32)
    with torch.no_grad():
        for s in range(0, len(pics), 16):
            chunk = pics[s:s + 16]
            px, sizes = [], []
            for p in chunk:
                with Image.open(os.path.join(data_root, p)) as img:
                    px.append(to_pixel_values(img, 256))
                    sizes.append(img.size)
            pt = m.vision_patches(torch.stack(px).to(dev)).float().cpu().numpy()
            for k, (w, h) in enumerate(sizes):
                patches[s + k] = pt[k]
                geom[s + k] = patch_geometry((0, 0, w, h), w, h, 256, 16).numpy()
            e54.progress(out, "global", s + len(chunk), len(pics))
    path = os.path.join(out, "exp66_global.npz")
    np.savez(path, patches=patches, geom=geom, pictures=np.array(pics))
    register(out, "global", {"exp66_global.npz": sha(path)})
    return {"pictures": len(pics)}


def word_rows(records: List[Dict], texts: List[str]) -> List[int]:
    """For each text, the index of a record with that question text (its words were embedded from the text alone)."""
    first: Dict[str, int] = {}
    for i, r in enumerate(records):
        if "text" in r and r["text"] not in first:
            first[r["text"]] = i
    missing = sorted(set(texts) - set(first))
    if missing:
        raise SystemExit("no embedded words for %s" % missing)
    return [first[t] for t in texts]


def all_inputs(out: str):
    """(Inputs, records, key -> index): EXP62's (exp49 + exp61 + EXP62's parents, their registered features), then exp66's
    train_C / train_C_parent with this run's patches and the words of an EXP61 question with the same text."""
    import numpy as np
    import torch
    inp, recs, _ = e62.all_inputs(E62)
    e54.check_registered(out, "global")
    z = np.load(os.path.join(out, "exp66_global.npz"))
    pics = z["pictures"].tolist()
    new = [q for q in jsonl(Q66) if q["set"] in ("train_C", "train_C_parent")]
    rows = word_rows(recs, [q["text"] for q in new])
    n_pics = inp.patches.shape[0]
    patches = torch.cat([inp.patches, torch.from_numpy(z["patches"])])
    geom = torch.cat([inp.geom, torch.from_numpy(z["geom"])])
    words = torch.cat([inp.words, inp.words[rows]])
    mask = torch.cat([inp.mask, inp.mask[rows]])
    pic_index = list(inp.pic) + [n_pics + pics.index(q["picture"]) for q in new]
    records = recs + new
    at = {q["key"]: i for i, q in enumerate(records)}
    if len(at) != len(records):
        raise SystemExit("duplicate record keys")
    return sel.Inputs(patches, geom, pic_index, words, mask), records, at


def stage_train(out: str = OUT) -> Dict:
    import torch

    from devision.reading import write_bundle
    st = check_start(out, data_root="")
    inp, recs, at = all_inputs(out)
    targets = torch.tensor([sel.target_of(q["box"], q["width"], q["height"]) for q in recs])
    batches = [[at[k] for k in b] for b in jsonl(PLAN)]
    init_path = os.path.join(e61.E53, "selector.pt")
    if sha(init_path) != e61.SELECTOR_SHA:
        raise SystemExit("EXP53's selector changed")
    init = torch.load(init_path, map_location="cpu")
    t0 = time.time()
    model, stats = e62.train_plan(init, inp, targets, batches, on_step=lambda k: e54.progress(out, "train", k, len(batches)))
    torch.save(model.state_dict(), os.path.join(out, "selector.pt"))
    trained = sorted({i for b in batches for i in b})
    with open(os.path.join(out, "boxes.jsonl"), "w") as fh:
        for i, f in zip(trained, sel.predict(model, inp, trained)):
            q = recs[i]
            px, bad = sel.pixel_box(f, q["width"], q["height"])
            fh.write(json.dumps({"key": q["key"], "set": q["set"], "box": px, "degenerate": bad}) + "\n")
    bundle = write_bundle(os.path.join(out, "bundle"), os.path.join(e58.NEW_RUN, "stage2"), os.path.join(out, "selector.pt"),
                          os.path.join(E64, "head.pt"),
                          {"selector": {"path": os.path.join(out, "selector.pt"), "init": e61.SELECTOR_SHA, "recipe": st["recipe"]},
                           "head": {"path": os.path.join(E64, "head.pt"), "sha256": HEAD_SHA}})
    info = {**stats, "train_records": len(trained), "seconds": time.time() - t0, "selector_sha256": sha(os.path.join(out, "selector.pt")),
            "bundle_reading_json": sha(os.path.join(out, "bundle", "reading.json")), "bundle": bundle["sha256"], "plan_sha256": sha(PLAN)}
    write_json(os.path.join(out, "train.json"), info)
    register(out, "selector", {f: sha(os.path.join(out, f)) for f in ("selector.pt", "boxes.jsonl", "train.json")})
    return info


def new_selector_sha(out: str) -> str:
    """The trained selector: file, registry, train.json and the bundle name one file, whose head is EXP64's."""
    reg = e54.check_registered(out, "selector")
    actual = sha(os.path.join(out, "selector.pt"))
    tj = json.load(open(os.path.join(out, "train.json")))
    b = json.load(open(os.path.join(out, "bundle", "reading.json")))["sha256"]
    if not (actual == reg["selector.pt"] == tj["selector_sha256"] == b["selector.pt"] == sha(os.path.join(out, "bundle", "selector.pt"))):
        raise SystemExit("selector.pt / train.json / bundle / registry do not name one selector")
    if b["head.pt"] != HEAD_SHA or tj["bundle_reading_json"] != sha(os.path.join(out, "bundle", "reading.json")):
        raise SystemExit("the bundle is not the recorded one with EXP64's head")
    return actual


# ------------------------------------------------------------------ reads

def identity(out: str, name: str, new_sha: str) -> Dict:
    src = name.rsplit("_", 1)[1]
    ident = {"start_sha256": sha(os.path.join(out, "START.json")), "set": name, "head_sha256": HEAD_SHA}
    if src == "label":
        ident["boxes"] = "data/exp66/questions.jsonl " + json.load(open(M66))["questions_sha256"]
    else:
        ident["selector_sha256"] = new_sha if src == "new" else C92
    return ident


def write_reads(out: str, name: str, recs: List[Dict], new_sha: str) -> None:
    f = "%s.jsonl" % name
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": identity(out, name, new_sha)}) + "\n" + "".join(json.dumps(x) + "\n" for x in recs))
    register(out, "reads_" + name, {f: sha(os.path.join(out, f))})


def reference(group: str) -> Dict[str, Dict[str, Dict]]:
    """EXP64's registered reads with c92 + this head for the group (checked against EXP64's registry)."""
    name, key = REF[group]
    f = os.path.join(E64, name + ".jsonl")
    if e54.check_registered(E64, "reads_" + name).get(name + ".jsonl") != sha(f):
        raise SystemExit("EXP64's %s is not registered" % name)
    lines = jsonl(f)
    if lines[0]["identity"].get("head_sha256") != HEAD_SHA or lines[0]["identity"].get("selector_sha256") != C92:
        raise SystemExit("EXP64's %s is not c92 + 0e275bdf" % name)
    return {key(x): {r["question"]: r for r in x["reads"]} for x in lines[1:]}


def close_group(out: str, group: str, results: Dict[str, List[Dict]], new_sha: str) -> Optional[Dict]:
    """A finished group: its read files written and registered; for rl_tune / tune_V / dev24 the c92 reads' reproduction
    of EXP64 recorded and registered (repro_<group>.json); a failure raises only after both are on disk."""
    for n in GROUPS[group]:
        write_reads(out, n, results[n], new_sha)
    if group not in REF:
        return None
    rep = e64.reproduce(results[group + "_c92"], reference(group), REF[group][1])
    f = "repro_%s.json" % group
    write_json(os.path.join(out, f), rep)
    register(out, "repro_" + group, {f: sha(os.path.join(out, f))})
    if not rep["reproduced"]:
        raise SystemExit("%s: c92 + EXP64's head does not reproduce EXP64's registered reads (%d mismatches, %d missing) -- stopped"
                         % (group, len(rep["mismatches"]), len(rep["missing_pictures"])))
    return rep


def tune_c_by_picture() -> Dict[str, List[Dict]]:
    by: Dict[str, List[Dict]] = defaultdict(list)
    for q in jsonl(Q66):
        if q["set"] == "tune_C":
            by[q["picture"]].append(q)
    return by


def stage_reads(out: str = OUT, data_root: str = "data") -> Dict:
    import copy

    import torch

    from devision.reading import TEMPLATES, reads_for, serialise
    from devision.train.evaluate import _probabilities
    st = check_start(out, data_root)
    new_sha = new_selector_sha(out)                                              # before any load
    if sha(os.path.join(E64, "head.pt")) != HEAD_SHA or sha(os.path.join(E62, "selector.pt")) != C92:
        raise SystemExit("EXP64's head / EXP62's selector changed")
    reading, st58, pics58 = e58.load_reading(e59.E58, data_root=data_root)      # EXP57 + EXP58 bundle, verified
    dev = next(reading.decider.model.parameters()).device
    reading.head.load_state_dict(torch.load(os.path.join(E64, "head.pt"), map_location="cpu"), strict=True)
    reading.head.to(dev).eval()
    sels = {"new": copy.deepcopy(reading.selector), "c92": copy.deepcopy(reading.selector)}
    sels["new"].load_state_dict(torch.load(os.path.join(out, "selector.pt"), map_location="cpu"), strict=True)
    sels["c92"].load_state_dict(torch.load(os.path.join(E62, "selector.pt"), map_location="cpu"), strict=True)
    for s in sels.values():
        s.to(dev).eval()
    heads = {"head": reading.head}
    m, cfg = reading.decider.model, reading.decider.cfg
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    vby: Dict[str, List[Dict]] = defaultdict(list)
    for q in jsonl(e61.V_QUESTIONS):
        if q["set"] == "tune_V":
            vby[q["picture"]].append(q)
    cby = tune_c_by_picture()
    pics = {p: sha(os.path.join(data_root, p)) for p in list(vby) + list(cby)}  # check_start just verified them
    total, done = len(tune) + len(vby) + len(cby) + 24, 0
    results: Dict[str, List[Dict]] = defaultdict(list)

    def auto(kv, img, ts, who):
        reading.selector = sels[who]
        return e64.read_heads(reading, kv, img.size[0], img.size[1], ts, None, heads)["head"]

    def tick():
        nonlocal done
        done += 1
        e54.progress(out, "reads", done, total)
    for rid in sorted(tune):                                                      # rl_tune
        img = e60.picture(data_root, tune[rid]["image"], pics58)
        kv = e64.encode(m, cfg, img)
        ts = reads_for(tune[rid]["questions"])
        for who in sels:
            results["rl_tune_" + who].append({"id": rid, "reads": auto(kv, img, ts, who)})
        tick()
    close_group(out, "rl_tune", results, new_sha)
    for p in sorted(vby):                                                         # tune_V (EXP61 layouts)
        img = e60.picture(data_root, p, pics)
        kv = e64.encode(m, cfg, img)
        ts = e61.templates_of(vby[p])
        for who in sels:
            results["tune_V_" + who].append({"picture": p, "reads": auto(kv, img, ts, who)})
        tick()
    close_group(out, "tune_V", results, new_sha)
    for p in sorted(cby):                                                         # tune_C (new layouts), + label boxes
        img = e60.picture(data_root, p, pics)
        kv = e64.encode(m, cfg, img)
        ts = e61.templates_of(cby[p])
        for who in sels:
            results["tune_C_" + who].append({"picture": p, "reads": auto(kv, img, ts, who)})
        lab = e64.read_heads(reading, kv, img.size[0], img.size[1], ts, {q["text"]: q["box"] for q in cby[p]}, heads)["head"]
        results["tune_C_label"].append({"picture": p, "reads": lab})
        tick()
    close_group(out, "tune_C", results, new_sha)
    dev24 = []
    for it in e59.items():                                                        # dev24: new and c92; decide with new
        row = it["row"]
        img = e60.picture(data_root, row["image"], pics58)
        kv = e64.encode(m, cfg, img)
        ts = reads_for(row["questions"])
        a = {who: auto(kv, img, ts, who) for who in sels}
        for who in sels:
            results["dev24_" + who].append({"id": row["id"], "reads": a[who]})
        table = [[*TEMPLATES[x["question"]], x["value"]] for x in a["new"]]
        state = serialise(row.get("state_text") or "", table)
        res = reading.decider.decide(state, {it["key"]: row["questions"][it["qid"]]}, image=None)
        dev24.append(e59.record(it, _probabilities(res["answers"][it["key"]]), {"diagnostics": {"reads": a["new"], "table": table, "state": state}}))
        tick()
    reading.selector = sels["c92"]
    close_group(out, "dev24", results, new_sha)
    e59.check_diagnostics(dev24, e59.items())
    f = "details_dev24.jsonl"
    with open(os.path.join(out, f), "w") as fh:
        fh.write(json.dumps({"identity": {"start_sha256": sha(os.path.join(out, "START.json")), "selector_sha256": new_sha, "head_sha256": HEAD_SHA,
                                          "model": st["model"], "temperature": "none (raw)"}}) + "\n" + "".join(json.dumps(r) + "\n" for r in dev24))
    register(out, "decide_dev24", {f: sha(os.path.join(out, f))})
    return {"pictures": done}


# ------------------------------------------------------------------ report

def question_sets() -> Dict[str, List[Dict]]:
    s = e61.question_sets()
    s["tune_C"] = [dict(q, _pic=q["picture"], _template=q["text"]) for q in jsonl(Q66) if q["set"] == "tune_C"]
    return s


def templates_by_key() -> Dict[str, Dict[str, List[str]]]:
    t = e64.templates_by_key()
    cby = tune_c_by_picture()
    return {"rl_tune": t["rl_tune"], "tune_V": t["tune_V_auto"], "tune_C": {p: e61.templates_of(v) for p, v in cby.items()}, "dev24": t["dev24_auto"]}


KEYS = {"rl_tune": lambda x: x["id"], "tune_V": lambda x: x["picture"], "tune_C": lambda x: x["picture"], "dev24": lambda x: x["id"]}


def read_items(out: str, name: str, new_sha: str) -> List[Dict]:
    f = os.path.join(out, name + ".jsonl")
    if e54.check_registered(out, "reads_" + name).get(name + ".jsonl") != sha(f):
        raise SystemExit("%s not registered" % name)
    lines = jsonl(f)
    want = identity(out, name, new_sha)
    if any(lines[0]["identity"].get(k) != v for k, v in want.items()) or set(lines[0]["identity"]) != set(want):
        raise SystemExit("%s: identity does not bind" % name)
    return lines[1:]


def validity(out: str, new_sha: Optional[str]) -> Dict:
    """Every required read file present, registered, bound (START, set, selector or label boxes, head), structurally
    complete (templates once each, value = format of decoded); c92's reads reproducing EXP64's on rl_tune / tune_V /
    dev24 (recomputed here). missing -> incomplete; anything else wrong -> invalid."""
    res: Dict[str, Any] = {"groups": {}, "missing": [], "invalid": []}
    want = templates_by_key()
    for group, names in GROUPS.items():
        items: Dict[str, List[Dict]] = {}
        for n in names:
            if not os.path.exists(os.path.join(out, n + ".jsonl")):
                res["missing"].append(n)
                continue
            try:
                if new_sha is None:
                    raise SystemExit("no verified trained selector")
                items[n] = read_items(out, n, new_sha)
                seen = set()
                for x in items[n]:
                    k = KEYS[group](x)
                    if k in seen or k not in want[group]:
                        raise SystemExit("%s: picture %s twice / unknown" % (n, k))
                    seen.add(k)
                    e64.check_reads_list(x["reads"], want[group][k])
                if seen != set(want[group]):
                    raise SystemExit("%s: %d pictures missing" % (n, len(set(want[group]) - seen)))
            except SystemExit as e:
                res["invalid"].append("%s: %s" % (n, e))
        if group in REF and group + "_c92" in items:
            try:
                rep = e64.reproduce(items[group + "_c92"], reference(group), KEYS[group])
            except SystemExit as e:
                res["invalid"].append("%s reference: %s" % (group, e))
                continue
            res["groups"][group] = {k: v for k, v in rep.items() if k != "mismatches"} | {"mismatches": len(rep["mismatches"])}
            if not rep["reproduced"]:
                res["invalid"].append("%s: c92 does not reproduce EXP64" % group)
    res["dev24_decide"] = "completed (report only)" if os.path.exists(os.path.join(out, "details_dev24.jsonl")) else "missing (report only)"
    res["state"] = "invalid" if res["invalid"] else "incomplete" if res["missing"] else "valid"
    return res


def no_worse(new: Dict, ref: Dict) -> Any:
    pairs = [(new["exact"], ref["exact"])] + [(new["exact:" + f]["value"], ref["exact:" + f]["value"]) for f in e61.FIELDS]
    if any(a is None or b is None for a, b in pairs):
        return "incomplete"
    return bool(all(a >= b - 0.03 for a, b in pairs))


def gates(res: Dict) -> Dict:
    g: Dict[str, Any] = {}
    for gate, s in (("old_layout", "rl_tune"), ("exp61_layout", "tune_V")):
        x = res.get(s) or {}
        g[gate] = no_worse(x["new"], x["c92"]) if x.get("new") and x.get("c92") else "incomplete"
    dv = res.get("dev24")
    g["dev24_speed"] = "incomplete" if not dv else bool(dv["new"]["speed"] >= 18)
    g["dev24_distance"] = "incomplete" if not dv else bool(dv["new"]["distance"] >= 20)
    return g


def branch(g: Dict, valid: Optional[Dict] = None) -> str:
    if valid is not None and valid["state"] != "valid":
        return ("%s: the run's required evidence is %s (%s) -- no verdict"
                % (valid["state"], valid["state"], "; ".join(valid["invalid"] or ["missing %s" % valid["missing"]])))
    missing = [k for k, v in g.items() if v == "incomplete"]
    if missing:
        return "incomplete: %s -- missing is not a low score" % missing
    if g["old_layout"] is False:
        return "old-layout regression beyond 0.03 against c92 + the same head: this selector is not usable"
    if g["exp61_layout"] is False:
        return "EXP61-layout (tune_V) regression beyond 0.03 against c92 + the same head: this selector is not usable"
    if g["dev24_distance"] is False:
        return "the real-picture distance protection fails (< 20 / 24): this selector is not usable"
    if g["dev24_speed"] is False:
        return "generated layouts hold, real speed stays below 18 / 24: this recipe does not close the real-picture gap (only this recipe)"
    return ("the selector recipe passes with EXP64's head: next (separate permission) the live reading + decision combination; "
            "dev24 / the dev slices are repeatedly used development diagnostics, not a confirmation; not a candidate")


def tune_c_boxes(out: str, new_sha: str) -> Dict:
    """Descriptive: on tune_C, the share of auto boxes holding the drawn value (its text box), per source and field."""
    qs = {(q["picture"], q["text"]): q for q in jsonl(Q66) if q["set"] == "tune_C"}
    res: Dict[str, Dict[str, float]] = {}
    for who in ("new", "c92"):
        acc: Dict[str, List[float]] = defaultdict(list)
        for x in read_items(out, "tune_C_" + who, new_sha):
            for r in x["reads"]:
                q = qs[(x["picture"], r["question"])]
                bx, by, bw, bh = r["box"]
                t = q["text_box"]
                acc[q["field"]].append(float(bx <= t[0] + 1 and by <= t[1] + 1 and bx + bw >= t[2] - 1 and by + bh >= t[3] - 1))
        res[who] = {f: round(sum(v) / len(v), 4) for f, v in sorted(acc.items())}
    return res


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        check_start(out, data_root="", code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    new_sha = None
    try:
        new_sha = new_selector_sha(out)
        res["train"] = json.load(open(os.path.join(out, "train.json")))
        res["states"]["train"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["train"] = "missing / invalid: %s" % e
    sets = question_sets()
    recs: Dict[str, List[Dict]] = {}
    for group in ("rl_tune", "tune_V", "tune_C"):
        for n in GROUPS[group]:
            try:
                if new_sha is None:
                    raise SystemExit("no verified trained selector")
                recs[n] = e61.slice_records(out, n, sets[group], KEYS[group], identity(out, n, new_sha))
                res["states"][n] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"][n] = "missing / invalid: %s" % e
        res[group] = {n.rsplit("_", 1)[1]: e61.stats(recs[n]) for n in GROUPS[group] if n in recs}
        if group + "_new" in recs and group + "_c92" in recs:
            res[group]["new_minus_c92"] = {f: e61.paired(recs[group + "_new"], recs[group + "_c92"], None if f == "all" else f)
                                           for f in ("all",) + e61.FIELDS}
    try:
        hn, hc = e64.dev24_hits(out, "dev24_new", HEAD_SHA), e64.dev24_hits(out, "dev24_c92", HEAD_SHA)
        fld = e60.field_of
        cnt = lambda hs: {f: sum(v for k in hs for t, v in hs[k].items() if fld(t) == f) for f in e61.FIELDS}  # noqa: E731
        res["dev24"] = {"new": cnt(hn), "c92": cnt(hc), "speed_new_minus_c92": e60.speed_difference(hc, hn)}
        res["states"]["dev24"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["states"]["dev24"] = "missing / invalid: %s" % e
    if new_sha is not None and "tune_C_new" in recs and "tune_C_c92" in recs:
        try:
            res["tune_C_boxes_hold"] = tune_c_boxes(out, new_sha)
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["tune_C_boxes_hold"] = "invalid: %s" % e
    res["validity"] = validity(out, new_sha)
    res["gates"] = gates(res)
    res["branch"] = branch(res["gates"], res["validity"])
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "validity", "gates", "branch", "candidate_eligible") if k in res}, indent=1, default=str)
    for s in ("rl_tune", "tune_V", "tune_C"):
        for who, v in (res.get(s) or {}).items():
            if isinstance(v, dict) and "exact" in v:
                text += "\n%s %s: exact %s %s" % (s, who, v["exact"] and round(v["exact"], 4),
                                                  {f: v["exact:" + f]["value"] and round(v["exact:" + f]["value"], 3) for f in e61.FIELDS})
    if "dev24" in res:
        text += "\ndev24 new %s c92 %s" % (res["dev24"]["new"], res["dev24"]["c92"])
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "global", "train", "reads", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1))
    elif a.stage == "global":
        print(json.dumps(stage_global(), indent=1))
    elif a.stage == "train":
        print(json.dumps(stage_train(), indent=1))
    elif a.stage == "reads":
        print(json.dumps(stage_reads(), indent=1))
    else:
        try:
            report(finalize=a.finalize)
        except Exception as e:
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write("REPORT FAILED (invalid, no verdict): %s\n" % e)
            print("REPORT FAILED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
