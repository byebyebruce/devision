"""exp62: the selector with paired parent / crop supervision (critic EXP62-PREP-01; docs/experiments/2026-10-19-exp62-paired-plan.md).
One new selector from EXP53's end (8f10f1fd; not from EXP61's c98becfc), EXP53's recipe (L1 + GIoU, AdamW 1e-3 / 0.01,
warm-up 100, cosine, seed 0, 2,000 updates, last step), batches from the fixed plan data/exp62/plan.jsonl: 24 parent /
crop pairs + 16 fit_O / fit_L replay rows per update. Vision, character head, decision and the reading rule frozen;
everything after training is EXP61's (same reads, statistics, gates and branch, through scripts/exp61/exp61.py).

  start   model-free: EXP61's identities (EXP53, EXP60, EXP58 / EXP57, EXP56, data/exp61, exp49) + EXP61's run (START /
          registry / report / its variant features) + data/exp62 (MANIFEST, parents, plan, parent pictures).
  global  MPS: EXP57's frozen vision patches of the 400 parent pictures (the crops' patches are EXP61's registered ones;
          a parent question uses its crop twin's question words: the same text).
  train   CPU: the plan, the new selector, its boxes (old / new) on every selector set, a new reading bundle.
  reads   MPS: EXP61's reads stage unchanged (rl_tune new, tune_V old / new / oracle, EXP60's 24 through Reading.decide).
  report  model-free: EXP61's report and pre-written gates / branch, plus (descriptive, not gates) parent box validity,
          new - EXP61 per field on rl_tune / tune_V (case clusters), and where EXP61's 38 mass regressions went.

    uv run python scripts/exp62/exp62.py start --deadline EPOCH | global | train | reads | report [--finalize CODES]
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List

HERE = os.path.dirname(__file__)
OUT, E61 = "runs/exp62", "runs/exp61"
E61_SHA = {"START.json": "909269d0785e9e3850f2d9131c710fd814b5e607950b97ff55e21de61c9842d3",
           "registry.json": "1014fa6cb016f15e14aae04f86d1088b58fe40bf69c9b3843a6717c0f6d22cfb",
           "report.json": "ddfa5594b67a04afe92e330a19fb0ae4427bde5978c68edc4eee44246c07253d"}
P_MANIFEST, PARENTS, PLAN = "data/exp62/MANIFEST.json", "data/exp62/parents.jsonl", "data/exp62/plan.jsonl"
FILES = ["scripts/exp62/exp62.py", "scripts/exp62/build.py", "scripts/exp62/run.sh", "scripts/exp61/exp61.py", "scripts/exp61/build.py",
         "scripts/exp53/selector.py", "scripts/exp60/exp60.py", "scripts/exp58/live.py", "src/devision/reading.py",
         "src/devision/infer.py", "src/devision/model.py", "src/devision/image.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e61 = _load("exp61", "../exp61/exp61.py")
sel, e54, e58, e59, e60 = e61.sel, e61.e54, e61.e58, e61.e59, e61.e60
sha, write_json, register, jsonl = e61.sha, e61.write_json, e61.register, e61.jsonl
E61_CHECK_START = e61.check_start                                   # EXP61's own check (report swaps the module name)


# ------------------------------------------------------------------ identities

def bound_e61() -> Dict[str, str]:
    bad = [f for f, h in E61_SHA.items() if sha(os.path.join(E61, f)) != h]
    if bad:
        raise SystemExit("EXP61's %s changed" % bad)
    for files in json.load(open(os.path.join(E61, "registry.json"))).values():
        for f, h in files.items():
            if sha(os.path.join(E61, f)) != h:
                raise SystemExit("EXP61's registered %s changed" % f)
    return dict(E61_SHA)


def check_parents(root: str = "data") -> Dict[str, str]:
    """data/exp62 against its MANIFEST (lists always; parent picture bytes with root)."""
    m = json.load(open(P_MANIFEST))
    if sha(PARENTS) != m["parents_sha256"] or sha(PLAN) != m["plan_sha256"]:
        raise SystemExit("data/exp62 parents / plan differ from their MANIFEST")
    if m["exp61"]["questions"] != sha(e61.V_QUESTIONS):
        raise SystemExit("data/exp62 was built on another data/exp61")
    if root:
        pics = sorted({q["picture"] for q in jsonl(PARENTS)})
        import hashlib
        if hashlib.sha256(json.dumps({p: sha(os.path.join(root, p)) for p in pics}, sort_keys=True).encode()).hexdigest() != m["pictures_sha256"]:
            raise SystemExit("parent pictures differ from the MANIFEST")
    return {"manifest": sha(P_MANIFEST), "parents": m["parents_sha256"], "plan": m["plan_sha256"], "pictures": m["pictures_sha256"]}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    st58 = e58.check_run(e59.E58, data_root=data_root, code=False)
    if st58["bundle"]["selector.pt"] != e61.SELECTOR_SHA:
        raise SystemExit("EXP58's bundle does not hold EXP53's selector")
    ident = {"exp53": e61.bound_e53(), "exp60": e61.bound_e60(), "variants": e61.check_variants(data_root), "exp49_questions": sha(e61.QUESTIONS),
             "exp56_reads": e54.check_registered(e61.E56, "reads")["reads.jsonl"], "model": st58["model"], "bundle": st58["bundle"],
             "exp61": bound_e61(), "parents": check_parents(data_root)}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    m = json.load(open(P_MANIFEST))
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, **ident,
         "recipe": {"init": "runs/exp53/selector.pt", "steps": len(jsonl(PLAN)), "batch": "24 parent / crop pairs + 16 replay (data/exp62/plan.jsonl)",
                    "lr": sel.LR, "weight_decay": sel.WD, "warmup": sel.WARMUP, "seed": sel.SEED, "loss": "L1(cx, cy, w, h) + (1 - GIoU)",
                    "device": "cpu", "records": m["records"], "exposures": m["plan"]},
         "requests": {"global_pictures": m["parents"]["pictures"], "reads": {"rl_tune_new": 600, "tune_V_old": 600, "tune_V_new": 600,
                                                                          "tune_V_oracle": 600, "dev24_new_decide": 24}},
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline,
         "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = E61_CHECK_START(out, data_root, code)                     # EXP61's identities, as EXP61 checks them
    if bound_e61() != st["exp61"] or check_parents(data_root) != st["parents"]:
        raise SystemExit("EXP61's run / data/exp62 differ from START")
    return st


# ------------------------------------------------------------------ global / train

def stage_global(out: str = OUT, data_root: str = "data") -> Dict:
    import numpy as np
    import torch
    from PIL import Image

    from devision.image import patch_geometry, to_pixel_values
    from devision.infer import Decider, best_device
    st = check_start(out, data_root)
    if e58.fz.weights(e58.NEW_RUN) != st["model"]:
        raise SystemExit("EXP57's weights are not START's: refused before loading")
    pics = sorted({q["picture"] for q in jsonl(PARENTS)})
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
    path = os.path.join(out, "parent_global.npz")
    np.savez(path, patches=patches, geom=geom, pictures=np.array(pics))
    register(out, "global", {"parent_global.npz": sha(path)})
    return {"pictures": len(pics)}


def all_inputs(out: str):
    """(Inputs, records, key -> index): exp49 + exp61 questions with their registered features (EXP53's, EXP61's), then
    the parents with this run's patches and their crop twin's words."""
    import numpy as np
    import torch
    e54.check_registered(E61, "global")
    inp, qs = e61.combined_inputs(E61)
    e54.check_registered(out, "global")
    z = np.load(os.path.join(out, "parent_global.npz"))
    pics = z["pictures"].tolist()
    par = jsonl(PARENTS)
    idx = {q["key"]: i for i, q in enumerate(qs)}
    n_pics = inp.patches.shape[0]
    patches = torch.cat([inp.patches, torch.from_numpy(z["patches"])])
    geom = torch.cat([inp.geom, torch.from_numpy(z["geom"])])
    crop_i = [idx[q["pair"]] for q in par]
    words = torch.cat([inp.words, inp.words[crop_i]])
    mask = torch.cat([inp.mask, inp.mask[crop_i]])
    pic_index = list(inp.pic) + [n_pics + pics.index(q["picture"]) for q in par]
    records = qs + par
    return sel.Inputs(patches, geom, pic_index, words, mask), records, {q["key"]: i for i, q in enumerate(records)}


def train_plan(init_state: Dict, inp, targets, batches: List[List[int]], on_step=None):
    """EXP53's optimisation (selector.train) with the batches taken from the fixed plan."""
    import torch
    torch.manual_seed(sel.SEED)
    model = sel.Selector(inp.patches.shape[-1], inp.words.shape[-1])
    model.load_state_dict(init_state, strict=True)
    steps = len(batches)
    opt = torch.optim.AdamW(model.parameters(), lr=sel.LR, weight_decay=sel.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / sel.WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    losses = []
    for step, idx in enumerate(batches):
        loss = sel.box_loss(model(*inp.batch(idx)), targets[idx])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite selector loss at step %d" % step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if on_step and (step + 1) % 50 == 0:
            on_step(step + 1)
    return model.eval(), {"steps": steps, "exposures": sum(map(len, batches)),
                          "final_loss_250": sum(losses[-250:]) / min(250, len(losses)) if losses else None}


def stage_train(out: str = OUT) -> Dict:
    import torch

    from devision.reading import write_bundle
    st = check_start(out, data_root="")
    inp, recs, at = all_inputs(out)
    targets = torch.tensor([sel.target_of(q["box"], q["width"], q["height"]) for q in recs])
    batches = [[at[k] for k in b] for b in jsonl(PLAN)]
    init = torch.load(os.path.join(e61.E53, "selector.pt"), map_location="cpu")
    t0 = time.time()
    model, stats = train_plan(init, inp, targets, batches, on_step=lambda k: e54.progress(out, "train", k, len(batches)))
    torch.save(model.state_dict(), os.path.join(out, "selector.pt"))
    old = sel.Selector(inp.patches.shape[-1], inp.words.shape[-1])
    old.load_state_dict(init, strict=True)
    with open(os.path.join(out, "boxes.jsonl"), "w") as fh:                  # EXP61's format (exp49 + exp61 questions) ...
        with open(os.path.join(out, "boxes_parents.jsonl"), "w") as fp:       # ... and the parents apart
            for who, mdl in (("old", old.eval()), ("new", model)):
                for q, f in zip(recs, sel.predict(mdl, inp, list(range(len(recs))))):
                    px, bad = sel.pixel_box(f, q["width"], q["height"])
                    (fp if q["set"] == "train_P" else fh).write(json.dumps({"selector": who, "key": q["key"], "set": q["set"], "box": px, "degenerate": bad}) + "\n")
    bundle = write_bundle(os.path.join(out, "bundle"), os.path.join(e58.NEW_RUN, "stage2"), os.path.join(out, "selector.pt"), e58.HEAD,
                          {"selector": {"path": os.path.join(out, "selector.pt"), "init": e61.SELECTOR_SHA, "recipe": st["recipe"]},
                           "head": {"path": e58.HEAD, "sha256": e58.HEAD_SHA}})
    info = {**stats, "train_records": len({k for b in batches for k in b}), "seconds": time.time() - t0,
            "selector_sha256": sha(os.path.join(out, "selector.pt")), "bundle_reading_json": sha(os.path.join(out, "bundle", "reading.json")),
            "bundle": bundle["sha256"], "plan_sha256": sha(PLAN)}
    write_json(os.path.join(out, "train.json"), info)
    register(out, "selector", {f: sha(os.path.join(out, f)) for f in ("selector.pt", "boxes.jsonl", "boxes_parents.jsonl", "train.json")})
    return info


def stage_reads(out: str = OUT, data_root: str = "data") -> Dict:
    check_start(out, data_root)
    return e61.stage_reads(out, data_root)


# ------------------------------------------------------------------ report

def exp61_records(name: str, qs: List[Dict], key) -> List[Dict]:
    st61 = json.load(open(os.path.join(E61, "START.json")))
    h61 = json.load(open(os.path.join(E61, "train.json")))["selector_sha256"]
    return e61.slice_records(E61, name, qs, key, e61.read_identity(E61, name, h61, st61))


def extras(out: str) -> Dict:
    """Descriptive only (not gates): parent box validity, new - EXP61 per field, the fate of EXP61's mass regressions."""
    res: Dict[str, Any] = {}
    par = {q["key"]: q for q in jsonl(PARENTS)}
    acc: Dict[str, List[float]] = {}
    ins: Dict[str, List[float]] = {}
    for b in jsonl(os.path.join(out, "boxes_parents.jsonl")):
        q = par[b["key"]]
        acc.setdefault(b["selector"], []).append(sel.iou_xywh(b["box"], q["box"]))
        x, y, w, h = b["box"]
        t = q["text_box"]
        ins.setdefault(b["selector"], []).append(float(x <= t[0] + 1 and y <= t[1] + 1 and x + w >= t[2] - 1 and y + h >= t[3] - 1))
    res["parent_boxes"] = {k: {"mean_iou": round(sum(v) / len(v), 4), "text_inside": round(sum(ins[k]) / len(ins[k]), 4)} for k, v in acc.items()}
    sets = e61.question_sets()
    h = e61.new_selector_sha(out)
    st = json.load(open(os.path.join(out, "START.json")))
    for slice_, name, key in (("rl_tune", "rl_tune_new", lambda x: x["id"]), ("tune_V", "tune_V_new", lambda x: x["picture"])):
        new = e61.slice_records(out, name, sets[slice_], key, e61.read_identity(out, name, h, st))
        old = exp61_records(name, sets[slice_], key)
        res["%s_minus_exp61" % slice_] = {f: e61.paired(new, old, None if f == "all" else f) for f in ("all",) + e61.FIELDS}
        if slice_ == "rl_tune":
            base = {q["key"]: q for q in e61.old_rl_tune(sets["rl_tune"])}
            o61 = {q["key"]: q for q in old}
            reg = [q for q in new if q["field"] == "mass" and base[q["key"]]["right"] and not o61[q["key"]]["right"]]
            res["exp61_mass_regressions"] = {"n": len(reg), "now_right": sum(q["right"] for q in reg)}
    return res


def report(out: str = OUT, finalize: str = "") -> Dict:
    import copy
    setattr(e61, "check_start", check_start)                                 # EXP61's report on this run's START checks
    try:
        res = e61.report(out, finalize)
    finally:
        setattr(e61, "check_start", E61_CHECK_START)
    if res.get("states", {}).get("train") == "completed" and all(v == "completed" for v in res["states"].values()):
        try:
            res["exp62"] = extras(out)
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["exp62"] = {"state": "invalid: %s" % e}
    res = copy.deepcopy(res)
    e61.write_report(out, res)
    if "exp62" in res:
        with open(os.path.join(out, "report.txt"), "a") as fh:
            fh.write("exp62 %s\n" % json.dumps(res["exp62"], default=str))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "global", "train", "reads", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files",)}, indent=1))
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
