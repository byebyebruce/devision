"""Round 13: one list of every picture the v12 model's training lineage saw (plan
docs/experiments/2026-10-08-r13-binding-plan.md, 1.1), so dev / test sets can be built only from pictures the
project never trained, aligned or calibrated on.

    uv run python scripts/data/r13/identity.py --root data

Reads the configs of rounds v1-release-0.1 ... v12-ext (the run's own `run.json` when it is there, else
configs/<round>.yaml) and collects every file a training stage used: stage `data` (training / alignment),
`val` (monitoring and temperature fit) and `eval` (periodic monitoring). Each file's `image_id`s are mapped to one
canonical picture identity:

- a Visual Genome id of a COCO picture becomes the COCO id (Visual Genome's image_data.json, fetch.vg_to_coco;
  the same mapping as data/v5/image_identity.json), so "vg:2384383" and "coco:450707" are one picture;
- derived pictures inherit their source: "coco:128691:flip" (a mirrored copy) is "coco:128691";
- every other namespace keeps its own id ("objects365:21141", "img:<hash>", ...).

Some history pictures carry no traceable source id but may be COCO / Visual Genome photos (TallyQA and
Vision-Flan rows from the Cauldron, A-OKVQA's hashed COCO 2017 files, PixMo web photos). Their 64-bit dHashes
are stored too, so a candidate picture within 2 bits of one of them (`near_seen`) also counts as seen (the
round-9 audit found 0-1 bits for the same photo, 2 mixed, 4 different photos).

Writes data/r13/identity.json (canonical id -> {sources, seen_in_training}), data/r13/identity_hashes.json and
data/r13/identity_report.md. Limitation: the files are read as they are now, not as snapshots taken when each
round trained; what SigLIP2 / Laya / ModernBERT saw in pretraining is not traceable and not claimed.

Use from other scripts:

    import identity
    ident = identity.load("data")
    ident.canonical("vg:2384383")   # -> "coco:450707"
    ident.seen("vg:2384383")        # -> True when any id of that picture is in the project's training history
    ident.near_seen(dhash_value)    # -> True when the photo is within 2 bits of an untraceable history photo
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DATA_SCRIPTS))
sys.path.insert(0, DATA_SCRIPTS)

# v12-ext's initialisation lineage, oldest first. v1 is not an ancestor of v12 (v3 starts from v2's alignment
# stage) but its files are included anyway: being too strict only shrinks the unseen pool.
ROUNDS = ["v1-release-0.1", "v2-align-lora", "v3-data2", "v5-data3", "v6-flip-lr", "v7-sqa", "v8-vsr-v7w",
          "v9b-lr-replay", "v11-v9b-recovery", "v12-ext"]
NEAR_BITS = 2
# namespaces whose pictures have no traceable COCO / Visual Genome id but may be such photos
UNTRACED = ("tallyqa", "img", "vision_flan", "pixmo_count")
DIAGRAMS = ("scienceqa", "ai2d", "tqa")       # not photos: a near dHash is a shared template, not the same picture

OUT_DIR = "r13"


# ---------------------------------------------------------------- identity rules (pure)

def canonical(image_id: str, vg2coco: Optional[Dict[str, str]] = None) -> str:
    """One name per picture. With no mapping given, the default list's mapping is used (load())."""
    if vg2coco is None:
        return _default().canonical(image_id)
    parts = image_id.split(":")
    if parts[0] in ("coco", "vg") and len(parts) > 2:      # "coco:128691:flip" -> "coco:128691"
        image_id = ":".join(parts[:2])
    return vg2coco.get(image_id, image_id)


def seen(image_id: str) -> bool:
    """True when the picture is in the project's training history (default list, see load())."""
    return _default().seen(image_id)


def stage_files(config: dict) -> List[dict]:
    """[{"path", "role", "stage"}] for every file a training stage of `config` read."""
    out = []
    for st in config.get("stages") or []:
        kind = st.get("kind", "train")
        if st.get("data"):
            out.append({"path": st["data"], "role": "align" if kind == "align" else "train", "stage": st["name"]})
        if st.get("val"):
            out.append({"path": st["val"], "role": "val (monitoring, temperature fit)" if kind == "train"
                        else "val (alignment monitoring)", "stage": st["name"]})
        ev = st.get("eval") or {}
        for v in (ev.values() if isinstance(ev, dict) else ev):
            out.append({"path": v, "role": "eval during training (monitoring)", "stage": st["name"]})
    return out


def init_chain(configs: Dict[str, dict], last: str) -> List[str]:
    """Round names from `last` back through each first stage's `init` ("runs/<round>/<stage>")."""
    chain, name = [], last
    while name in configs and name not in chain:
        chain.append(name)
        stages = configs[name].get("stages") or []
        init = next((s.get("init") for s in stages if s.get("init") and str(s["init"]).startswith("runs/")), None)
        name = init.split("/")[1] if init else None
    return chain


class Identity:
    def __init__(self, pictures: Dict[str, dict], vg2coco: Dict[str, str], hashes: Iterable[int] = ()):
        self.pictures = pictures
        self.vg2coco = vg2coco
        self._hashes = sorted(set(int(h) for h in hashes))
        self._arr = None

    def canonical(self, image_id: str) -> str:
        return canonical(image_id, self.vg2coco)

    def seen(self, image_id: str) -> bool:
        return self.canonical(image_id) in self.pictures

    def sources(self, image_id: str) -> List[str]:
        return list((self.pictures.get(self.canonical(image_id)) or {}).get("sources", []))

    def near_seen(self, dhash_value: int, bits: int = NEAR_BITS) -> bool:
        if not self._hashes:
            return False
        import numpy as np
        if self._arr is None:
            self._arr = np.array(self._hashes, dtype=np.uint64)
        return int(np.bitwise_count(self._arr ^ np.uint64(int(dhash_value))).min()) <= bits


_DEFAULT: Dict[str, Identity] = {}


def load(root: str = "data") -> Identity:
    """The identity list built by this script (data/r13/identity.json + identity_hashes.json)."""
    path = os.path.join(root, OUT_DIR, "identity.json")
    if not os.path.exists(path):
        raise FileNotFoundError("%s missing: run scripts/data/r13/identity.py --root %s" % (path, root))
    d = json.load(open(path))
    from fetch import vg_to_coco
    hp = os.path.join(root, OUT_DIR, "identity_hashes.json")
    hashes = [int(v) for v in json.load(open(hp))["hashes"].values()] if os.path.exists(hp) else []
    ident = Identity(d["pictures"], vg_to_coco(root), hashes)
    _DEFAULT.setdefault("default", ident)
    return ident


def _default() -> Identity:
    return _DEFAULT.get("default") or load()


def dhash_file(path: str) -> Optional[int]:
    from PIL import Image
    from lv_bench import dhash
    try:
        with Image.open(path) as im:
            return dhash(im)
    except OSError:
        return None


def dhash_files(root: str, rels: Sequence[str], workers: int = 8) -> Dict[str, int]:
    from concurrent.futures import ProcessPoolExecutor
    rels = sorted(set(rels))
    with ProcessPoolExecutor(workers) as ex:
        values = list(ex.map(dhash_file, [os.path.join(root, r) for r in rels], chunksize=64))
    return {r: v for r, v in zip(rels, values) if v is not None}


# ---------------------------------------------------------------- build

def read_config(name: str, root_repo: str) -> dict:
    run = os.path.join(root_repo, "runs", name, "run.json")
    if os.path.exists(run):
        return json.load(open(run))["config"]
    import yaml
    return yaml.safe_load(open(os.path.join(root_repo, "configs", name + ".yaml")))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--repo", default=REPO, help="repository root (configs/, runs/)")
    a = p.parse_args(argv)
    from fetch import vg_to_coco
    v2c = vg_to_coco(a.root)
    configs = {r: read_config(r, a.repo) for r in ROUNDS}
    config_src = {r: ("runs/%s/run.json" % r if os.path.exists(os.path.join(a.repo, "runs", r, "run.json"))
                      else "configs/%s.yaml" % r) for r in ROUNDS}
    files: Dict[str, dict] = {}
    for r in ROUNDS:
        for f in stage_files(configs[r]):
            rel = f["path"][len("data/"):] if f["path"].startswith("data/") else f["path"]
            e = files.setdefault(rel, {"roles": set(), "uses": []})
            e["roles"].add(f["role"])
            e["uses"].append("%s/%s" % (r, f["stage"]))
    pictures: Dict[str, dict] = {}
    by_ns: Counter = Counter()
    untraced: set = set()
    gqa_qids: set = set()
    for rel in sorted(files):
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            raise SystemExit("history file missing: %s" % path)
        lines, ids = 0, set()
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                lines += 1
                s = json.loads(line)
                ids.add(s["image_id"])
                ns = s["image_id"].split(":")[0]
                img = s.get("image", "")
                if ns in UNTRACED and img and not any(d in img for d in DIAGRAMS):
                    untraced.add(img)
                if str(s.get("source", "")).startswith("gqa") and s["id"].startswith("gqa:"):
                    gqa_qids.add(s["id"].split(":", 1)[1])
        canon = {canonical(i, v2c) for i in ids}
        for c in canon:
            pictures.setdefault(c, {"sources": [], "seen_in_training": True})["sources"].append(rel)
        files[rel].update(lines=lines, image_ids=len(ids), pictures=len(canon))
    for c in pictures:
        by_ns[c.split(":")[0]] += 1
    print("history: %d files, %d pictures; hashing %d untraced photos" % (len(files), len(pictures), len(untraced)),
          flush=True)
    t0 = time.time()
    hashes = dhash_files(a.root, sorted(untraced))
    out = os.path.join(a.root, OUT_DIR)
    os.makedirs(out, exist_ok=True)
    meta = {"built": time.strftime("%Y-%m-%d %H:%M"), "rounds": ROUNDS, "config_sources": config_src,
            "init_chain_from_v12": list(reversed(init_chain(configs, "v12-ext"))),
            "files": {k: {"roles": sorted(v["roles"]), "uses": v["uses"], "lines": v["lines"],
                          "image_ids": v["image_ids"], "pictures": v["pictures"]} for k, v in sorted(files.items())},
            "pictures_by_namespace": dict(by_ns.most_common()), "gqa_question_ids": sorted(gqa_qids),
            "vg_to_coco": "raw/vg/image_data.json.zip (fetch.vg_to_coco)", "near_bits": NEAR_BITS,
            "untraced_photos": len(untraced), "untraced_hashed": len(hashes),
            "limits": ["history files are read as they are now under data/, not as snapshots from when each round "
                       "trained", "external pretraining (SigLIP2, Laya, ModernBERT) is not traceable and not claimed",
                       "pictures without a source id (%s) are matched only by dHash <= %d bits"
                       % (", ".join(UNTRACED), NEAR_BITS)]}
    json.dump({"meta": meta, "pictures": pictures}, open(os.path.join(out, "identity.json"), "w"))
    json.dump({"near_bits": NEAR_BITS, "hashes": {k: str(v) for k, v in sorted(hashes.items())}},
              open(os.path.join(out, "identity_hashes.json"), "w"))
    write_report(os.path.join(out, "identity_report.md"), meta, len(pictures), time.time() - t0)
    print("wrote %s/identity.json (%d pictures), identity_hashes.json (%d), identity_report.md"
          % (out, len(pictures), len(hashes)))


def write_report(path: str, meta: dict, n_pictures: int, hash_seconds: float) -> None:
    L = ["# Round 13 picture identity list", "",
         "Built %s by `scripts/data/r13/identity.py` (plan 1.1). %d canonical pictures in the project's training "
         "history (`data/r13/identity.json`)." % (meta["built"], n_pictures), "",
         "Init chain of v12-ext (oldest first, from each round's first stage `init`): %s. Rounds read: %s "
         "(v1-release-0.1 is not an ancestor of v12; its files are included to be strict)."
         % (" -> ".join(meta["init_chain_from_v12"]), ", ".join(meta["rounds"])), "",
         "## History files", "", "| file | lines | image ids | canonical pictures | roles | used by |", "|---|---|---|---|---|---|"]
    for k, v in meta["files"].items():
        L.append("| `data/%s` | %d | %d | %d | %s | %s |" % (k, v["lines"], v["image_ids"], v["pictures"],
                                                         "; ".join(v["roles"]), ", ".join(v["uses"])))
    L += ["", "## Canonical pictures by namespace", "", "| namespace | pictures |", "|---|---|"]
    L += ["| %s | %d |" % kv for kv in meta["pictures_by_namespace"].items()]
    L += ["", "GQA question ids appearing in these files: %d (used by `gqa_bind.py` to report truly new questions)."
          % len(meta["gqa_question_ids"]), "",
          "Untraced photos (namespaces %s, diagrams excluded): %d, of which %d hashed (dHash, %.0f s). A candidate "
          "within %d bits of one counts as seen (`Identity.near_seen`)."
          % (", ".join(UNTRACED), meta["untraced_photos"], meta["untraced_hashed"], hash_seconds, meta["near_bits"]),
          "", "## Rules", "",
          "- Visual Genome ids of COCO pictures map to the COCO id (`raw/vg/image_data.json.zip`); derived ids "
          "(`coco:N:flip`) inherit the source picture; other namespaces keep their own id.",
          "- Roles: stage `data` (training, alignment), `val` (monitoring and temperature fit), `eval` (periodic "
          "monitoring during training). Post-training evaluation sets are not history.",
          "- Configs come from each round's `runs/<round>/run.json` (the configuration actually run) when present, "
          "else `configs/<round>.yaml`: %s." % ", ".join("%s: %s" % kv for kv in meta["config_sources"].items()),
          "", "## Limitations", ""]
    L += ["- " + x for x in meta["limits"]]
    L += ["- Edited / pasted pictures of later rounds (VisMin, CounterCurate, pasted backgrounds) must map their "
          "source id themselves before calling `seen()`; this list only knows ids that appear in the files."]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
