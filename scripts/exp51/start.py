"""exp51 START (critic EXP51-R1-02), model-free, before the first model forward: checks the vision identity (X / T / C),
the question / region files against data/exp51/MANIFEST.json and EXP50-T's model / config, then writes
runs/exp51/START.json -- the full git HEAD and whether the tree is dirty, the sha256 of every file the run executes or
imports for its result (this experiment's scripts, EXP49's readout, EXP41's interval, image / model / infer), the
model / config / vision digest, questions, regions, grid, seed / steps / devices, start and absolute deadline. Refused if
START.json exists (fresh only).

    uv run python scripts/exp51/start.py --deadline EPOCH
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(__file__)
OUT = "runs/exp51"
FILES = ["scripts/exp51/start.py", "scripts/exp51/regions.py", "scripts/exp51/extract.py", "scripts/exp51/probe.py",
         "scripts/exp51/vision_identity.py", "scripts/exp51/run.sh", "scripts/exp49/probe.py", "scripts/exp41/report.py",
         "src/devision/image.py", "src/devision/model.py", "src/devision/infer.py"]
QUESTIONS, REGIONS, MANIFEST = "data/exp49/questions.jsonl", "data/exp51/regions.json.gz", "data/exp51/MANIFEST.json"


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def start(deadline: int, out: str = OUT, vision=None) -> dict:
    sys.path.insert(0, HERE)
    import extract  # type: ignore
    import probe  # type: ignore
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    ident = vision if vision is not None else __import__("vision_identity").check()
    man = json.load(open(MANIFEST))
    if sha(QUESTIONS) != man["questions_sha256"] or sha(REGIONS) != man["regions_sha256"]:
        raise SystemExit("questions / regions are not the MANIFEST's")
    ck = os.path.join(extract.RUN, "stage2")
    weights = {f: sha(os.path.join(ck, f)) for f in extract.T_SHA}
    if weights != extract.T_SHA:
        raise SystemExit("EXP50-T is not 1a8e5439 / 99c6246e")
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, "model": {"run": extract.RUN, **weights}, "vision_identity": ident,
         "questions_sha256": man["questions_sha256"], "regions_sha256": man["regions_sha256"], "grid": man["grid"],
         "conditions": list(probe.CONDS), "head": {"seed": probe.p49.SEED, "steps": probe.p49.STEPS, "batch": probe.p49.BATCH,
                                                    "lr": probe.p49.LR, "weight_decay": probe.p49.WD, "device": "cpu"},
         "extract_device": "auto (mps)", "started": time.strftime("%F %T"), "deadline": deadline,
         "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    os.makedirs(out, exist_ok=True)
    json.dump(s, open(path, "w"), indent=1)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline", type=int, required=True)
    print(json.dumps(start(ap.parse_args().deadline), indent=1))
