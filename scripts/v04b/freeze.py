"""Candidate B endpoint identity (critic V04-B-R2-01). After training and before the first screening evaluation the
chain freezes the final weights and config (the config holds the fitted temperatures): `freeze` writes
<run>/FROZEN.json with their sha256. After each completed screening evaluation `record NAME` adds that summary's and
details' sha256, checking first that the weights / config are still the frozen ones. `check` (used by the report)
verifies the weights, the config and every recorded file are unchanged; any difference is refused.

    uv run python scripts/v04b/freeze.py freeze runs/v04b-fit
    uv run python scripts/v04b/freeze.py record runs/v04b-fit sqa_temp
    uv run python scripts/v04b/freeze.py start runs/exp39-report.json runs/exp39-h/stage1/model.safetensors <sha>
"""
import hashlib
import json
import os
import sys


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weights(run: str) -> dict:
    ck = os.path.join(run, "stage2")
    return {"model.safetensors": sha(os.path.join(ck, "model.safetensors")), "config.json": sha(os.path.join(ck, "config.json"))}


def path(run: str) -> str:
    return os.path.join(run, "FROZEN.json")


def freeze(run: str) -> dict:
    if os.path.exists(path(run)):
        raise SystemExit("%s exists" % path(run))
    out = {"weights": weights(run), "evals": {}}
    json.dump(out, open(path(run), "w"), indent=1)
    return out


def check(run: str, names=None) -> dict:
    """The frozen identity, after checking that the weights / config and the recorded files (or `names` of them,
    which must all be recorded) are unchanged."""
    if not os.path.exists(path(run)):
        raise SystemExit("no frozen identity in %s" % run)
    fr = json.load(open(path(run)))
    if weights(run) != fr["weights"]:
        raise SystemExit("the weights or config of %s changed after the freeze" % run)
    for n in (names if names is not None else fr["evals"]):
        if n not in fr["evals"]:
            raise SystemExit("%s was not recorded under the frozen identity" % n)
        for f, h in fr["evals"][n].items():
            if not os.path.exists(os.path.join(run, "eval", f)) or sha(os.path.join(run, "eval", f)) != h:
                raise SystemExit("%s changed after it was recorded" % f)
    return fr


def record(run: str, name: str) -> dict:
    fr = check(run)
    files = [name + ".json", name + ".details.jsonl"]
    fr["evals"][name] = {f: sha(os.path.join(run, "eval", f)) for f in files}
    json.dump(fr, open(path(run), "w"), indent=1)
    return fr


def start_allowed(report_path: str, start_file: str, given: str) -> str:
    """'' when exp39's report passes its four conditions for the very weights B starts from: the report's H
    model sha, the actual start file's sha and the sha given to the chain all agree; else the reason."""
    r = json.load(open(report_path))
    h = (r.get("identity") or {}).get("H", {}).get("model.safetensors")
    actual = sha(start_file) if os.path.exists(start_file) else None
    if r.get("invest_in_512") is not True:
        return "exp39 did not meet its four conditions"
    if not (h == actual == given):
        return "start weights disagree: report %s, file %s, given %s" % (h, actual, given)
    return ""


if __name__ == "__main__":
    cmd, run = sys.argv[1], sys.argv[2]
    if cmd == "start":
        why = start_allowed(run, sys.argv[3], sys.argv[4])
        print(why or "start allowed")
        sys.exit(1 if why else 0)
    if cmd == "freeze":
        print(json.dumps(freeze(run)))
    elif cmd == "record":
        record(run, sys.argv[3])
    else:
        check(run)
