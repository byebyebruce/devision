"""exp51: are the SigLIP towers of EXP48-X, EXP50-T and EXP50-C the same tensors (critic EXP51-PREP-01)? Reads only the
safetensors header and the raw bytes of every `vision.` tensor, one at a time (no torch, no model): per run, a sha256
over (name, dtype, shape, bytes) in name order, and the tensor count. Any difference: refused.

    uv run python scripts/exp51/vision_identity.py
"""
import hashlib
import json
import struct
import sys

RUNS = {"X": "runs/exp48-x/stage2/model.safetensors", "T": "runs/exp50-t/stage2/model.safetensors",
        "C": "runs/exp50-c/stage2/model.safetensors"}


def vision_digest(path: str, prefix: str = "vision.") -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        h = hashlib.sha256()
        names = sorted(k for k in header if k.startswith(prefix))
        size = 0
        for k in names:
            meta = header[k]
            a, b = meta["data_offsets"]
            f.seek(base + a)
            h.update(json.dumps([k, meta["dtype"], meta["shape"]]).encode())
            h.update(f.read(b - a))
            size += b - a
    return {"tensors": len(names), "bytes": size, "sha256": h.hexdigest()}


def check(runs: dict = RUNS) -> dict:
    d = {k: vision_digest(p) for k, p in runs.items()}
    if len({v["sha256"] for v in d.values()}) != 1 or not d["X"]["tensors"]:
        raise SystemExit("vision towers differ: %s" % d)
    return d


if __name__ == "__main__":
    print(json.dumps(check(), indent=1))
    sys.exit(0)
