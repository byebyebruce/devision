"""exp38 (critic EXP38-PREP-01 item 1): the same checkpoint read at another input size. Copies a devision
checkpoint and changes only its config -- image_size (e.g. 512) and interpolate_pos_encoding (SigLIP's interpolated
position embeddings; the tower keeps its own 256 table) -- every weight file byte-identical. Refuses to overwrite.

    uv run python scripts/exp38/convert.py runs/exp37-l/stage1 runs/exp38-init-512 --image-size 512
"""
import argparse
import hashlib
import json
import os
import shutil
import sys


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert(src: str, dst: str, image_size: int) -> dict:
    if os.path.exists(dst):
        raise SystemExit("%s exists" % dst)
    shutil.copytree(src, dst)
    cfg_path = os.path.join(dst, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["image_size"] = image_size
    cfg["interpolate_pos_encoding"] = image_size != 256
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    out = {"source": src, "image_size": image_size, "interpolate_pos_encoding": cfg["interpolate_pos_encoding"],
           "files_identical": {}, "config_sha256": sha(cfg_path)}
    for base, _, files in os.walk(src):
        for f in files:
            rel = os.path.relpath(os.path.join(base, f), src)
            if rel != "config.json":
                out["files_identical"][rel] = sha(os.path.join(src, rel)) == sha(os.path.join(dst, rel))
    if not all(out["files_identical"].values()):
        raise SystemExit("a weight file changed")
    out["model_sha256"] = sha(os.path.join(dst, "model.safetensors"))
    json.dump(out, open(os.path.join(dst, "CONVERTED.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--image-size", type=int, default=512)
    a = ap.parse_args()
    r = convert(a.src, a.dst, a.image_size)
    print(json.dumps({k: v for k, v in r.items() if k != "files_identical"}, indent=1), file=sys.stdout)
