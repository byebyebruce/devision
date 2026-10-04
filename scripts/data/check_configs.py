"""Check that every data file the given training configs use exists and is not empty.

    uv run python scripts/data/check_configs.py --root data configs/scratch.yaml configs/v6-flip-lr.yaml ...

Reads each YAML's stages (data / val / eval) and evaluate section (sets, calibration_fit, history,
image_identity); paths under data/ are looked up under --root. Exits non-zero listing what is missing.
"""
import argparse
import os
import sys
from typing import List

import yaml


def data_paths(cfg: dict) -> List[str]:
    out = []
    for st in cfg.get("stages") or []:
        out += [st.get("data"), st.get("val")] + list((st.get("eval") or {}).values())
    ev = cfg.get("evaluate") or {}
    for spec in (ev.get("sets") or {}).values():
        out.append(spec["path"] if isinstance(spec, dict) else spec)
    out += [ev.get("calibration_fit"), ev.get("image_identity")] + list(ev.get("history") or [])
    return [p for p in out if isinstance(p, str) and p.startswith("data/")]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("configs", nargs="+")
    a = p.parse_args(argv)
    bad = []
    for c in a.configs:
        with open(c) as f:
            for path in data_paths(yaml.safe_load(f)):
                local = os.path.join(a.root, path[len("data/"):])
                if not os.path.exists(local) or os.path.getsize(local) == 0:
                    bad.append("%s: %s" % (c, local))
    if bad:
        print("missing or empty:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("all data files of %d configs present" % len(a.configs))


if __name__ == "__main__":
    main()
