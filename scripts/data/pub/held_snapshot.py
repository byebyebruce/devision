"""Write held_snapshot.json.gz (next to this file, committed): what the held-out check needs, without the project's
data -- the dHash of every held-out photo and the pixel hash of every held-out diagram (common.HeldOut plus the sets
in held.EXTRA). Hashes only, no pictures or questions. Regenerate on the machine with the data when held-out sets
change:

    uv run python scripts/data/pub/held_snapshot.py --root data

The public sources carry no COCO / Visual Genome ids, so the picture checks (photos within NEAR_BITS, identical
diagrams) are the whole held-out test for them; the snapshot is enough for that.
"""
import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from held import PubHeld, SNAPSHOT  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    a = p.parse_args()
    h = PubHeld(a.root)
    snap = {"built": time.strftime("%Y-%m-%d %H:%M"), "missing_here": h.missing,
            "photo_dhash": sorted(str(v) for v in h.photo_hashes().values()),
            "diagram_pixels": sorted(h.diagram_pixels())}
    with gzip.open(SNAPSHOT, "wt") as f:
        json.dump(snap, f)
    print("%s: %d photo hashes, %d diagram hashes, %.1f MB" % (SNAPSHOT, len(snap["photo_dhash"]),
          len(snap["diagram_pixels"]), os.path.getsize(SNAPSHOT) / 1e6))


if __name__ == "__main__":
    main()
