"""Fetch the official eight public examples, with pinned source and asset hashes.

These are interface smoke cases, not the public 300 and not an official score.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from chart_data import write_rows


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def main():
    repo = "fstandhartinger/model-market-comparison"
    rev = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/main"))["sha"]
    base = f"https://raw.githubusercontent.com/{repo}/{rev}/"
    out = Path("data/v05/external")
    (out / "images").mkdir(parents=True, exist_ok=True)
    if (out / "jev_examples.MANIFEST.json").exists():
        raise SystemExit("Frozen examples already exist")
    manifest = json.loads(fetch(base + "public/image-jev/examples/manifest.json"))
    code = fetch(base + "lib/image-jev-public-examples.mjs").decode()
    # Evaluate a pinned data module in an isolated JS context, without Node I/O.
    js = 'const vm=require("node:vm");let s="";process.stdin.on("data",x=>s+=x);process.stdin.on("end",()=>console.log(vm.runInNewContext(s.replaceAll("export ","")+";JSON.stringify(PUBLIC_IMAGE_JEV_EXAMPLES)",{}, {timeout:1000})));'
    examples = json.loads(subprocess.check_output(["node", "-e", js], input=code, text=True, timeout=10))
    by_asset = {r["asset"]: r for r in manifest["examples"]}
    assert len(examples) == manifest["count"] == 8
    rows = []
    for ex in examples:
        meta = by_asset[ex["image"]]
        raw = fetch(base + "public" + ex["image"])
        if hashlib.sha256(raw).hexdigest() != meta["asset_sha256"]:
            raise ValueError("Official image hash mismatch: " + ex["key"])
        path = out / "images" / (ex["key"] + ".webp")
        path.write_bytes(raw)
        labels = {r["label"]: r["text"] for r in ex["options"]}
        assert len(set(labels.values())) == len(labels)
        q = {"type": "choice", "instructions": ex["question"], "criteria": dict.fromkeys(labels.values())}
        gold = {text: float(label == ex["correctLabel"]) for label, text in labels.items()}
        rows.append({"id": "jev-example:" + ex["key"], "source": "imagejev-public-examples",
                     "kind": ex["category"], "image_id": "jev-example:" + ex["key"],
                     "image": str(path.relative_to("data")), "questions": {"q": q}, "gold": {"q": {"probabilities": gold}},
                     "license": ex["license"], "official_item": ex["sourceItemId"]})
    result = write_rows(out / "jev_examples.jsonl", rows)
    result.update(repo=repo, revision=rev, official_manifest=manifest,
                  protocol="Eight public examples, choice interface smoke only; no official Capability score")
    (out / "jev_examples.MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "official_manifest"}, indent=2))


if __name__ == "__main__":
    main()
