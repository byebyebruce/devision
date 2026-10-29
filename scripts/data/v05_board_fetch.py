"""Resume-safe download of the TRAIN-split sources behind the Decision Index board-gap data (scripts/data/v05_board_*.py).

    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \\
        uv run --no-sync python scripts/data/v05_board_fetch.py --root data [--only cord infovqa hateful arxivqa]

Everything goes to <root>/raw/board/<name>/ through huggingface_hub.hf_hub_download (local_dir; partial files resume,
finished files are skipped). hf-mirror.com currently 308-redirects to huggingface.co; the owner accepted that for
datasets. Only train splits (plus the dev / test *id lists* of Hateful Memes, used to exclude those images) are fetched:

  cord      naver-clova-ix/cord-v2           data/train-*.parquet           (the board uses validation + test)
  infovqa   HuggingFaceM4/the_cauldron        infographic_vqa/train-*        (the board uses InfographicVQA validation)
  docvqa    HuggingFaceM4/the_cauldron        docvqa/train-*                 (document reading; no board benchmark uses DocVQA)
  hateful   neuralcatcher/hateful_memes       train.jsonl + img/<train ids>, dev/test jsonl (board: dev_seen + test_seen)
  arxivqa   MMInstruction/ArxivQA             arxivqa.jsonl + images.tgz (8 GB; CharXiv-like figure QA)

Progress (files done / total, GB on disk, elapsed) is appended to <root>/raw/board/fetch.log every file and every ~60 s.
Revisions are pinned below and written to <root>/raw/board/<name>/REVISION.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPOS = {
    "cord": ("naver-clova-ix/cord-v2", "7f0115a4b758a71d6473b8d085751692da2fef98", lambda f: f.startswith("data/train-")),
    "infovqa": ("HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605",
                lambda f: f.startswith("infographic_vqa/")),
    "docvqa": ("HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605",
               lambda f: f.startswith("docvqa/")),
    "hateful": ("neuralcatcher/hateful_memes", "d201c488dc7024623d1ecbcc987b3f132c4c2e12", None),   # special-cased
    "arxivqa": ("MMInstruction/ArxivQA", "85a6dca0e2bdc6f0268ae519be8913f83a83cafd",
                lambda f: f in ("arxivqa.jsonl", "images.tgz", "README.md")),
}
HATEFUL_META = ("train.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl", "test_seen.jsonl", "test_unseen.jsonl",
                "LICENSE.txt", "README.md")


def log(root: str, msg: str) -> None:
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(root, "raw", "board", "fetch.log"), "a") as f:
        f.write(line + "\n")


def du(path: str) -> float:
    total = 0
    for d, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return total / 1e9


def fetch(root: str, name: str, workers: int) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    repo, rev, keep = REPOS[name]
    out = os.path.join(root, "raw", "board", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "REVISION"), "w") as f:
        f.write("%s@%s via %s\n" % (repo, rev, os.environ.get("HF_ENDPOINT", "huggingface.co")))
    if name == "hateful":
        for m in HATEFUL_META:
            hf_hub_download(repo, m, repo_type="dataset", revision=rev, local_dir=out)
        want = {json.loads(l)["img"] for l in open(os.path.join(out, "train.jsonl")) if l.strip()}
        have = set(HfApi().list_repo_files(repo, repo_type="dataset", revision=rev))
        files = sorted(want & have)   # the repo lacks ~1.8k of the 8.5k train images
        log(root, "hateful: %d train images listed, %d present in the repo" % (len(want), len(files)))
    else:
        files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev) if keep(f))
    todo = [f for f in files if not os.path.exists(os.path.join(out, f))]
    log(root, "%s: %d files, %d to fetch" % (name, len(files), len(todo)))
    done, t0, stop = [0], time.time(), threading.Event()

    def ticker():
        while not stop.wait(60):
            log(root, "%s: %d/%d files, %.2f GB on disk, %.0f s" % (name, done[0], len(todo), du(out), time.time() - t0))

    threading.Thread(target=ticker, daemon=True).start()

    def one(f):
        for attempt in range(5):
            try:
                hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=out)
                break
            except Exception as e:   # network hiccup -> retry (hf_hub_download resumes the partial file)
                log(root, "%s: retry %d %s: %s" % (name, attempt + 1, f, str(e)[:160]))
                time.sleep(10 * (attempt + 1))
        else:
            log(root, "%s: FAILED %s" % (name, f))
        done[0] += 1
        if len(todo) < 50 or done[0] % 500 == 0:
            log(root, "%s: %d/%d files (%s)" % (name, done[0], len(todo), f))

    with ThreadPoolExecutor(workers if name == "hateful" else 2) as ex:
        list(ex.map(one, todo))
    stop.set()
    log(root, "%s: done, %.2f GB, %.0f s" % (name, du(out), time.time() - t0))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", nargs="*", default=list(REPOS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.root, "raw", "board"), exist_ok=True)
    if "hf-mirror" not in os.environ.get("HF_ENDPOINT", ""):
        sys.exit("set HF_ENDPOINT=https://hf-mirror.com (CLAUDE.md: downloads go through the domestic mirror)")
    for name in a.only:
        fetch(a.root, name, a.workers)


if __name__ == "__main__":
    main()
