"""Resume-safe download of the GUI-action sources from ModelScope (domestic mirror; hf-mirror.com currently answers
every path with a 308 redirect to huggingface.co, so it is not used and huggingface.co is never contacted).

    uv run --no-sync python scripts/data/v05_gui_fetch.py --root data [--only aitw-general mind2web aitw-shopping]

Each repo's file list comes from the ModelScope tree API; every file goes to
<root>/v05/gui/raw/<name>/<path> through `curl -C -` (resumes a partial file) and is accepted only when its size
equals the listed size; a finished file is skipped on the next run. Progress (bytes done / total, MB/s, ETA) is
appended to <root>/v05/gui/raw/fetch.log every file and every ~60 s.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

API = "https://www.modelscope.cn/api/v1/datasets/%s/repo"
REPOS = {
    # name: (ModelScope repo, sub-directory to fetch, file filter)
    "aitw-general": ("russwest/AitW", "data/general", None),
    "mind2web": ("osunlp/Multimodal-Mind2Web", "data", None),
    "aitw-shopping": ("russwest/AitW", "data/web_shopping", None),
}


def tree(repo: str, root: str):
    url = (API % repo) + "/tree?" + urllib.parse.urlencode({"Revision": "master", "Root": root, "PageSize": 1000})
    with urllib.request.urlopen(url, timeout=120) as r:
        files = json.load(r)["Data"]["Files"]
    return sorted((f["Path"], int(f["Size"])) for f in files if f["Type"] == "blob")


def file_url(repo: str, path: str) -> str:
    return (API % repo) + "?" + urllib.parse.urlencode({"Revision": "master", "FilePath": path})


def log(out: str, msg: str):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(out, "fetch.log"), "a") as f:
        f.write(line + "\n")


def fetch(root: str, names):
    out = os.path.join(root, "v05", "gui", "raw")
    os.makedirs(out, exist_ok=True)
    for name in names:
        repo, sub, _ = REPOS[name]
        files = tree(repo, sub)
        total = sum(s for _, s in files)
        dest_dir = os.path.join(out, name)
        log(out, "%s: %s/%s, %d files, %.2f GB" % (name, repo, sub, len(files), total / 1e9))
        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_dir + ".files.json", "w") as f:
            json.dump({"repo": repo, "root": sub, "files": files}, f, indent=1)
        start, start_done = time.time(), sum(min(os.path.getsize(os.path.join(dest_dir, p)), s)
                                             for p, s in files if os.path.exists(os.path.join(dest_dir, p)))
        for path, size in files:
            dest = os.path.join(dest_dir, path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and os.path.getsize(dest) == size:
                continue
            stop = threading.Event()

            def progress():
                while not stop.wait(60):
                    done = sum(os.path.getsize(os.path.join(dest_dir, p)) for p, _ in files
                               if os.path.exists(os.path.join(dest_dir, p)))
                    rate = (done - start_done) / max(1.0, time.time() - start)
                    log(out, "%s: %.2f / %.2f GB, %.1f MB/s, ETA %.1f h" % (
                        name, done / 1e9, total / 1e9, rate / 1e6, (total - done) / max(rate, 1) / 3600))

            t = threading.Thread(target=progress, daemon=True)
            t.start()
            for attempt in range(20):
                subprocess.run(["curl", "-s", "-L", "-C", "-", "--retry", "5", "--retry-delay", "10",
                                "--speed-limit", "10000", "--speed-time", "120", "-o", dest, file_url(repo, path)])
                if os.path.exists(dest) and os.path.getsize(dest) == size:
                    break
                if os.path.exists(dest) and os.path.getsize(dest) > size:
                    os.remove(dest)
                time.sleep(15)
            stop.set()
            ok = os.path.exists(dest) and os.path.getsize(dest) == size
            log(out, "%s: %s %s (%.2f GB)" % (name, "done" if ok else "FAILED", path, size / 1e9))
        log(out, "%s: finished" % name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", nargs="*", default=list(REPOS), choices=list(REPOS))
    a = ap.parse_args(argv)
    fetch(a.root, a.only)


if __name__ == "__main__":
    sys.exit(main())
