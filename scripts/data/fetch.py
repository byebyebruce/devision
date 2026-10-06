"""Download training / evaluation data.

- GQA and POPE from the Hugging Face mirrors (lmms-lab/GQA, lmms-lab/POPE), one parquet shard at a time.
- VQAv2 questions and annotations from the official S3 bucket; COCO images one by one from
  images.cocodataset.org, only for the samples actually selected.

Images are written under the data root at the paths the converters expect
(gqa/images/<id>.jpg, coco/<split>/COCO_<split>_<id>.jpg); downloads live under <root>/raw
(Hugging Face datasets in <root>/raw/hf, a hub cache directory).
"""
import io
import json
import os
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

GQA_REPO, POPE_REPO = "lmms-lab/GQA", "lmms-lab/POPE"
VQAV2_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_%s_%s_mscoco.zip"  # % (kind, "Train"|"Val")
VG_IMAGE_DATA_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/%s"  # % "train2014/COCO_train2014_<id>.jpg"


COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
GQA_SCENE_GRAPHS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"


def ensure_file(url: str, path: str) -> str:
    """`path`, downloaded from `url` first if it is not there (written to a temporary name, then moved)."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        retrieve(url, path)
    return path


def retrieve(url: str, path: str) -> None:
    """Download `url` to `path` (through a temporary name), printing progress about every 10%."""
    print("downloading", url, flush=True)
    start, shown = time.time(), [-1]

    def hook(blocks: int, block_size: int, total: int) -> None:
        got = blocks * block_size
        step = got * 10 // total if total > 0 else got // (200 << 20)   # tenths, or every 200 MB if size unknown
        if step > shown[0]:
            shown[0] = step
            of = " / %.0f MB" % (total / 1e6) if total > 0 else ""
            print("  %s: %.0f MB%s, %.0f s" % (os.path.basename(path), got / 1e6, of, time.time() - start), flush=True)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, path + ".part", hook)
    os.replace(path + ".part", path)


def run_with_progress(label: str, items: Sequence[Any], one: Callable[[Any], str], workers: int = 16) -> List[str]:
    """`one(item)` for every item in a thread pool; returns the non-empty results (the failures). Prints how
    many are done about every 5% and every 60 s, so a long download is visibly alive."""
    total, done, failed = len(items), 0, []
    print("%s: %d to check" % (label, total), flush=True)
    start = last = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for r in pool.map(one, items):
            done += 1
            if r:
                failed.append(r)
            now = time.time()
            if done == total or done % max(1, total // 20) == 0 or now - last >= 60:
                last = now
                print("  %s: %d / %d (%d failed), %.0f s" % (label, done, total, len(failed), now - start), flush=True)
    return failed


def coco_annotations(root: str) -> str:
    """data/raw/coco/annotations_trainval2014.zip (instances and captions of COCO 2014)."""
    return ensure_file(COCO_ANNOTATIONS_URL, os.path.join(root, "raw", "coco", "annotations_trainval2014.zip"))


def hf_dataset_files(root: str, repo: str, pattern: str) -> List[str]:
    """Local paths of the files of Hugging Face dataset `repo` matching the glob `pattern` (downloaded to
    data/raw/hf on first use)."""
    import fnmatch
    from huggingface_hub import HfApi
    names = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern))
    if not names:
        raise SystemExit("no file of %s matches %s" % (repo, pattern))
    return [hf_hub_download(repo, n, repo_type="dataset", cache_dir=os.path.join(root, "raw", "hf")) for n in names]


def _parquet(root: str, repo: str, filename: str):
    return pq.read_table(hf_hub_download(repo, filename, repo_type="dataset",
                                         cache_dir=os.path.join(root, "raw", "hf")))


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)


def gqa_records(root: str, split: str, image_shards: List[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """(question id, record) for the balanced `split` questions whose images are in `image_shards`."""
    ids = set()
    for shard in image_shards:
        for row in _parquet(root, GQA_REPO, "%s_balanced_images/%s" % (split, shard)).to_pylist():
            _write(os.path.join(root, "gqa", "images", "%s.jpg" % row["id"]), row["image"]["bytes"])
            ids.add(row["id"])
    instructions = "%s_balanced_instructions/%s-00000-of-00001.parquet" % (split, split)
    table = _parquet(root, GQA_REPO, instructions)
    table = table.filter(pc.field("imageId").isin(list(ids)))
    for rec in table.select(["id", "imageId", "question", "answer", "types", "semantic"]).to_pylist():
        yield rec["id"], rec


def pope_records(root: str, shard: str = "test-00000-of-00003.parquet") -> Iterator[Dict[str, Any]]:
    for row in _parquet(root, POPE_REPO, "data/" + shard).to_pylist():
        split = row["image_source"].split("_")[1]
        _write(os.path.join(root, "coco", split, row["image_source"] + ".jpg"), row["image"]["bytes"])
        yield {k: row[k] for k in ("question_id", "question", "answer", "image_source", "category")}


def _download_json_zip(url: str, cache: str) -> Dict[str, Any]:
    """The single JSON file inside a zip at `url`, cached under `cache`."""
    path = os.path.join(cache, os.path.basename(url))
    if not os.path.exists(path):
        retrieve(url, path)
    with zipfile.ZipFile(path) as z:
        (name,) = [n for n in z.namelist() if n.endswith(".json")]
        return json.load(io.TextIOWrapper(z.open(name), encoding="utf-8"))


def vqav2_records(root: str, split: str = "Train") -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(question, annotation) pairs of VQAv2 train2014 ("Train") or val2014 ("Val"). No images are
    downloaded here."""
    cache = os.path.join(root, "raw", "vqav2")
    questions = {q["question_id"]: q for q in
                 _download_json_zip(VQAV2_URL % ("Questions", split), cache)["questions"]}
    for ann in _download_json_zip(VQAV2_URL % ("Annotations", split), cache)["annotations"]:
        yield questions[ann["question_id"]], ann


def vg_to_coco(root: str) -> Dict[str, str]:
    """"vg:<id>" -> "coco:<id>" for the Visual Genome (GQA) images that are also COCO images."""
    data = _download_json_zip(VG_IMAGE_DATA_URL, os.path.join(root, "raw", "vg"))
    return {"vg:%d" % r["image_id"]: "coco:%d" % r["coco_id"] for r in data if r.get("coco_id")}


def download_coco_images(root: str, samples: Sequence[Dict[str, Any]], workers: int = 16) -> List[str]:
    """Fetch the COCO image of each sample that is not on disk yet. Returns the ids that failed."""
    def one(s) -> str:
        path = os.path.join(root, s["image"])
        if os.path.exists(path):
            return ""
        rel = s["image"].split("/", 1)[1]  # coco/train2014/x.jpg -> train2014/x.jpg
        try:
            with urllib.request.urlopen(COCO_IMAGE_URL % rel, timeout=30) as r:
                _write(path, r.read())
            return ""
        except OSError:
            return s["id"]

    return run_with_progress("COCO images", samples, one, workers)
