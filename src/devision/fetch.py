"""Download a small MVP dataset from the Hugging Face mirrors (lmms-lab/GQA, lmms-lab/POPE).

Only the image shards that are needed are downloaded; images are written under the data root at
the paths the converters expect (gqa/images/<id>.jpg, coco/val2014/<name>.jpg).
"""
import os
from typing import Any, Dict, Iterator, List, Tuple

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

GQA_REPO, POPE_REPO = "lmms-lab/GQA", "lmms-lab/POPE"


def _parquet(repo: str, filename: str):
    return pq.read_table(hf_hub_download(repo, filename, repo_type="dataset"))


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)


def gqa_records(root: str, split: str, image_shards: List[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """(question id, record) for the balanced `split` questions whose images are in `image_shards`."""
    ids = set()
    for shard in image_shards:
        for row in _parquet(GQA_REPO, "%s_balanced_images/%s" % (split, shard)).to_pylist():
            _write(os.path.join(root, "gqa", "images", "%s.jpg" % row["id"]), row["image"]["bytes"])
            ids.add(row["id"])
    instructions = "%s_balanced_instructions/%s-00000-of-00001.parquet" % (split, split)
    table = _parquet(GQA_REPO, instructions)
    table = table.filter(pc.field("imageId").isin(list(ids)))
    for rec in table.select(["id", "imageId", "question", "answer", "types", "semantic"]).to_pylist():
        yield rec["id"], rec


def pope_records(root: str, shard: str = "test-00000-of-00003.parquet") -> Iterator[Dict[str, Any]]:
    for row in _parquet(POPE_REPO, "data/" + shard).to_pylist():
        split = row["image_source"].split("_")[1]
        _write(os.path.join(root, "coco", split, row["image_source"] + ".jpg"), row["image"]["bytes"])
        yield {k: row[k] for k in ("question_id", "question", "answer", "image_source", "category")}
