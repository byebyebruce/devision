"""Command-line entry points (see CLAUDE.md for usage)."""
import argparse
import json
import os
from typing import Iterable, List

from .data import Sample, convert_gqa, convert_pope, convert_vqav2, select


def _write_jsonl(path: str, samples: Iterable[Sample]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[Sample]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_main(argv=None) -> None:
    """Small MVP set: GQA train shard -> train.jsonl, GQA testdev -> val.jsonl, POPE -> pope.jsonl."""
    from .fetch import gqa_records, pope_records

    p = argparse.ArgumentParser(description=fetch_main.__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--train-limit", type=int, default=500)
    p.add_argument("--val-limit", type=int, default=200)
    p.add_argument("--pope-limit", type=int, default=300)
    p.add_argument("--train-shards", nargs="+", default=["train-00000-of-00021.parquet"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    val = select((convert_gqa(q, r) for q, r in gqa_records(a.root, "testdev", ["testdev-00000-of-00001.parquet"])),
                 a.val_limit, set(), a.seed)
    pope = select((convert_pope(r) for r in pope_records(a.root)), a.pope_limit, set(), a.seed)
    held_out = {s["image_id"] for s in val + pope}
    train = select((convert_gqa(q, r) for q, r in gqa_records(a.root, "train", a.train_shards)),
                   a.train_limit, held_out, a.seed)
    for name, samples in (("train", train), ("val", val), ("pope", pope)):
        print("%s: %d samples" % (name, _write_jsonl(os.path.join(a.root, name + ".jsonl"), samples)))


def convert_main(argv=None) -> None:
    """Convert official GQA / VQAv2 / POPE files into a selected, balanced JSONL."""
    p = argparse.ArgumentParser(description=convert_main.__doc__)
    p.add_argument("--gqa", nargs="*", default=[], help="GQA *_balanced_questions.json files")
    p.add_argument("--vqav2-questions", help="v2_OpenEnded_mscoco_<split>_questions.json")
    p.add_argument("--vqav2-annotations", help="v2_mscoco_<split>_annotations.json")
    p.add_argument("--vqav2-split", default="train2014")
    p.add_argument("--pope", nargs="*", default=[], help="POPE JSONL records")
    p.add_argument("--exclude", nargs="*", default=[], help="JSONL sample files whose images must not appear")
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    def samples():
        for path in a.gqa:
            with open(path) as f:
                for qid, rec in json.load(f).items():
                    yield convert_gqa(qid, rec)
        if a.vqav2_questions:
            with open(a.vqav2_questions) as f:
                questions = {q["question_id"]: q for q in json.load(f)["questions"]}
            with open(a.vqav2_annotations) as f:
                for ann in json.load(f)["annotations"]:
                    yield convert_vqav2(questions[ann["question_id"]], ann, a.vqav2_split)
        for path in a.pope:
            for rec in read_jsonl(path):
                yield convert_pope(rec)

    exclude = {s["image_id"] for path in a.exclude for s in read_jsonl(path)}
    print("wrote %d samples" % _write_jsonl(a.out, select(samples(), a.limit, exclude, a.seed)))


def train_main(argv=None) -> None:
    """Fine-tune from Laya + SigLIP2 on a JSONL of samples and save a checkpoint."""
    from .decider import Decider
    from .model import ModelConfig, from_pretrained
    from .train import TrainConfig, train

    p = argparse.ArgumentParser(description=train_main.__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--val", help="JSONL used for per-epoch accuracy and temperature fitting")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--laya", default="convaiinnovations/laya")
    p.add_argument("--vision", default="google/siglip2-base-patch16-256")
    p.add_argument("--visual-shuffle", type=int, default=2, help="2: 64 visual tokens, 1: 256")
    p.add_argument("--model-name", default="devision-0.1")
    for field, default in vars(TrainConfig()).items():
        p.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
    a = p.parse_args(argv)

    model, tok, cfg = from_pretrained(a.laya, a.vision, ModelConfig(model_name=a.model_name,
                                                                    visual_shuffle=a.visual_shuffle))
    config = TrainConfig(**{k: getattr(a, k) for k in vars(TrainConfig())})
    report = train(Decider(model, tok, cfg), read_jsonl(a.data), a.data_root, a.out, config,
                   val_samples=read_jsonl(a.val) if a.val else None)
    with open(os.path.join(a.out, "train_report.json"), "w") as f:
        json.dump(report, f)


def eval_main(argv=None) -> None:
    """Accuracy, ECE and latency of a checkpoint on a JSONL of samples, through decide()."""
    from .decider import Decider
    from .evaluate import evaluate

    p = argparse.ArgumentParser(description=eval_main.__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out")
    a = p.parse_args(argv)

    result = evaluate(Decider.load(a.checkpoint), read_jsonl(a.data), a.data_root)
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


def serve_main(argv=None) -> None:
    """Serve POST /v1/systemone on CPU."""
    import uvicorn

    from .decider import Decider
    from .server import create_app

    p = argparse.ArgumentParser(description=serve_main.__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    uvicorn.run(create_app(Decider.load(a.checkpoint)), host=a.host, port=a.port)
