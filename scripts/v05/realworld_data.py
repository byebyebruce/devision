"""RealWorldQA native MC and yes/no only. Never invent test distractors."""
import hashlib
import json
from pathlib import Path
import re

from chart_data import pixels, write_rows


def convert(question, answer):
    question = re.sub(r"\s*Please answer directly with .*?$", "", question, flags=re.S).strip()
    matches = list(re.finditer(r"(?<![A-Za-z0-9])([A-F])\.\s+", question))
    if matches:
        letters = [m.group(1) for m in matches]
        if len(matches) < 2 or letters != list("ABCDEF"[:len(matches)]) or answer.strip() not in letters:
            return None
        options = [question[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(question)].strip()
                   for i, m in enumerate(matches)]
        if any(not x for x in options) or len(set(x.casefold() for x in options)) != len(options):
            return None
        q = {"type": "choice", "instructions": question[:matches[0].start()].strip(), "criteria": dict.fromkeys(options)}
        gold = {o: float(i == letters.index(answer.strip())) for i, o in enumerate(options)}
        return q, gold
    if answer.strip().lower() in ("yes", "no"):
        return {"type": "noul", "instructions": question}, {"false": float(answer.strip().lower() == "no"),
                                                             "true": float(answer.strip().lower() == "yes")}
    return None


def main():
    import pyarrow.parquet as pq
    source = json.loads(Path("runs/v05-prep/realworld_source.json").read_text())
    dest = Path("data/v05/external")
    if (dest / "realworld.MANIFEST.json").exists():
        raise SystemExit("Frozen RealWorldQA already exists")
    (dest / "images").mkdir(parents=True, exist_ok=True)
    rows, total = [], 0
    for f in sorted((Path(source["snapshot"]) / "data").glob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=16):
            for r in batch.to_pylist():
                sid = f"realworldqa:{total}"
                total += 1
                converted = convert(r["question"], r["answer"])
                if converted is None:
                    continue
                h, _ = pixels(r["image"]["bytes"])
                path = dest / "images" / (h + ".png")
                from PIL import Image
                import io
                with Image.open(io.BytesIO(r["image"]["bytes"])) as im:
                    im.convert("RGB").save(path)
                q, gold = converted
                rows.append({"id": sid, "source": "realworldqa", "kind": q["type"], "image_id": "realworldqa:" + h,
                             "image": str(path.relative_to("data")), "questions": {"q": q},
                             "gold": {"q": {"probabilities": gold}}, "original_question": r["question"],
                             "original_answer": r["answer"]})
    meta = write_rows(dest / "realworld_native.jsonl", rows)
    meta.update(source=source, original_rows=total, excluded_open_or_unparseable=total - len(rows),
                protocol="Native-MC and native yes/no subset only; not official full RealWorldQA accuracy",
                role="monitoring", types={t: sum(r["kind"] == t for r in rows) for t in ("choice", "noul")})
    (dest / "realworld.MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
