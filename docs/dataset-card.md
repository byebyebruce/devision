---
license: cc-by-4.0
language:
- en
task_categories:
- visual-question-answering
pretty_name: LookFirst
size_categories:
- 100K<n<1M
tags:
- vqa
- calibration
- spatial-reasoning
- grounding
- system-one
- jev
configs:
- config_name: exist
  data_files: [{split: train, path: data/exist/train.parquet}, {split: validation, path: data/exist/validation.parquet}, {split: test, path: data/exist/test.parquet}]
- config_name: position
  data_files: [{split: train, path: data/position/train.parquet}, {split: validation, path: data/position/validation.parquet}, {split: test, path: data/position/test.parquet}]
- config_name: relation
  data_files: [{split: train, path: data/relation/train.parquet}, {split: validation, path: data/relation/validation.parquet}, {split: test, path: data/relation/test.parquet}]
- config_name: size
  data_files: [{split: train, path: data/size/train.parquet}, {split: validation, path: data/size/validation.parquet}, {split: test, path: data/size/test.parquet}]
- config_name: vqa_yesno
  data_files: [{split: train, path: data/vqa_yesno/train.parquet}, {split: validation, path: data/vqa_yesno/validation.parquet}, {split: test, path: data/vqa_yesno/test.parquet}]
- config_name: vqa_choice
  data_files: [{split: train, path: data/vqa_choice/train.parquet}, {split: validation, path: data/vqa_choice/validation.parquet}, {split: test, path: data/vqa_choice/test.parquet}]
- config_name: gqa
  data_files: [{split: train, path: data/gqa/train.parquet}, {split: validation, path: data/gqa/validation.parquet}, {split: test, path: data/gqa/test.parquet}]
---

# LookFirst

**Visual yes/no and multiple-choice questions that the words alone cannot answer.** Every question is built so that the question text, the object names and the answer position carry no hint: for each group of questions that share a wording or a pair of object names, every answer occurs equally often. A model has to look at the picture to beat chance — and when it does, you can tell, because swapping the picture brings it back to chance.

LookFirst was built to train and test [deVision](https://github.com/byebyebruce/devision), a calibrated visual decision model in the Jev `/v1/systemone` format. The questions use the same typed format: `noul` (does the statement hold?) and `choice` (which option?), with a probability for every option.

## What makes it different

- **No language prior.** In COCO, "the dining table is at the bottom of the picture" holds 96% of the time and "the person is bigger than the tennis racket" 100%, so most spatial questions generated from boxes can be answered from the names. Here, per category (position), per unordered pair of names (relation, size) and per wording (VQAv2, GQA), every answer occurs equally often; groups that only ever go one way are dropped. Existence questions use objects that co-occur with the scene as negatives, not random ones.
- **Measured question-only baselines.** For every test set, the accuracy of answering from the wording alone (most frequent answer for the same wording in all training data) is published below and in `question_only_baselines.json`. They sit at chance.
- **Disjoint pictures.** No picture is in two splits, across all configs, with COCO and Visual Genome ids of the same picture matched (half of Visual Genome is COCO). Test pictures are COCO val2014 / VQAv2 val / GQA val pictures.
- **Soft labels.** VQAv2 yes/no questions keep the annotators' agreement as the probability (7 of 10 said yes → 0.7), for calibration training; questions where annotators split 5–5 are left out of the test set.
- **Grounding probes.** Position and relation questions carry an `axis` (`lr` left/right, `tb` top/bottom), so a model's spatial ability can be read per direction. The relation train split adds 1,784 above/below questions from GQA scene graphs, because COCO yields only ~1,500 balanced ones.

## Configs

| Config | What is asked | Built from | Train | Validation | Test |
|---|---|---|---|---|---|
| `exist` | Is there a *X* in the image? (negatives co-occur with the scene) | COCO instances | 44,398 | 1,472 | 1,120 |
| `position` | Is the *X* on the left or the right / top or bottom of the image? | COCO boxes | 31,988 | 1,088 | 1,274 |
| `relation` | Is the *X* to the left of / above the *Y*? (yes/no and two-way) | COCO boxes + GQA scene graphs | 10,600 | 200 | 1,950 |
| `size` | Which looks bigger, the *X* or the *Y*? | COCO boxes | 4,608 | 162 | 1,100 |
| `vqa_yesno` | VQAv2 yes/no questions, balanced per wording, soft labels | VQAv2 | 55,714 | 1,948 | 1,000 |
| `vqa_choice` | VQAv2 questions of 13 kinds (colour, count, room, sport, …) as multiple choice with plausible distractors | VQAv2 | 36,058 | 1,591 | 1,420 |
| `gqa` | GQA balanced questions (yes/no → `noul`, "choose" → `choice`), incl. spatial | GQA | 14,832 | 569 | 992 |

## Baselines

Accuracy on the test splits. *Question only*: the most frequent answer for the same wording in all training data. *deVision*: round 3 (`v3-data2`, 518M, trained on these train splits), through `decide()` on CPU; *mismatched*: the same model with every picture swapped for another one.

| Test set | Chance | Question only | deVision | deVision, mismatched pictures |
|---|---|---|---|---|
| exist | 0.500 | 0.508 | 0.958 | 0.494 |
| position | 0.500 | 0.481 | 0.730 (top/bottom 0.920, left/right 0.542) | 0.500 |
| relation | 0.500 | 0.485 | 0.519 | 0.481 |
| size | 0.500 | 0.532 | 0.841 | 0.503 |
| vqa_yesno | 0.500 | 0.518 | 0.714 | 0.520 |
| vqa_choice | 0.321 | 0.418 | 0.892 | 0.361 |
| gqa | 0.500 | 0.524 | 0.755 | 0.543 |

Left/right questions remain at chance for this model: the relation and left/right position sets are the hard part of LookFirst.

## Fields

| Field | Meaning |
|---|---|
| `id` | unique question id |
| `question_type` | `noul` (statement holds?) or `choice` |
| `question` | the question (Jev `instructions`) |
| `options` | `choice`: the options in the order they are asked; `noul`: `["false", "true"]` |
| `probabilities` | gold probability of each option (soft for VQAv2 yes/no) |
| `answer` | the most probable option |
| `image_id` | `coco:<id>` or `vg:<id>` (Visual Genome / GQA) |
| `image_file` | path the picture is stored under by `fetch_images.py` |
| `image_source` | `coco_train2014`, `coco_val2014` or `visual_genome` |
| `source`, `kind`, `axis`, `group` | generator, ability (e.g. `color`, `relation`), direction (`lr` / `tb`), and the balancing group (wording or names) |

## Pictures

The pictures are not redistributed. Download them from COCO and Visual Genome:

```bash
pip install pyarrow huggingface_hub
huggingface-cli download lukbit/lookfirst --repo-type dataset --local-dir lookfirst
python lookfirst/fetch_images.py --out images                      # all configs and splits
python lookfirst/fetch_images.py --out images --config exist --split test
```

## Use with deVision

A row as a Jev request and as a deVision training sample:

```python
def to_request(row, image_root):
    q = {"type": row["question_type"], "instructions": row["question"]}
    if row["question_type"] == "choice":
        q["criteria"] = {o: None for o in row["options"]}
    return {"state": [{"type": "image", "url": f"file://{image_root}/{row['image_file']}"}], "questions": {"q": q}}

def to_sample(row):
    q = {"type": row["question_type"], "instructions": row["question"]}
    if row["question_type"] == "choice":
        q["criteria"] = {o: None for o in row["options"]}
    return {"id": row["id"], "source": row["source"], "kind": row["kind"], "axis": row["axis"],
            "image_id": row["image_id"], "image": row["image_file"], "questions": {"q": q},
            "gold": {"q": {"probabilities": dict(zip(row["options"], row["probabilities"]))}}}
```

## Sources and license

The annotations of this dataset are released under **CC BY 4.0**. They are derived from:

- [COCO](https://cocodataset.org) 2014 instance annotations (CC BY 4.0); pictures under their Flickr terms.
- [VQAv2](https://visualqa.org) (CC BY 4.0).
- [GQA](https://cs.stanford.edu/people/dorarad/gqa/) balanced questions and scene graphs (CC BY 4.0), on [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/) pictures (CC BY 4.0).

Please cite those datasets alongside this one.

## Known limitations

- Visual Genome names are noisy: in a spot check of 16 scene-graph above/below questions, the directions all matched the boxes, but about one in four named an object ambiguously (several of them in the picture) or wrongly.
- VQAv2 soft labels reflect annotator disagreement, which includes genuinely ambiguous questions.
- Balancing per group removes common, easy questions; accuracy here is not comparable to the original VQAv2 / GQA benchmarks.
