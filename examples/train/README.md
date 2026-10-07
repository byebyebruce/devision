# 训练数据示例

这里是几条最小的训练数据，用来说明格式，并能端到端跑通一次训练：

```bash
uv run devision-pipeline configs/example.yaml    # CPU 上约 1–2 分钟；首次会下载基础模型（约 2.5 GB）
```

需要训练依赖：本仓库里 `uv sync` 已包含；单独安装时用 `pip install "devision[train]"`。

它依次跑阶段 1（对齐）、阶段 2（决策题）和评测，产出 `runs/example/`。数据只有十几条，模型学不到东西，只用来确认格式和流程。真实规模的从头训练见 `configs/scratch.yaml`，发布版 v0.2 的最后一轮见 `configs/v12-ext.yaml`，每一轮的数据和结果见 `docs/training-log.md`。

## 文件

| 文件 | 用在 | 内容 |
|---|---|---|
| `align.jsonl` | 阶段 1 `devision-align` | 图片 + 描述（caption） |
| `train.jsonl` | 阶段 2 `devision-train --data` | 决策题 + 标准答案 |
| `val.jsonl` | 阶段 2 `--val`，以及评测 | 同上；训练过程中定期评测，结束时拟合每种题型的温度 |
| `shapes/*.png` | 图片 | 合成的彩色形状；另外两张照片是上一级目录里的 web demo 示例图 |

图片路径都相对于 `data_root`（这里是 `examples/`）。

## 阶段 1：caption 样本

每行一张图，`captions` 是一条或多条描述，训练时每次随机取一条，遮掉一半的词让模型看图猜：

```json
{"id": "example-cap:dog", "source": "example", "image_id": "example:1-dog-skier", "image": "1-dog-skier.jpg",
 "captions": ["A golden dog runs across the snow toward the camera.", "A woman on cross-country skis follows a dog through a snowy field."]}
```

## 阶段 2：决策题样本

`questions` 就是 Jev 请求里的 `questions`；`gold` 给出每道题的标准答案，是一个**概率分布**。可选的 `state_text` 是和图片一起给模型的文字（如 ScienceQA 的提示），推理时相当于 `state`。例子：

```json
{"id": "example:1", "source": "example", "image_id": "example:1-dog-skier", "image": "1-dog-skier.jpg",
 "questions": {"q": {"type": "noul", "instructions": "Is there a dog in the image?"}},
 "gold": {"q": {"probabilities": {"false": 0.0, "true": 1.0}}}}
```

`train.jsonl` 里每种写法都有一条：

| 写法 | 例子 | 说明 |
|---|---|---|
| `noul`，硬标签 | "Is there a dog in the image?" → `true: 1.0` | 是非题；键固定是 `false` / `true` |
| `noul`，软标签 | "Is the dog running?" → `true: 0.8` | 例如 10 个标注者里 8 个说是（VQAv2 的做法），模型学的是这个概率，有利于校准 |
| `choice` | "Which sport do these people play?" → `tennis` | `criteria` 的键是选项，`gold` 的键必须和它一致 |
| `choice`，带选项说明 | `"criteria": {"skiing": "moving on skis", "running": "on foot, no skis", "sitting": null}` | 说明文字会一起给模型看，帮它区分相近的选项 |
| 一张图多道题 | `"questions": {"lying": {...}, "count": {...}}` | 题目 id 自己起；每道题都要有对应的 `gold` |
| 位置题 | "Is the red circle to the left or to the right of the blue square?" | 选项可以是短语 |

## 字段规则

- `id`：全局唯一。
- `source`：数据来源，评测时按它分别统计准确率。
- `image_id`：带命名空间的图片 id，例如 `coco:42`、`vg:2384884`。**评测集里的图片不能出现在任何训练文件里**，就是靠它检查的。这里 `val.jsonl` 用的两张图（`green-square-yellow-circle`、`yellow-square-bottom`）不在 `align.jsonl` 和 `train.jsonl` 里。
- 一个样本只能有一张图；题型只支持 `noul` 和 `choice`。
- 选项的顺序要打乱，不要让正确答案总在同一个位置。

## 生成真实规模的数据

训练代码只认这种格式（见 `src/devision/train/samples.py`）。把 GQA、VQAv2、COCO 等公开数据集下载并转换成这种格式的脚本在仓库的 [`scripts/data/`](../../scripts/data/)：`bash scripts/data/build_all.sh data` 一次生成从头训练和第 3–9 轮用到的数据，第 12 轮的扩展数据包用 `bash scripts/data/ext/build_ext.sh data`。详见主 README 的「自己训练」。
