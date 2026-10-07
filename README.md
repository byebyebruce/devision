# deVision

**deVision = Decision + Vision**：看图做决策。名字取 **De**cision 的开头和 **Vision** 的全部，两个词共享同一个 `-sion` 结尾，拼起来正好是 "deVision"。
- **Vision**：看图，由 SigLIP2 视觉编码器负责。
- **Decision**：做决策，由 Laya 的 ModernBERT 和决策头负责。它不生成文字，而是直接给出每个选项的校准概率。

它相当于一个能看图的 Jev：输入一张图片和若干英文问题，输出 Jev 格式的结构化答案（`noul` 是非题 / `choice` 选择题）和校准过的概率。同一张图的多道题共用一次图片编码，没有 GPU 也能跑。

- 模型：[`lukbit/devision`](https://huggingface.co/lukbit/devision)（Hugging Face，当前版本 v0.2）
- 设计与范围：[`docs/spec/vision-decision-mvp.md`](docs/spec/vision-decision-mvp.md)
- 每一轮训练的改动和结果：[`docs/training-log.md`](docs/training-log.md)

## 安装

需要 Python 3.11 及以上。

```bash
pip install "devision @ git+https://github.com/byebyebruce/devision"            # 推理 + HTTP 服务 + web demo
pip install "devision[train] @ git+https://github.com/byebyebruce/devision"     # 再加训练、评测、校准
```

| 安装项 | 包含 |
|---|---|
| `devision` | 推理（`devision.load`、`predict` / `decide`）、HTTP 服务和 web demo（`devision-serve`） |
| `devision[train]` | 加上训练、评测、校准和实验流水线（`devision-align`、`devision-train`、`devision-eval`、`devision-calibrate`、`devision-compare`、`devision-pipeline`；依赖 peft、SwanLab 等） |

在本仓库里开发用 [uv](https://docs.astral.sh/uv/)：`uv sync` 装好全部依赖，命令都经 `uv run ...` 执行。

## Python

```python
import devision

decider = devision.load("lukbit/devision")   # Hugging Face 上的最新版；也可传本地 checkpoint 目录
# 固定版本：devision.load("lukbit/devision", revision="v0.2")
# 设备默认 auto（CUDA > MPS > CPU），可指定 device="cpu"；仓库私有时先 hf auth login，或传 token="hf_..."

result = decider.predict(            # 与 decider.decide(...) 相同，predict 是 Laya 的叫法
    image="kitchen.jpg",
    state="",                        # 文字背景；没有就传 ""
    questions={
        "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
        "room": {"type": "choice", "instructions": "Which room is this?",
                 "criteria": {"kitchen": None, "bathroom": "has a sink and a toilet", "bedroom": None}},
    },
)
print(result["answers"]["has_fork"]["noul"])        # 陈述成立的概率
print(result["answers"]["room"]["choice"])          # 概率最高的选项
print(result["answers"]["room"]["probabilities"])   # 每个选项的概率，和为 1
```

**`image`**：库会自动识别下面这些输入。

| 输入 | 示例 |
|---|---|
| HTTP(S) URL | `image="https://example.com/photo.jpg"` |
| 本地文件路径 | `image="photo.jpg"` 或 `image=Path("photo.jpg")` |
| Data URI | `image="data:image/png;base64,..."` |
| 纯 Base64 | `image=base64_string` |
| 图片文件的字节 | `image=image_bytes`（PNG / JPEG 文件内容） |
| PIL 图片 | `image=pil_image` |

**`state`**：必填的文字背景，可以是字符串、对象或数组，和 Jev 一样：

```python
result = decider.decide(
    image="photo.jpg",
    state={"note": "The customer says this item arrived damaged."},
    questions={"damaged": {"type": "noul", "instructions": "Is the item visibly damaged?"}},
)
```

- `state` 对象里叫 `image` 的字段只是文字，不会加载图片。
- 兼容 Jev 的旧写法 `state=[{"type": "image", "url": "https://..."}, "背景文字"]` 仍可用，但不能和 `image=` 同时给。
- 请求不合法时抛出 `devision.InvalidRequest`（HTTP 服务里是 422）；读本地文件失败时可能抛出 `OSError`。

## HTTP 服务和 web demo

```bash
devision-serve --checkpoint lukbit/devision --port 8000      # 在本仓库里：uv run devision-serve ...
# 浏览器打开 http://127.0.0.1:8000 是 web demo
```

| 参数 | 作用 |
|---|---|
| `--checkpoint` | 本地 checkpoint 目录或 Hugging Face 仓库 id |
| `--revision` | 仓库 id 时固定的 commit、tag 或分支（如 `v0.2`）；不给就用最新版 |
| `--host` / `--port` | 默认 `127.0.0.1:8000` |
| `--device` | `auto`（默认：有 CUDA / Apple GPU 就用，没有就用 CPU）、`cuda`、`mps` 或 `cpu` |
| `--threads` | CPU 推理线程数 |
| `--examples` | demo 的照片示例目录（默认 `examples/`） |
| `--no-demo` | 只开接口，不挂 web demo |

接口：

- `POST /v1/systemone`：做决策，请求和响应见下文。
- `GET /health`：返回 `{"status": "ok", "model": "devision-v0.2"}`。
- `GET /`：web demo。可以选示例图或上传图片、编辑多道题、看每个选项的概率，也能和"水平镜像 / 无图"对照比较（对照会再发一次请求）。界面是中文，问题和选项用英文。

### 请求

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "image": "https://example.com/kitchen.jpg",
  "state": "customer says the order arrived damaged",
  "questions": {
    "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
    "room": {
      "type": "choice",
      "instructions": "Which room is this?",
      "criteria": {"kitchen": null, "bathroom": "has a sink and a toilet", "bedroom": null}
    }
  }
}'
```

- **`image`**：http(s) URL、Data URI 或纯 Base64，和 Python 的 `image=` 对应；服务不读自己机器上的文件。不带图时省略，等同于纯文本的 Laya。
- **`state`**：文字背景，字符串、对象或数组；没有就传 `""`。兼容 Jev 的旧写法：在 `state` 数组里放图片片段 `{"type": "image", "url": "..."}` 或 `{"type": "image", "base64": "..."}`，不能和 `image` 同时给。
- **`questions`**：对象，键是你自己起的题目 id，一次可以问多道。
  - `noul`：是非题，`instructions` 写要判断的陈述或问题；`criteria` 可选，用 `{"false": "...", "true": "..."}` 改写两个选项的含义。
  - `choice`：单选题，`criteria` 的键是选项（2–255 个），值是 `null` 或一句说明，帮模型区分相近的选项。
  - `score` 不支持，返回 422。

### 响应

与 Jev 完全相同：

```json
{
  "model": "devision-v0.2",
  "answers": {
    "has_fork": {"type": "noul", "noul": 0.03},
    "room": {
      "type": "choice",
      "choice": "kitchen",
      "probabilities": {"kitchen": 0.81, "bathroom": 0.07, "bedroom": 0.12},
      "confidence": 0.72
    }
  },
  "usage": {"input_tokens": 180, "output_tokens": 0}
}
```

- `noul`：陈述成立的概率。
- `probabilities`：每个选项的概率，和为 1；`choice` 是概率最高的选项。
- `confidence`：按 Jev 的公式 `(n·p_max − 1)/(n − 1)`，0 表示和均匀猜一样，1 表示完全确定。
- 错误：请求格式不对、图片取不到或解码失败、选项放不进模型输入时，返回 422 和 `{"error": "..."}`。

**速度**：Mac CPU（4 线程）上单题约 0.15 秒；同一请求里的多道题共用一次图片编码，每题约 0.09 秒。有 GPU 时自动使用。

## 怎么用得好

**用概率做阈值，没把握的转给人或更强的模型。** 概率按（题型, 选项数）做了温度校准，v0.2 在留出测试集上的 ECE 约 0.02–0.06（见下方 Benchmark）。但校准程度因题而异，阈值请在自己的数据上确定并验证：看不同阈值下自动处理的比例和错误率。

```python
p = result["answers"]["has_fork"]["noul"]
if p > 0.9:
    handle_yes()
elif p < 0.1:
    handle_no()
else:
    escalate()   # 人工，或者更强的模型（系统 2）
```

**适合问的：**

- 图里有没有某样东西、有几个：有没有人、有没有破损、桌上有几个杯子（数量用选择题问）。
- 是什么、属于哪类：房间类型、动物种类、颜色、材质，选项不宜太多。
- 整体判断：室内还是室外、东西看起来是不是坏了。
- 照片里的上下、左右位置，用选择题问（"左边还是右边"）。

**不适合问的：**

- 在示意图、图表、地图上比较两处（哪根柱子高、磁铁哪一极、哪条线长）。这类题仍接近随机。
- 相对位置的是非题（"Is the cup to the left of the laptop?"），改成选择题效果好得多。
- 图中的小字（输入只有 256×256，这不是 OCR 模型）。
- 非英文问题、一次多张图、`score` 打分题。

**题目写清楚，选项互不重叠。** 选项意思接近时，在 `criteria` 里加一句说明。

## Benchmark

数字来自发布版 v0.2（训练轮次 `v12-ext`，训练过程见 [`docs/training-log.md`](docs/training-log.md)），全部经 `decide` 评测（MPS；CPU 上结果相同），已套用拟合温度。每一轮的结果和与上一轮的逐题配对比较都记在训练日志里。

### 与 laya-vision 对比（同一批题逐题配对）

laya-vision 是目前唯一公开的、能看图的 System One 模型。我们按它公开的逐题预测重建了它的评测集（`scripts/data/lv_bench.py`，不运行它的模型），在同一批题上逐题比较；差值 = deVision − laya-vision 201M，95% 区间按图片配对重采样（`devision-compare`）。

| 评测集 | 题数 | laya-vision 201M | deVision v0.2 | 差值 [95% 区间] |
|---|---|---|---|---|
| POPE random / popular / adversarial | 3 × 3,000 | 0.836 / 0.819 / 0.777（公开分，无逐题） | **0.891 / 0.868 / 0.791** | +5.5 / +4.9 / +1.4 个点 |
| VQAv2 是非题 | 4,887 | 0.717 | **0.725** | +0.8 [−0.9, +2.3] |
| A-OKVQA（常识推理，四选一） | 1,138 | 0.598 | **0.626** | +2.7 [−0.6, +6.0] |
| ScienceQA（带图题） | 2,097 | **0.824** | 0.766 | −5.8 [−7.9, −3.6] |

- "有没有某物"（POPE）领先，日常是非题和常识推理略高（区间含 0）；教科书插图仍低 5.8 个点。差距集中在必须看示意图的自然科学题（323 题，0.455 对 0.700），其中磁铁题最多。
- 校准：VQAv2、A-OKVQA 上 NLL 和 ECE 都低于 laya-vision；ScienceQA 的 ECE 0.022（对方 0.028）。
- 其他差别：参数 518M 对 201M；输入 256 px 对 512 px；我们在 Mac CPU 上单题约 150 ms，对方在 L4 GPU 上 41 ms（硬件不同，不可直接比）。逐项分析见 [`docs/research/laya-vision-gap.md`](docs/research/laya-vision-gap.md)。

### 本项目的测试集

`data/v2/test_*`：我们自己出的题，按组平衡（只看题目答不出来），图片与所有训练、dev 数据不重叠（数据集 LookFirst，私有）。VSR、Visual7W 两行按图另行留出（`data/v5/test_*`）；新图计数是第 11 轮新建、从没参与训练的图（`data/v11/test_count_fresh.jsonl`）。*配错图*：同一批题换成别的图片后的准确率。

| 测试集 | 题数 | 准确率 | 配错图 | ECE |
|---|---|---|---|---|
| 存在（有没有某物） | 1,120 | 0.938 | 0.493 | 0.020 |
| VQAv2 多选（13 类） | 1,420 | 0.898 | 0.399 | 0.026 |
| 尺寸（哪个更大） | 1,100 | 0.860 | 0.504 | 0.036 |
| POPE（3 档合计，`bench_pope`） | 8,676 | 0.851 | 0.527 | 0.058 |
| GQA val | 992 | 0.784 | 0.532 | 0.042 |
| 位置（上下 0.909，左右 0.836） | 1,274 | 0.872 | 0.493 | 0.025 |
| VQAv2 是非 | 1,000 | 0.710 | 0.518 | 0.022 |
| 相对位置（上下 0.838，左右 0.672） | 1,950 | 0.711 | 0.484 | 0.047 |
| VSR 物体关系（按图留出，`test_vsr`） | 904 | 0.679 | 0.481 | 0.048 |
| Visual7W 四选一（按图留出，`test_v7w`） | 1,000 | 0.721 | 0.422 | 0.030 |
| 新图计数（`test_count_fresh`） | 600 | 0.740 | 0.507 | 0.065 |

配错图后都回到随机或只看题目的水平，说明分数来自看图。左右关系：同一道左右题在原图和左右镜像图上同时答对的比例，位置题 72%、相对位置选择题 65%（v11 为 61% / 39%）；相对位置是非题 19%，仍低于随机的 25%。

## 已知限制

- **按文字找到图里两处再比较还不会**：示意图、图表、地图上的比较题（ScienceQA 磁铁题、FigureQA、MapQA）接近随机；相对位置是非题在原图和镜像图上同时答对只有 19%。诊断见训练日志 v12 一节。
- **"有没有某物"偏向答"有"**：v0.2 比上一版更常把标注为不存在的物体答成存在（存在题 0.955 → 0.938，POPE adversarial 0.791）。
- 只支持英文、单张图、`noul` / `choice`（`score` 返回 422）。
- 输入 letterbox 到 256×256，图中小字和细节会丢失。
- 校准是在本项目的数据上拟合的，换到差别很大的数据上，概率可能不准。

## 模型结构

```mermaid
flowchart TB
    subgraph IN["请求 (Jev 格式)"]
        IMG["image（或 state 中的图片片段）<br/>任意尺寸"]
        TXT["state 文字背景 + 每个问题的<br/>instructions / criteria"]
    end

    IMG --> LB["Letterbox<br/>保持比例缩放 + 灰边补成 256×256<br/>归一化 (x−0.5)/0.5"]
    LB --> VIT["SigLIP2-B/16-256 视觉塔<br/>93M · 冻结<br/>→ 16×16 = 256 个 patch × 768"]
    VIT --> PS["2×2 Pixel Shuffle<br/>→ 8×8 = 64 个 token × 3072"]
    PS --> PROJ["投影层 (新训练)<br/>LayerNorm → Linear 3072→1024 → GELU → Linear 1024→1024"]

    TXT --> SEQ["Laya build_sequence<br/>每个问题一条序列"]

    PROJ -- "64 个视觉 token<br/>(同一张图只编码一次，按问题数复制)" --> CAT
    SEQ -- "token embedding" --> CAT["拼接: [CLS] + 视觉 token + 其余文本 token"]

    CAT --> ENC["ModernBERT-large 双向编码器<br/>28 层 · hidden 1024 · 用 Laya checkpoint 初始化<br/>训练时加 LoRA (Wqkv / Wo / Wi)，保存时合并"]
    ENC --> TE["+ 题型 embedding (choice / score / noul)"]
    TE --> HEAD["Laya 决策头<br/>2 层 TransformerEncoder"]
    HEAD --> GATHER["取出每个 [MASK] 位置的向量<br/>(每个选项一个)"]
    GATHER --> SC["Scorer<br/>LayerNorm → Linear → GELU → Linear→1<br/>每个选项一个 logit"]
    SC --> SM["softmax(logits / T)<br/>T 按（题型, 选项数）在校准集上拟合"]
    SM --> OUT["组装 Jev 答案<br/>noul: P(true)<br/>choice: choice / probabilities / confidence = (n·p_max−1)/(n−1)"]
```

### 编码器看到的序列

每个问题单独构成一条序列（下例是 `choice`，3 个选项）：

```
[CLS] V1 V2 … V64  choice question: Is the person standing or sitting? [SEP]
      └─ 视觉 ─┘   └──────────── 题型 + instructions ───────────┘
[MASK] standing  [MASK] sitting: on a chair or the ground  [MASK] lying  [SEP]  <state 文本>  [SEP]
  ↑ 选项 0         ↑ 选项 1                                  ↑ 选项 2
```

- `noul` 固定两个选项，顺序为 `[false, true]`，默认文本是 `false: no, the statement does not hold` 和 `true: yes, the statement holds`；`criteria` 可覆盖这两段描述。
- 文本部分由 Laya 的 `build_sequence` 生成，上限 512 token（选项区 192）。视觉 token 由模型插在 `[CLS]` 之后，并自动平移 `[MASK]` 位置。
- 纯文本请求（不带图）不插视觉 token，行为与 Laya 相同。
- 选项放不进输入时返回 422，不会悄悄丢掉选项。

### 参数与训练

| 模块 | 规模 | 来源 | 训练时 |
|---|---|---|---|
| SigLIP2-B/16-256 视觉塔 | 93M（含未用到的 pooling head） | `google/siglip2-base-patch16-256` | 冻结 |
| 投影层 | 4M | 随机初始化 | 全量训练 |
| ModernBERT-large 编码器 | 395M | `convaiinnovations/laya` | LoRA（保存时合并） |
| 决策头（题型 embedding + 2 层 Transformer + scorer） | 27M | `convaiinnovations/laya` | 全量训练 |
| **合计** | **518M** | | |

- 训练目标：RLCD，沿用 Laya 官方单卡脚本。它由两部分组成：对加了噪声的 logit 做策略梯度，奖励用 log、spherical 和 RPS 三种 proper scoring rule；再加上对 gold 分布的 soft 交叉熵。
- 训练结束后，按（题型, 选项数）在校准集上用 LBFGS 拟合温度 T，范围限制在 [0.5, 5]；没有分桶温度时用题型温度。
- 两个阶段：
  - **阶段 1 对齐**（`devision-align`）：带图完形填空，只训投影层（也可给 ModernBERT 加 LoRA）。数据是 COCO train2014 的 caption。
  - **阶段 2 决策**（`devision-train`）：RLCD，训投影层、决策头和 ModernBERT 的 LoRA。之后每一轮从上一轮的最佳 checkpoint 接着训，逐步加入 COCO 自动出题（存在、位置、大小）、VQAv2、GQA、A-OKVQA、ScienceQA、VSR、Visual7W、左右镜像对和计数。
- **v0.2 是第 12 轮**（[`configs/v12-ext.yaml`](configs/v12-ext.yaml)）：从第 11 轮接着训一遍 353,830 道题，其中 315,342 道是扩展数据包，来自 13 个公开数据集（Objects365、TallyQA、CLEVR、CLEVR-Math、Super-CLEVR、FigureQA、MapQA、IconQA、SNLI-VE、Vision-Flan、VisOnlyQA、SpatialSense、PixMo-Count），在 Mac（MPS）上约 16 小时。来源和许可见 [`docs/research/data-pack-sources-2026-10-06.md`](docs/research/data-pack-sources-2026-10-06.md)。
- 实验过程和结论（包括失败的尝试）见 [`docs/experiments/2026-09-30-rlcd-plateau.md`](docs/experiments/2026-09-30-rlcd-plateau.md)。

## 自己训练

先装 `devision[train]`（本仓库里 `uv sync` 已包含）。

### 先跑通一个小例子

```bash
uv run devision-pipeline configs/example.yaml    # CPU 上约 1–2 分钟；首次会下载基础模型（约 2.5 GB）
```

用 [`examples/train/`](examples/train/) 里的十几条示例数据跑一遍对齐、决策训练和评测，产出 `runs/example/`。数据格式见 [`examples/train/README.md`](examples/train/README.md) 和 `src/devision/train/samples.py`。

### 实验用 YAML 编排

每一轮训练是一个 YAML（`configs/*.yaml`）：`name`、若干阶段（`kind: align|train`、`init`、`data` / `val` / `eval`、训练参数）和评测集。新实验复制一份 YAML，改参数和 `name`，第 13 轮起命名为 `r<轮次>-<描述>`，输出在 `runs/<name>/`。再跑同一个 YAML 是续跑：已完成的阶段和评测会跳过，训练中断后从 `resume.pt` 接着训。

```bash
uv run devision-pipeline configs/r13-xx.yaml --dry-run    # 先检查参数名和数据文件
nohup uv run devision-pipeline configs/r13-xx.yaml > runs/r13-xx.out 2>&1 &
```

训练默认上报 SwanLab（先 `uv run swanlab login`；配置里 `swanlab_project: ""` 关闭）。

### 在新机器上从头准备数据和训练

```bash
uv sync
bash scripts/data/build_all.sh data       # 从原始来源生成数据：下载几十 GB、几个小时；日志同时写到 data/build_all.log
bash scripts/download_models.sh           # 可选：先下载 Laya、SigLIP2、ModernBERT 的权重
uv run devision-pipeline configs/scratch.yaml        # 从头训练：align -> stage2a -> stage2b，device: auto（CUDA > MPS > CPU）
# 单卡 A100：uv run devision-pipeline configs/scratch-a100.yaml
```

- `build_all.sh` 生成 `scratch.yaml` 和第 3–9 轮配置用到的数据，最后核对这些配置引用的数据文件和图片都存在且非空，缺了会列出来并返回非零。中断后重跑会跳过已下载的部分。
- 第 11 轮的计数集用 `uv run python scripts/data/v11_count_eval.py --root data`，第 12 轮的扩展数据包用 `bash scripts/data/ext/build_ext.sh data`（每个来源一个脚本，可断点续传，重跑不会重复下载）。
- 生成的数据和本机的规则、比例相同，不保证逐条一致，也不会自动复现发布版的权重或分数。
- `scratch.yaml` 只训练到大致相当于第 5 轮的模型（输出 `runs/scratch/stage2b`）。后面各轮的配置是历史实验记录，`init` 指向本机的 `runs/` 路径，有的写死 `device: mps`；要在新机器上接着训，复制一份，把 `init` 改成实际的 checkpoint、`device` 改成 `auto` 或 `cuda`。数据目录不是 `data/` 时，配置里的数据路径也要一起改。

## 代码结构

```
src/devision/
├── __init__.py   对外入口 devision.load(...)（import devision 不加载 torch）
├── model.py      网络结构：SigLIP2 → 投影层 → ModernBERT + Laya 决策头
├── image.py      图片读取与 letterbox 预处理
├── infer.py      Decider：checkpoint 读写、decide / predict（推理入口）
├── serve.py      POST /v1/systemone 和 web demo 路由；命令 devision-serve
├── static/       web demo 页面
└── train/        训练、评测、校准、实验流水线（需 devision[train]）
configs/          每一轮训练的 YAML
scripts/data/     生成训练和评测数据的脚本（不随包分发）
scripts/release/  发布到 Hugging Face 的工具（见 release/README.md）
examples/         demo 示例图；examples/train/ 是训练数据格式示例
docs/             设计、调研、实验记录、训练日志
```

依赖只能单向：`train` 和 `serve` 建立在推理代码（`model` / `image` / `infer`）之上，推理代码不引用它们；demo 页面只通过 HTTP 调用 API。

## 许可

代码和模型权重用 [Apache-2.0](LICENSE)，与三个基座模型（Laya、SigLIP2、ModernBERT）一致。训练数据来自各公开数据集，各有条款，部分为非商用（如 ScienceQA 为 CC BY-NC-SA 4.0）；这些条款是否延伸到训练出的权重尚无定论，请按你的用途自行核对模型卡里列出的数据集。
