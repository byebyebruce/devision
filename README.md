# deVision

**deVision = Decision + Vision**：看图做决策。名字取 **De**cision 的开头和 **Vision** 的全部，两个词共享同一个 `-sion` 结尾，拼起来正好是 “deVision”。
- **Vision**：看图，由 SigLIP2 视觉编码器负责。
- **Decision**：做决策，由 Laya 的 ModernBERT 和决策头负责。它不生成文字，而是直接给出每个选项的校准概率。

它相当于一个能看图的 Jev：输入一张图片和若干英文问题，输出 Jev 格式的结构化答案（`noul` / `choice`）及校准概率。接口兼容 Jev `/v1/systemone`，`state` 里可以放 `{"type": "image", ...}`。

设计与范围见 [`docs/spec/vision-decision-mvp.md`](docs/spec/vision-decision-mvp.md)，常用命令见 [`CLAUDE.md`](CLAUDE.md)。

## 使用

给一张图和几道英文题，返回每道题的答案和校准过的概率。可以起 HTTP 服务，也可以在 Python 里直接调用。请求和响应格式与 Jev `/v1/systemone` 相同，唯一的扩展是 `state` 里可以放一张图。

### 安装

```bash
pip install "devision[serve] @ git+https://github.com/byebyebruce/devision"   # 推理 + HTTP 服务
pip install "devision @ git+https://github.com/byebyebruce/devision"          # 只在 Python 里推理
```

| 安装项 | 包含 |
|---|---|
| `devision` | 推理：`devision.load`、`Decider.predict` / `decide`、`devision-eval` |
| `devision[serve]` | 加上 HTTP 服务和 web demo（`devision-serve`） |
| `devision[train]` | 加上训练（`devision-align`、`devision-train`，依赖 peft、SwanLab） |

在本仓库里开发用 `uv sync`，开发依赖组已经包含上面所有内容。

### 在 Python 里调用

```python
import devision

decider = devision.load("runs/v1-release-0.1/stage2b")   # 本地 checkpoint 目录，或 Hugging Face 仓库 id
# devision.load("user/repo", revision="v0.1", token="hf_...")   # 固定版本 / 私有仓库

result = decider.predict(            # 和 decider.decide(...) 完全相同，predict 是 Laya 的叫法
    state=[{"type": "image", "url": "https://example.com/kitchen.jpg"}],
    questions={
        "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
        "room": {"type": "choice", "instructions": "Which room is this?",
                 "criteria": {"kitchen": None, "bathroom": "has a sink and a toilet", "bedroom": None}},
    },
)
```

请求不合法时抛出 `devision.InvalidRequest`（HTTP 服务里对应 422）。

### 启动 HTTP 服务

```bash
devision-serve --checkpoint runs/v1-release-0.1/stage2b --port 8000
```

| 参数 | 作用 |
|---|---|
| `--checkpoint` | 本地目录或 Hugging Face 仓库 id |
| `--revision` | 仓库 id 时固定的 commit、tag 或分支 |
| `--host` / `--port` | 默认 `127.0.0.1:8000` |
| `--threads` | CPU 推理线程数 |
| `--no-demo` | 只开接口，不挂 web demo |

接口：

- `POST /v1/systemone`：做决策，见下文。
- `GET /health`：返回 `{"status": "ok", "model": "devision-0.1"}`。
- `GET /`：web demo，可以上传图片、填题目、看概率（`--no-demo` 关闭）。

只跑 CPU，Mac 上单题大约 0.2 秒；同一请求里的多道题共用一次图片编码。

### 请求

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": [
    {"type": "image", "url": "https://example.com/kitchen.jpg"},
    "customer says the order arrived damaged"
  ],
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

- **`state`**：字符串、对象或数组。数组里最多一张图：`{"type": "image", "url": "https://..."}` 或 `{"type": "image", "base64": "..."}`；其余元素是文字背景信息。不带图时等同于纯文本的 Laya。
- **`questions`**：对象，键是你自己起的题目 id，可以一次问多道。
  - `noul`：是非题，`instructions` 写要判断的陈述或问题；`criteria` 可选，用 `{"false": "...", "true": "..."}` 改写两个选项的含义。
  - `choice`：单选题，`criteria` 的键是选项（2–255 个），值是 `null` 或一句说明，帮模型区分相近的选项。
  - `score` 不支持，返回 422。

### 响应

```json
{
  "model": "devision-0.1",
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

### 怎么用得好

**用概率做阈值，没把握的转给人或更强的模型。** 概率经过按题型的温度校准，但校准程度因题而异：在拟合温度用的 GQA / VQAv2 题上 ECE 约 0.03–0.04，在没参与拟合的 POPE 和 COCO 题上是 0.098 和 0.054。阈值请在自己的数据上确定并验证，不要假设 "说 80% 就八成对" 对所有问题都成立。

```python
p = result["answers"]["has_fork"]["noul"]
if p > 0.9:
    handle_yes()
elif p < 0.1:
    handle_no()
else:
    escalate()   # 人工，或者更强的模型（系统 2）
```

阈值请在自己的数据上定：看不同阈值下自动处理的比例和错误率。

**适合问的：**

- 图里有没有某样东西：有没有人、有没有破损、有没有某种物品。
- 是什么、属于哪类：房间类型、动物种类，选项不宜太多。
- 整体判断：室内还是室外、东西看起来是不是坏了。

**不要问的：**

- 某个东西在哪里、在另一个东西的哪一边。这类题现在是随机水平。
- 数数、读图中的小字（输入只有 256×256）。
- 非英文问题、一次多张图、`score` 打分题。

**题目写清楚，选项互不重叠。** 选项意思接近时在 `criteria` 里加一句说明。

## Benchmark

deVision 的数字来自发布的 checkpoint（`stage2-cocoqa`），全部经 `decide` 在 CPU 上评测，已套用拟合温度。laya-vision 的数字取自其公开评测页和模型卡（检索于 2026-10-01）；它是目前唯一公开的、能看图的 System One 模型。

### 与 laya-vision 对比

| 评测集 | deVision 0.1 | laya-vision 201M | laya-vision ModernVBERT-250M |
|---|---|---|---|
| POPE adversarial（"有没有 X"） | **0.773**（300 题） | **0.777**（3,000 题） | — |
| POPE popular / random | 未测 | 0.819 / 0.836 | — |
| VQAv2 yes/no（val 按图重切） | 0.648（1,000 题） | **0.715**（5,000 题） | **0.718**（5,000 题） |
| VSR（空间关系） | 未测；左右类题在随机水平 | **0.875**（160 题） | — |
| A-OKVQA（四选一） | 未测 | 0.598 | 0.652 |
| ScienceQA（图片子集） | 未测 | 0.822 | 0.790 |
| ECE | 拟合集上 0.025–0.042；未参与拟合：POPE 0.098、COCO 0.054 | 0.041（34 个集合）；POPE 0.048 | 0.022（22 个集合） |
| 参数 | 518M | 201M | 267M |
| 输入分辨率 | 256 px | 512 px | 512 px |
| 延迟（单题） | 160–190 ms，Mac **CPU**（历史测量，未记录线程数，非服务器） | 40.8 ms，L4 **GPU** | 32 ms，L4 GPU |
| 训练数据 | 约 18 万题 + 8.3 万张图的 caption | 182 万条 | 27 万题 |
| 训练硬件 | Mac（MPS），约 16 小时 | 1×H100，2 小时 | 1×A100，68 分钟 |
| 权重许可 | 待定 | CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 |

- POPE adversarial 0.773 对 0.777，**两边接近**；日常是非题低约 7 个点；空间关系差距最大；POPE 上我们的校准偏差（温度没在这类题上拟合）。
- 这些不是同一批题上的比较，没有做配对检验，不能据此判断谁显著更好：POPE 我们只测了 300 题，对方 3,000 题；两边的 VQAv2 都从官方 val 按图重切，但抽到的题不同。延迟一边是 CPU、一边是 GPU。我们自己的模型之间在同一批题上比较时，用 `devision-compare`（按图片配对重采样，给出差值的 95% 区间）。
- 差距主要来自：我们的 ModernBERT 只在纯文本上预训练过、连接层从零训练、只训 LoRA，而 laya-vision 的骨干在预训练时已经学过看图；以及训练数据少一个数量级、题型窄。逐项分析见 [`docs/research/laya-vision-gap.md`](docs/research/laya-vision-gap.md)。

### 本项目的评测集

| 评测集 | 用途 | 题数 | 准确率 | ECE |
|---|---|---|---|---|
| POPE adversarial | 未参与拟合 | 300 | 0.773 | 0.098 |
| VQAv2 val yes/no（正负各半） | **温度拟合集** | 1,000 | 0.648 | 0.025 |
| GQA testdev（是非 + 二选一） | **温度拟合集** | 1,000 | 0.635（是非 0.639，选择 0.628） | 0.042 |
| COCO 实例框出题（val2014） | 未参与拟合 | 1,000 | 0.761 | 0.054 |

温度是在 VQAv2 + GQA 这 2,000 题（`val_mix`）上拟合的，这两行的 ECE 不是未见数据上的结果。之后的评测用 v2 测试集（`data/v2/eval_*`：不含任何 v1 / v2 训练图片），并同时给出配错图和选项倒序两个对照。是否算 "未见过" 取决于被评的模型：例如对 v1 模型，`eval_pope` 和它拟合温度用的 `val_mix` 共享 10 张图、和训练中监测过的旧 POPE 共享 234 张图，只能当回归参考；评测流水线按题目和图片的实际重叠给每个集合标用途（calibration_fit / monitoring / heldout）。

COCO 出题按题型：有没有某物 0.920，大小 0.908，上下 0.814，左右 0.491（随机水平）。上下和大小的成绩大部分来自类别先验（只看类别不看图分别能到 0.757 和 0.844）。

## 已知限制

- **分不清 "文字说的那个东西在哪"**：左右类问题（"Is the cup to the left of the laptop?"）在随机水平。上下、大小题的成绩大部分来自类别先验。诊断见实验记录。
- 只支持英文、单张图、`noul` / `choice`（`score` 返回 422）。
- 输入 letterbox 到 256×256，图中小字和细节会丢失。

## 代码结构

```
src/devision/
├── model/     模型：网络结构、图片预处理、checkpoint 读写、Decider.decide（推理入口）
├── train/     训练：样本格式、对齐（阶段 1）、RLCD 训练（阶段 2，SwanLab 可视化）、评测；命令 devision-align / train / eval
├── serve/     服务：POST /v1/systemone API（CPU）；命令 devision-serve
└── demo/      Web demo：页面与示例图路由，由 devision-serve 挂载（--no-demo 可关）
examples/      demo 默认加载的示例图
```

依赖只能单向：`train`、`serve`、`demo` 都建立在 `model` 之上，三者之间互不引用。`model` 不依赖其他子包；`demo` 只通过 HTTP 调用 API，不引用任何 Python 代码。

## 模型结构

```mermaid
flowchart TB
    subgraph IN["请求 (Jev 格式)"]
        IMG["state 中的图片<br/>任意尺寸"]
        TXT["state 中的文本 + 每个问题的<br/>instructions / criteria"]
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
    SC --> SM["softmax(logits / T[题型])<br/>T 在 val 集上拟合"]
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
- 训练结束后，按题型在 val 集上用 LBFGS 拟合温度 T，范围限制在 [0.5, 5]。
- 训练分两个阶段：
  - **阶段 1 对齐**（`devision-align`）：带图完形填空，只训投影层。数据是 COCO train2014 全部 caption。
  - **阶段 2 决策**（`devision-train`）：RLCD，训投影层、决策头和 ModernBERT 的 LoRA。先在 6 万 GQA + 4 万 VQAv2 是非题上训，再在从 COCO 实例框自动出的题（有没有某物、位置、大小）上接着训。
  - 发布的 checkpoint 的完整步骤见 [`configs/v1-release-0.1.yaml`](configs/v1-release-0.1.yaml)，在 Mac（MPS）上共约 16 小时（在已有 `runs/v1-release-0.1/` 的机器上再跑是续跑，会跳过已完成的阶段）。新实验复制一份 YAML，改参数和 `name`（如 `v3-xx`，即 `runs/v3-xx/`）。
- 训练数据是 JSONL 样本，格式见 `src/devision/train/samples.py`；[`examples/train/`](examples/train/) 里有每种写法的示例数据，`uv run devision-pipeline configs/example.yaml` 能在 CPU 上一两分钟内跑通整个流程。生成数据的代码（GQA / VQAv2 / POPE / COCO 的下载与规则转换）不在仓库里。所有评测图片按 COCO 和 VG 两套 id 从训练数据中剔除。
- 实验过程和结论见 [`docs/experiments/2026-09-30-rlcd-plateau.md`](docs/experiments/2026-09-30-rlcd-plateau.md)。
