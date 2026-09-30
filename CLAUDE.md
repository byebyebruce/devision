# deVision (Decision + Vision)

视觉决策模型：图片 + 英文问题 → Jev 格式的结构化答案（带校准概率）。SigLIP2 视觉编码 + Laya 初始化的 ModernBERT-large 决策头。

- 设计与范围：`docs/spec/vision-decision-mvp.md`（改设计先改 spec）
- 调研：`docs/research/jev-api.md`（Jev 协议）、`docs/research/encoder-data.md`（编码器与数据集）

## 代码结构

- `src/devision/model/`：网络、预处理、checkpoint、`Decider.decide`；不依赖其他子包。
- `src/devision/train/`：样本格式（`samples.py`）、阶段 1 对齐（`align.py`，带图完形填空，只训投影层）、阶段 2 RLCD 训练（`rlcd.py`）、评测，依赖 model。
- 准备训练数据的代码（下载、转换、转换器测试）**不进仓库**：放在 `scripts/data/`（已 gitignore）。仓库只认 `samples.py` 里约定的 JSONL 格式。
- `src/devision/serve/`：`/v1/systemone` API，依赖 model。
- `src/devision/demo/`：web demo 页面和示例图路由，只通过 HTTP 调 API，不 import model、train、serve。
- 依赖只能单向，不要让 model 反向引用 train/serve/demo，也不要让 train、serve、demo 互相引用。
- 测试按包放：`tests/model/`、`tests/train/`；共享的 tiny 模型在 `tests/conftest.py`。

## 不可违背的约束

- **API 与 Jev `/v1/systemone` 兼容**：唯一扩展是 `state` 数组可含 `{type:"image", base64|url}`。响应结构不得偏离 Jev。
- **`confidence = (n·p_max−1)/(n−1)`**（Jev 口径），不用 Laya 的熵式定义。
- 推理只跑 **CPU**；训练 1×A100。
- 当前仅：英文、单图、letterbox 到 256×256、题型 `noul`/`choice`（`score` 返回 422）。
- **非商用项目**：可用 laya-vision（CC BY-NC-SA）做 baseline；许可相关改动需先确认。

## 测试

- 只从 seam 测外部行为：仓库内是 `decide(state, questions) → answers`；数据转换器的测试在 `scripts/data/`（`uv run pytest scripts/data`）。不测张量形状/层结构。
- HTTP 层是 `decide` 的薄封装，不单测。
- 端到端冒烟（`tests/train/test_pipeline.py`）：tiny 模型 + 合成红/蓝图样本 → 训练 → 存盘 → 加载 → 经 decide 评测，CPU 上几秒，随 `pytest` 一起跑。真实模型的同一流程用下面的 train → eval 命令（训练在 A100 上）。
- 评测集图片按 image id 从所有训练源剔除（COCO/VG 跨数据集泄漏）。

## Python 与依赖

- 用 **uv** 管理 Python 与依赖（Python 版本见 `.python-version`）。加依赖用 `uv add <pkg>`（开发依赖 `uv add --dev`），不要用 pip 或手改 `uv.lock`。
- 所有命令经 `uv run ...` 执行。

## 命令

```bash
uv run pytest -q                      # 全部测试（含 tiny 模型端到端冒烟，CPU 上几秒）
uv run pyright                        # 类型检查
uv run python scripts/data/prepare.py fetch --root data ...   # 仓库外：生成 data/{train,val,pope}.jsonl
uv run python scripts/data/captions.py --root data --exclude data/val.jsonl data/pope.jsonl   # 仓库外：生成 data/align_{train,val}.jsonl（COCO caption）
uv run devision-align --data data/align_train.jsonl --val data/align_val.jsonl --data-root data --out runs/align --run-name align   # 阶段 1
uv run devision-train --init runs/align --lr-new 1e-4 --data data/train.jsonl --val data/val.jsonl --data-root data --out runs/x --run-name x   # 阶段 2；A100 上 --device cuda，Mac 上 --device mps
uv run devision-eval --checkpoint runs/x --data data/pope.jsonl --data-root data --out runs/x/pope.json
uv run devision-serve --checkpoint runs/x --port 8000   # POST /v1/systemone；浏览器打开 / 是 web demo（--no-demo 关闭）
```

- `data/`（数据集）、`runs/`（checkpoint）不进 git；`examples/` 里的少量示例图进 git，供 web demo 默认加载。
- 训练默认上报 SwanLab（项目 `devision`）：loss / nll / 学习率 / 每 epoch 的 val 准确率 / 拟合温度。先 `uv run swanlab login` 登录（API key 只存在本机用户目录，**不要写进仓库**）；`--swanlab-project ""` 关闭。本地缓存 `swanlog/` 不进 git。
- 首次训练会从 Hub 下载 `convaiinnovations/laya` 与 `google/siglip2-base-patch16-256`；对齐阶段还会下载 `answerdotai/ModernBERT-large`（只取 MLM 头）。
- 不做阶段 1 直接训决策题，nll 会停在 ln2（见 `docs/experiments/2026-09-30-rlcd-plateau.md`）。
