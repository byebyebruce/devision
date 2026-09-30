# devision

视觉决策模型：图片 + 英文问题 → Jev 格式的结构化答案（带校准概率）。SigLIP2 视觉编码 + Laya 初始化的 ModernBERT-large 决策头。

- 设计与范围：`docs/spec/vision-decision-mvp.md`（改设计先改 spec）
- 调研：`docs/research/jev-api.md`（Jev 协议）、`docs/research/encoder-data.md`（编码器与数据集）

## 不可违背的约束

- **API 与 Jev `/v1/systemone` 兼容**：唯一扩展是 `state` 数组可含 `{type:"image", base64|url}`。响应结构不得偏离 Jev。
- **`confidence = (n·p_max−1)/(n−1)`**（Jev 口径），不用 Laya 的熵式定义。
- 推理只跑 **CPU**；训练 1×A100。
- 当前仅：英文、单图、letterbox 到 256×256、题型 `noul`/`choice`（`score` 返回 422）。
- **非商用项目**：可用 laya-vision（CC BY-NC-SA）做 baseline；许可相关改动需先确认。

## 测试

- 只从两个 seam 测外部行为：`decide(state, questions) → answers` 和数据转换器（原始记录 → Jev 格式样本）。不测张量形状/层结构。
- HTTP 层是 `decide` 的薄封装，不单测。
- 端到端冒烟（`tests/test_pipeline.py`）：tiny 模型 + 合成红/蓝图，转换 → 训练 → 存盘 → 加载 → 经 decide 评测，CPU 上几秒，随 `pytest` 一起跑。真实模型的同一流程用下面的 fetch → train → eval 命令（训练在 A100 上）。
- 评测集图片按 image id 从所有训练源剔除（COCO/VG 跨数据集泄漏）。

## Python 与依赖

- 用 **uv** 管理 Python 与依赖（Python 版本见 `.python-version`）。加依赖用 `uv add <pkg>`（开发依赖 `uv add --dev`），不要用 pip 或手改 `uv.lock`。
- 所有命令经 `uv run ...` 执行。

## 命令

```bash
uv run pytest -q                      # 全部测试（含 tiny 模型端到端冒烟，CPU 上几秒）
uv run pyright                        # 类型检查
uv run devision-fetch --root data     # 拉 MVP 小数据集：GQA 训练切片 / GQA testdev / POPE → data/*.jsonl
uv run devision-convert ...           # 官方 GQA / VQAv2 / POPE 文件 → 平衡后的 JSONL（--exclude 剔除评测图）
uv run devision-train --data data/train.jsonl --val data/val.jsonl --data-root data --out runs/x   # A100 上 --device cuda
uv run devision-eval --checkpoint runs/x --data data/pope.jsonl --data-root data --out runs/x/pope.json
uv run devision-serve --checkpoint runs/x --port 8000   # POST /v1/systemone
```

- `data/`、`runs/` 不进 git。
- 首次训练会从 Hub 下载 `convaiinnovations/laya` 与 `google/siglip2-base-patch16-256`。
