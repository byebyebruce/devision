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
- 推理只跑 **CPU**；训练在本地 Mac（MPS）。
- 当前仅：英文、单图、letterbox 到 256×256、题型 `noul`/`choice`（`score` 返回 422）。
- **非商用项目**：可用 laya-vision（CC BY-NC-SA）做 baseline；许可相关改动需先确认。

## 测试

- 只从 seam 测外部行为：仓库内是 `decide(state, questions) → answers`；数据转换器的测试在 `scripts/data/`（`uv run pytest scripts/data`）。不测张量形状/层结构。
- HTTP 层是 `decide` 的薄封装，不单测。
- 端到端冒烟（`tests/train/test_pipeline.py`）：tiny 模型 + 合成红/蓝图样本 → 训练 → 存盘 → 加载 → 经 decide 评测，CPU 上几秒，随 `pytest` 一起跑。真实模型的同一流程用下面的 train → eval 命令（训练在 Mac 上，`--device mps`）。
- 评测集图片按 image id 从所有训练源剔除；VG（GQA）里约一半是 COCO 图，剔除时两套 id 都要对上（`convert.same_images`，映射来自 VG 的 `image_data.json`）。
- v2 数据的图片用途互不重叠：训练文件、`dev_*`（选模型和拟合温度，按图片哈希约 3%）、`eval_*`（最终测试，COCO val2014，同时避开 v1 和 v2 的训练图片）。报告准确率时同时给出 MANIFEST 里的「只看题目」基线和配错图对照（YAML `evaluate.controls: true`）。
- 评测集：`val_mix`（GQA testdev 1000 + VQAv2 val 1000，用于 `--val` 与拟合温度，指标按来源拆分）、`pope`（300，只评不拟合）。旧的 `val.jsonl`（200 题）只为和早期实验对比而保留。

## Python 与依赖

- 用 **uv** 管理 Python 与依赖（Python 版本见 `.python-version`）。加依赖用 `uv add <pkg>`（开发依赖 `uv add --dev`），不要用 pip 或手改 `uv.lock`。
- 所有命令经 `uv run ...` 执行。
- 依赖分层（参照 Laya）：核心依赖只够推理；`serve`（fastapi、uvicorn）和 `train`（peft、swanlab）是 extras，`model` 包不能 import 它们。开发依赖组也包含这些 extras，所以 `uv run` 下全部可用；新增服务或训练依赖时，`uv add --optional <extra>` 之外再 `uv add --dev` 一次。
- 对外入口：`devision.load(...)` → `Decider.predict` / `decide`；`import devision` 不加载 torch。

## 命令

```bash
uv run pytest -q                      # 全部测试（含 tiny 模型端到端冒烟，CPU 上几秒）
uv run pyright                        # 类型检查
uv run python scripts/data/prepare.py fetch --root data ...   # 仓库外：生成 data/{train,val,pope}.jsonl
uv run python scripts/data/captions.py --root data --exclude data/val.jsonl data/pope.jsonl   # 仓库外：生成 data/align_{train,val}.jsonl（COCO caption）
uv run python scripts/data/evalsets.py --root data --gqa 1000 --vqav2 1000   # 仓库外：生成 data/val_{gqa,vqav2,mix}.jsonl，并从 train.jsonl 剔除评测图
uv run python scripts/data/bigtrain.py --root data --gqa 60000 --vqav2 40000 --out data/train_100k.jsonl   # 仓库外：更大的阶段 2 训练集
uv run python scripts/data/cocoqa.py --root data --split train2014 --limit 60000 --out data/cocoqa_train.jsonl   # 仓库外：从 COCO 实例框出题（val2014 + --out data/val_cocoqa.jsonl 是评测集）
uv run python scripts/data/v2_build.py --root data   # 仓库外：一条命令重建 v2 数据集（data/v2/：训练、dev_*、eval_*、MANIFEST.json），任何验收失败即中止；见 docs/research/data-quality.md
uv run devision-align --data data/align_train.jsonl --val data/align_val.jsonl --data-root data --out runs/align --run-name align   # 阶段 1
uv run devision-train --init runs/align --lr-new 5e-5 --lr-head 5e-5 --lr-lora 1e-4 --warmup 500 --data data/train_100k.jsonl --val data/val_mix.jsonl --data-root data --eval pope=data/pope.jsonl --out runs/x --run-name x   # 阶段 2；Mac 上加 --device mps
uv run devision-eval --checkpoint runs/x --data data/pope.jsonl --data-root data --out runs/x/pope.json   # --shuffle-images：配错图对照
uv run devision-serve --checkpoint runs/x --port 8000   # POST /v1/systemone；浏览器打开 / 是 web demo（--no-demo 关闭）
uv run devision-pipeline configs/x.yaml [--dry-run] [--from STAGE] [--force]   # 按 YAML 跑一整个实验
```

- **实验用 YAML 编排**（`configs/*.yaml`，`src/devision/train/pipeline.py`）：每个新实验复制一份 YAML 改参数，不要再写训练脚本。一个 YAML = `name` + 若干 `stages`（`kind: align|train`、`init` 指向前面的阶段或路径、`data` / `val` / `eval`、`params` 即训练 CLI 的参数名，`common` 是所有阶段共用的参数）+ `evaluate`（评测集和额外命令，`{checkpoint}` / `{name}` 会被替换）。阶段 N 输出到 `runs/<name>-N/`，日志 `runs/logs/<name>-N.log`，SwanLab run 名 `<name>-N`。启动前检查所有参数名、类型和数据文件；已完成的阶段自动跳过（`--from` 从某阶段起重跑）。长任务用 `nohup uv run devision-pipeline configs/x.yaml > runs/logs/x.log 2>&1 &`。

- `data/`（数据集）、`runs/`（checkpoint）不进 git；`examples/` 里的少量示例图进 git，供 web demo 默认加载；`examples/train/` 是格式示例数据（含合成形状图），配 `configs/example.yaml` 端到端跑通，改样本格式时要同步更新它。
- 训练默认上报 SwanLab（项目 `devision`），两个阶段同一套分组（`src/devision/train/tracking.py`），每 `--log-every` 步一行：`train/`（窗口内均值：loss、accuracy、grad_norm；阶段 2 另有 nll、sigma）、`lr/`（各参数组）、`perf/`（step_s、samples_per_s、progress_pct、eta_h）、`sys/`（cpu_pct、proc_cpu_pct、ram_used_pct、ram_available_gb、swap_used_gb、proc_rss_gb、gpu_util_pct、gpu_mem_gb、torch_mps_gb）。评测行另记：阶段 1 每 `--eval-every` 步记 val 上真实/错配图片的补词准确率与 nll 及两者之差（`val/mlm_acc_gap`）；阶段 2 从第 0 步起每 `--eval-every` 步和每个 epoch 末，在 val 和每个 `--eval` 集上记准确率（总体/noul/choice，混合来源时按来源拆分）、nll、ECE，结束时记拟合温度和套用温度后的指标（`final/...`）。只有训练（`devision-align` / `devision-train`）上报；`devision-eval`、分析脚本和测试都不上报，结果写成 JSON / 文本文件。先 `uv run swanlab login` 登录（API key 只存在本机用户目录，**不要写进仓库**）；`--swanlab-project ""` 关闭。本地缓存 `swanlog/` 不进 git。
- 首次训练会从 Hub 下载 `convaiinnovations/laya` 与 `google/siglip2-base-patch16-256`；对齐阶段还会下载 `answerdotai/ModernBERT-large`（只取 MLM 头）。
- 不做阶段 1 直接训决策题，nll 会停在 ln2（见 `docs/experiments/2026-09-30-rlcd-plateau.md`）。
- 阶段 2 用默认学习率长训练（上万步）会塌缩成 50/50 输出；用上面命令里的低学习率加预热。发布的 checkpoint 怎么训出来的见 `configs/release-0.1.yaml`。
- `--checkpoint` / `--init` / `Decider.load` 都接受本地目录或 Hugging Face 模型仓库 id。
