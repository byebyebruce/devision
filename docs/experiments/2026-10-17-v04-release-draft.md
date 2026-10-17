# v0.4 完整评测与发布草案（候选未锁定，成绩未填）

日期：2026-10-17。依据：交付计划 `docs/experiments/2026-10-17-v04-delivery-plan.md` 中的"发布证据准备"一节和 `release/README.md`。本文只写步骤、命令和要核的证据。候选 checkpoint 与全部成绩都留待锁定后填写。exp32 完成前不启动任何额外的模型作业，也不提前发布。

## 1. 锁定候选（评论家与 Engineer 共同确认后填写）

| 项 | 内容 |
|---|---|
| 候选 | TBD（暂定 `runs/exp32-fit/stage2`） |
| 来源与步数 | TBD：起点链 r15 → exp30 对齐 → exp32 A2 → exp32 B（实际步数） |
| 数据版本 | real_layout MANIFEST（questions_train `2adf8dd0…`），exp32 train_mix `f5ba82c7…`，align_mix `da011a1b…`，exp30 align_mix `a43037e3…` |
| 权重指纹 | TBD：`model.safetensors` 的 sha256 |
| 交付判定 | TBD：exp32 report 的结局，以及 transfer 的准确率 / 看图收益区间 |
| 代价 | TBD：7 个保护面板 |

如果候选换成别的 checkpoint，只改 `configs/v04-full-eval.yaml` 的 `checkpoint` 和 `release/v0.4/release.yaml` 的 `round` / `stage` 两处。

## 2. 完整评测（29 个集合，与 v0.3 相同）

- 配置 `configs/v04-full-eval.yaml`：29 个集合、对照（配错图 / 选项倒序）、温度拟合集（`data/r13b/val.jsonl`）和用途判定都与 `configs/r15-vision.yaml` 相同。
- `history` 另外加入候选训练时监测过的集合：real_layout 调参集、exp21 的 4 个 ScienceQA 开发集和 dev16。
- 两份配置给 29 个集合判出的用途逐一相同（`build_plan` 已比对）。
- 每题都能与 `runs/r15-vision/eval/` 配对。
- 结果写到候选轮次的 `eval/`。注意：dry-run 时 `runs/exp32-fit/run.json` 还不存在，输出路径显示为 `stage2/`；exp32 训练后就会是 `runs/exp32-fit/eval/`。
- **复用已有结果前**，先跑 `uv run python scripts/release/check_eval.py configs/v04-full-eval.yaml`，对每个集合及其每个对照逐一核对：
  - checkpoint 是否为配置里的候选；
  - 是否用拟合后的温度、对照类型是否正确、用途是否一致；
  - 数据文件当前的 sha256；
  - 逐题明细是否完整、无重复（倒序对照只覆盖选择题）。
  - 只有文件存在不算。exp32 chain 已经评过的 7 个保护面板，只要核对通过就可以复用，对照会补跑。
- 用 v0.3 自己的结果试过这个检查：29 个集合加全部对照，0 个问题。
- 评完后：
  - `scripts/lv_report.py <eval> --prev runs/r15-vision/eval`，生成 laya-vision 详细比较，包括 ScienceQA 按学科 / 选项数 / 必须看图的切片；
  - 每个集合跑 `devision-compare` 对 v0.3 逐题配对；
  - 写入训练日志和 `docs/research/laya-vision-gap.md`。
- 用时：v0.3 的同一套评测实测约 2 小时 16 分钟（14 日 18:06–20:22），按 2.5 小时安排。

## 3. 交付证据（`release/v0.4/delivery_vs_v0.3.json`）

- 主目标：直接取 exp32 report 的 transfer 读出（真实 ScienceQA 温度 + 磁力开发题，68 : 114，准确率区间和看图收益区间），以及 7 个保护面板，不重算、不改口径。
- 泛化证据：
  - real_layout 确认集：冻结、按规范案例隔离，只评一次，按族配对；
  - 全量 `bench_lv_scienceqa` 及其"必须看图"切片。
- 交付计划里的"新冻结泛化集"是否就以确认集充当，请评论家确认。确认集和训练版式相似，这一点会如实披露。

## 4. 概率质量

- 候选的温度是在 B 阶段结束时于 `data/r13b/val.jsonl` 上拟合的，与 v0.3 同一来源；没有在任何最终测试集上拟合。
- 校准前后对比：`uv run devision-calibrate --checkpoint <候选> --data data/r13b/val.jsonl --data-root data`，只打印、不写盘，给出各桶校准前后的 NLL / ECE。
- 报告完整评测里每个集合的 accuracy / NLL / ECE，以及按选项数分层。

## 5. CPU 延迟（候选自己测，不复用 v0.3 的数字）

```bash
OMP_NUM_THREADS=4 uv run devision-eval --checkpoint <候选> --data data/r12/cpu_latency.jsonl --data-root data \
  --out <候选轮次>/cpu/single.json --device cpu --no-group
cp <候选轮次>/cpu/single.json release/v0.4/cpu_latency.json
```

- 与 v0.3 用同一个 200 题夹具，4 线程，空闲机器。v0.3 是 P50 174.9 / P95 190.0 ms。
- 结构没有变（256 输入、64 个视觉 token），预计相近，以实测为准。
- 不在 exp32 或完整评测运行时测，避免抢占资源。

## 6. 打包与验证

1. `uv run python scripts/release/hf.py build release/v0.4`：生成 `runs/release/v0.4/`，并写回 results / MANIFEST / smoke。
2. `uv run python scripts/release/hf.py verify release/v0.4`：核对文件哈希和模型卡 metadata，并在 CPU 上与源 checkpoint 的答案一致。这一步只验证打包；能力证据以第 2–4 节为准。
3. 从临时 checkout 用当前 PyPI 版本（v0.2.1）加载 v0.4，在 CPU 上比对 smoke 答案，验证向后兼容（同 v0.3）。
4. `uv run --with modelscope-hub python scripts/release/ms.py build release/v0.4`。
5. `uv run pytest -q` 和 `uv run pyright`。
6. 写 `release/v0.4/validation.md`：只记录实际执行过的检查和结果。
7. 模型卡里的数字都由 build 从 `runs/<候选>/eval/` 读出；`notes.md` 里的 TBD 全部按实测替换，不复制 v0.3 的数字。

推理代码不变时不发新的 PyPI 包；模型版本和包版本分开。

## 7. 发布（双方都满意后执行，负责人的冲刺授权已覆盖）

1. 评审通过的 `develop` 合入 `master`，隔离无关的工作区改动（俄罗斯方块 / 贪吃蛇 demo、README、AGENTS.md、serve.py、static 等未提交改动都不带入）。
2. `git tag -a model-v0.4`，推送 master 和 tag。
3. Hugging Face：
   - 先 `hf.py publish release/v0.4` 演练，再加 `--push`。已有的版本 tag 不覆盖，v0.3 保留。
   - 发布后 `hf download lukbit/devision --revision v0.4` 到临时目录，再 `verify --folder`，核对远端哈希。
4. ModelScope：`ms.py publish release/v0.4`，先演练再加 `--push`。脚本会核对远端大小和 sha256、建 v0.4 tag，并核对打了 tag 的文件。
5. 回执：`runs/release/` 下的上传回执和远端核验结果不进 git，凭据也不进 git。
6. 仓库可见性保持现状，不做任何改动。

## 8. 排程（以 exp32 按时完成为前提）

| 时间 | 事项 |
|---|---|
| 约 16:30 | exp32 报告；评论家决定是否锁定候选，或改做限定修复 |
| 锁定后约 2.5 小时 | 完整评测 + laya-vision 比较 |
| 随后约 30 分钟 | 校准报告、CPU 延迟、build / verify / ModelScope build |
| 10-18 上午 | 证据评审；模型卡和 notes 定稿 |
| 10-18 18:00 前 | 合入、打 tag、双平台发布与远端核验 |

## 已准备（本提交）

- `configs/v04-full-eval.yaml`：29 个集合的只评测配置，dry-run 通过，用途与 v0.3 一致。
- `scripts/release/check_eval.py` 和 `tests/release/test_check_eval.py`（2 项）：复用结果前核对身份、数据哈希和逐题完整性。
- `release/v0.4/release.yaml`、`notes.md`：草案，候选和数字都标为 TBD。
