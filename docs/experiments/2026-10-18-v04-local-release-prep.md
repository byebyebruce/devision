# v0.4 发布准备：局部视图路线（候选无关，CPU / tiny；评论家 V04-LOCAL-PREP-01、EXP50-NOTE-01）

日期：2026-10-18。隔离 worktree `../devision-v04local`，分支 `v04-local-prep`，从 52ff548（EXP50 运行代码）拉出。不应用到正在运行的 EXP50，主仓库的 src / config / scripts、安装环境和 watch 都没动。没有加载真实权重，没有启动 v0.3 确认评测，没有发布任何东西。**这里没有候选**：EXP50 的 T / C 都不是候选，EXP50 即使通过也只是候选训练的依据。

## 1. NOTE-01：极小 / 细长图片的非零视图（3598ab5）

- `image.view_rects` 取整后，只有当某个视图的跨度为空时（宽或高只有几个像素的图片），才改成落在图内的 1 像素跨度（`image._span`）；跨度 ≥ 1 像素的矩形原样返回。
- 证据（CPU）：
  - 1 × 1、1 × 300、300 × 1、2 × 3、5 × 1000、1000 × 4、11 × 7，每种都配 n12 / 0.25、n16 / 0.5、n4 / 0.5：视图都在图内，宽高 ≥ 1，打包像素有限。
  - 尺寸扫描（宽 20–1400、高 20–900，加上实际尺寸），凡是旧规则矩形非空的，新旧完全相同。
  - EXP50 用到的全部 2,826 张图（train、val_quarter、fit_O / L、rl_tune / L、exist、science；共 1,077 种尺寸）：12 / 0.25 的矩形与运行代码逐个相同（0 个不同）。
  - 1 × 1、1 × 300、400 × 2 的图片经普通 decide 正常作答（tiny 局部模型）。
  - 默认旧模式不变（原有夹具）。
- 测试：`tests/train/test_local_views.py` 和 `tests/model/`，55 项通过（worktree 源码，主仓库 venv 的解释器，PYTHONPATH 指向 worktree；没有改安装环境）。
- 未覆盖：真实权重下极小图的答案质量；只验证了能处理、不出错。

## 2. 未发布的局部视图包（3598ab5 构建）

- 版本：`0.3.0.dev1`（pyproject 唯一的版本号；PEP 440 开发版，不会被误当作正式版）。正式版本号在发布前核实是否空闲（计划 0.3.0），再改 pyproject、打 `v0.3.0` 标签。
- 构建：`uv build --offline -o dist`（hatchling），产物不进 git：
  - `devision-0.3.0.dev1-py3-none-any.whl`：sha256 `25bdebe01e9f66b3314485d54f594da9b8e68ea745276ac074d6492a9101ca4b`
  - `devision-0.3.0.dev1.tar.gz`：sha256 `91261f69660486413eb80ba8b1c6c70a9ea8a47bc63138d74f0245a66e959a1c`
- 内容：wheel 里的 23 个文件，即 `devision/` 下全部源码（image / model / infer / serve、train/*、static 的 demo 四个文件），逐个与 git 3598ab5 的 blob 哈希相同。
  - 包含局部视图路径：`local_pixels`、`unpack_local`、`_span`、`visual_inputs`、`enable_local_views`、`QuestionConditioning.add_geometry`。
  - 主仓库里他人未提交的 serve.py / demo.html 改动和 snake / tetris 文件不在包里（worktree 从提交拉出）。
  - 不复用 00:55 的旧 wheel。
- 隔离安装：`/tmp/v04venv`（Python 3.12，`uv pip install --offline` 装这个 wheel）。
  - `import devision` 不加载 torch，版本 0.3.0.dev1，路径在该 venv 里。
  - tiny smoke（`python -I`，不用仓库源码）：旧模式和局部视图（12 / 0.25 / T）的 tiny checkpoint 保存 → `devision.load` → decide，在 560 × 405、650 × 140、1 × 1、1 × 300 上与保存前答案相同。
  - config 里写入 `local_views 12 / local_overlap 0.25 / local_source T`。
- `release/v0.4/release.yaml`：写明局部视图候选需要 `min_package_version: 0.3.0`（读取 local_views 的最低包），沿用已验收的卡片机制。正式版本发布前再核实；没有局部视图的候选不设。
- **不能当作真实兼容证据**：v0.3 和候选的真实 smoke / API / CPU 延迟，要等模型空闲、另行排程。tiny 通过不代表真实兼容。旧包 0.2.1 读局部视图 checkpoint 的行为没有测（预期不能正确读取），所以必须设最低版本。

## 3. 发布草案占位与关键路径

- `release/v0.4/notes.md` 的 training 一节改成占位：不再预写旧三阶段配方、视觉顶 4 层训练或"256 输入、64 token"等同当前结构。也不预写 EXP50 的 T / C 为候选。
- 历史准备记录（发布草案第 9–10 节等）保留原样。主仓库的 `docs/experiments/2026-10-17-v04-release-draft.md` 有他人未提交的改动，这里不碰，关键路径写在本文件。

### 全量校准的入口与身份（在最终 freeze 之前）

- **入口**：

  ```
  devision-calibrate --checkpoint runs/<候选>/stage2 --data data/r13b/val.jsonl --data-root data --device mps --out runs/<候选>-cal/stage2
  ```

  - `data/r13b/val.jsonl` 共 3,892 题。
  - `--out` 写一份副本，只改 config 的 `temperature` / `temperature_by_options`，权重原样复制。
  - 同时写 `calibrate_report.json`：源 checkpoint、`data_sha256`、代码版本、拟合前后各桶的 NLL / ECE。
  - 不用 `--in-place`。
- **身份核对**（freeze 前）：
  - 副本 `model.safetensors` 的 sha 等于候选训练终点；
  - config 除两个温度字段外与终点相同；
  - `calibrate_report.json` 的 data_sha256 等于 val.jsonl 的实际 sha，且题数为 3,892。
  - 然后 `scripts/v04b/freeze.py freeze runs/<候选>-cal`，此后所有评测（29 集、确认集、概率质量、CPU 延迟、打包）都用这个冻结身份。
- **不能替代**：pipeline 的 `calibration_fit` 只是用途标注，不执行拟合。训练里的 val 拟温（EXP50 是 val_quarter 244 题）不能替代全量校准；如果候选训练本身就以 r13b/val 全量拟温，就核验它的 train_report（calibration_fit_set、题数、sha）后复用。
- **不允许**：freeze 或全评之后不再换温度；换了就是新身份，要重评。
- **耗时估计**：局部视图每题多编码 12 张图，3,892 题约 15 分钟（MPS）；无局部视图的候选约 5 分钟。

### 到 23:59 的条件关键路径（依赖 → 模型时间；括号内是可并行的 CPU 事项）

1. EXP50 终点（最迟 16:23）→ 评论家按预写条件裁决。若没有臂达到 progress，局部视图路线不进候选，本节后续不启动；v0.3 保持。（同时可做：本文件第 1–2 项、模型卡文字。）
2. 候选训练与筛查（原图迁移、七保护、dev16）：方案由评论家定，模型时间未定，按 1–2 小时估。（同时可做：包和卡片的 CPU 准备，不能做任何依赖候选权重的事。）
3. 全量 3,892 题校准，约 15 分钟 → 身份核对 → freeze。
4. 29 集评测加 laya-vision 比较，约 3–4 小时（v0.3 实测 2 小时 16 分钟，局部视图每请求多约 0.15 秒）。之后是附加证据：新确认集，候选和 v0.3 各一次，≤ 10 分钟，候选锁定后才开启（`configs/v04-v03-confirm.yaml` 的规定）；原始概率质量约 3 分钟。（同时可做：模型卡结构、build 脚本的 dry-run。）
5. CPU 延迟实测（候选自己测，约 10 分钟，机器空闲）；隔离环境装正式 wheel，跑 v0.3 和候选的真实 smoke / API；build / verify；PyPI（先包）→ HF / ModelScope（后模型）→ 远端核验，约 1–1.5 小时。
- **例**：16:30 开始候选训练 1–2 小时，校准和 freeze 约 18:00–19:00，全评约 21:00–23:00 结束，第 5 项约 22:00–24:30。只有每一步一次通过、全评吞吐不差于估计时才在 23:59 前完成。这是有风险的估计，不是承诺；发布覆盖不删减。

## 4. 本轮不做的

- 不启动 v0.3 新确认评测，不查看任何新确认预测（候选锁定后才开启）。
- 不发布包或权重，不推送标签。
- 不改 EXP50 活动链。
