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
- 内容：wheel 共 23 个文件，其中 `devision/` 下 18 个源码/资源文件（image / model / infer / serve、train/*、static 的 demo 四个文件）逐个与 git 3598ab5 的 blob 哈希相同；其余是构建生成的 dist-info。
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
  uv run devision-calibrate --checkpoint runs/<候选>/stage2 --data data/r13b/val.jsonl --data-root data --device mps --out runs/<候选>/stage2-cal
  ```

  - `data/r13b/val.jsonl` 共 3,892 题。
  - `--out` 写一份副本，只改 config 的 `temperature` / `temperature_by_options`，权重原样复制。
  - 同时写 `calibrate_report.json`：源 checkpoint、`data_sha256`、代码版本、拟合前后各桶的 NLL / ECE。
  - 不用 `--in-place`。
- **身份核对**（freeze 前）：
  - 副本 `model.safetensors` 的 sha 等于候选训练终点；
  - config 除两个温度字段外与终点相同；
  - `calibrate_report.json` 的 data_sha256 等于 val.jsonl 的实际 sha，且题数为 3,892。
  - 校准副本放在已有真实 `run.json` 的候选轮次内，作为 `stage2-cal`；不改原 `stage2` 或筛查冻结记录。最终评测配置的 checkpoint 改为这个副本，release.yaml 的 round 仍为该候选轮次、stage 改为 `stage2-cal`。这样 pipeline 和发布工具都使用 `runs/<候选>/eval/`；不能新建一个没有 run.json 的 `<候选>-cal` 目录，却假定 pipeline 会写入它的 eval 子目录。
  - **最终发布冻结**用 `scripts/release/check_eval.py freeze --extras configs/v04-extra-evidence.yaml configs/v04-full-eval.yaml`，在首次最终评测前生成 `eval/IDENTITY.json`，绑定整个 checkpoint 与温度表；随后按主发布草案第2节执行 `preflight --quarantine → pipeline → run-extras → seal → check`，各核验步骤均带同一 extras 配置。旧筛查输出不直接冒充最终输出，由 preflight 按原规则隔离重评。CPU、打包同样指向校准后的副本。
  - `scripts/v04b/freeze.py` 的 FROZEN.json 只绑定实验筛查的 model/config，不能替代上述发布身份与封存流程。
- **不能替代**：pipeline 的 `calibration_fit` 只是用途标注，不执行拟合。训练里的 val 拟温（EXP50 是 val_quarter 244 题）不能替代全量校准；如果候选训练本身就以 r13b/val 全量拟温，就核验 train_report 的 calibration_fit_set / calib_items，并用当次运行身份或数据 manifest 中的校准文件 SHA 与实际文件核对后复用。train_report 本身没有校准数据 SHA，不能只靠它证明数据身份。
- **不允许**：freeze 或全评之后不再换温度；换了就是新身份，要重评。
- **耗时估计**：局部视图每题多编码 12 张图，3,892 题约 15 分钟（MPS）；无局部视图的候选约 5 分钟。
- **尚未测量的资源限制**：独立 calibrate 调用 `_val_logits` 的默认32题前向，并不继承本轮训练的 eval_batch=8；15分钟及其峰值内存均未实测。后续申请真实校准许可时须明确采用的批量和资源预算；如需暴露较小批量，应另行实现并验证等价，不能向当前CLI传不存在的参数。这里不启动校准、不改活动链或默认行为。

### 到 23:59 的条件关键路径（评论家 V04-LOCAL-TIME-01 后按实测重算；替换早先"全评 3–4 小时"的未测估计）

- **请求量**：v0.3 的 29 集及对照共 79 份摘要（runs/r15-vision/eval/），79,711 次请求，133,689 题。分组上限 32（v04-full-eval 的设置），每请求平均 1.68 题；请求时间合计 2.16 小时，墙钟 2 小时 16 分钟。
  - 主集合 29 份：30,167 请求；错图 29 份：30,121 请求；逆序 21 份：19,423 请求。
  - 若改用分组上限 8（EXP50 链的设置），16 份摘要有超过 8 题的图片，其中 POPE 两个集合每图 18 题，请求会增加约 4,000 次（约 5%）；全评按配置用 32。
- **局部视图每请求的额外耗时（实测，同集合、同分组 8、MPS）**：T 减 X 的每请求平均耗时为 257–278 ms。每请求 1.29 题时约 257 ms，4 题时约 278 ms，近似为 247 ms + 7.7 ms × 题数。
- **全评估计**：2.16 小时 + 79,711 × 0.247 秒 + 133,689 × 0.0077 秒 = 2.16 + 5.47 + 0.29 ≈ 7.9 小时，再加进程加载约 0.1–0.3 小时。按每请求额外 240–290 ms 的区间算，约 7.7–9.0 小时。
  - **未知项**：18–32 题的大组是把 ≤ 4 题外推出来的（条件模块对每题展开的 3,328 个 K/V 投影会随题数增加）；最终候选的结构也可能不同。这是估计，不是实测。
- **全量校准**：共3,892题。实际 calibrate 调用 rlcd._val_logits，默认每批32题前向，并非逐题 decide 请求；约20分钟仅作粗估，实际内存和用时尚未测量，批量限制仍按前述要求核实。
- **串行关键路径**：
  1. EXP50 终点 16:23；
  2. 候选训练与筛查：时长未知（EXP50 每臂 576 次更新是 26.5 分钟；候选还要加迁移 / 保护 / dev16 筛查，时长随方案）；
  3. 校准和 freeze 约 0.35 小时；
  4. 全评约 8–9 小时；
  5. CPU / API / 打包 / 发布约 1–1.5 小时。
  - 候选训练假设为0，从16:23起按全评7.7–9.0小时、校准约0.35小时、收尾1–1.5小时复算，为次日约01:25–03:15；补测、候选训练/筛查等尚未计入。这是有假设的估计，不是严格最早时点。**当前本机串行路径没有可信的23:59前完整交付排程。**
- **一个最小的等价复用方案（不实现）**：在同一个 devision-eval 进程里，把主集合和逆序对照放进同一张图的同一次 decide 请求，图片的 13 路视觉编码只做一次。逆序对照的题目与主集合用同一张图，只是选项顺序反过来。
  - **省时来源**：逆序 19,423 次请求的额外编码，约 19,423 × 0.25 秒 ≈ 1.35 小时。错图对照每题配的是别的图，这种方式不能共享；跨请求缓存图片特征则内存不可行（每图 3,328 × 768，fp16 约 5 MB，大集合要几十 GB）。
  - **身份与等价**：同一冻结 checkpoint，摘要和明细仍按集合和对照分别写出；decide 对每题独立打分，视觉输入是同一个张量，逐题概率只可能有批组成带来的 1e-6 量级浮点差异。验证需要 tiny 夹具，加上一个小集合在固定权重上的逐题对比（这要跑模型，属于以后的许可）。
  - **内存**：一次请求最多 64 题，展开后的 K/V 投影约 0.9 GB。需要上限 32，或者把条件模块的 K/V 投影移到展开之前（这又是一处代码改动）。
  - **结论**：即使采用，全评仍约 6.5–7.5 小时，救不回 23:59。它还违背负责人"离线评测不为提速加缓存 / 去重机制"的既定偏好。**不建议现在实现**，提速没有可承诺的途径。
  - 任何复用都不能替代最终候选普通 decide 的真实 smoke 和 CPU 延迟实测，缓存吞吐也不能当作产品的单次延迟。

## 4. 本轮不做的

- 不启动 v0.3 新确认评测，不查看任何新确认预测（候选锁定后才开启）。
- 不发布包或权重，不推送标签。
- 不改 EXP50 活动链。

评审补正说明（评论家，2026-10-18）：上述发布冻结入口、校准副本位置和数据身份来源已在本轮直接作文档补正；不表示相关校准、冻结或全评已经执行。代码与构建产物仍为3598ab5，文档基于e6e0b3f补正，尚未提交。
