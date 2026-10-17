# v0.4 完整评测与发布草案（候选未锁定，成绩未填；评审 V04-FALLBACK-01 更新）

日期：2026-10-17。依据：交付计划 `docs/experiments/2026-10-17-v04-delivery-plan.md` 中的"发布证据准备"一节和 `release/README.md`。本文只写步骤、命令和要核的证据。候选 checkpoint 与全部成绩都留待锁定后填写；不提前发布。

## 1. 锁定候选（评论家与 Engineer 共同确认后填写）

**现状（2026-10-17 18:45）**：还没有候选。
- exp32 的筛查结论是"没显出提升"，exp36 两臂都没达到继续条件，都不进入 v0.4。
- 下一项训练是 exp37（数字可读性对照）。
- v0.3 继续作为稳定回退版本保留；不把 v0.3 或任何未过筛查的权重改名为 v0.4，也不会把它们称为能力进步。

| 项 | 内容 |
|---|---|
| 候选 | TBD（配置里是占位 `runs/CANDIDATE/stage2`） |
| 来源与步数 | TBD：完整的起点链和各阶段实际步数 |
| 数据版本 | TBD：各阶段数据文件的 sha256 |
| 权重指纹 | TBD：`model.safetensors` 的 sha256（freeze 时写入 IDENTITY.json） |
| 筛查 | TBD：筛查实验名和报告；报告所用权重的 sha256 取自报告本身或与报告哈希绑定的运行身份（`delivery.py --screen / --screen-report / --screen-identity`），与冻结权重不一致或缺证据时标为不可验证 |
| 代价 | TBD：7 个面板 |

锁定候选时要同步更新：
- `configs/v04-full-eval.yaml` 的 `checkpoint` 和 `history`；
- `release/v0.4/release.yaml` 的 `round`、`stage` 和 evidence 里的训练配置；
- `notes.md`；
- 本节的来源链和数据版本。

之后重新 freeze。

## 2. 完整评测（29 个集合，与 v0.3 相同）

- 配置 `configs/v04-full-eval.yaml`：29 个集合、对照（配错图 / 选项倒序）、温度拟合集（`data/r13b/val.jsonl`）和用途判定都与 `configs/r15-vision.yaml` 相同。
- `history` 另外加入候选训练时监测过的集合：real_layout 调参集、exp21 的 4 个 ScienceQA 开发集和 dev16。
- 用途：除 `bench_lv_scienceqa` 对本候选明确标为 monitoring（V04-REL-R1-05）外，其余 28 个集合判出的用途与 r15 逐一相同（`build_plan` 已比对）。
- 每题都能与 `runs/r15-vision/eval/` 配对。
- 结果写到候选轮次的 `eval/`（候选轮次有 run.json 时）。
- **执行顺序**（V04-REL-R1-01 / 02，`scripts/release/check_eval.py`）：
  1. `freeze`：候选第一次做最终评测之前，把 checkpoint 目录里每个文件（权重、带温度的 config、tokenizer 等）的 sha256 和温度表写入 `<eval>/IDENTITY.json`。已有一份不同的身份时拒绝覆盖。
  2. `preflight --quarantine`（隔离时同时撤销对应的 seal 记录，旧记录写进隔离目录的 QUARANTINE.json，V04-REL-R2-01）：把每个计划输出（集合及其每个对照）分为三类：
     - ok：已在当前身份下 seal，文件哈希不变，内容检查通过；
     - missing：不存在，交给 pipeline 新跑；
     - invalid：存在但证明不了身份（freeze 之前写的、权重或温度不同），或内容不对。
     invalid 的文件移到 `<eval>/_invalid/<时间>/`，pipeline 会重新评。因为 pipeline 只要看到摘要存在就跳过，所以必须先清掉 invalid。
  3. `devision-pipeline configs/v04-full-eval.yaml`：补跑 missing。
  4. `seal`：确认 checkpoint 仍与 IDENTITY 一致；对 freeze 之后写出、内容检查通过的输出，记下文件哈希，写入 `RESULTS.json`。
  5. `check`：全量核查——每个计划输出都在、已 seal、未改动、内容通过。
  - 附加证据（V04-REL-R2-04）：29 个集合以外的证据列在 `configs/v04-extra-evidence.yaml`，每条写明数据、温度模式（fitted / raw）、对照和用途：
    - sqa_temp / force 及其错图版；
    - dev16、syn_erased；
    - 校准集的 fitted / raw 两份；
    - bench_lv_scienceqa 的 raw；
    - 新确认集的真图、固定错图和倒序。
    - 上面每一步都加 `--extras configs/v04-extra-evidence.yaml`，再加一步 `run-extras` 补跑缺的附加证据，与 29 个集合同一个冻结身份、同样 seal 和 check。
    - calibration.py / delivery.py 只读通过 check 的输出，按名字取。
  - 完整命令：
    ```bash
    # 在 bash 里执行（zsh 直接粘贴时 $E 不会拆成两个参数）
    bash -c 'C=configs/v04-full-eval.yaml; X=configs/v04-extra-evidence.yaml
      uv run python scripts/release/check_eval.py freeze --extras $X $C &&
      uv run python scripts/release/check_eval.py preflight --quarantine --extras $X $C &&
      uv run devision-pipeline $C &&
      uv run python scripts/release/check_eval.py run-extras --extras $X $C &&
      uv run python scripts/release/check_eval.py seal --extras $X $C &&
      uv run python scripts/release/check_eval.py check --extras $X $C'
    ```
    freeze 只在候选训练完成、锁定之后执行，不提前。
- 内容检查（每个输出；每条明细的 image_id 必须等于数据文件的 image_id，V04-REL-R3-01）：
  - 摘要的 run 字段：checkpoint、数据 sha256、fitted、对照类型、用途；摘要的温度表与冻结的相同。
  - 明细：每题恰好一次。数据里重复出现的题号（如 r13b/val 的 242 个）按选项逐份配对，每一份都要在（V04-REL-R2-03）；倒序对照只覆盖选择题；题型和 gold 与数据文件相同；选项与数据文件相同（倒序对照则顺序相反）；图片与数据文件相同；配错图对照必须正好是 `picture_pairs(samples, seed=0)` 指定的那张，且 seed 记录为 0（V04-REL-R2-02）；概率只覆盖这些选项、都是 [0, 1] 内的有限值、和为 1；prediction 是 argmax；correct 与 gold 一致。
  - 摘要的题数、准确率与明细一致。
- 候选训练流程在 freeze 之前评过的集合证明不了身份，按规则在 preflight 时移走、重评。
- 夹具（`tests/release/test_check_eval.py`，7 项）：
  - 同一路径换权重或改温度后拒绝复用；
  - freeze 之前写的结果被隔离；
  - 同编号错 gold、缺概率、缺选项、缺题都会被拒绝；
  - 完整的新结果能通过。
- v0.3 的历史结果没有冻结身份：`check --legacy configs/r15-vision.yaml` 只做内容检查，29 个集合加全部对照 0 个问题，输出里明确写"identity not proven"。身份改用已发布权重的 MANIFEST（`release/v0.3/MANIFEST.json`，model.safetensors `110fea95…`）佐证，并披露历史元数据的这一限制，不默认重跑基线。
- 评完后：
  - `scripts/lv_report.py <eval> --prev runs/r15-vision/eval`，生成 laya-vision 详细比较，包括 ScienceQA 按学科 / 选项数 / 必须看图的切片；
  - 每个集合跑 `devision-compare` 对 v0.3 逐题配对；
  - 写入训练日志和 `docs/research/laya-vision-gap.md`。
- 用时：v0.3 的同一套评测实测约 2 小时 16 分钟（14 日 18:06–20:22），按 2.5 小时安排。

## 3. 交付证据（`release/v0.4/delivery_vs_v0.3.json`，`scripts/release/delivery.py`）

delivery.py 只读通过 check 的输出（`check_eval.verified`），全部从已 seal 的同题明细重算；不直接取候选训练流程在 freeze 之前评的记录（V04-REL-R2-04）。

- **主目标**：用候选已 seal 的 `extra_sqa_{temp,force}{,_mism}` 和 v0.3 的 exp27 记录（只做内容检查，历史身份），按 exp21 的 readout.transfer 重算 68 : 114 准确率变化和看图收益变化及区间。
  - 候选自己的筛查标签：`--screen <实验> --screen-report <报告> [--screen-identity <运行身份>]`。报告所用权重的 sha256 只从报告自身的 `model_sha256` 读取，或从运行时写下、并记有该报告文件 sha256 的身份文件读取，不接受手填。只有它等于冻结的 model.safetensors 时才带入标签，缺证据或不一致都标为不可验证（V04-FALLBACK-01 / R1-01）。另给出 `screening_report_matches`。
  - 所以候选的训练 chain 要在出报告时把 `model_sha256` 写进报告，或同时写一个记有报告 sha256 的身份文件。正在运行的 exp37 不改。
  - 筛查标签不等于发布条件全部满足；不改各实验预先约定的主判定。
- **新能力保留**（V04-REL-R1-04）：
  - 同一候选已 seal 的 `extra_dev16`：读数和比较的点估计、按族重采样的区间，各自按原规则判定（通过 = 点估计 ≥ 0.80 且下界 ≥ 0.75；失败 = 点估计 < 0.60 且上界 < 0.65）。
  - `extra_syn_erased`：擦除控制，有效条件是读数 < 0.50 且比较 < 0.55；无效时 dev16 的判定作废。
- **保护**：29 集里已 seal 的 7 个面板，对 v0.3 的区间和预写代价。
- **身份**：IDENTITY.json 里 model.safetensors / config.json 的哈希。
- 夹具（`tests/release/test_release_evidence.py`）：擦除控制有效时 dev16 正常判定，无效时作废；transfer 重算后与筛查报告比对；两个候选不能互借对方的筛查标签。

## 3b. 新冻结确认集（V04-REL-R1-05、V04-CONF-01 / 02；模型在候选锁定后才评，只评一次）

- 不用 rl_confirm 加历史 bench 来充当：
  - rl_confirm 保留为"同一生成器、未见过的数值 / 大小组合"的证据；
  - `bench_lv_scienceqa` 的分项曾用来选研发方向，对本候选标为 monitoring，历史报告不回写。
- **物体属性**（已冻结，V04-CONF-01）：
  - 来源：ScienceQA 官方 test（snapshot `f18b0a70…`）的 Compare properties of objects / Identify properties of an object。
  - 隔离后剩 75 题 / 75 图。confirm.jsonl `5e5c8e2d…`，confirm_mism.jsonl `d9e55e18…`（分项内按图固定置换）。
  - 原定的图表、图解两个分项，隔离后都是 0 题：气候图都是 train / val 换色重绘，图解是完全相同的图或 JevBench 用过的行。详见 `data/v04_confirm/MANIFEST.json`、`EXCLUSIONS.json`。
- **图表读数**（ChartQA 改编的图表单值四选一确认子集，V04-CONF-02、V04-CHART-R1-01）：
  - 来源：ChartQA 官方仓库固定提交 `044eabfc…` 的 test_augmented。按 SHA256(imgname + query) 遍历前 225 题，保留直接读单个数值的 80 题 / 80 图，四选一，干扰项都取同一 CSV 列。
  - 80 张图都看过：执行方 21 张、评论家 27 张、Spec 评审 32 张。70 题图上印着被问的数值；10 题要从柱高 / 点位估读，其中 multi_col_40409 读的是堆叠段的厚度。
  - 冻结：chartqa.jsonl `e129226b…`，chartqa_mism.jsonl `e059d318…`，EXCLUSIONS.json `8902145e…`。
    - 与评审时的版本（`82c1c082…` / `e316cce7…`）相比，题目、选项、gold、图片、题序和错图映射逐条相同，只补了 value_printed / checked_by 两个字段。
    - 图片、256 预览和计数的哈希都没变；隔离记录复用已审的那份，没有重扫 75 万张图。
  - 它不是官方 ChartQA 成绩，也不是 80 题精确 OCR 测试；数值在 256 输入里约 2–4 px（按 12 px 字高估算，不是逐图测量）。
- **评测**：
  - 候选：`configs/v04-extra-evidence.yaml` 里的 `confirm_property{,_mism,_rev}` 和 `confirm_chartqa{,_mism,_rev}`，与 29 集同一个冻结身份。
  - v0.3：`configs/v04-v03-confirm.yaml`（没有常规集合）加 `configs/v04-confirm-extras.yaml`，只评这 6 项新确认项，题单和模式与候选完全相同。它的 freeze / run-extras / seal / check 只覆盖这 6 项，不重跑 v0.3 的 29 集。
  - 两个模型加对照合计 ≤ 10 分钟。
- **报告**：`scripts/release/confirm_report.py`。
  - 两个分项分别报告：准确率、看图收益（真图 − 固定错图）及其变化、候选 − v0.3 差值，以及选项倒序时选择改变的比例。
  - 区间按原图 bootstrap。两份错图文件的 image_id 约定不同：属性集记的是展示的那张图（原图在 mismatched_from），图表集保留原图（展示图在 shown_image_id）。统计一律按 sample_id 从真图数据文件恢复原图，两种约定都适用。
  - 图表集的解读（V04-CHART-R1-02）：不看图、只选四项中最大的值就能对 49 / 80 = 61.25%，只选最小的值是 17 / 80；正确答案的位置分布是 25 / 15 / 25 / 15。所以准确率超过 25% 不能当作看图的证据，要看真图 − 错图的收益及其变化。可以在同一批明细里把 70 道印字题和 10 道估读题分列描述，10 题不做显著性检验，不设门槛。
  - 输入文件带 `extra_` 前缀：
    ```bash
    uv run python scripts/release/confirm_report.py --data data/v04_confirm/confirm.jsonl --name extra_confirm_property \
      --v03 runs/r15-vision/eval --cand <候选轮次>/eval --out release/v0.4/confirm_property.json
    uv run python scripts/release/confirm_report.py --data data/v04_chartqa/chartqa.jsonl --name extra_confirm_chartqa \
      --v03 runs/r15-vision/eval --cand <候选轮次>/eval --out release/v0.4/confirm_chartqa.json
    ```
- 这项证据不替换 68 : 114 主指标，不增设事后的通过阈值，也不能单独证明通用 OCR。第一次的结果要保留；结果一旦用于下一轮开发，就失去盲测资格。

## 4. 概率质量（V04-REL-R1-03，`scripts/release/calibration.py`）

- 在同一份冻结权重上，比较 T = 1（`devision-eval --no-temperature`）与候选实际存储的温度（`devision-eval`），给出 accuracy / NLL / ECE，并按（题型, 选项数）分层。不重新拟合，不写 checkpoint。
- 覆盖两个集合，分别注明用途：
  - 校准集 `data/r13b/val.jsonl`（calibration_fit）：温度就在这里拟合，改善是样本内的；
  - 完整 `bench_lv_scienceqa`（monitoring）：没有参与拟温，但参与过研发选择。
- 训练报告只记录了校准后的最终指标，没有 T = 1 的原始记录（r15 的 train_report 已核实），所以原始概率用 `--no-temperature` 补跑：两个集合，约 3 分钟。
- 不为这一项重跑 29 个集合的全部对照。

## 5. CPU 延迟（候选自己测，不复用 v0.3 的数字）

```bash
OMP_NUM_THREADS=4 uv run devision-eval --checkpoint <候选> --data data/r12/cpu_latency.jsonl --data-root data \
  --out <候选轮次>/cpu/single.json --device cpu --no-group
cp <候选轮次>/cpu/single.json release/v0.4/cpu_latency.json
```

- 与 v0.3 用同一个 200 题夹具，4 线程，空闲机器。v0.3 是 P50 174.9 / P95 190.0 ms。
- 结构没有变（256 输入、64 个视觉 token），预计相近，以实测为准。
- 不在训练或完整评测运行时测，避免抢占资源。

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

## 8. 排程（V04-FALLBACK-01）

| 时间 | 事项 |
|---|---|
| 10-17 约 21:15 | exp37 结果出来，当晚定下唯一的夜间候选配方 |
| 10-18 08:00 | 候选完成的目标 |
| 10-18 12:00 | 候选锁定的最迟内部控制点（需要限定修复时） |
| 锁定后约 2.5 小时 | 完整评测 + laya-vision 比较 |
| 随后约 30 分钟 | 校准、CPU 延迟、build / verify / ModelScope build |
| 10-18 18:00 | 内部发布目标：合入、打 tag、双平台发布与远端核验 |
| 10-18 23:59 | 负责人给的实际最晚截止 |

到锁定点仍没有合格模型时，逐项报告缺口和最后能做的动作，不把研究候选或旧权重称为已完成目标。

## 已准备

- `configs/v04-full-eval.yaml`：29 个集合的只评测配置。checkpoint 现在是占位，锁定候选后再填。
- `scripts/release/check_eval.py`（freeze / preflight / seal / check）和 `tests/release/test_check_eval.py`（7 项）。
- `scripts/release/calibration.py`、`scripts/release/delivery.py` 和 `tests/release/test_release_evidence.py`（3 项）。
- `scripts/data/v04_confirm.py`：新冻结确认集。
- `release/v0.4/release.yaml`、`notes.md`：草案，候选和数字都标为 TBD。
