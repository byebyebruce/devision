# EXP76 方案：R 端点的普通决策诊断（PANEL-DECISION-PREP-01；CPU 准备完成，未运行模型）

日期：2026-10-20。不算一轮，不产出可采用的模型。worktree `devision-panel-decision`，分支 `panel-decision-prep`（从 6764043 拉出）。

## 问题

EXP75 的 R 端点在直接读数上达到了预写的投入线：异值两端全对 523/837 = 0.625，R − D1 +0.391 [0.357, 0.425]，真减错 +0.625 [0.591, 0.660]。它能否用于普通的选项决策（readA / readB / cmp），同时保住 K 文字消费、封存的温度 / 磁力读数状态，以及 exist / science？这是学习和保护的诊断，不是发布判定。EXP74 原来的 0.30 结局不变，原 D2 和它的守卫不使用、不修改、不绕过。

## 设计

- **起点**：只有 R（`runs/exp75-panel-r/r`，model b7f471fc…，config 7fb24ad3…，align_report 2721ecd6…）。前置检查只要求 R 完成并通过三项投入线（EXP75 compare.json d69461b7… 记录为 completed / worth_decision_diagnostic）；不要求双臂完整，N 的 invalid 状态不被当作任何东西。
- **配置**（`configs/exp76-panel-decision.yaml`）：照搬已审的 EXP74 D2（configs/exp74-panel-d2.yaml），只改 name 和 init（夹具核过其余完全相同）：
  - 数据 data/panel_learn/train.jsonl（7c9dbb97…，8,430 题）；fit / dev16 / erased、PROTECT 的 K 文字切片、封存 read_state 的四个监测集；
  - mixed_input CE，1 遍 1,054 次更新，micro / eval 8，LoRA r16（lr 5e-5），决策头 lr 1e-5，warmup 100，seed 1；
  - 只训练 LoRA 和决策头，冻结投影层、conditioning、SigLIP 和全部读数组件；
  - 之后用 devision-eval 跑 exist / science 探针。
- **报告**：复用已验收的 D2 报告（scripts/panel_learn/report.py 的 d2_report，PANEL-D2-R1/R2 的全部有效性检查），只做两处接线：
  - 运行目录和配置由命令行传入（`report.py d2 runs/exp76-panel-decision runs/v04-k/eval configs/exp76-panel-decision.yaml`），检查对象是新实验，不是旧路径；
  - 新增"versus"：终点相对本次运行自己的起点（step0 监测，也就是做任何决策更新之前的 R）和相对封存候选（封存 read_state 的准确率、封存的 exist / science），只是描述性的点差，不进入任何规则。
- **预写判定**沿用原 D2：
  - fit 读数 / 比较 pass 要求点估计 ≥ 0.80 且 CI 下界 ≥ 0.75；
  - dev16 用同一口径作开发报告；
  - 擦除对照必须无效；
  - 保护：K 文字 ≥ 0.95、rs_temp ≥ 39/41、rs_force ≥ 0.458、exist / science 掉不超过 3 个点；
  - 结局 A / B / C / D，证据不全为 incomplete，证据不对为 invalid。
- **预检**（`scripts/panel_decision/preflight.py`，不加载模型；失败时 exit 12，不训练）：
  - R 的三个文件和 EXP75 的 compare / report / R numbers 都按 SHA 核对，并要求 compare 把 R 记为 completed 且值得进入决策诊断；
  - 训练数据和监测集按各自 MANIFEST 核对；
  - 配置从 R 开始；
  - rlcd 对本配置规划 1,054 次更新；
  - EXP76 目录是新的。
  - 已在真实文件上只读运行，ok（运行后删除了它写出的 preflight 文件，runs/ 没有留下东西）。
- **runner**（`scripts/panel_decision/run.sh`）：只启动一次（`runs/exp76-panel-decision.started`）；先做进程检查和预检；硬上限 5,700 s（95 分钟），训练加探针在截止前 120 s 结束，最后 120 s 留给不加载模型的报告；不串接其他任何东西。

## 预算

沿用已核的 1,054 次更新：训练 45.7–52.7 分钟（PROTECT-K 同路径实测 2.6 s / 次，面板行估计最多 3.0 s）；监测 3 轮共约 17 分钟；exist / science 约 5 分钟；加载约 1.5 分钟；合计 **69–76 分钟**，上限 95 分钟。

## 结论范围

即使 fit 通过，也只说明 R 在训练族上能用于普通决策；未见族、七个保护面板、实时组合、校准和发布验证都需要各自的有效证据。没学会就按预写分支报告，不自动加遍数直到通过。RELEASE-GAP-01 仍未关闭。

## 验证（CPU，已完成）

- `scripts/panel_decision/test_panel_decision.py` 7 passed（连续 3 次）：
  - 配置与 EXP74 D2 只差 name 和 init；
  - 预检在真实文件上通过；夹具中通过；
  - 拒绝 3 种：另一个 R config、EXP75 不认为 R 值得、EXP76 目录已存在；
  - runner：预检失败时 exit 12、不调用 pipeline、不写启动标记；正常时顺序为 预检 → pipeline → 报告（带新目录和配置参数）；只启动一次；上限。
  - runner 夹具里把 ps 换成桩：之前有一次间歇失败，最可能的原因是本机其他进程的命令行里出现了这些脚本路径，被进程守卫正确地拦下。守卫本身的检测由引入它时的夹具覆盖。
- `scripts/panel_learn/test_panel_learn.py` 23 passed：原有 22 项，加 1 项"versus"（终点对 step0 和封存值的点差）。
- pyright 0。没有跑全仓测试。
