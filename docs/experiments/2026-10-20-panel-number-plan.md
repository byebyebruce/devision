# EXP75 方案：数字全遮 vs 原随机遮罩，同预算对照（PANEL-NUMBER-PREP-01；CPU 准备完成，未运行模型）

日期：2026-10-20。不算一轮，不产出可采用的模型。worktree `devision-panel-number`，分支 `panel-number-prep`（从 59f7fad 拉出）。

## 问题

D1 已有直接的依图读数（异值两端全对 0.234，真图减错图 +0.234 [0.207, 0.264]），但两端正确率不足（历史阶段 A 是 0.693）。把监督集中到两个完整数字，是否比同预算的额外随机遮罩训练更有效？

## 设计

- **两臂**都从 EXP74 D1 端点（`runs/exp74-panel-d1/d1`，f2efe49d…）各自独立初始化：
  - **N**（`configs/exp75-panel-n.yaml`）：A、B 两个数字的全部 token（含分隔符）同时遮掉，是唯一的目标；量名、单位、标签都可见。
  - **R**（`configs/exp75-panel-r.yaml`）：原随机遮罩（0.5）。
  - 两份配置只在 name、stage 名和 `number_mask_sources: exp26-panels` 上不同（夹具核过）。其余与 D1 相同：原 20,000 张训练图、1,000 张验证图，batch 16，微批 0，投影层 2e-5、LoRA r16 / alpha 32 / 1e-4，warmup 100，seed 0，eval_every 625，原随机验证。
  - 各 1 遍，1,250 次更新。两臂都新建 LoRA 和优化器，不是 EXP26 第二遍的精确恢复。
- **更新的参数**：只有投影层和新 LoRA（保存时合并）。SigLIP、conditioning、决策头和读数组件都不动，不加 decision_data。
- **最小实现**（`src/devision/train/align.py`）：
  - number_spans 和 _number_rows 增加面板格式 `A: <量> <数> <单位>. B: <量> <数> <单位>.`（PANEL_PATTERN，规则与 scripts/panel_bind/probe.py 的解析相同），NUMBERS_EXPECTED 加上 Panel = 2；
  - 原来的 Sample / Pair（real-layout）语义不变；
  - 先完成原随机遮罩的抽样，再覆盖 N 的数字行，所以两臂的题序和随机流完全相同；
  - 缺数、跨度不精确或超长都在前向之前拒绝。
- **CPU 回放**（`scripts/panel_number/counts.py`，真实 tokenizer，不加载模型）：照 align 的随机流重放全部 1,250 批。两臂的批次顺序摘要相同（a3ac7c71…）；20,000 条训练 caption 全部通过 _number_rows，没有一条被拒绝。
  - 受监督 token：**N 数字 62,084、其余 0；R 数字 31,092、其余 106,882**。
  - 不为对齐目标 token 数去改 loss 权重，以免多出一个变量；N 的目标 token 少，这一点照实报告。

## 读出（复用已验收的面板探针，不改口径）

- 每个臂的端点：scripts/panel_bind/probe.py，在固定的 1,000 张图、真 / 错图映射上跑 random 和 numbers 两种模式。
  - random 必须对回该臂自己训练终点的 align_report（step 1,250，±0.003），否则该臂判 invalid；
  - 报告的全部检查照旧（题键完整唯一、family / 图片 / gold、由明细重算并对回 summary、缺失或截断的状态）。
  - scripts/panel_bind/report.py 只改了一处：端点集合从 START 读取，默认值不变（原 19 个夹具仍然通过）。
- D1 和 EXP26 直接复用 PANEL-BIND-RUN-01 已验收的明细，按 SHA 固定（d1 numbers e18bf3f2…、exp26 numbers 41c1c998…、report d7c4e0e4…），不重新加载。
- **比较**（`scripts/panel_number/compare.py`）：
  - 主读数是异值 837 族的"两端都对"（严格口径：两段数字的全部 token 都正确）；
  - 按 family 重采样 2,000 次（seed 0）的配对区间：N − R、各臂 − D1、各臂 − EXP26、各臂的真图减错图；
  - 同答、串到另一端、对调、逐端和同值分层都保留，只作诊断，不替代主判据。

## 输入同源与逐臂有效性（PANEL-NUMBER-R1-01 / R1-02）

- **训练前预检**（`compare.py preflight`，不加载模型；runner 的第一步，失败时以 exit 12 停止，什么都不训练，原因写进 `runs/panel-number-preflight.json`）：
  - 已批准的 D1 model / config / tokenizer（f2efe49d… / 7fb24ad3… / f3458e25…）；
  - train / val（731999b8… / 6a36f5bc…）；
  - 已验收的 panel-bind START（3c4280cd…）及其中登记的 val、图片摘要和 D1；
  - 实际 1,000 张验证图的字节摘要（31e1840f…）；
  - 固定的 D1 / EXP26 明细和 report。
- **比较前**：新 START 的 val sha 和图片摘要必须等于冻结值；每个臂的 numbers 中每个 (family, 真 / 错图) 的 (id, gold, 图片) 必须与固定 D1 明细完全相同，否则该臂判 invalid、不参与任何比较。同一个 family 换了 caption 或图片，不能再和旧成绩配对。
- **逐臂训练报告**（`compare.py start`）：
  - align_report 缺失记为 missing；1,250 个 loss 不全记为 incomplete；截断或无法读取、最后一次验证不在 1,250 步、或数值不有限，都记为 invalid；
  - 状态写进 `runs/panel-number-arms.json`；只有有效的臂进入 START；
  - runner 的探针名单直接取自 START 的端点，与 START 一致；
  - 坏臂不会挡住另一臂；最终比较保留各臂状态，有效臂照常读出，但不给双臂归因；没有有效臂时，比较报告记为 incomplete。

## 投入判断（运行前写下，不看结果改）

- **某臂值得进入"新的决策诊断方案"**：异值两端全对 ≥ 0.60，且（该臂 − D1）的下界 > 0，且该臂真图减错图的下界 > 0。
- **数字监督的额外归因**：N − R 点差 ≥ 0.10 且下界 > 0。
- 结局：
  - 两臂都不过 → 停止这笔额外对齐，不自动加遍数；
  - 只有 R 过 → 承认额外的原配方训练有效，不归功于数字监督；
  - N 过但归因未过 → 只报端点进步，不说数字监督优于多训；
  - N 过且归因过 → 数字监督有额外贡献。
  - 有臂缺证或复现无效 → 该臂不参与归因；两臂不都完成就判 incomplete。
- 这些是这次追加训练的投入线（0.60 接近历史的 0.693，但仍有差距），不是发布线。不追溯改变 EXP74 的 0.30 失败或 dev16 的原要求。
- 两臂结束后由评审决定是否值得新的 D2；不自动运行旧 D2，不开始全评。

## 预算

- 按本机 D1 实测：47.5 分钟一臂（其中一致性检查约 1 分钟不在这次里）。训练 1,250 步加 3 次验证约 46.5 分钟 × 2 = 93 分钟；START（2 个约 2 GB 的权重加 1,000 张图的哈希）约 0.5 分钟；两个探针各约 3.8 分钟，共 7.6 分钟；比较约 0.5 分钟。合计约 **102 分钟**。
- **硬上限 120 分钟**，最后 120 s 留给不加载模型的比较。验证次数（每臂 step 0 / 625 / 1,250 共 3 次）、题和模式都不删，余量约 18 分钟。
- N 和 R 的前向开销相同（同样的 token），反向开销也基本相同。
- runner（`scripts/panel_number/run.sh`）：
  - 只启动一次；启动前两个臂的目录和 runs/panel-number 都必须不存在；
  - 串行执行：N → R → 为已存在的端点写 START → 探针 → 比较；
  - 每个模型步骤在截止前 120 s 结束（TERM，15 s 后 KILL）；
  - 某臂失败或被停止，它在报告里记为 missing，另一臂照常运行并报告。

## 已知限制

单种子；全遮会泄露数字的 token 长度；固定错图同时改变量名和数值，真减错不只隔离标签绑定；开发集已多次使用。即使通过，也不代表普通 decide 或真实输入的迁移；以后进入决策训练，必须检验普通 decide、已封存 read_state 的迁移保留和保护项，不能借用 MLM 成绩或旧候选的全评。RELEASE-GAP-01 仍未关闭。

## 验证（CPU，已完成）

- `scripts/panel_number/test_panel_number.py` 25 passed（R1 之后；新增 15 项：真实文件预检通过；caption / 图片字节 / D1 组件 / panel-bind START 四种变化在训练前被拒；同 family 改 gold、另一份 START 都不参与比较；坏臂保留状态、另一臂照常读出、不归因；训练报告 6 种状态；START 只纳入有效臂；runner 在预检失败时不训练，探针只跑 START 里的臂）。最初的 10 项：
  - 21,000 条面板 caption 都恰好给出 A、B 两个数字（与探针解析一致）；
  - _number_rows 只把这两段设为目标、全部遮掉、其余 token 可见；
  - 旧的 real-layout caption 仍给出 4 个字段；缺数和非面板 caption 被拒绝；
  - 两份配置只差遮罩开关，其余与 D1 相同；
  - tiny 模型上带面板数字遮罩的对齐能跑完；
  - 比较的 5 种结局（stop / R only / N 未归因 / N 归因 / incomplete）；
  - runner：已有臂目录时拒绝、顺序、没有端点时不跑探针、只启动一次、上限。
- 已有测试：test_align_numbers、test_align_micro、test_panel_learn_path、panel_bind 共 32 passed；pyright 0。没有跑全仓测试。
