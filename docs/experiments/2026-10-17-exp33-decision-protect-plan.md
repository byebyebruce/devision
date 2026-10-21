# exp33 方案（草案）：在对齐阶段加入旧决策监督，挡住保护代价（已实现、未运行；等 exp32 结果再选）

日期：2026-10-17。依据：评论家 EXP30-RESULT-01 / EXP30-NEXT-01 第 5 项。exp30 在对齐阶段加了真实图文回放，比 exp24A 好 5–18 个点，但对 v0.3 仍有三项代价：science −4.85、count_wide −10.33、flip −5.83。exp32 会回答"B 阶段的 12,000 道旧决策回放能不能在完整候选上恢复"。只有当 exp32 有迁移收益、但仍有保护代价时，才用本方案；exp32 无代价，或没有迁移收益时，本方案不运行。

## 要回答的问题

**发生漂移的是对齐阶段（align.py 只有 MLM 目标）。在这个阶段每一步同时加入旧决策题的直接监督，能不能在读字门槛不掉的前提下，把 exp30 剩下的保护代价挡住？**

## 与已有尝试的区别

- 不是加更多 COCO 说明文字：那和 exp30 的回放是同一种目标（MLM），exp30 已经显示它挡不住全部代价。
- 不是重复 exp18：exp18 是在决策训练（rlcd.py）里，对旧回放块加 λ = 1 的教师 KL，结果无效（P1 +0.0007 [−0.0030, +0.0044]）。这里改的是两件事：
  - 位置：放在对齐阶段，也就是漂移实际发生的地方；
  - 目标：用 gold 的直接决策监督（选项上的交叉熵），而不是模仿教师输出。

## 设计（与 exp30 只差一项：对齐时加入旧决策题）

- **起点**：r15（与 exp30 相同）。
- **对齐部分（不变）**：exp30 的 40,000 条混合说明文字（`data/exp30/align_mix.jsonl`，`a43037e3…`），batch 16，mask 0.5，2,500 步，投影层 2e-5，LoRA r16 1e-4，视觉塔冻结，seed 0。
- **加入的决策部分**：
  - 数据：v0.3 的训练集 `data/r14b/train.jsonl`（81,875 题：ext 21,002、count_balanced 12,000、pixmo 12,000、existence 10,000、general 9,998、science 7,875、count_original 6,000、mirror 2,000、vsr 1,000），对 v0.3 来说全是旧任务。按块分层抽 20,000 题、保持原比例（每步 8 题 × 2,500 步），固定种子。和 exp32 B 的 12,000 题回放可能重叠，这没有问题：两者都来自 r14b，沿用 r14b 对全部监测 / 评测图片的隔离。
  - 损失：L = L_MLM + w · L_dec。
    - L_dec 是这 8 题在各自选项上的 gold 交叉熵：经过 Laya 决策头，按题平均，不加 RLCD 噪声。
    - w = 1，开训前冻结，不扫。
    - 两项都是按 token / 按题的平均，量级相近：exp30 末段 MLM 约 1.8，r15 在旧任务上的 CE 约 0.7。
  - 训练的参数：与对齐相同，只训投影层和 ModernBERT LoRA。Laya 决策头冻结，用 r15 的权重。这样决策监督只能通过"保持表示与原决策头兼容"来起作用，不会让决策头去迁就新表示，也不改变 exp32 B 阶段的分工。视觉塔冻结。
- **记录**：L_MLM、L_dec、决策题准确率的窗口均值；读字门槛每 500 步评一次（同 exp30）。

## 判定（开训前写下，不改）

- 读字门槛：exp24 规则，终点"真图 − 换图"差 ≥ 0.30。
- 保护：7 个面板对 v0.3 配对，≤ −3 且上界 < 0 记为代价；同时报告对 exp30 的配对差。
- 结局：
  - 门槛未过 → "决策监督挤掉了读字"；
  - 门槛通过、对 v0.3 无预写代价 → "通过保护筛查"：以它为起点重做 exp32 的 A2 + B，A2 也加同样的决策监督，另写修订，评论家放行后运行；
  - 门槛通过、science / count_wide / flip 三项都比 exp30 好（下界 > 0）但仍有代价 → "有改善但不够"，由评论家决定是否调 w 或改成更新决策头；
  - 其他 → "对齐阶段的直接决策监督没有挡住代价"。
- 只报告：ScienceQA 温度 / 磁力 / 浓度开发题（这个 checkpoint 不评判迁移）。

## 实施（已写、未运行；评论家 EXP33-DESIGN-01 准备放行，训练仍等 exp32 结果和实现复核）

- `src/devision/train/align.py`：AlignConfig 新增 `decision_data`、`decision_weight`（默认 1.0）、`decision_micro_batch`（默认 8），CLI 和 YAML 自动获得对应参数。
  - 决策题用 rlcd 的 `_items`（`question_item` 构造，同一 gold / 选项 mask 语义）和 `collate_items`。
  - 用 rlcd 的 `_forward` 前向：决策头冻结，但不在 no_grad 下，梯度能到投影层和 LoRA。
  - 在原始 logits 上算 gold CE（与 `_rlcd_loss` 的 CE 项同一公式），与 MLM 相加后只做一次 backward / clip / optimizer.step。
  - 决策题按自己的种子顺序抽（seed + 1），caption 采样和 mask 的随机流不变。
  - 不给 decision_data、或 weight 为 0 时，与原来只有说明文字的路径完全相同。
- `tests/train/test_align_decisions.py`（tiny 模型，CPU）：
  - weight 0 与不给决策数据的运行逐步损失相同、保存的权重逐位相同；
  - 加入决策题后：第一步的说明文字损失不变（同一批、同一 mask）；训练轨迹改变；投影层和 LoRA（合并进 ModernBERT）有变化；视觉塔和决策头与起点逐位相同；存盘、加载正常。
- `scripts/exp33/build.py` → `data/exp33/decisions.jsonl`：从 r14b 按块比例抽 20,000 题（seed 33，最大余数法），sha256 `a67df880…`。各块：ext 5,130、count_balanced 2,931、pixmo 2,931、existence 2,443、general 2,442、science 1,924、count_original 1,466、mirror 489、vsr 244。
- `configs/exp33-align-decide.yaml`：与 exp30 的 YAML 只差 name 和三个决策参数，dry-run 通过。
- `scripts/exp33/chain.sh`：用 2 小时绝对截止（`runs/exp33.deadline`，重启沿用），对齐和每个评测都在 `capped` 下运行，到点终止整个进程树。
- `scripts/exp33/report.py` + `test_exp33.py`：抽样比例和四种结局。
- 注意（EXP33-DESIGN-01）：w = 1 是单次预定选择；两种损失按 token / 按题平均，不代表梯度量级相近，不据此扫参。

## 预算

- 每步多 8 道决策题的前向和反向。exp30 是 1.85 秒 / 步；决策 micro batch 8 的前向 + 反向在 r15 上约 0.6 秒。估计约 2.45 秒 / 步，2,500 步约 102 分钟，加评 15 个集合约 9 分钟，合计约 1 小时 50 分钟。
- 上限 2 小时：用 exp32 的 `capped` 绝对截止机制，到点停止，不延长。
- 代码实现和测试约 1–1.5 小时，可以在 exp32 训练期间完成，不占 GPU。是否启动由评论家在 exp32 结果出来后决定。

## 已知限制

- 单种子、单个 w；决策头冻结是一种选择，不能代表"更新决策头"的效果。
- 回放题和保护面板来自同一类任务，挡住代价不等于对未见任务也不漂移；最终仍以完整候选的 7 面板和 29 个集合为准。
