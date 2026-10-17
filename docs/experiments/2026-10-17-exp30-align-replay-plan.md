# exp30 方案：对齐阶段加真实图文回放，能不能挡住旧能力下降（单因素短筛；代码与测试已写，待评审，未运行模型）

日期：2026-10-17。依据：评审 V04-PREP-02（exp28 运行中证据）。exp24-A（从 r15 只做读字对齐）在 exist、science 两个保护面板上分别比 v0.3 低 9.78 [−12.35, −7.15]、10.23 [−13.88, −6.67] 个点，所以最早的保护代价出现在对齐阶段。这一点由评论家用 `runs/exp28-diagnose/eval/exp24A/` 和 exp27 的 v03 记录复算得出。本方案只在 exp28 / exp29 的完整结果出来、评论家放行后才运行。

## 要回答的问题

**在对齐阶段（align.py，被遮说明文字的补全训练）里，把一半的合成读字图换成真实照片 + 说明文字，其余全部不变，旧能力的下降能不能挡住，读字能力还在不在？**

## 设计：与 exp24 阶段 A 只差训练数据

- **不变**：起点 r15，投影层学习率 2e-5，LoRA r16 / 1e-4，视觉塔冻结，MLM 损失，batch 16，warmup 100，mask 0.5，seed 0，总优化步数 2,500。验证集仍是 exp24 的 1,000 张留出读字图，所以门槛可以直接比较。
- **只改数据**（`configs/exp30-align-replay.yaml`，与 `configs/exp24-align.yaml` 只差 name、data、epochs，已 diff）：
  - exp24 的 20,000 张读字图 + 20,000 张真实图（COCO train2014 照片和它们的说明文字），打乱后训一遍（40,000 = 2,500 步）。
  - **取舍**：总步数相同，合成读字图的曝光从 40,000 次（20,000 张 × 2 遍）降到 20,000 次（每张 1 遍），会如实报告。
- **图片隔离**（`scripts/exp30/isolate.py` → `data/exp30/ISOLATION.json`）：
  - 来源池是 `data/align_full_train.jsonl` 的 82,583 张 COCO train2014 图。
  - 排除所有监测、校准、开发、测试、bench 集合用到的图：按 image_id 排除，VG id 先映射到 COCO id（`data/v5/image_identity.json`），镜像图算原图；同时按文件路径排除。覆盖的集合有：r13b 三个面板与 val、ext、v11 两个计数、v6 全部、v5 dev / test、v2 dev / test / bench_pope、lv_bench、r13 gqa dev / test、r13 bench、exp14 pixmo 面板、v10 dev_sqa、exp21 / 22 / 28 的诊断集、早期的 val_* 和 pope。
  - 排除的 id 37,918 个、路径 40,420 个。池中被排除 2,531 张，剩 80,052 张，固定 seed 30 抽 20,000 张（sha256 `faa977b8…`）。逐集合的排除数见 ISOLATION.json，例如 r13b val 1,411 张、exist 面板 572 张、flip 596 张。
  - 混合文件 `data/exp30/align_mix.jsonl` sha256 `a43037e3…`。

## 判定（运行前写下，结果出来后不改；`scripts/exp30/report.py`）

- **读字保留**：exp24 的门槛，即留出读字图上混合补词的"真图 − 换图"差 ≥ 0.30。
- **旧能力**：用 devision-eval 评 7 个保护面板，exist / science / ext 宏平均 / count_wide / count_narrow / flip / vsr。对 v0.3（exp27 记录）配对，≤ −3 且上界 < 0 记为代价；对 exp24A（exp28 记录，同样的对齐、但没有回放）配对报告。
- **结局**（按顺序）：
  - 门槛未过 → "读字丢了"；
  - 对 v0.3 没有任何代价 → "回放挡住了"，v0.4 的对齐阶段采用真实图文回放；
  - exist、science 都高于 exp24A（下界 > 0）但仍有代价 → "回放有帮助但不够"，下一步在对齐阶段加小范围的决策回放 / 原模型输出约束（另写方案）；
  - 其他 → "回放没帮助"，同样转向决策回放 / 输出约束。
- 只报告：ScienceQA 温度 / 磁力 / 浓度开发题的准确率。这个 checkpoint 只做了对齐，不评判迁移。
- 说明文字门槛通过不代表决策能力受到了保护（评审 V04-PREP-02），所以两项分开判断。

## 已知限制

- 单种子、单一比例（1 : 1），不扫比例。
- 合成读字曝光减半，读字门槛可能因此变低；这是本设计的已知取舍。
- 回放用的是说明文字补全（MLM），和决策任务不是同一个目标；挡不住时，下一步用决策回放，不默认同一种回放一定有效。

## 实施（已写好，未运行模型）

- `scripts/exp30/isolate.py`：生成隔离池和混合文件。数据已生成。
- `configs/exp30-align-replay.yaml`：`--dry-run` 通过，带 `--init runs/r15-vision/stage2 --epochs 1 --data data/exp30/align_mix.jsonl`，按 align 的步数公式是 2,500 步。
- `scripts/exp30/chain.sh`：对齐 → 门槛（只记录、不中止）→ 评 15 个集合 → 报告。
- `scripts/exp30/report.py`。
- `scripts/exp30/test_exp30.py`（5 项）：
  - 隔离会按 id（VG 映射到 COCO、镜像算原图）和路径排除，混合文件包含两半，样本不够时报错；
  - 报告的四种结局。
- pyright 0 错误。

## 预算（估计）

对齐约 80 分钟（exp24 阶段 A 实测 79 分钟，步数相同），评 15 个集合约 9 分钟，合计约 1.5 小时，上限 2 小时。每小时检查一次，并挂上完成通知。

命令：`nohup bash scripts/exp30/chain.sh > runs/exp30.out 2>&1 &`
