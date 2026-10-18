# exp43 方案：对象名称、空间指代与文字转录对照（准备验收通过，已获单次运行许可；实际启动见回执）

日期：2026-10-18。依据：评论家 EXP43-PREP-01..04。

## 问题

EXP42-C 在原图 tune 上异值两对象同时对只有 0.013，fit_L 也只有 0.042。

- exp37 的 L 图只放大了数字；"Sample A/B"、"Pair 1/2" 和字段名仍是原来的小字。图中位置也不总是 A / Pair 1 在左。
- 所以有两个环节还没分开：一是"对象名称 → 图中区域"，二是"视觉数值 → 决策读出"。
- 小标签**不是**已确认的根因，而是本诊断要隔离的因素之一。

本诊断用三种问法对照：

| 问法 | 图 | 文字 |
|---|---|---|
| 名称（原题） | 原图 / L 图 | 原题 |
| 空间 | 同一张原图 / L 图 | 只把对象名称换成它画在左边还是右边 |
| 文字转录 | 空白图 | 原题 + 两对象的完整数值 |

- 空间问法需要知道真实布局，转录给出了真实数值，所以**两者都是 oracle 诊断条件**：不是产品预处理，不是迁移提升，也不作候选或发布门槛。
- 空间问法阳性，只说明绕开对象名称有用，不单独证明标签读不清；阴性也不证明视觉特征缺失，字段小字和新问法本身仍是限制。

## 固定对象

- 只评 `runs/exp42-c/stage2`：model `65969078…`、config `e5421f88…`。
- 选 C 是为了诊断"纯 CE 集中拟合之后"的这条路径，不是挑模型。不追加别的模型。

## 数据（`scripts/exp43/build.py` → `data/exp43/`；MANIFEST.json）

- **名称问法不新评**：复用 EXP42-C 在温度 1 下评过的 fit_O / fit_L / rl_tune / rl_tune_L 记录。报告重核它们的冻结身份（`freeze.py check runs/exp42-c`）和逐题完整性。
- **范围**：只用读数题（质量 / 速度 / 距离），比较题不新增。
  - fit：EXP42 固定的 144 个单元，原图和 L 各 576 题；
  - tune：300 族、600 张图、1,800 道读数题，原图和 L 逐题对应。
  - 不按预测选样，不碰 confirm 或新冻结确认集，tune 不进训练。
- **空间问法的替换表**（只替换指代，其余不变）：

  | 原题 | 空间问法 |
  |---|---|
  | What is the mass of each particle in sample A/B? | What is the mass of each particle in the sample on the left/right? |
  | What is the average particle speed in sample A/B? | What is the average particle speed in the sample on the left/right? |
  | What is the distance between the magnets in Pair 1/2? | What is the distance between the magnets in the pair on the left/right? |

  - 磁力用 "the pair on the left/right"，而不是评论家原文的 "the left/right pair of magnets"，因为后者在这句里会出现 "the magnets in the left pair of magnets" 这样的重复。意思相同，与温度的 "the sample on the left" 句式一致。如需按原文改，只要改 TEMPLATES 一行并重建。
- **左右从哪里来**：
  - 用生成器自己的布局标志（`a_left` / `p1_left`，只读重放 `real_layout.families()`），不按 gold 或预测推断，也不默认 A / Pair 1 在左。
  - 全部 880 张图再与 exp37（以及 exp40 补画）元数据中的数字框位置交叉核对，全部一致。
  - 构造时逐题断言：生成器给出的数值等于该题的 gold。
- **左右分布**：A / Pair 1 实际画在右边的情况——tune 温度 164/600、磁力 80/300；fit 温度 44/192、磁力 32/96。只报告分层，用来防止固定一侧的偏好掩盖失败。
- **转录**：
  - 空白图 `exp21/images/blank.png` + 原 state_text + 换行 + 两个对象的完整数值。
    - 温度：`Sample A: mass of each particle M u, average particle speed V m/s. Sample B: …`
    - 磁力：`Pair 1: the magnets are D <单位> apart. Pair 2: …`
  - 对象顺序按 sha256(family) 决定，约一半是 B / Pair 2 在前；同一族的两个视图顺序相同。
  - 同一张图的所有题共用同一段转录。不只给被问对象，不写"答案是"；数值来自生成器场景（`rl.fmt` 格式），逐题核对等于 gold。孪生图的数值变化同时体现在转录和 gold 上。
  - 文件里标 `"oracle": true`，模型看不到这个标记。
- **六份新输入**（共 7,128 题）：

  | 文件 | 题数 | sha256 |
  |---|---|---|
  | fit_spatial_O | 576 | `e940b6dc…` |
  | fit_spatial_L | 576 | `de9ff530…` |
  | tune_spatial_O | 1,800 | `151227ca…` |
  | tune_spatial_L | 1,800 | `737fa38e…` |
  | fit_transcript | 576 | `29764f41…` |
  | tune_transcript | 1,800 | `9bdd0b8d…` |

  - 每行保留原 id / qid（orig_id）以便逐题配对，并记录被问对象和它的左右（objects / sides）。
- **12 个对照例子**：`runs/exp43-preview/examples.md`。按固定规则选：温度 3 张 A 在右、2 张速度行在上、1 张默认；磁力 3 张 Pair 1 在右、3 张默认。每张列出名称问法、空间问法、左右、gold 和转录。

## 读出（`scripts/exp43/report.py`；运行前写定）

- **口径**：全部走 decide、温度 1。
- **身份**（不满足就拒绝）：
  - exp42-C 是冻结身份，且等于 65969078 / e5421f88；
  - 每个摘要的 checkpoint、`none (raw)`、control none、data sha 都正确；
  - 逐题无漏无重，gold / image_id 一致，概率、argmax 和 correct 一致（复用 exp41 的 check_details）；
  - 转录记录的 image_used 必须是空白图。
  - 缺的集合记为 incomplete。
- **统计**：沿用 EXP42 的四项——异值单题、异值两对象同时对、值改变的孪生两视图同时对、A/B（或左右）同答；字段等权宏平均。
- **比较**：
  - 名称 → 空间（fit 和 tune、O 和 L 分开）、名称 → 转录：差值按 case 连接簇重抽；
  - 逐题转移：改变数、错→对、对→错；
  - 按 A / Pair 1 实际画在左还是右分层报告。
- **诊断筛查**（不是发布阈值）：
  - **文字端可用**：tune 转录的异值单题宏平均 ≥ 0.90、两对象 ≥ 0.80、孪生 ≥ 0.80。
  - **空间救回**（O、L 分开判）：tune 空间问法两对象 ≥ 0.20，且比同画法的名称问法高 ≥ 0.10、下界 > 0。只有 L 过线，不算真实输入修好；fit 只作辅助说明。
- **证据模式**（交评论家决定，不自动接链）：
  - 1 = 文字端可用，且原图空间救回；
  - 2 = 文字端可用，但原图空间没救回；
  - 3 = 文字端不可用；
  - 其余为 incomplete。
  - 对应的后续训练方向见评论 EXP43-PREP-03，由评论家落实。缺主项不给归因，接近阈值不重试，不为涨分改模板。
- **报告冒烟**：用真实复用的 EXP42-C 名称记录跑报告（新集合缺失）。读出与 EXP42 完全一致（fit_O 0.503 / 0.052，rl_tune 0.393 / 0.013），模式为 incomplete。

## 运行与预算（`scripts/exp43/run.sh`）

- **启动前检查**：没有其他模型进程；exp42-C 冻结检查和 sha 正确；6 份数据的 sha 与 MANIFEST 一致；runs/exp43 原本不存在。
- **截止**：启动后 15 分钟，最后 90 秒留给报告。每个 devision-eval 都是有界运行（到点 TERM，15 秒后 KILL）。
- **顺序**：tune_transcript → tune_spatial_O → tune_spatial_L → fit_spatial_O → fit_spatial_L → fit_transcript。
- **每个请求最多 8 题**（`--max-questions 8`，评论家 EXP43-R1-01）。转录都用同一张空白图，转录相同的图会被合并成一个请求，默认上限 32 下 tune_transcript 最大组会到 32 题；改为 8 后共 405 个请求。其余五集每请求最多 4 题。
- 三个 tune 主项（转录、空间 O、空间 L）中任何一个缺失，证据模式都记为 incomplete；已完成项的筛查保留（EXP43-R1-02）。
- 不训练，不拟合温度，不重试，跑完不接任何别的运行。
- **预算**：EXP42 起点的 4 集共约 5,400 题、5.7 分钟，约 0.064 秒 / 题（含加载）。按此估计 7,128 题约 7.6 分钟，加 6 次加载约 9 分钟，在 15 分钟上限内。

## CPU 验证

- `scripts/exp43/test_exp43.py`，5 项：
  1. 空间问法只改了对象名称：选项及其顺序、gold、图片、state_text 都不变，确实出现了 A / Pair 1 在右的情况，问题里不再有 "sample A/B" 或 "Pair 1/2"；
  2. 左右由布局标志决定，不默认 A 在左；
  3. 转录：两个对象都给出；被问对象的那一段里有它自己的 gold；每张图一段转录；两视图对象顺序相同；四种起始对象都出现；不含 "answer"；
  4. 报告（假的冻结 C + 假记录）：筛查、模式、逐题转移、分母；
  5. 转录记录 image_used 不是空白图时拒绝；缺集合时记为 incomplete。
- pyright 0。没有运行模型。


## 第2轮评审决定（2026-10-18 09:10:17 +08:00）

固定c6dd18b，EXP43-R1-01/02通过。正式放行EXP43-RUN-01：唯一一次当前C的六组诊断，温度1、每请求最多8题、总链15分钟，失败不重试、不自动接下一实验。磁力空间措辞接受；三个tune主项缺任一项仍incomplete。本记录不代表已经启动。完整验收范围见长期交付文档本时刻记录。

## 结果（2026-10-18 09:10:28 – 09:17:49；`runs/exp43.out`、`runs/exp43/{eval,report.json}`、`runs/exp43-report.txt`、`runs/exp43-status.txt`）

**运行**
- 放行 EXP43-RUN-01，代码 c6dd18b。截止 09:25:28；09:17:45 final report exit 0。
- 6 项全部 completed；设备 mps:0；温度 none (raw)；control none；数据 sha 与 MANIFEST 一致。
- 报告 sha256：report.json `60d9ac71…`、report.txt `e4b685ce…`。身份检查通过（exp42-C 为 65969078 / e5421f88），没有缺项。

**读出**（字段等权宏平均）

| 集合 | 异值单题 | 两对象同时对 | 孪生 | 同答 |
|---|---|---|---|---|
| rl_tune（名称，原图） | 0.393 | 0.013 | 0.045 | 0.970 |
| tune_spatial_O | 0.392 | 0.011 | 0.040 | 0.957 |
| rl_tune_L（名称，L） | 0.409 | 0.010 | 0.038 | 0.970 |
| tune_spatial_L | 0.413 | 0.006 | 0.026 | 0.960 |
| **tune_transcript** | **0.972** | **0.943** | **0.943** | 0.055 |
| fit_O / fit_spatial_O | 0.503 / 0.477 | 0.052 / 0.014 | 0.111 / 0.104 | |
| fit_L / fit_spatial_L | 0.507 / 0.488 | 0.042 / 0.017 | 0.073 / 0.080 | |
| fit_transcript | 0.976 | 0.951 | 0.951 | 0.049 |

- tune_transcript 两对象同时对，逐字段：质量 135/138、速度 176/191、距离 79/85。
- **差值**（case 簇区间）：
  - 空间 − 名称：tune O 两对象 −0.002 [−0.018, +0.012]，tune L −0.004 [−0.015, +0.004]；fit O −0.038 [−0.068, −0.010]，fit L −0.024 [−0.049, −0.001]。
  - 转录 − 名称：tune 两对象 +0.930 [+0.893, +0.958]，fit +0.899 [+0.862, +0.932]。
- **逐题转移**（异值题，名称 → 新问法）：
  - tune 空间 O：828 题中改变 100，错→对 36、对→错 38；
  - tune 空间 L：改变 65，错→对 26、对→错 23；
  - tune 转录：改变 524，错→对 504、对→错 14。
- **按 A / Pair 1 实际左右**（tune 异值答对）：
  - 名称 O：A 在左 238/610、A 在右 76/218；
  - 空间 O：239/610、73/218；
  - 转录：589/610、215/218。
- **筛查**：text_usable = true；spatial_rescue_O = false；spatial_rescue_L = false。**证据模式 2**。

**读法**（oracle 诊断，只有一个端点，不作候选或发布证据）
- 文字端可用：同一个 EXP42-C 决策头，在空白图上拿到两对象的数值文字后，几乎总能按对象名称取出正确的值（两对象同时对 0.94），A/B 同答从约 0.97 降到 0.055。
- 绕开对象名称、改成左右指代，在原图和 L 图上都没有救回（两对象同时对仍约 0.01，区间跨 0）；fit 上空间问法反而略低。
- 所以按预写的分支，不再把问题只归于"对象标签读不清"。在这个端点上，"从图中按问题取出对应数值"的视觉读出没有建立起来；"按名称从文字中取值"这一环是可用的。
- 这不证明视觉特征缺失：图中数值仍是小字，空间问法本身也是新措辞。
- 下一步训练方向（模式 2：问题条件下的数值直接监督，与决策共同训练）由评论家落实，我不自动接链。
