# EXP59 方案：在 24 张转录图上定位真实迁移失败（EXP59-PREP-01；CPU 准备，未运行）

日期：2026-10-18。这是 EXP58 之后提出的机制诊断，不是重试 EXP58，也不是确认集；candidate_eligible 恒为 false。只在评审另发 EXP59-RUN-01 后运行。

## 固定的输入

- **题目**：`scripts/exp21/sqa_transcribed.json` 的 24 张图（温度 12、磁力 12；2026-10-15 用 random.Random(0) 选定），取自当前 dev_sqa 题单，每图一题，原选项、顺序和 state 不变。
  - 不按 EXP58 的错题重新挑选；EXP58 在这个子集上是温度 6/12、磁力 7/12，只作参考值。
- **端点**：与 EXP58 相同的 EXP57 端点（962503fb / 09a5b447）和 EXP58 的 bundle（70bd5aa8）、图片清单（4e29b380）。
  - start 核对：EXP58 的 START / report / registry / fidelity 哈希及其登记的每个文件；`e58.check_run`（模型、bundle、5,855 张图片字节、源题单）。
  - 不建新 bundle，不改 EXP58 的任何文件。
- **数字金标**：`data/exp28/fields.jsonl` 的 72 个字段，正好覆盖每题读取路径用到的模板；每个金标选项文本都要出现在该行的转录里。原注释「由模型目视核对、未经人工复核」保留。
- **磁力事实**（`scripts/exp59/facts.json`）：12 张磁力图逐张目视（2026-10-18，Claude Code，未经人工复核）。
  - 只写每组磁铁的形状和相对大小；不写极性（题目只问大小）、力的关系、规则或答案；
  - start 会拒绝含 force / strong / weak / greater / magnitude / attract / repel 的描述。
- 这些都不进训练。

## 三种条件（共 60 个请求，raw 温度 1）

| 条件 | 请求 | 输入 | 调用 |
|---|---|---|---|
| P | 24 | EXP58 原请求：同一 key、实际图片、行自己的 state 文本 | `Reading.decide(image=实际图片, diagnostics)`；保存框、解码串、有效性、值、表、完整 state、概率 |
| G_num | 24 | 同样的模板和顺序，数字换成转录金标（exp55 的千位格式，`reading.serialise`） | 普通 `decide(image=None)` |
| G_facts | 12（磁力） | G_num 的 state 加一行 "Also shown in the picture: …"（12 条事实见下） | 普通 `decide(image=None)` |

- **P 先复现**：先与 EXP58 已核的同题记录比较，预测相同、概率差 ≤ 1e-5。
  - 不符就停止：G_num / G_facts 不运行，报告给 invalid / incomplete，容差不放宽。
- **G 的输入**：G_num / G_facts 在 start 时离线写好并登记；金标数字只出现在 G 的输入里，P 路径拿到的只有图片和行自己的文字。

## 报告（从逐题明细复算）

- **读取**（P，温度 / 磁力分开）：72 个字段的 exact / invalid；所有需要的数字都读对的图片数；两个对象金标不同、却读成相同值的次数。框只作定位观察，没有 oracle 框，不报 IoU。
- **准确率**：P / G_num / G_facts 分别列出，注明分母。
- **逐题转移和配对差**：G_num − P（温度、磁力各一）、G_facts − G_num（磁力），按图片配对。
- **逐题明细**：每题在三个条件下对不对，以及数字是否全部读对。

## 预写分支（可以同时成立；每类 12 题，区间宽，差异不显著不等于排除）

- **P 数字错而 G_num 改对**：读取端有修复价值。下一步看框落点，再决定最小的选区 / 字符对照，不宣称唯一原因。
- **数字全对仍错，或 G_num 仍错**：文字消费 / 规则问题保留；后续在训练源里准备同类监督，不再只训读数。
- **磁力 G_num 错而 G_facts 改对**：先修「完整事实到达决策」的路径。
- **G_facts 仍错**：完整事实的消费也没有建立。
- 不新设候选通过线，不自动接训练或 B2。

## 运行（`bash scripts/exp59/run.sh`，等许可）

- 顺序：start（不加载模型）→ run（MPS，watch 规则同 EXP58）→ report。
- 硬上限 10 分钟，最后 120 秒留给报告。
- 启动一次就写 `runs/exp59.started`，永不删除；不续跑，不重试；失败就 finalize；没有 START 或身份无效时也写出可读的报告。
- 预计：加载约 1 分钟；24 个读取请求按 EXP58 B1 的 0.33 秒 / 请求约 10 秒；36 个文字请求不到 10 秒。合计约 2 分钟。

## CPU 夹具（`scripts/exp59/test_exp59.py`，6 项，全过；假 Reading / Decider，真实 EXP58 产物只读）

- 24 题 / 72 金标 / 12 条事实；G_facts 是 G_num 的扩展，事实里没有 force 字样。
- diagnostics 不改变 `Reading.decide` 的答案。
- P 不复现时停止：只写 P 明细，G_num 不写；报告给 invalid / incomplete。
- P 复现（假对象返回 EXP58 的原概率）时：
  - P 的 24 次调用拿到的都是真实图片和行自己的 state；
  - G 的 36 次调用都是 image=None，G_num 的 state 正是登记的输入；
  - 三个条件都 completed，candidate_eligible false。
- 改了明细（state 被改，未重新登记）或漏题（重新登记）都拒绝，报告记为 missing / invalid。

## 12 条磁力事实（G_facts 追加的一行）

| 题 | 事实 |
|---|---|
| test-3898 | Pair 1: a large circular magnet and a large circular magnet. Pair 2: a large circular magnet and a small circular magnet. |
| test-2030 | Pair 1: a small bar magnet and a large bar magnet. Pair 2: a large bar magnet and a large bar magnet. |
| test-3589 | Pair 1: two bar magnets. Pair 2: two bar magnets. All four magnets are the same size. |
| test-4069 | Pair 1: two bar magnets. Pair 2: two bar magnets. All four magnets are the same size. |
| test-2097 | Pair 1: a ring magnet and a diamond-shaped magnet. Pair 2: a ring magnet and a diamond-shaped magnet. The two ring magnets are the same size, and the two diamond-shaped magnets are the same size. |
| test-362 | Pair 1: two ring magnets. Pair 2: two ring magnets. All four magnets are the same size. |
| test-1376 | Pair 1: two bar magnets. Pair 2: two bar magnets. All four magnets are the same size. |
| test-2479 | Pair 1: a small diamond-shaped magnet and a small diamond-shaped magnet. Pair 2: a large diamond-shaped magnet and a small diamond-shaped magnet. |
| test-2360 | Pair 1: two bar magnets. Pair 2: two bar magnets. All four magnets are the same size. |
| test-2054 | Pair 1: a solid circular magnet and a ring magnet. Pair 2: a solid circular magnet and a ring magnet. The two solid circular magnets are the same size, and the two ring magnets are the same size. |
| test-4157 | Pair 1: a diamond-shaped magnet and a bar magnet. Pair 2: a diamond-shaped magnet and a bar magnet. The two diamond-shaped magnets are the same size, and the two bar magnets are the same size. |
| test-3665 | Pair 1: two bar magnets. Pair 2: two bar magnets. All four magnets are the same size. |

## 第 1 轮修正（EXP59-R1-01 / 02；CPU，未运行）

- **R1-01**：新增 `check_diagnostics`，P 登记后立即检查，report 读取 P 时再查一次。逐题核对：
  - reads 正好是 reads_for 要求的模板，按顺序，各一次；
  - 每个读取有字段和 4 个数字的框；
  - valid 时解码串是严格数字且 value = fmt(解码串)；invalid 时 value = unreadable，真实的解码失败保留；
  - table 由 reads 按顺序构成；state = 行自己的文字 + serialise(table)。
  - 结构缺失 / 重复 / 链条不符：P 判为 invalid，不做读取或消费的归因；真实的 unreadable 计为读数失败。
- **R1-02**：branches 只统计明确的 True / False。G_num 或 G_facts 缺失 / 无效的题不进入转移，所以 G_num 无效而 G_facts 有效时，不会报「补事实改对」。
- **夹具**（13 项）：
  - 成功夹具改为结构完整的 72 个字段（temp 48 / 48、force 24 / 24 exact）；
  - 真实的 unreadable 计为读数失败（exact 47，invalid 1），P 仍是 completed；
  - 缺模板、重复模板、表不符、state 不符、把 invalid 修成数字，这五种都在「合法写出并登记」之后仍被拒绝，P 记为 invalid，不出读取统计；
  - G_num 无效而 G_facts 有效时，没有「facts」分支；
  - branches 单元用例：缺 G_num 不计，G_num False → G_facts True 照常计数，缺 G_facts 不计 still。


## 局部验收与运行许可

2026-10-18 23:17:19 +08:00，评论家：0793892的EXP59-R1-01/02通过，放行EXP59-RUN-01唯一一次 bash scripts/exp59/run.sh；24图、60请求、600秒硬上限及末120秒报告不变。P完整性/保真失败即停止，不续跑/重试，不接B2/训练/校准/发布；candidate_eligible=false。完整记录归档于主仓库长期交付文档。许可不是实际启动记录。

## EXP59 结果（23:19:38–23:20:02，完整运行）

- **运行**：HEAD 0793892，START afb92f7e，report 844e4154；全部阶段退出码 0，24 秒。
- **P 保真**：24 题与 EXP58 全同，概率差 0。
- **读数**（实时，对转录金标）：
  - 温度 23 / 48：质量 23 / 24，**速度 0 / 24**；数字全对的图 0 / 12；
  - 磁力距离 16 / 24；数字全对的图 5 / 12。
  - 速度读出的多半像质量数值（1,300 → 43 / 46）。一张图上观察到速度框落在质量行和速度行之间，没有覆盖到速度行（只看了一张图，没有 oracle 框，不是归因）。
- **准确率**：

| 类别 | P | G_num | G_facts |
|---|---|---|---|
| 温度（12 题） | 6 | 10 | — |
| 磁力（12 题） | 7 | 6 | 5 |

  - 温度 G_num − P：+0.333 [0, +0.667]，转移 W→R 5 / R→W 1；
  - 磁力 G_num − P：−0.083；磁力 G_facts − G_num：−0.083（W→R 1 / R→W 2）。
- **分支**：
  - 读取端修复有价值：6 题数字错、换成金标后改对，其中温度 5 题；
  - 消费 / 规则问题保留：3 题数字全对仍错，8 题在金标数字下仍错；
  - 磁力：完整事实的消费没有建立，7 题在完整事实下仍错，只有 1 题因补事实改对。
- candidate_eligible = false。
