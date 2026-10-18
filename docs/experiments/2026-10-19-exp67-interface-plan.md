# EXP67 方案：区域 token 包与连续相对几何的 2 × 2 接口诊断（EXP67-PREP-01；CPU 准备，未运行）

日期：2026-10-19。依据 EXP66 终点裁决（EXP66-RESULT-01 / 02）：

- EXP64 的固定头：c92 自动区域下速度 16 / 24，oracle 区域下 24 / 24；但距离从 21 / 24 降到 10 / 24。
- EXP66 换了选择器，速度降到 4 / 24。
- region() 同时改变三样东西：被选的 token 集合、离散的 row / col、连续的相对 geom。所以继续改版式，也说不清该修选择器还是读出头对几何的依赖。
- 本实验只检验这个接口：不改模型结构、训练配方或输入图像。
- 这是使用 oracle 框的事后机制诊断，不是产品推理，也不是独立确认。candidate_eligible 恒为 false。

## 冻结的部分

- EXP57 视觉（962503fb）、n12 / 0.25 局部视图、EXP64 头 0e275bdf、严格数字解码。
- EXP60 的 24 张已转录开发图，72 个字段，全部保留（不只挑速度，也报告距离在紧框下的退化）。
- 不训练、不改值、不跑普通比较题。
- 两套框：
  - A：EXP64 已登记的 c92 自动框（runs/exp64/dev24_auto_new.jsonl 每个读数的 box）；
  - O：EXP60 冻结的 oracle 框（scripts/exp60/oracle_boxes.json）。START 另核对 EXP64 的 oracle 读数用的正是这些框。

## 四个条件（`scripts/exp67/exp67.py`）

- **因子 S（选择包）**：被选 token 的 index、顺序、mask，以及该集合自己的离散 row / col，即 region(box_S) 的输出。整包一起替换，不声称只改了像素内容。
- **因子 G（几何参照框）**：只决定连续 geom 相对哪个框计算：
  - geom = [(cx − bx) / bw, (cy − by) / bh, pw / bw, ph / bh]；
  - (cx, cy, pw, ph) 是该 token 所在视图的原始绝对 patch 几何；
  - 不换视觉值、不重选 token、不改离散 row / col、不截断坐标、不借金标。
- **条件**：AA、AO、OA、OO，第一个字母是 S，第二个是 G。
  - AA / OO 直接用 region() 的原输出（保真）；
  - AO / OA 只替换 geom；
  - padding / mask 语义与原头相同（每个条件的 token 数就是 S 的 token 数）。
- **张量契约**：读出头 CharHead 的输入是逐 token 的 (x, mask, rel, geom)。混合条件保持 x、mask、rel 逐位不变，只换 4 维 geom。没有遇到无法保持 rel 的情况。
- **执行**：每张图的视觉只编码一次（EXP64 的 encode），之后四个条件都在同一份 kv 上走头。

## 保真与停止

- 真实前向之前，START 绑定代码、24 张图片、两套框（含来源文件的哈希）、头、EXP57 / EXP58 和 EXP64 的登记参考。
- 每张图读完后立即核对：
  - AA 必须复现 EXP64 dev24_auto_new 的 decoded / valid / value；
  - OO 必须复现 EXP64 dev24_oracle_new；
  - 任何一处不符就立即停止：写出已有的四个条件文件（标为 stopped）和 repro.json 并登记，报告给 invalid；不调阈值继续。
- **每个条件的文件**（cond_XX.jsonl，读完即登记）逐字段记录：
  - 读数和框、G 框；
  - token 的 index / rel / geom 哈希、token 数，geom 各维的范围；
  - 中心落在 G 框外的 token 数（保留坐标、只计数，不截断）；
  - 全部数值都有限；
  - 原始字符 logits（含 EOS 那一类）。
- 金标只在报告里用来统计。报告另算金标序列的平均对数概率，只作描述，从不作为输入。

## 报告与预写解释（不加载模型）

- 有效性在报告中重新计算：四个文件都已登记、身份绑定、状态为 completed、结构完整、数值有限，AA / OO 对 EXP64 的复现也重算。缺失给 incomplete，其他问题给 invalid，都不下结论。
- 按质量 / 速度 / 距离分别报告：
  - 各条件的正确数；
  - 逐题转移：AA→AO、OA→OO、AA→OA、AO→OO、AA→OO；
  - 差值：AO − AA 与 OO − OA（连续 geom 效应），OA − AA 与 OO − AO（S 包效应），交互 (OO − OA) − (AO − AA)，以及 OO − AA。
  - 区间：该领域 12 张图重抽，同一张图的两个对象放在一起，不当独立样本（r41.interval，seed 0，2000 次）。
- 预写解释（每个字段分别给，距离单独报告，不由速度代判）：
  - geom 的两个差值下界都 > 0：连续几何有稳定正证据，考虑训练读出头的几何稳健性；
  - S 的两个差值下界都 > 0：支持优先修定位 / 选取；
  - 两者都成立：分别记录；
  - 单因子都不成立，但 OO − AA 和交互的下界 > 0：记为「存在正交互，单因素的作用仍不确定、随另一因子的设置而变」。简单效应全部照报；不能推成「只有两者一起才有效」（评审 EXP67-R1-03）；
  - 其他情况：未定位。混合输入本身可能超出训练分布，阴性结果不能据此排除某一部件。
- 任何一个差值都不能推广成整条架构的唯一原因。

## 不加载模型的演练（/tmp）

- START 的身份全部通过：24 张图、72 个字段；两套框覆盖相同的模板。
- 72 个字段上，geom_against 在各自的框上与 region() 逐位相同。
- 中心落在 G 框外的 token 数（72 个字段合计）：AA 3,536、AO 10,901、OA 706、OO 1,597。同源条件也有框外 token，因为 region 按相交格子选取，局部视图之间又有重叠。
- 没有读数时，报告给 incomplete、不下结论。

## 预算与编排（`bash scripts/exp67/run.sh`）

- 顺序：start（不加载模型）→ run（MPS，在 watch 下）→ report（不加载模型）。
- 估计：加载约 30 秒，24 次编码加 4 × 24 次头前向，不到 1 分钟；硬上限 10 分钟，最后 120 秒报告。
- 只运行一次：不扫参、不重试、不自动接后续；与 EXP65 串行（run.sh 检查到其他模型进程就不启动）。
- 只在 EXP67-RUN-01 之后运行。

## CPU 夹具（`scripts/exp67/test_exp67.py`，第 1 轮修正后共 15 项，全过；pyright 0 错误）

- 几何契约：
  - 随机框上，同源条件等于 region() 的输出；
  - 混合条件的 index / view / rel 不变，geom 符合公式，数值有限；
  - 远离 G 框的 token 保留坐标并计数，不截断。
- 读出头的张量构造与 EXP64 的 read_heads 给出相同的读数（随机头、随机 kv）。
- 差值、转移和四类解释分支的用例。
- START：重复启动被拒绝；改动 oracle 框后，EXP60 的身份核对和「EXP64 oracle 读数用的框」核对都会拒绝。
- 报告：
  - 把 EXP64 的已登记读数代入后为 valid，AA 速度 16、OO 速度 24、OO 距离 10，与 EXP64 一致；
  - 缺一个文件给 incomplete；AA 改动一个读数给 invalid；状态为 stopped 给 invalid。
- run.sh stub：顺序正确，run 失败时直接 finalize，默认上限 600 秒、留 120 秒。

## 第 1 轮修正（评审 EXP67-R1-01 / 02 / 03；未运行）

- **R1-01 logits 的数值核对**：
  - run：每次头前向后、解码前，检查 logits 形状为 [字段数, SLOTS 5, CLASSES 12] 且全部有限。不满足时，把已有结果落盘（状态 stopped）并登记，然后停止；不把它当作普通错答。
  - report：对每个读数重新检查已存 logits 的形状和有限性（核的是实际数值，不是 finite 标志），再用 argmax、严格解码和 fmt 重算 decoded / valid / value，与记录逐项比对；任一不符，该条件 invalid，不给效应。
  - 夹具：valid 用例改为与读数自洽的 one-hot logits。新增 4 个拒绝用例：AO 含 NaN、OA 含 Inf、AO 少一行（尺寸错）、OA 的读数合法但不是其 logits 的解码。四个用例中 AA / OO 都照常复现，报告仍给 invalid，没有效应。
- **R1-02 金标来源**：
  - bound_gold 核对 exp59 的 TRANSCRIBED（scripts/exp21/sqa_transcribed.json）和 FIELDS（data/exp28/fields.jsonl）：实际 sha256 必须等于 EXP60 START.sources 冻结的值（838c4c4d… / 2fd94777…）；EXP60 的 START 本身由 bound_e60 绑定；
  - 结果写进本轮 START 的 gold 字段；check_start 每次都重核，start、run（加载前）、report（统计前）都会调用它。
  - 夹具：
    - 把 fields 里一道题的金标换成转录中另一个已有数字（这能通过 exp59 自己的检查），被 bound_gold 拒绝，报告给 invalid，不出正确数；
    - 改动转录文件同样被拒绝；
    - 原文件正常通过。没有重建任何数据。
- **R1-03 正交互的解释**：分支改为「存在正交互，单因素的作用仍不确定、随另一因子的设置而变」，统计算法不变。
  - 夹具：单因素点值为正、区间跨 0，而联合和交互的下界都 > 0，此时给出这条解释，文字不含「只有」。
- **未改的部分**：逐图增量落盘（非阻断说明，本轮不扩展）、四个条件、预算和编排；EXP65 的绑定文件没有动。

## EXP67 结果（04:13:55–04:14:18，完整运行，valid）

- **运行**：HEAD 9d94539，START f3db6c7d，report ec4fe5f0；EXP65 全链结束、确认没有其他模型进程后才启动；全部退出码 0。
- **保真**：AA 和 OO 逐字复现 EXP64；四个条件的 logits 核对全部通过；金标源与 EXP60 一致。

| 条件（S 包 + G 框） | 质量 | 速度 | 距离 |
|---|---|---|---|
| AA（c92 + c92） | 24 | 16 | 21 |
| AO（c92 + oracle） | 24 | 16 | 20 |
| OA（oracle + c92） | 24 | 24 | 9 |
| OO（oracle + oracle） | 24 | 24 | 10 |

- **速度**：
  - 连续 geom 效应为 0（AO − AA、OO − OA 的区间都是 [0, 0]）；
  - S 包效应 +0.333 [+0.125, +0.583]（两种背景下相同），交互为 0；
  - 预写解释：S 包——优先修定位 / 选取。
- **距离**：
  - oracle 的 S 包反而有害：OA − AA −0.500 [−0.708, −0.292]，OO − AO −0.417 [−0.625, −0.250]；
  - geom 和交互的区间下界都不大于 0；
  - 预写解释：未定位（因为两个 S 差值是负的，不是正证据）。
- **质量**：四个条件都是 24 / 24。
- **边界**：这是 24 张反复使用过的开发图上用 oracle 框做的事后机制诊断，不是产品路径或确认；混合输入可能超出训练分布。速度的结论只是：在这组图上，读出头对连续相对几何不敏感，差别来自被选的 token 和离散 row / col。距离单独看：紧框选出的 token 更差。candidate_eligible = false。
