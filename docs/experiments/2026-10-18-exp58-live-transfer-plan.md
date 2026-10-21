# EXP58 方案：EXP57 端点 + 读取组件的实时保真与真实任务迁移（LIVE-TRANSFER-PREP-01；CPU 准备，未运行）

日期：2026-10-18。只在 EXP57 的预写分支允许（fit、G、P、读数保留、exist / science 全部通过），并且另行获得模型许可后才运行；不与 EXP57 串联。本文件不放行任何运行。

## 为什么要单独一条通路

现有 `devision-eval` 只加载 Decider，并把图片嵌在 state 里调用 decide；`Reading.decide` 在 image=None 时会回退到普通 decide（`src/devision/reading.py:322`）。所以只换 checkpoint，或只换加载器而沿用现有调用方式，都测不到新的「自动读取 + 决策」组合。EXP58 的每个请求都经 `ask()`：

- 读出当前请求实际图片的字节：真图，或 `*_mism.jsonl` 行里固定的错图；
- 以 PIL 图像作为 `image=` 交给 `Reading.decide`，同时传文字 state；
- 图片不放进 state，image 也从不为 None。

请求分组与 devision-eval 相同（`evaluate._requests`：同一图片文件、同一文字 state、每组至多 32 题、按样本顺序），v0.3 的记录也是这样分组生成的。

## 身份

- **决策端点**：EXP57 冻结后的 `runs/exp57/stage2`。`NEW_SHA` 目前是占位，等 EXP57 终点后从 `FROZEN.json` 填入并送审；占位状态下 start 直接拒绝（夹具已覆盖）。
- **start 另外核对**：EXP57 的 `report.json` 分支必须是「全部通过」那一支。
- **读取组件**：选择器 `runs/exp53/selector.pt`（`8f10f1fd`）、字符头 `runs/exp51/head_T.pt`（`0b548142`）。视觉塔和词嵌入跟着端点走：EXP57 的 export 检查保证它们与 EXP50-T 逐位相同。
- **bundle**：start 时用 `write_bundle` 新建 `runs/exp58/bundle`，绑定 EXP57 的权重 / config / tokenizer。现有冻结件都不改，准备阶段也不建真实 bundle。
- **START 绑定**：
  - 模型、bundle 的三个文件、EXP56 的 START / registry / pictures；
  - B 阶段 11 个数据文件，v0.3 的 11 份明细；
  - 代码哈希、请求数；
  - 温度：raw，只在内存里设置。
- **逐题明细的身份**：集合、START、数据 sha、模型、bundle、温度、调用路径。

## 阶段与请求数（`live.py counts`，不加载模型）

| 阶段 | 内容 | 请求 | 题 | 走读取路径 |
|---|---|---|---|---|
| A | 原 600 张 rl_tune 图，每图一个请求（行内全部题） | 600 | 2,400 | 600 张图都读 |
| B1 | dev_sqa temp / force 及其固定错图 | 41 + 120 + 41 + 120 = 322 | 322 | 全部（temperature 82，magnets 240） |
| B2 | 七保护：exist 572 / science 269 / ext 2,771 / count_wide 300 / count_narrow 300 / flip 1,192 / vsr 165 | 5,569 | 7,145 | 只有 science 的 20 题（temperature 9，magnets 11），其余 7,125 题走普通路径 |

## 判据（沿用已有规则，不新设门槛）

- **A 保真**（`fidelity`，全部满足才算通过）：
  - 1,800 个实时读取（解码串、有效性、值）与 EXP56 的 `reads.jsonl` 完全相同；
  - 600 个实际 state 与 EXP56 的 `inputs_P` 相同；
  - 2,400 题与 EXP57 的 `details_P`（同一端点、EXP56 的固定表）预测相同，概率差 ≤ 1e-5。
  - 不通过就停在这里，B 阶段不启动；也不放宽容差去容忍回退路径。
- **B1 迁移**（exp21 `readout.transfer`，68:114 加权，基线是 v0.3 的同题记录）：
  - 准确率变化 ≥ +10 点且下界 > 0，看图收益（真图 − 错图）的变化下界 > 0，即 v0.4 的目标；
  - temp / force 分开报告，各附实际路由数和错误分布；
  - 不改成只报温度，磁铁的限制保留：事实表里只有距离，没有极性 / 大小。
- **B2 保护**（与 exp32 / v04b 相同）：`paired_metric` 对 v0.3 同题记录；ext 用 13 个来源的宏平均，flip 去掉 `:flip` 后与原图配对；差值 ≤ −3 点且上界 < 0 记为代价。science 里被路由的 20 题另报。
- **温度**：v0.3 的记录用的是拟合温度，本轮用 raw = 1（EXP57-NOTE-01）。按题缩放温度不改变 argmax，所以准确率和看图收益可以直接比较；NLL / ECE 不比较。

## 复用与新跑

- **全部新跑**：七保护和 B1 都经同一个 Reading 对象重新评测。
- **不复用的理由**：
  - EXP57 的 exist / science 是普通 Decider 的结果；
  - science 里有 20 题会走读取，所以不能拿普通 science 冒充组合结果；
  - 另外五项 EXP57 根本没评过；
  - exist 虽然没有题走读取（经 Reading 时等于普通 decide），但为了一条路径、不论证等价，也重跑，只多约 3 分钟。
- **v0.3 基线**：复用已核的同题记录（`runs/exp27-transfer/eval/v03`），哈希写进 START。

## 预算与分支（各部分分开许可，先失败的不继续）

| 运行 | 依据 | 估计 | 硬上限 |
|---|---|---|---|
| A + B1（`run.sh A`） | 加载约 1 分钟；EXP56 实测 P50 0.463 秒 / 请求 × 922 个请求 | 约 9 分钟 | 20 分钟，最后 120 秒报告 |
| B2（`run.sh B2`） | EXP50-T 实测 exist 572 个请求 196 秒，即 0.34 秒 / 请求 × 5,569 | 约 33 分钟 | 45 分钟，最后 120 秒报告 |

- **watch 与收尾**：live / transfer / guard 都在 watch 下运行，规则同 EXP56 / EXP57。只跑新目录，不续跑、不重试；任何一步失败都转到 `report --finalize`，缺项记为 incomplete。
- **分支**：
  - A 不保真：新端点下的实时通路没有成立，B 不判；
  - B1 未达目标：temp / force 分开报告，不拿保护结论代替；
  - B1 达标但保护缺项：不称候选；
  - 有保护代价：只做保护修复，不发布；
  - 全过：仍然只是开发集上的结果。dev_sqa 是反复参与开发的监测集，不是新冻结的确认集。v0.4 还需要新的确认集、全量校准 / 评测，以及 CPU / API / 安装检查。

## CPU 准备的验证（`scripts/exp58/test_live.py`，4 项，假对象，不加载模型）

- 真图和固定错图都以 `image=` 到达 Reading，像素颜色核对正确。
- 真实的 `Reading.decide` 中：
  - 被支持的题：在「文字 + 数值表」上决策，读取之后 image = None；
  - 不支持的题：经普通 decide，并带上图片；
  - 请求里没有被支持的题时，不读取。
- 占位端点下 start 拒绝，且不留下任何文件。
- 分组等于 devision-eval 的 `_requests`。

## 第 1 轮修正（EXP58-R1-01–03；CPU，未运行）

- **R1-01 图片身份与 START 契约**：
  - start 写 EXP58 自己的 `pictures.json`：A 的 600 张取自已冻结的 EXP56 清单（不在清单里就拒绝）；B 的每一行按源题单的实际 image 字段（真图和固定错图）在 start 时计算哈希。
  - START 的 `pictures` / `bundle` 结构与 `e56.check_combination` 的契约一致。
  - `check_run` 在加载前和报告时都核对以下各项：代码、bundle 文件、清单；EXP57 权重 / config = START = NEW_SHA；bundle 的 reading.json 绑定的是这份权重 / config；源题单与 rl_tune；在加载前还逐张核对清单里每张图的字节。
  - `ask` 必须拿到清单，A 和 B 的每次读取都核实际字节，不在清单里的图直接拒绝。
- **R1-02 保真闭环**：`verify_fidelity` 每次使用时都从登记证据重新计算，不信缓存的 True：
  - 新的读取经 EXP56 的 `check_reads`：已登记，每图一次，模板齐全，值是其解码串的格式，inputs_P 正是读取的序列化，details_P 绑定 START 的 bundle / 图片清单；
  - 新的明细经 `live_records`：已登记；身份绑定本轮 START、已登记的 inputs_P、模型、raw、bundle、清单；用 `check_details` 核对唯一全集，选项顺序也核；
  - 再与 EXP56 的读取 / state、EXP57 的 details_P（START 绑定其哈希）比较。
  - `fidelity.json` 只是一份记录：B 阶段开始前和 report 时都重新计算。A 或组合核对不过时，报告给 invalid / incomplete，不给迁移或候选结论。
- **R1-03 一次启动与收尾**：
  - 每个部分在做任何事之前都写 `runs/exp58-<部分>.started`，含截止、HEAD；B2 的标记里还记它接续的 START 哈希。标记从不删除，存在就拒绝再次启动。
  - B2 要求 A 已启动且有 START；不再以「成功产物不存在」代替「是否启动过」。
  - report 在没有 START、组合无效、基线异常时都写出可读的 incomplete / invalid 报告；main 兜底，任何异常也写 report.txt。
- **CPU 夹具**（`test_live.py` 共 19 项，全过；假权重和假 Reading，不加载模型）：
  - 非占位的完整链：start → 加载前核对 → live → fidelity 通过；
  - 改 B 图片、换清单、改源题单，都在使用前拒绝；B 请求也核字节（加载后改图、清单外的图）；
  - 保真拒绝：改读取但不重新登记、改预测但不重新登记、重复、漏项、外来身份、读取串相同而实际表不同；
  - 过期的 fidelity 记录：B 拒绝启动，报告给 invalid；
  - 没有 START 时报告给 incomplete；
  - stub 链：start 失败时 finalize，A 不能再次启动；B2 在写出第一个集合前被截止停止，子进程组被杀，然后 finalize，B2 也不能再次启动。

## 事后迁移探索模式（EXP58-EXPLORE-PREP-01；CPU，未运行）

- **背景**：EXP57 按预写规则整体未通过：fit = False，原报告和分支都保留。评审裁决（2026-10-18 22:45:30）认为同域的比较收益有效，先用低成本诊断检验它能否迁移到真实输入。这是**看过 EXP57 结果之后**提出的探索，不让 EXP57 或原投资规则追溯通过。
- **端点**：`NEW_SHA` 填入 EXP57 冻结端点 962503fb / 09a5b447。
- **默认模式**：仍要求 EXP57 走全部通过那一支，所以对这次 fit 失败会拒绝。
- **`explore-after-exp57-fit-fail` 模式**（`run.sh A explore` → `start --mode explore-after-exp57-fit-fail`）：
  - 只接受精确的身份：权重 / config = 962503fb / 09a5b447；EXP57 `report.json` 的 sha256 = e5a8565d；五个门槛正好是 fit false、G_tune / P_tune / reading / screens true；分支以 fit 优先的 "fit not learned" 开头。任何一项不符都拒绝，没有通用的 force 开关；
  - START 记录 `mode`：事后性质、目的、来源报告哈希、全部原门槛和原分支、candidate_eligible = false；
  - `check_run` 在加载前和报告时都重新核对，报告在启动后被改动就给 invalid；
  - report 一律带上 mode 和 candidate_eligible = false，分支前缀为 "post-hoc exploration (EXP57 fit failed, kept; not a candidate, no release)"；即使 B1 达标也不输出候选或发布资格；
  - B2 在探索模式下被拒绝。
- **范围不变**：
  - A：600 张图实时保真，1,800 / 600 / 2,400 和概率差 ≤ 1e-5，都不放宽；A 不成立就停；
  - B1：322 个请求，68:114 准确率和看图收益，temp / force 分项，原统计门槛；
  - 硬上限 20 分钟，最后 120 秒报告，只启动一次。
- **CPU 夹具**（`test_live.py` 共 28 项，全过；不加载模型、不建真实 bundle）：
  - 在真实 runs/exp57 上只读核哈希：默认模式拒绝，探索模式接受（e5a8565d，fit false）；
  - 假环境里，默认模式拒绝 fit 失败且不写 START；探索模式写出事后记录，B2 被拒绝；
  - 报告哈希、缺 screens、G false、fit true、分支、端点任一不符，都拒绝且不写 START；
  - 启动后改报告，组合就是 invalid，candidate_eligible 仍为 false；
  - 伪造的完整阳性结果（A 通过、B1 达标、七保护无代价）在探索模式下仍是 candidate_eligible false，只有默认模式的全部通过才可能为 true。


## 探索模式验收与运行许可

2026-10-18 22:51:15 +08:00，评论家：fc53c1d局部验收通过，放行EXP58-RUN-01唯一一次：在本隔离树执行 `bash scripts/exp58/run.sh A explore`，实际启动+1200秒上限，末120秒报告，全部既有身份/保真/停止规则不变。只含A+B1，不含B2或训练；EXP57 fit失败及candidate_eligible=false保留。许可不等于已经启动。完整评论和回执归档于主仓库长期交付记录。

## EXP58 结果（探索模式 A + B1，22:51:40–22:58:26，完整运行）

- **运行**：
  - HEAD fc53c1d，START 505516bd，bundle 70bd5aa8，report d3ada400；
  - 全部阶段退出码 0，约 7 分钟（上限 20 分钟）；live 2.19 图/秒，B1 3.04 请求/秒。
- **A 保真：通过**。读取 1800 / 1800、state 600 / 600 都与 EXP56 相同；2400 题与 EXP57 details_P 一致，最大概率差 0。
- **B1 迁移：未达目标**（对 v0.3 同题记录）：

| 集合 | v0.3 | EXP58 |
|---|---|---|
| temp 真图 | 19 / 41 | 23 / 41 |
| temp 错图 | 19 / 41 | 17 / 41 |
| force 真图 | 43 / 120 | 47 / 120 |
| force 错图 | 47 / 120 | 45 / 120 |

  - 68:114 加权：准确率变化 +0.057 [−0.048, +0.163]，看图收益变化 +0.086 [−0.026, +0.192]；
  - 分项：temp 准确率 +0.098 [−0.146, +0.317]，force +0.033 [−0.067, +0.142]；
  - 322 题全部走读取路径。
- **报告分支**：post-hoc exploration（EXP57 fit 失败，保留；不是候选，不发布）：dev_sqa 上 68:114 迁移目标未达。candidate_eligible = false；B2 没有运行。
- **限制**：B1 明细没有保存每题的读取值，所以真实 ScienceQA 图片上读数的对错无法从本次产物判定；temp 只有 41 题，区间很宽。
