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
