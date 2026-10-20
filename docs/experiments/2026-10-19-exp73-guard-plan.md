# EXP73 方案：固定 EXP71-on 的完整读取组合，评七个保护面板（EXP73-PREP-01；CPU 准备，未运行）

日期：2026-10-19。依据 EXP71-RESULT-01：

- EXP71-on 在 dev_sqa 上达到 v0.3 的开发迁移目标（准确率 +16.67 点 [+7.03, +25.91]，看图收益 +25.40 点）。
- 下一步是看这个完整组合在七个保护面板上有没有代价。
- 这是 EXP71 开发目标达成后另行批准的风险评估。EXP58 探索模式不允许 B2 的禁令保持不变：那次运行和报告都不改，本实验是新的、独立的一次。candidate_eligible 恒为 false。

## 固定的组合（直接使用 EXP71 冻结的 bundle_on）

- EXP57 决策端点 962503fb / 09a5b447（EXP58 的探索身份：fit 未过，START 里披露）；
- 选择器 c92b7b7e、字符头 0e275bdf、校正器 199b5367、局部视图 12 / 0.25 / T；
- 现行问题路由和表格式（devision.reading 的 POLICY / TEMPLATES / HEADER）；
- 用经验证的入口 Reading.load(EXP57, runs/exp71/bundle_on) 加载，核对校正器确实在。
- EXP71 已完成的 A + B1 证据直接引用，不再重跑；不需要再跑一遍 off 条件（on − off 的归因已有 EXP71 的对照；保护的主要参考是 v0.3）。
- 不混入 K、人工事实或新的决策权重。

## 七个面板与请求数（`uv run python scripts/exp73/exp73.py counts`）

| 面板 | 请求 | 题 | 路由到读取链 |
|---|---|---|---|
| exist | 572 | 736 | 0 |
| science | 269 | 577 | 20（温度 9、磁力 11） |
| ext | 2,771 | 3,408 | 0 |
| count_wide | 300 | 300 | 0 |
| count_narrow | 300 | 300 | 0 |
| flip | 1,192 | 1,544 | 0 |
| vsr | 165 | 280 | 0 |
| **合计** | **5,569** | **7,145** | **20** |

- 题单、分组（EXP58 的 devision-eval 分组，同一张图、同一段文字、最多 32 题）与 v0.3 记录的口径相同；路由按实际读取链的 domain_of。
- **预算**：EXP71 实测请求速度约 2.3（含读取）到 3.1 个 / 秒；本次几乎全是普通请求，按约 3 个 / 秒估计约 31 分钟，加载和报告约 1 分钟，合计约 32 分钟。历史 B2 约 5,569 个请求、33 分钟，与此相符。硬上限 45 分钟、最后 120 秒报告；与 EXP72 等其他模型任务串行。

## 阶段（`scripts/exp73/exp73.py`、`run.sh`）

- **start**（不加载模型）：
  - 绑定 EXP71 的运行（START da4c3cec… / report 6869596d… / registry dc93c7a1… 及其全部登记文件）和 bundle_on（reading.json 以及 selector / head / corrector 文件本身，必须等于 EXP71 START 里记录的 on bundle）；
  - 绑定 EXP57 冻结端点和探索记录、EXP58 的运行、七个面板文件、v0.3 的七份逐题明细；
  - 图片清单取 EXP58 已绑定的条目（5,094 张），带数据根时逐张核字节（约 3.3 秒）。
- **guard**（MPS，watch 下）：七个面板的每道题都走 Reading.decide（实际图片作为 image=）。答案、每个请求的读数 / 表 / state 都按面板登记。
  - 本实验自带 run_set：yes / no（noul）题的选项按 EXP58 原样记为 ["false", "true"]。EXP71 的 run_set 只适用于选择题（对 dev_sqa 正确），保持原样不改。
- **report**（不加载模型）：
  - 每个面板先核登记、身份、逐题完整，以及每题用的是该行自己的图；缺失 → incomplete，不符 → invalid，都没有保护结论；
  - 对 v0.3 沿用 EXP58 的代价规则：paired_metric；ext 用 13 个来源的宏平均；flip 原图和翻转图一起重抽；代价 = 差值 ≤ −3 点且区间上界 < 0；
  - science 的 20 道路由题和 557 道普通题分开描述，不用小切片顶替整个面板；
  - 「没有预写代价」不等于等效。

## 预写分支

1. 有面板缺失或无效 → 没有保护结论。
2. 有面板出现预写代价 → 先修保护，不做任何后续；不是候选。
3. 七个面板都没有预写代价 → 仍是开发结果；独立确认、完整校准 / 全评、CPU / API / 安装检查都还没做；不是候选。

## CPU 夹具（`scripts/exp73/test_exp73.py`，6 项，全过；pyright 0 错误；用假读取器，取 count_wide / vsr 两个面板以节省时间）

- START 绑定 EXP71 的 bundle_on；校正器文件被改后 check_start 拒绝。
- 守护阶段：每道题都在该行自己的图上作答。
  - 假读取器复现 v0.3 的答案 → 差值为 0、没有代价；
  - count_wide 上把原来答对的题全改错 → count_wide 出现代价、走「保护代价」分支。
- 缺一个面板 → incomplete；某条记录被改成用别的图作答（并重新登记）→ invalid；两种都没有保护结论。
- run.sh stub：顺序 start → guard → report；guard 失败时 finalize；默认 2,700 秒、留 120 秒。
- 写夹具时发现：EXP71 的 run_set 对 yes / no 题会因为没有 criteria 而出错（EXP71 只跑选择题，没受影响），所以本实验改用自带的 run_set。
