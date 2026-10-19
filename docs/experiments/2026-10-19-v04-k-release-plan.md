# v0.4 K 组合的发布准备（V04-K-RELEASE-PREP-01；CPU 准备，不是任何模型运行的许可）

日期：2026-10-19。依据 LIVE-K-RESULT-01 和 V04-K-RELEASE-PREP-01。

唯一的验证来源：PROTECT 端点 185b9496 / 4f9ccf48 + EXP71 的 selector / head / corrector + LIVE-K 的 K 规则。现在不继续训练，也不再挑选端点。LIVE-K 报告里 candidate_eligible = false 的历史结论保持不变，EXP57 / EXP73 的失败记录也保留。

worktree devision-k-release，分支 k-release-prep，从 devision-livek-run 4a1eea2 拉出。已有的运行和产物只读。

## PREP-01：发布来源与 K 保留（提交 dec19a4，已单独送审）

- 内容：scripts/release/rebind.py 的 "livek" 来源（固定的 LIVE-K 运行、13 项状态 completed、三个门槛为 true、K bundle 及其规则哈希 59478275、PROTECT 端点、EXP71 组件）；bind 保留 K；发布流程只接受 livek 来源；旧的 no-K 流程不变。
- 夹具：tests/release/test_k_source.py。
- 补充 `temp_bundle`：校准用的临时 bundle 写在候选目录之外，组件和 K 规则来自已核对的源，绑定候选当前的 config。它只用来收集路由路径的原始概率，绝不作为证据；一旦 config 定稿，它就不再匹配，加载会被拒绝（有夹具）。

## PREP-02：校准编排（按 prep03 第 5.1–5.3、5.5 节）

### rl_calib（scripts/release/rl_calib.py）

- 规则：用 real_layout 的场景、孪生和渲染器，换新 seed 20261019，按「温度、磁力」交替抽族；每个视图一张图，只保留比较题。
- 隔离：一个族的场景和孪生的 canonical case 都不能出现在 data/ 下任何 JSONL 的 case 字段里（覆盖 real_layout 的 train / tune / confirm、facts、exp6x / exp7x、protect 等，共 5,282 个 case），也不能是转写过的 ScienceQA 场景；rl_calib 内部的 case 也不重复。
- **数量不足，如实报告**（`--dry-run` 实际结果，没有画任何图）：
  - 温度：300 族，满额；
  - 磁力：**只有 104 族**，计划是 300 族，少 196 族。原因是磁力的 case 空间很小：3 个单位的距离组合加大小、相等，一共 769 个 case，其中 646 个已经被现有数据用掉，只剩 123 个空闲；
  - 合计 404 族、808 张图，与已用 case 的重叠为 0。
  - 我没有为了凑数放宽隔离（例如允许与 real_layout train 的 case 重叠）。磁力一侧的区间会更宽，规则照常按「合并 + 各域点估计」判定。如果评审认为必须凑满 300 族，最小的替代方案是只对磁力放宽为「与 tune / confirm / 开发集的 case 不相交，但允许与 train 的 case 重叠」；这样做的代价是可能因为训练见过同一数值组合而高估校准。我不采用这个方案。
- 完整生成要等规则验收后再运行：`uv run python scripts/release/rl_calib.py --root data`，只用 CPU，约 1–2 分钟，写入 data/rl_calib/。
- 确认集还没有生成。以后生成独立渲染器的合成确认时，必须同时排除 rl_calib 的 case（MANIFEST 会列出这些 case）。

### 路由校准分析（scripts/release/route_calib.py）

- **collect**：用已验证的入口 Reading.load(候选, 临时 bundle, full_rules) 加载，设为原始温度，经正式的 evaluate_records 跑 rl_calib；每道比较题都必须 route = read（否则拒绝）；每条记录接上它那一行的 family；组合没有 K 规则就拒绝。
- **analyse**：
  - 先核对：原始概率必须有限且严格大于 0（0 或非有限值一律拒绝，不夹紧）；必须是读取题、有 family；概率和为 1，gold 在选项里。
  - 三套概率都来自同一次原始输出：raw（T = 1）；ordinary（候选 config 按题型 / 选项数取的温度，即 3,892 题普通校准选出的温度）；routed（每个域一个温度，按 sha256(family) 分两折，在一折上拟合、在另一折上评，样本外合并）。
  - 每个族（两个视图一起）计算 Δ_R / Δ_O / Δ_B，给合并值和每个域的值，各自用 2,000 次族重抽（seed 0）直接得到 95 % 区间。
  - 四个分支完全按 5.3（含 R4 的 Δ_B 条件），容许幅度 0.02 nats / 题；ECE（15 分箱）只作报告，不参与判定。
  - 分支 1 选中时，用全部数据为每个域拟合温度；产品改动 temperature_by_route 另行送审。
- 夹具 tests/release/test_route_calib.py（11 项全过）：
  - 5.3 写下的两个反例分别落在分支 3 和分支 4；「ECE 下降但 NLL 升高」不会选中路由温度；
  - 合成的过度自信数据（真实温度 2、config 温度 1）选中分支 1，全数据拟合的温度在 1.6–2.5 之间；已校准的数据走分支 2；
  - config 的分桶温度就是 ordinary 温度；
  - 0、NaN、未读取、缺 family、gold 不在选项里，都会被拒绝；同一族的两个视图在同一折；
  - collect：用真实渲染器生成 2 族的 tiny rl_calib，经 tiny K 组合和正式评测路径，全部是读取题、family 接上、温度为原始温度；没有 K 的组合被拒绝。

### 真实校准运行（需要单独许可）

1. `rebind.py source --kind livek --run runs/livek --decision runs/protect-k/stage2 --bundle runs/livek/bundle_k --round runs/v04-k`，然后 `clone`。候选目录 runs/v04-k/cand，runs/v04-k 下放 run.json。
2. 普通校准：`devision-calibrate --checkpoint runs/v04-k/cand --data data/r13b/val.jsonl --data-root data --in-place`（3,892 题，全部普通题；这时还没有 reading/）。同时在 config 里把 model_name 定为发布名。
3. 生成 rl_calib（CPU）。
4. `rebind.temp_bundle(cand, runs/v04-k/meta/calib_bundle)`；`route_calib.py collect`（808 个读取请求）；`route_calib.py analyse`。
5. 按分支处理：分支 2 → 保留现行温度，进入 bind；分支 1 → 先送审 temperature_by_route 的产品改动，再写入 config；分支 3 / 4 → 按 5.3 停止或另行裁定，不放行。
6. `rebind.py bind`，然后 check_eval freeze（之后的步骤见 PREP-03）。

- 请求数与预算（估算）：

| 步骤 | 请求 / 题 | 依据 | 估计 |
|---|---|---|---|
| 普通校准 | 3,892 题（带图，每题一次前向） | 带局部视图的图片前向约每秒 3–6 题（还没有实测这个入口） | 约 11–22 分钟 + 加载 |
| collect | 808 个读取请求 | LIVE-K live_p 实测每秒 2.15 个 | 约 6.3 分钟 + 加载 |
| analyse | — | CPU | < 1 分钟 |
| 报告 | — | | 约 1 分钟 |

  - 合计约 20–32 分钟。建议总上限 45 分钟，最后 180 秒留给报告，一次启动；runner 沿用已有的 start / watch / report 框架，在许可前再实现并送审。普通校准的吞吐以首窗为准，预计赶不上就判 incomplete，不删题。

## PREP-03：交付顺序与能力声明

- **当前来源的映射**：v0.4 候选的来源是 LIVE-K / PROTECT（PROTECT 端点 + EXP71 组件 + K）。EXP57（fit 未过）和 EXP73（保护代价）的失败记录保留，不改写；旧端点的保真结果不当作新候选的证据。
- **顺序**：
  1. 来源与 K 生命周期验收（PREP-01）；
  2. 普通校准 + 路由校准（PREP-02）；
  3. 最终组合和代码身份冻结（bind → freeze，记录运行身份）；
  4. 已验收的独立渲染器合成确认，加 v0.3 同题对照（120 张图 + 240 个基线请求，约 5–8 分钟）。冻结之后才生成或揭示确认答案；判据和样本数不变；失败按 5.3 否决当前组合，不在确认集上调参；它也填补不了真实来源的缺口；
  5. 29 个集合加 extras 的全评（约 8.6–9 小时）、CPU 延迟、真实候选的干净安装与接口核验；
  6. 发布判定。
  - 便宜的合成确认放在全评之前。confirm_property、confirm_chartqa 及其对照照常保留；已暴露的集合不改名为 heldout。真实来源缺口如实披露，但不因为等一个还没取得的来源而停下能做完的工程准备。
- **剩余关键路径（估算，每项都要单独许可）**：

| 步骤 | 请求 | 普通 / 读取 | 估计 |
|---|---|---|---|
| 校准（PREP-02） | 3,892 题 + 808 | 普通 / 读取 | 约 20–32 分钟 |
| 冻结 | — | — | < 1 分钟 |
| 合成确认 + v0.3 基线 | 约 120 + 120 错图 + 240 | 读取 / 普通 | 约 5–8 分钟 |
| 全评（29 集合 + extras，含对照） | 约 55,900 题（29 集）+ 12,468 题（extras） | 读取题约 726 道，其余普通 | 约 8.6–9 小时（盘点粗估） |
| CPU 延迟 + 干净安装 / 接口核验 | 少量 | — | 约 15–20 分钟 |
| 加载与报告 | — | — | 每段 1–3 分钟 |

  - 不以旧期限为由删集合，也不把准备包称为正式 v0.4。
- **模型卡的说法**（定稿时使用）：
  - 写成「自动区域读取 + 确定性 K 计算 + 决策模型」，不说模型学会了乘方；
  - 分开写三个数：实时读数正确率 93.7 %、读数题选择准确率 96.7 %、温度比较准确率 93.0 %（LIVE-K 的实时 P）；
  - dev_sqa 的温度 100 %、磁力 50.8 % 明确标为开发集成绩；
  - 保护只说「没有触发预写的代价线」；
  - 最终温度改变之后，概率证据必须属于最终身份，不沿用 LIVE-K 原始温度下的数字。
