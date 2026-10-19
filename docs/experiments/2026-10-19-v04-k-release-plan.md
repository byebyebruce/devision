# v0.4 K 组合的发布准备（V04-K-RELEASE-PREP-01；CPU 准备，不是任何模型运行的许可）

日期：2026-10-19。依据 LIVE-K-RESULT-01 和 V04-K-RELEASE-PREP-01。

唯一的验证来源：PROTECT 端点 185b9496 / 4f9ccf48 + EXP71 的 selector / head / corrector + LIVE-K 的 K 规则。现在不继续训练，也不再挑选端点。LIVE-K 报告里 candidate_eligible = false 的历史结论保持不变，EXP57 / EXP73 的失败记录也保留。

worktree devision-k-release，分支 k-release-prep，从 devision-livek-run 4a1eea2 拉出。已有的运行和产物只读。

## PREP-01：发布来源与 K 保留（提交 dec19a4，已单独送审）

- 内容：scripts/release/rebind.py 的 "livek" 来源（固定的 LIVE-K 运行、13 项状态 completed、三个门槛为 true、K bundle 及其规则哈希 59478275、PROTECT 端点、EXP71 组件）；bind 保留 K；发布流程只接受 livek 来源；旧的 no-K 流程不变。
- 夹具：tests/release/test_k_source.py。
- 补充 `temp_bundle`：校准用的临时 bundle 写在候选目录之外，组件和 K 规则来自已核对的源，绑定候选当前的 config。它只用来收集路由路径的原始概率，绝不作为证据；一旦 config 定稿，它就不再匹配，加载会被拒绝（有夹具）。

## PREP-02：校准编排（按 prep03 第 5.1–5.3、5.5 节）

### rl_calib（scripts/release/rl_calib.py；第 1 轮按 R1-01 修正）

- **场景与题目**：用 real_layout 的场景、孪生和渲染器，新 seed 20261019，按「温度、磁力」交替抽族；每个视图一张图，只保留比较题。
- **隔离**：一个族的两个视图的 canonical case 都必须同时满足：
  - 不出现在 data/ 下任何 JSONL 的 case 字段里（共 5,282 个）；
  - 不是转写场景；
  - 在 rl_calib 内部不重复；
  - 落在固定哈希分区的「calib」一侧：sha256('v04-route-calib-partition:' + repr(case)) < 0.5 为 calib，否则为 confirm。confirm 一侧留给冻结后才生成的合成确认。
- **磁力数值池**：原值池几乎用完（769 个 case 用掉 646 个，剩余的 109 / 7 / 7 分别是 distance / size / equal）。所以只在这个新构建器里用更密的固定值池，单位、范围和小数精度都与原来相同，用整数刻度生成，旧值都包含在内：
  - in：0.25–5.00，步长 0.05，共 96 个值；
  - cm：0.10–8.00，步长 0.05，共 159 个值；
  - mm：5.5–70.2，步长 0.1，共 648 个值。
  - 绘制用 real_layout 原来的渲染器，只在画图时把值池换进去（force_geometry 按值在单位值表里的排名决定间距，所以同一单位只有一种映射，间距随距离增大），画完就恢复。real_layout 的默认行为和旧数据都不变。
- **分层**：磁力的基础场景分为 distance / size / equal 各 100 族。equal 的孪生按原规则变成 distance，所以最终视图的分布是 distance 300、size 200、equal 100。
- **实际数量**（--dry-run，没有画图）：
  - 温度 300 族、磁力 300 族，共 1,200 张图；
  - 与已用 case 的重叠为 0；落在 confirm 一侧的 calib case 为 0；
  - 跳过的抽取：转写场景 16、与已有 case 重叠 154、rl_calib 内部重复 137、落在 confirm 一侧 1,136。
- **确认余量**（confirm 一侧未被使用的磁力 case）：distance 113,225，size 431，equal 419，都远多于合成确认需要的每类 20 个。温度的 case 空间约 2,670 万，已用 4,636 个。
- 完整生成要等规则验收后再运行：`uv run python scripts/release/rl_calib.py --root data`，只用 CPU，约 2–3 分钟，写入 data/rl_calib/。确认集还没有生成。
- **夹具**：
  - tests/release/test_rl_calib.py：值池的范围、个数，且包含旧值；分区；分层；画出了旧值池里没有的值，图中标注文字等于数值和单位；间距随距离单调增加；real_layout 的值池在构建后恢复。
  - tests/release/test_k_source.py 里的 tiny 校准用的就是这个构建器（6 + 6 族）。

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
| collect | 1,200 个读取请求 | LIVE-K live_p 实测每秒 2.15 个 | 约 9.3 分钟 + 加载 |
| analyse | — | CPU | < 1 分钟 |
| 报告 | — | | 约 1 分钟 |

  - 合计约 23–35 分钟。建议总上限 45 分钟（这是待 runner 首窗实测的估计），最后 180 秒留给报告，一次启动；runner 沿用已有的 start / watch / report 框架，在许可前再实现并送审。普通校准的吞吐以首窗为准，预计赶不上就判 incomplete，不删题。

## PREP-03：交付顺序与能力声明

- **当前来源的映射**：v0.4 候选的来源是 LIVE-K / PROTECT（PROTECT 端点 + EXP71 组件 + K）。EXP57（fit 未过）和 EXP73（保护代价）的失败记录保留，不改写；旧端点的保真结果不当作新候选的证据。
- **顺序**：
  1. 来源与 K 生命周期验收（PREP-01）；
  2. 普通校准 + 路由校准（PREP-02）；
  3. 最终组合和代码身份冻结（bind → freeze，记录运行身份）；
  4. 合成确认（确认方案已经验收，独立渲染器本身的实现还要另行验收），加 v0.3 同题对照：候选的真图 120、错图 120、选项倒序 120 个请求（倒序是原方案就有的对照，不是新加的），读数题照常保留；v0.3 的真图 / 错图 240 个请求。约 6–9 分钟。冻结之后才生成或揭示确认答案；判据和样本数不变；失败按 5.3 否决当前组合，不在确认集上调参；它也填补不了真实来源的缺口；
  5. 29 个集合加 extras 的全评（约 8.6–9 小时）、CPU 延迟、真实候选的干净安装与接口核验；
  6. 发布判定。
  - 便宜的合成确认放在全评之前。confirm_property、confirm_chartqa 及其对照照常保留；已暴露的集合不改名为 heldout。真实来源缺口如实披露，但不因为等一个还没取得的来源而停下能做完的工程准备。
- **剩余关键路径（估算，每项都要单独许可）**：

| 步骤 | 请求 | 普通 / 读取 | 估计 |
|---|---|---|---|
| 校准（PREP-02） | 3,892 题 + 1,200 | 普通 / 读取 | 约 23–35 分钟 |
| 冻结 | — | — | < 1 分钟 |
| 合成确认 + v0.3 基线 | 候选：真图 120 + 错图 120 + 选项倒序 120；v0.3：真图 / 错图 240 | 读取 / 普通 | 约 6–9 分钟 |
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

## PREP-02 第 1 轮修正（V04-K-PREP-02-R1-01–04）

- R1-01：rl_calib 改用更密的固定值池，加上 calib / confirm 分区，恢复为 300 + 300 族、1,200 张图（见上文 rl_calib 一节）。
- R1-02：ordinary 温度改用产品自己的 pick_temperature，按精确的桶名取（包括 choice:11+），取不到时退回题型温度；温度必须有限且大于 0。夹具用了普通校准真实会产生的桶名，而且 11+ 排在前面。
- R1-03：正式的 collect 在加载之前核对并登记校准身份（来源、候选推理文件和 config、绑定这个 config 的临时 bundle、题单、全部图片、代码），连同 records 一起写进 CALIB.json。正式的 analyse 先重算身份并核对 records 的哈希，再核对完整性（每题恰好一次，选项、gold、family、域都对得上，两个域都有，每族两个视图都有），之后才做统计。temp_bundle 和 identity 都调用 rebind.candidate_problems（候选权重不变、config 只有白名单字段有差异）。夹具覆盖：config 改动、records 改动、题单改动、缺一个域、缺一个视图、重复题、gold 错、少题，以及非白名单的 config 字段。
- R1-04：NLL 改用 log p / T 的稳定 logsumexp。p = 1e-30、T = e^−3 时结果为 1374.230651，有限，与参考值一致；原始概率为 0 或 NaN 仍然拒绝。
- R1-03 剩余两处（R2）：
  - A：新增 bundle_problems，在 identity 里（collect 加载之前）和 analyse 重算 identity 时都会调用。逐项核对临时 bundle 的实际 selector / head / corrector 文件，以及 reading.json 里记录的哈希，都等于已批准来源的 bundle_on；K 规则、规则哈希和 dims 与源 bundle 相同；sources 指回当前的 source.json 和候选。三个组件的实际哈希登记在 CALIB 里。
  - B：collect 跑完后登记 runtime_code（实际载入的 devision 模块，含路径、内容哈希和版本）。register 和 analyse 都按模块名比较内容：每个模块必须等于本实现里同名的源文件，必需的模块必须载入；路径只用来追溯。固定源码表里缺文件时拒绝，不再静默跳过。
  - 夹具：
    - 自洽但外来的 head（包内哈希已更新，绑定同一个候选；这个 bundle 本身可以被 Reading.load 加载）在加载前被拒绝；
    - 登记之后改动组件，analyse 拒绝；
    - 实际运行的模块内容不同、缺少必需模块，都不能登记；登记之后实现被改，analyse 拒绝；
    - 内容相同、路径不同的模块可以通过；
    - 固定源码缺失时拒绝。

## 校准 runner（scripts/calib/calib.py、run.sh；CPU 实现，不是运行许可）

- 阶段依次为 start → calibrate → rl_calib → temp_bundle → collect → analyse → report。
  - **start**（不加载模型）：
    - 核对 LIVE-K 来源（rebind kind livek）；val 必须正好 3,892 题；
    - runs/v04-k-calib、runs/v04-k、data/rl_calib 都必须还不存在；
    - 写 START（代码哈希、val 哈希、来源及其固定哈希、截止时间）；
    - 之后只动文件：source --kind livek、clone 到 runs/v04-k/cand（runs/v04-k 下有 run.json）。
  - **calibrate**（MPS）：
    - 在同一进程里调用 devision-calibrate 的 --in-place；每次前向都写一行进度，给 watch 的停滞规则用；
    - 把 calibrate_report.json 移出候选目录；把 model_name 定为 devision-v0.4；
    - 核对候选的权重仍是来源的，config 只有 temperature / temperature_by_options / model_name 有差异。
  - **rl_calib**（CPU）：用已验收的构建器写入 data/rl_calib。300 + 300 族、1,200 张图、重叠 0、分区正确，任何一项不符就拒绝。
  - **temp_bundle**（CPU）：rebind.temp_bundle，写在候选目录之外。
  - **collect**（MPS）：先用 route_calib.identity 核对并登记身份，再加载；原始温度；每个请求写一行进度；登记 records 和 runtime_code。
  - **analyse**（CPU）：analyse_registered，结果写入 route_calibration.json。
  - **report**（不加载模型）：
    - 逐个阶段核对登记的文件（缺失 → incomplete，被改 → invalid，都先于任何数字）；
    - 重新计算 analyse，结果必须与已写文件完全相同；
    - 给出普通校准前后的温度、路由分支，以及每个分支允许的下一步。
  - 本 runner 不做 bind、freeze、确认或发布。
- 总上限 2,700 秒（45 分钟），最后 180 秒留给报告；calibrate 和 collect 在 watch 下运行；一次启动，不重试。
- 预算按前面的表：约 23–35 分钟。普通校准的吞吐以首窗为准，预计到不了就判 incomplete，不删题。
- 夹具 tests/release/test_calib_runner.py（5 项全过）：
  - 在临时工作区里用假的 LIVE-K 来源、tiny val（8 题）和真实构建器生成的 tiny rl_calib（6 + 6 族）走完全部阶段：真实的普通校准、进度写到最后一题、报告文件移出候选、发布名已设、候选里没有 reading/；collect 的进度达到全部 24 个请求；报告的各阶段都是 completed，分支来自重算的分析；
  - 缺 collect → incomplete；records 被改 → invalid；start 时来源不对 → 拒绝；
  - run.sh 的阶段顺序、collect 失败时 finalize 且不再跑 analyse、上限 2,700 / 180。
- 用真实文件做只读核对（没有创建任何东西）：LIVE-K 来源核对通过；val 共 3,892 题；三个输出目录都不存在。

## 校准结果（V04-K-CALIB-RUN-01，2026-10-19 12:08:41–12:38:25，完整；HEAD 07694e6，START 2c02a535…）

- 运行：一次启动，没有重试或延长；截止 12:53:41，实际 12:38:25 report 退出码 0；6 个阶段都是 completed；结束后没有残留进程。
  - calibrate 12:08:46–12:29:45：3,892 题，首窗每秒 3.42 题；
  - rl_calib 和 temp_bundle：CPU；
  - collect 12:30:16–12:37:59：1,200 个请求，首窗每秒 2.96 个；
  - analyse：CPU。
- **普通校准**（3,892 题；之前为 [0.5, 1.0, 1.0]，choice:3-5 为 0.5）：
  - choice 题型温度 1.960，noul 2.279；
  - 分桶：choice:2 2.410、choice:3-5 2.174、choice:11+ 1.446、noul:2 2.279。
- **路由校准**（rl_calib：600 族 / 1,200 题，样本外）：
  - NLL：raw 1.192、ordinary 0.703、routed 0.618；ECE（只作报告）：0.176、0.101、0.048。
  - 合并：Δ_R −0.085 [−0.118, −0.054]；Δ_O −0.489 [−0.580, −0.402]；Δ_B −0.574 [−0.697, −0.456]。
  - 温度域：Δ_R −0.048 [−0.083, −0.012]，Δ_O −0.355，Δ_B −0.402；磁力域：Δ_R −0.123 [−0.175, −0.071]，Δ_O −0.623，Δ_B −0.746。
  - 样本外拟合的温度：温度域 3.40 / 3.08，磁力域 4.33 / 4.95；全数据拟合：温度域 3.238，磁力域 4.665。
- **预写分支 1：另设路由温度。**
  - 条件：合并 Δ_R ≤ −0.02 且上界 < 0；各域 Δ_R 点估计 ≤ +0.01；Δ_B 上界 ≤ +0.02（实际为 −0.456）。
  - 下一步：temperature_by_route 的产品改动先送审，通过后再写入 config（温度域 3.238、磁力域 4.665），然后 bind → freeze。这些都还没做。
- 边界：这只说明路由路径的概率校准有改善，不是候选资格；路由温度要等产品改动验收之后才进入组合；最终组合的概率证据属于最终身份。
