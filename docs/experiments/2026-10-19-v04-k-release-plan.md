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

## temperature_by_route（V04-ROUTE-TEMP-PREP-01；CPU 产品实现，没有写真实候选）

- **ModelConfig.temperature_by_route**：默认是空的；只接受 "temperature" / "magnets" 两个键，值必须是有限且大于 0 的数（bool、0、负数、NaN、Inf、未知域都会在构造或加载时被拒绝）。为空时 to_dict 不写这个字段，所以新保存的 config 与以前逐字节相同，旧 config 也照常加载。
- **应用位置**：infer._decide 新增按题的 route_temperature，在 logits → softmax 这一步只应用一次，替换这道题的题型 / 分桶温度。Reading.decide 对每道实际走 read 路由的题，按题目文本确定的域查表（所以与有没有 K 块、读数成败、题号和选项位置都无关）；普通题、没有图的请求、reading off 都不受影响。confidence 由最终概率计算。
- **原始温度**：use_raw_temperatures 同时清空题型温度、分桶温度和路由温度。
- **记录与身份**：
  - evaluate 的 temperature 记录在不为空时加入 by_route，check_eval.temperature_table 用同样的规则，所以冻结表和评测记录一致；
  - config 一变，组合身份就跟着变，按既有流程需要重新 bind / freeze；
  - rebind 的白名单本来就包含这个字段。
- **正式写入**：scripts/release/route_temps.py。
  - 来源只能是已验收的那次校准运行（ROUTE-TEMP-R1-01 修正后）：runs/v04-k-calib 的 START.json / registry.json / report.json 三个 SHA 固定在脚本里（ACCEPTED）；再按 registry 核对 route_calibration.json、CALIB.json、records.jsonl 和 rl_calib 题单，按 CALIB 和 registry 核对写入前的候选 config；直接读出报告里的两个全量温度，不重新拟合，也不重算 bootstrap（不加载 route_calib）；必须是分支 1，且两个温度都是有限且大于 0 的数。
  - 只写入登记时的那个候选，只能在 bind 之前写，只改这一个字段，不能写第二次。
  - 不重新核对那次运行的代码身份：本改动有意修改推理模块，那次运行的代码在运行时已经核过，并已由 V04-K-CALIB-RESULT-01 验收。
  - 用真实文件做只读核对（没有写入任何东西）：source_temperatures(runs/v04-k-calib) 得到 {'temperature': 3.238142943837961, 'magnets': 4.664590270988126}，与登记的报告完全一致。
- **夹具**：
  - tests/model/test_route_temperature.py（11 项）：
    - 读取题的概率等于 softmax(logits / T_domain)，普通题用题型温度，confidence 来自最终概率；
    - 打乱题目顺序结果不变；
    - 没有 K、读数失败时仍按域取温度；
    - 普通题、没有图、reading off、旧 config 都不变，空字段不会被写入 config；
    - 原始温度时全部为 1；6 种非法值都被拒绝；
    - save / load 后概率相同；正式 evaluate 记录的 by_route 等于 check_eval 的冻结表，空的时候两边都没有这一项。
  - tests/release/test_route_temps.py（10 项，R1 修正后）：
    - 写入的值精确等于 3.238142943837961 / 4.664590270988126，只改这一个字段；route_calib 被设为一加载就失败，读取和写入仍然成功；
    - 以下情况都被拒绝：另一套内部自洽的分支 1 校准、report / registry / route_calibration / records / 题单被替换、不是分支 1、写到别的候选、候选已有 reading/、候选 config 不是登记时的那份、写第二次。
- 全部 tests 445 项通过，pyright 0 错误。真实候选的 config 没有写，没有 bind / freeze，没有运行模型。

## 合成确认的 CPU 准备（V04-K-CONFIRM-PREP-01；worktree devision-k-confirm，分支 k-confirm-prep，从 07694e6 拉出）

- **scripts/release/confirm.py**：
  - **独立渲染器**：自己的画布、圆角容器加盖子、带描边的粒子、圆角条形和圆形磁铁、带箭头的距离线；字体用 Verdana / Georgia Bold / Trebuchet MS（real_layout 用的是 Arial），布局也不同。从不调用 real_layout 的 temp_render / force_render（夹具会把这两个函数换成一调用就失败）。画面上只有对象、标签、数值和单位：小磁铁就画得小，不写任何能提示答案的文字。
  - **事实定义沿用已验收的规则**：温度用 real_layout 的数值范围和 M × V²；磁力用 force_items 的规则和四种措辞（金标随措辞翻转）；数值取自 rl_calib 的密集值池；题目和金标都由 temp_items / force_items 生成。
  - **集合构成**：
    - 温度 60 张：same mass / same speed / both differ / equal 各 15 张；前三类里 A 和 B 谁更热交替出现，相差不超过 1；每张图 1 道比较题加 4 道读数题；
    - 磁力 60 张：distance / size / equal 各 20 张；每张图 1 道比较题加 2 道距离读数题；size 类照常计入主要分母，单独描述。
  - **隔离**：每张图的 case 必须同时满足：落在 rl_calib 分区的 confirm 一侧；不被任何数据文件使用；不在 rl_calib 里（有 MANIFEST 时读取，否则按固定 seed 重新确定）；不是转写场景；在集合内唯一。配额不足时直接报错，不放宽隔离。
  - **固定错图**：每个域内部做一次错位排列（不会配到自己），作为配错图对照。
  - **正式入口 generate**：
    - 在画任何东西之前，先用 freeze_problems 核对：IDENTITY.json 完整（files / reading / code），是这个 checkpoint 的；checkpoint 的整棵目录树仍与它一致；reading bundle 和 LIVE-K lineage 与冻结时相同（check_eval.reading_identity，即 rebind 的发布来源）；带 K 规则。
    - 输出目录必须是新的。
    - seed 取 int(sha256(IDENTITY.json)[:16], 16)，与本次运行的排除记录（数据文件和 rl_calib）一起确定这个集合。这是流程约束：只在冻结之后、按一次许可、从一份冻结输入生成一次；它不是「冻结前没人能算出来」的密码学保证。
    - 输出 confirm.jsonl、confirm_mism.jsonl 和 MANIFEST（类别、温度题答案的分布、隔离结果、渲染器代码哈希和字体、身份哈希、请求契约）。
  - **preview**：必须给出 seed，每个域最多 4 张图，id 以 preview: 开头，只用来检查渲染器，不是确认集。
  - **outcome**：预写的主要判定。每张图一道比较题，120 张图等权，按图重抽 2,000 次（seed 0）：候选减 v0.3 的准确率 ≥ +0.10 且下界 > 0，并且候选自身「真图减错图」的下界 > 0。这是候选自己的看图收益，不是 68:114 那种增量。三组记录覆盖的图片不一致时，不给结论。
- **请求契约**（每次模型运行都要单独许可）：候选的真图 / 固定错图 / 选项倒序各 120 个请求；v0.3 的真图 / 同一组错图各 120 个请求，共 600 个请求。按 LIVE-K 实测的读取吞吐（每秒 2.2–3.0 个）和普通请求吞吐，约 4–6 分钟，另加两次加载约 1–2 分钟。
- **夹具**（tests/release/test_confirm.py，7 项全过，约 23 秒；只加载了一个 tiny 冻结候选的文件，没有加载任何模型权重）：
  - preview（7 张）：只用独立渲染器（real_layout 的渲染器一被调用就失败）；截获的绘制文字包含题目里的全部数值和单位；题目数量正确；超过 4 张会被拒绝；
  - 金标：温度按 M × V²，与指定的更热一侧一致，equal 为 TEMP_SAME；磁力的 8 种措辞里，金标随措辞翻转；
  - 选择：类别数量正确，A / B 差不超过 1，case 唯一且都在 confirm 一侧，注入的已用 case 不会被选中；
  - 错图是每个域内部的错位排列；
  - 正式入口：tiny LIVE-K 式候选的冻结身份可以通过 freeze_problems（没有生成集合）；以下情况都在任何选择或绘制之前被拒绝，输出目录也不会被创建：没有身份文件、身份不完整、身份属于另一个 checkpoint、来源不是发布来源、冻结后改动了组件；输出目录已经存在也会拒绝；
  - outcome：通过、准确率差不够、没有看图收益、记录集不一致这四种情况。
- 没有生成正式确认集，也没有在正式 seed 上遍历。preview 只在 /tmp 和 pytest 的临时目录里画过 7 张以内的图。

### CONFIRM-PREP 第 1 轮修正（V04-K-CONFIRM-R1-01 / R1-02、NOTE-01）

- R1-01：freeze_problems 在任何选择或绘制之前调用 code_problems，核对冻结的 code：
  - 必须是真实的运行身份记录（modules 每项都有 64 位 sha256，并有 version 和 python）；
  - 必需模块必须齐全；
  - 每个模块的内容必须等于本仓库当前同名源文件（复用 route_calib.runtime_problems，按模块名和内容比较，路径只用来追溯）。
  - 夹具：伪造的非空 code、缺 devision.image、冻结后 reading.py 的内容变了，都在选择之前被拒绝，输出目录不会被创建；同内容换路径可以通过。
- R1-02：outcome 改为接收正式集合的 picture id（picture_ids 从 confirm.jsonl 读取），要求正好 120 个互不相同的 id；三组记录都必须正好覆盖这 120 个 id，值必须是显式的 bool。
  - 夹具：三组共同缺一图、只剩一图、全空、同样数量的外来 id、值为 "yes" 或 1，在 expected 取完整 / 一张 / 空三种情况下都没有结论；完整情况下的通过 / 未通过 / 没有看图收益三个分支保持不变。
- NOTE-01：seed 的措辞收窄为流程约束（见上文）。

### 确认 runner（V04-K-CONFIRM-PREP-02；CPU 实现，不是运行许可）

- scripts/confirm/confirm_run.py 和 run.sh。阶段依次为 start → generate → cand_true → cand_mism → cand_rev → v03_true → v03_mism → report。
  - **start**（不加载模型）：
    - 必须给出 CONFIRM_IDENTITY / CONFIRM_CHECKPOINT；
    - confirm.freeze_problems 必须为空（完整的冻结身份、冻结代码就是本实现、K、LIVE-K lineage）；
    - v0.3 基线是 runs/r15-vision/stage2，model.safetensors 和 config.json 的哈希都已固定（110fea95… / 3a10bacd…）；
    - 写 START，并登记代码、身份和候选权重；只启动一次。
  - **generate**（CPU）：confirm.generate 只运行一次，必须正好 120 张图、480 道题；登记 MANIFEST、confirm.jsonl、confirm_mism.jsonl，以及全部图片哈希的摘要。
  - **五个模型条件**（MPS，串行，在 watch 下）：
    - 候选经 devision.load(reading="on") 加载，必须是 mode = on 且带 k_rule，使用自己冻结的温度；
    - v0.3 经 devision.load(reading="off") 加载，使用自己的温度；
    - 都走正式的 evaluate_records：真图和错图条件的 control 为 none；错图条件直接用 confirm_mism.jsonl 里的固定映射，不在评测时重新抽；选项倒序条件的 control 为 reversed；
    - 每个请求写一行进度；每个条件的明细都带身份行（START、题单、权重、control、reading 状态、runtime_code），并登记。
  - **report**（不加载模型）：
    - 先核对 START 和身份，再核对题单和图片没有变化；
    - 每个条件：已登记、身份绑定、runtime_code 是本实现；每题恰好一次，用的是它那一行的图（错图条件就是固定的那张错图）；correct 是显式的 bool。缺失为 incomplete，其余问题为 invalid，都先于任何数字；
    - 主要判定：confirm.outcome，用正式 120 个 id 的比较题；
    - 读数题和比较题按域和类别、倒序对照、v0.3 的错图，都只作报告；
    - 分支文字注明「合成确认，不是未见真实图片上的确认」。
- 请求：5 个条件 × 120 个请求 = 600 个请求，5 × 480 = 2,400 道题。按 LIVE-K 和校准实测的每秒 2.2–3.4 个请求，约 4–5 分钟，加上两种模型各加载两次约 2–3 分钟，共约 6–8 分钟。
- 总上限 1,200 秒（20 分钟），最后 120 秒留给报告；一次启动，不重试，后面不接任何任务。
- 夹具 tests/release/test_confirm_run.py（6 项，只用假模型，没有生成确认集）：
  - 用假的 120 张图集合（小图、480 道题、confirm.mismatched 产生的固定错图）走完五个条件：错图条件用的确实是固定映射；倒序条件是 480 题；主要判定为通过（候选减 v0.3 为 +0.5）；
  - 缺一个条件 → incomplete；明细被改、或者把一题改到别的图上（连登记也一起改）→ invalid；
  - start 遇到没有冻结的组合、或 v0.3 文件不是固定的那份，都会拒绝；同一个阶段不能跑两次；
  - run.sh：阶段顺序；cand_rev 失败时 finalize 且不再继续；缺 CONFIRM_IDENTITY 时直接退出、不调用任何阶段；上限 1,200 / 120。

## 整合与冻结（V04-K-FREEZE-PREP-01，2026-10-19 13:08–13:16；worktree devision-v04-freeze，分支 v04-freeze）

- **整合**：从 route-temp 9c9a920 拉出，合并确认 runner 1ccbca3（f6cff48）；只有本计划文档冲突，两边都是追加的章节，两节都保留。136da37 加了一个 tiny 夹具：候选 config 有 temperature_by_route 时，check_eval 冻结表和 confirm 的 temperature_of 都带上 by_route；模型运行时没有它就判为 invalid。全量 491 tests 通过，数据脚本 228 项通过，pyright 0 错误。
- **全评配置**（6be0963）：checkpoint 改为 runs/v04-k/cand；history 在原 17 个之外加入 46 个非训练集合，来自 exp30 到 protect-k / LIVE-K / 两次校准的脚本和配置（各阶段 --val / --eval、runner 的筛选集、读数拟合、路由校准题单），训练文件不列入。29 个集合、15 个 extras、显式角色和对照都不变；用 build_plan 比较，加入前后 29 个集合的角色完全相同。test_count_fresh 和 test_bind_gqa 各与旧 history 共享 1 张 COCO 图，这在 v0.3 的配置里就已存在，它们的显式 heldout 不改。
- **来源**：对现有 runs/v04-k/reading_lineage/source.json（c9c7d12e…，等于校准 registry 登记的值）做只读核对，rebind.check_source 和 candidate_problems 都通过；没有重新 clone，也没有重做校准。
- **写值**：写入前的 config 原字节留存在 runs/v04-k/meta/config_before_route_temps.json（186f175c…，只读）。route_temps.py 写入 temperature 3.238142943837961、magnets 4.664590270988126；config 从 186f175c… 变为 7fb24ad3…，只多了这一个字段。
- **bind**：rebind bind 成功，K 规则在；reading.json 9f397a0d…，rules 59478275…；lineage.json ff3667fa…；model.safetensors、encoder、vision、tokenizer 与来源相同。
- **freeze**：check_eval freeze configs/v04-full-eval.yaml --extras configs/v04-extra-evidence.yaml，写出 runs/v04-k/eval/IDENTITY.json（c14863e2…）。温度表包含 by_route；code 的 9 个模块都指向 devision-v04-freeze/src，7 个必需模块齐全。confirm.freeze_problems、rebind.lineage_problems 都没有发现问题；temperature_table 等于 IDENTITY。
- 之后的合成确认、全评和安装核验都必须在这个 worktree 的代码下运行，冻结后不再改动它的推理代码。runs/v04-k-confirm、data/v04_k_confirm 和一次性 started 标记都还不存在；没有加载模型，没有生成确认集。

## 合成确认结果（V04-K-CONFIRM-RUN-01，2026-10-19 13:19:12–13:24:35，完整；HEAD 05c83b4，START 3f482ef2…）

- 只运行了一次，五个条件都是 completed，report 54c2e041…。分支：synthetic confirmation passed（合成确认，不是新真实来源上的确认）。
- 主判据（120 道比较题按图配对，2,000 次图片重采样）：
  - 候选 − v0.3 = +0.133 [+0.042, +0.233]（56/120 对 40/120）；
  - 候选自身 真图 − 错图 = +0.125 [+0.025, +0.233]。
  - 两项都通过。
- 只报告的项：
  - 读数选择题准确率（按图聚类；不是原始数字读取的准确率）：候选真图 force 0.800、temp 0.533；v0.3 真图 force 0.517、temp 0.342；
  - 倒序翻转率：force 0.117、temp 0.353（原写“翻转在答错的题之间”是错的，见下面的事后披露）；
  - equal 类比较题：候选 temp 2/15、force 2/20，是明显弱点。
- 下一阶段（全评）等评审核实后再开始。

### 事后描述性披露（V04-K-CONFIRM-NOTE-01 / 02；只读复算，不改原 report、不改门槛、不在确认集上调参）

- **倒序翻转**：按同题的答案内容配对 cand_true / cand_rev 的明细。
  - temp 全部 300 题中有 106 次翻转：错→对 38、对→错 41、错→错 27。其中比较题 14 次（5 / 6 / 3），读数题 92 次（33 / 35 / 24）。
  - force 180 题中有 21 次：错→对 10、对→错 6、错→错 5。
  - 总体准确率相近，是对→错和错→对互相抵消的结果，不说明逐题稳健；temp 的答案明显受选项位置影响。
- **读数是选择题准确率，不是数字读取成功率**：report 里的 readings_by_picture（temp 0.533、force 0.800）是读数选择题的准确率。直接按模板的对象 / 字段从 cand_true 的 read_table 取值，去掉千位分隔符后与金标数值比较：
  - temp 只有 1/240 精确一致（unreadable 47）；
  - force 57/120（unreadable 14）。
  - 120 张图的读表在倒序前后完全相同（读表不依赖选项）。
  - 例子：temp000 图上两边质量都是 6，速度为 1410 / 380；读表把 1410 / 380 放进了质量字段，速度读成 1 / unreadable。
  - 这说明在新画法上，对象 / 字段的定位或读出泛化明显有问题；仅凭这些记录不能把原因唯一归到 selector、字符头或后续使用环节。
- **范围**：合成确认支持的是本集合上相对 v0.3 的增益和看图收益（比较题总体只有 46.7%，equal 类 2/15、2/20），不能外推为两个域的各子类都改善或都可以可靠使用。之后如果针对这些失败做开发，本确认集要转为已暴露的开发证据，不能再当独立确认。

## 包身份衔接（V04-K-PACKAGE-PREP-01；CPU 方案、工具和 tiny 夹具，待评审；worktree devision-v04-materials）

- **问题**：冻结 IDENTITY 的 code.version 是 0.3.0.dev1，发布计划用正式包 0.3.0。evaluate.code_problems 严格比较 version / python，install_check 用它对照冻结身份，所以模块逐字相同的 0.3.0 也会被拒绝。devision.__version__ 只来自包元数据（src/devision/__init__.py:18），只改 pyproject 版本时，模块内容可以完全不变。
- **做法**：scripts/release/package_bridge.py 写一份明确的打包衔接记录 PACKAGE_BRIDGE.json。只有发布安装检查 install_check --bridge 使用它；全局 code_problems、普通评测和活动 IDENTITY 都不改。
  - **make**（--identity、--wheel、--source 冻结源码树、可选 --sdist）：
    - 目标只能是冻结开发版本对应的正式版（0.3.0.dev1 → 0.3.0；rc、post、local 或其他版本都拒绝），包名必须是 devision；
    - wheel（和 sdist）里每个冻结模块的内容逐字等于冻结值；wheel 里 devision/ 下其他所有文件（评测没有加载的 serve、train/*、static 等）逐个等于冻结源码树；
    - 记录绑定：IDENTITY 的 SHA、两个版本、冻结的 Python、完整的模块映射、wheel 的 SHA 和逐文件 SHA、sdist 的 SHA、uv.lock 的 SHA（安装时用的依赖版本）。
  - **record_problems**（安装之前）：记录必须对应这份 IDENTITY（SHA、from 版本、Python、模块映射完全相同），to 版本是正式版，uv.lock 没变，要安装的 wheel 的 SHA 和逐文件 SHA 等于记录。
  - **installed_problems**（安装之后）：安装进程报告的版本必须是记录的正式版；然后对"只把版本替换成记录值的冻结 code"用原来严格的 code_problems 比较（模块内容、多出的模块、必需模块、Python 都照旧严格）。
  - **install_check --bridge**：只能和 --identity 一起用；先做 record_problems，不通过就不安装；通过后安装记录里的那个 wheel（不重新 build），之后照旧比较读表、路由、状态和全部概率，记录里写入衔接记录的 SHA。
- **夹具**：tests/release/test_package_bridge.py 19 项，都是假源码树、wheel、sdist 和 IDENTITY，不安装、不加载模型：
  - 正式版通过衔接；同一个已安装 code 用原 code_problems 仍然只报版本差异；
  - make 拒绝：冻结模块被改、未加载文件被改、0.3.1、rc、post、包名不同、缺模块、sdist 模块不同；
  - 记录拒绝：另一份 IDENTITY、模块被改、目标被改、uv.lock 变化、另一个 wheel；
  - 安装后拒绝：版本、Python、模块内容不同，或多出模块；
  - install_check：记录不对时不安装；安装的是记录的 wheel；参考 code 只替换了版本；非发布检查使用 --bridge 时拒绝。
- **未做**：没有真实构建 0.3.0、没有上传、没有在真实候选上做安装 smoke。正式 0.3.0 是否可用按发布前流程核实。

### 第 1 轮评审后的修正（PACKAGE-R1-01..03）

- **R1-01**：install_check 只要给了 --bridge 就必须校验，不看解析出的值是什么。null、非对象、缺字段、类型不对、JSON 读不出来，都在 install 和 probe 之前拒绝，绝不退回无 bridge 的路径（自己 build 这棵树）；另有一道守卫，没有核过的 wheel 就不安装。
- **R1-02**：正式衔接必须是 wheel + sdist 一对；wheel-only 只能作开发记录，release 检查拒绝。sdist 的完整 src/devision 文件集合逐个对照批准来源清单（多文件、少文件、内容不同都拒绝）；README.md、LICENSE 等于批准提交里的文件；pyproject.toml 只允许 version 那一行不同，并且必须写目标版本；sdist 里不能有别的文件。使用记录时，重新计算旁边实际 wheel 和 sdist 的 SHA，并重做内容核对。
- **R1-03**：批准来源固定在脚本里：APPROVED[IDENTITY c14863e2…] = {commit 05c83b4…, src/devision 19 个文件的清单摘要 ccfa0d9f…}。make 从任意 clone（--repo）读取这个提交，清单必须等于固定摘要，并且包含 9 个冻结模块且内容一致。wheel 和 sdist 都对照这一份清单；记录里的 source（commit、清单、摘要）在使用时也要等于固定值。同样内容放在别的路径下照样通过。
- **验证**：tests/release/test_package_bridge.py 39 passed；项目 pyright 0 errors。只读核对真实来源：用 devision-v04-freeze 读 05c83b4，得到 19 个文件，清单等于 APPROVED，9 个冻结模块全部一致。没有真实构建、上传或安装。
- **R2-01**（第 2 轮之后）：APPROVED 同时固定 extra_sha256 = 26102aec…0b02（05c83b4 的 README.md、LICENSE，以及去掉版本值后的 pyproject.toml）。make 核对提交取出的 extra；使用记录时，source.extra 的摘要和记录的 extra_sha256 都必须等于固定值，所以同时改 sdist 的 pyproject、记录的 extra 和 sdist 的 SHA，也会在安装前被拒绝。40 passed，pyright 0；只读复算真实 extra 摘要与固定值一致。
- **GITIGNORE-01**（实际打包后）：sdist 构建会把根目录的 .gitignore 放进去，它与 05c83b4 的 .gitignore 逐字相同。APPROVED 为同一身份单独固定 gitignore_sha256 = 00d1f148…37c4，其余固定值不变。sdist 只允许这一个文件名，内容必须等于固定值；make 和使用记录时都按固定值核对实际字节，记录里登记的哈希只用于追溯。make 另外核对批准提交里的 .gitignore 等于固定值；其他多出的文件仍然拒绝。41 passed，pyright 0。
