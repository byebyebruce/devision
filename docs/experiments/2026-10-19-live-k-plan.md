# LIVE-K：把 K 规则接进实时读取组合（LIVE-K-PREP-01；CPU 准备，不是运行许可）

日期：2026-10-19。依据 PROTECT-MID-01 / LIVE-K-PREP-01。

背景：
- PROTECT 的文字训练和 B / C 读出都用 EXP72 的 K 输入：表后附加「mass × speed²」块。
- 现在产品的读取链（reading.py）只把读到的表 serialise 后交给普通决策，没有 K。
- 本文件给出最小接入和后续评测设计。PROTECT 正在运行，devision-protect 的 HEAD、它的 START 绑定的文件和运行产物都没有动。

## 实现（worktree devision-livek，分支 live-k-prep，从 v04-reading-entry 99a917e 拉出）

- `devision.reading.K_RULE`：规则的完整声明，包括 header、quantity、对象 A / B、输入的两个字段、公式（精确整数）、什么情况下有定义（四个值都可读且都是整数）、作用范围（只用于温度题）。
- `k_block(table)`：只从实际读到的表计算 EXP70 的 K 块，与 scripts/exp70/build.py 的 k_block 格式逐字相同：样本 A 在前、B 在后，千位分隔，原 header。
  - 有值缺失、unreadable 或不是整数时返回 None，这道题按无 K 处理，照常作答。
  - 不从题号、选项、gold 或任何缓存取值，也不输出比较结论。
- **默认关闭**：只有 bundle 的 reading.json 声明了 `k_rule` 才打开（`write_bundle(..., k_rule=True)`）。
  - K_RULE 写进 declared_rules，所以 rules_sha256 和 reading.json 的哈希都会变，K 组合与无 K 组合是两个不同的身份，check_eval 的身份链会区分它们。
  - Reading.load 会把声明的 K_RULE 与代码常量逐项比对，不一致就拒绝。
  - 旧 bundle（包括 PROTECT 的 A 所用的、重新绑定的无 K 组合）行为完全不变。
- **作用范围**：在一个请求里，只有温度域的读取题看到「表 + K」；磁力读取题和普通题保持原契约（磁力用无 K 的表 state，普通题走原图片路径）。
  - 诊断里记录 k_block 和实际 state。
  - 关闭时路径与原来逐位相同：已有的 60 项 model 测试全部通过。
- 经已有的统一入口生效：devision.load → Reading → eval / serve；reading_status 里多一项 k_rule。

## CPU 验证（tests/model/test_reading_k.py，6 项全过；不加载真实权重，不做真实前向）

- **600 张自动表**：data/facts/tune_P.jsonl 的每个温度 state，用 k_block 处理它自己的表，得到的结果与 data/protect_p/P_K.jsonl 逐字相同；其中 3 行没有 K（2 行有不可读的值，1 行是小数），300 行磁力不变。
- 规则本身：给出的整数表得到正确的 K；unreadable 或 1.80 → 无 K；14.0 按整数处理（与 EXP70 一致）；没有温度表时无 K。
- 用 tiny 模型和固定读数（替换 read）验证：
  - 关闭时没有 K，打开时只有温度题的 state 带 K；
  - 磁力题和普通题的答案在开和关两种情况下完全相同；
  - route 不变；
  - 换题号、换选项顺序时 K 和 state 不变；
  - 顶层 image= 和 state 图片片段得到相同的 K。
- K 组合与无 K 组合的 reading.json 和规则哈希都不同；改动 K_RULE 声明后加载被拒绝。
- 经 devision.load 加载带 K 的 checkpoint，reading_status 显示 k_rule = true，诊断里有 K 块。
- 全部 tests 366 项通过，pyright 0 错误。

## 后续评测设计（只是设计；前提是 PROTECT 的 A / B / C 全过，且本实现另行验收、另行许可）

1. **组合**：PROTECT 新端点 + EXP71 的三个组件 + K 规则，用 write_bundle(k_rule=True) 生成，得到新身份。PROTECT-A 用的是无 K 组合，它的成绩不能当作 K 组合的保护结果。
2. **P 状态保真**：用新组合在 tune_P 那 300 张温度图上实时读取，得到的 state 与 EXP56 已登记表加 K 的结果逐题对照。读取器不同（EXP71 带校正器，EXP56 的读取器不带），所以这一步报告的是读数差异，不是等价性；C 题的同端点保真靠「给定 P_K state 时的决策与 PROTECT 已登记的 C_P_new 相同」来证明（文字前向，同一端点，同一 state）。
3. **迁移**：dev_sqa 的真图 / 配错图相对 v0.3，沿用原 68:114 迁移目标（准确率差 ≥ +10 点且下界 > 0，看图收益下界 > 0）。
4. **保护**：K 只改变温度读取题的 state。普通题和磁力题与 PROTECT-A 用的无 K 组合逐位相同（上面的夹具已证明这一点），所以七个面板中这些题可以直接引用 PROTECT-A 的明细，并核对答案逐题相同；science 里的 9 道温度读取题必须用 K 组合重新作答，重新算 science 的代价。如果实现改变了一个请求里普通题的输入，这种复用就不成立，要报告范围、整面板重跑。
5. 不生成也不打开新的确认集；候选冻结之前不暴露。
6. 对外只能说「自动读数 + 确定性数值计算 + 模型决策」的组合，不能说模型自己学会了乘方运算。

## 第 1 轮修正（LIVE-K-R1-01 / R1-02；NOTE-01 推迟）

- **R1-01（每题的实际输入）**：在温度 + 磁力的混合请求里，温度题用「表 + K」、磁力题用「只有表」，是两份不同的 state，但原来的诊断只记了带 K 的那一份，evaluate.py 又把它写给了所有读取题。
  - 现在诊断里有 `states`（按题记录各自的实际输入）；`state` 只在所有读取题共用同一输入时才填，否则为 None，不再冒充所有题的共同输入。
  - evaluate_records 按题写入 `read_state = states[题]`。不带 K 时 `state` 和以前完全一样。
- **R1-02（换图后 K 跟随新图）**：新增两张像素不同的图，假读取按实际展示的图返回两套不同的表。
- 新夹具 `test_evaluate_records_logs_each_questions_actual_input`，分 none / mismatched 两种对照：
  - 经正式的 evaluate_records 跑「温度 + 磁力 + 普通」混合请求，记录每次 `_decide` 的输入；
  - 温度题明细 = 展示图的表 + 该表算出的 K，磁力题明细 = 只有表，两者都等于实际的 `_decide` 输入；普通题走图片路径，没有 read_state；
  - 配错图时 K 来自换过的那张图；两张图的 K 不同。
- `test_states_fall_back_without_k`：不带 K 规则，或 K 没有定义时，每道读取题都是同一份不带 K 的 state，`state` 就是它。
- 结果：test_reading_k 9 项全过；全部 tests 369 项通过；pyright 0 错误。
- NOTE-01（scripts/release/rebind.py 重新绑定时没有传 k_rule）：推迟到 K 候选进入发布生命周期时一起做；现在绑定的 EXP71 来源本来就不带 K，不影响已验收的路径。

## runner（LIVE-K-RUNNER-PREP-02；CPU 编排，不是运行许可）

worktree devision-livek-run，分支 live-k-runner（live-k-prep 57532f6 合并了 protect-prep 9f0a038，没有冲突）。脚本 scripts/livek/livek.py 和 run.sh。

- **身份（PREP-02-01）**：start 在任何前向之前写下 START，内容包括：
  - PROTECT 的 START / report / registry 三个固定哈希，以及 registry 登记的全部文件；report 的 15 项状态都是 completed，A / B / C 都是 true；
  - 冻结端点 185b9496 / 4f9ccf48；PROTECT 登记的无 K bundle 6459eb99；
  - EXP71 的 bundle_on；
  - start 时用 write_bundle(k_rule=True) 把同一份 EXP71 组件写成 K bundle（runs/livek/bundle_k）：组件已核对，k_rule、rules_sha256 和 reading.json 都是新身份，PROTECT 的无 K bundle 不动；
  - tune_P、P_K、dev_sqa 的四个文件、七个面板、v0.3 明细、PROTECT 登记的 C_P_new；
  - 本次会打开的全部 1,100 多张图片的 sha256（与 EXP58 清单里重合的部分要一致）；
  - 代码。
- **阶段（PREP-02-02）**：
  1. **fid_text**：新端点在 P_K 的 600 个给定 state 上作答，必须与 PROTECT 已登记的 C_P_new 预测相同、概率差 ≤ 1e-5，否则停止，后面的阶段都不跑（run.sh 夹具已验证）。
  2. **live_p**：K 组合在 tune_P 的全部 600 张图上实时读取。每行的输入是去掉表之后的原文（夹具已验证：原文加 EXP56 的表重新 serialise 后，逐字等于原 P state）。登记每个请求的读数、表、K 块，以及每题实际的 state。
     - 读全部 600 张，而不是只读 300 张温度图：旧门槛的 P_use 还要看磁力的距离读数，磁力题不受 K 影响。
     - 门槛沿用 C：温度比较 ≥ 0.65、P all ≥ 0.866、P_use。其中 read_right 按实时表计算（比较时去掉千位分隔符，因为 EXP56 的表和 target 都没有分隔符）。
     - 另外描述：实时表与 EXP56 表的一致率、实时读数的正确率、温度请求中有 K 的数量。
  3. **transfer**：K 组合在 dev_sqa 的温度 / 磁力真图和固定错图上作答；错图自己读、自己算 K，不复用真图的表。相对 v0.3 用 exp21 的 68:114 加权，判定：准确率差 ≥ +10 点且下界 > 0，看图收益差下界 > 0；两个域分开报告。
  4. **guard_science**：K 组合重跑整个 science 面板（269 个请求，其中有 9 道温度读取题）。
  5. **report**：
     - 先查身份和保真；
     - 每份明细都核对：已登记、身份绑定、源题逐题唯一、每题用的是它自己那一行的图、reads 来自那张图，而且每道读取题的 state 等于「请求原文 + 那张图的表（温度题再加由该表算出的 K）」，被改动就判 invalid；
     - 保护：science 用本次重算的结果；其余六个面板引用 PROTECT-A 已登记的明细，报告里写明文件和哈希。复用的依据是：这六个面板没有一道题会路由到读取（start 时用 domain_of 逐题核对，science 如果出现在复用名单里会被拒绝），所以 K 规则碰不到它们；端点、温度、EXP58 的请求分组和 Reading.decide 的普通路径都相同。
     - 分支：保真不过 → 停止；有缺失或无效 → 没有结论；有保护代价 → 不继续；迁移没达标 → 不继续；P 门槛不过 → 不继续；全部通过 → 开发结果。之后的独立确认、校准和全量发布都要分别许可；不是候选。
- **请求数与预算（PREP-02-03）**：

| 阶段 | 请求 | 依据 | 估计 |
|---|---|---|---|
| start（图片哈希、K bundle） | — | | < 0.5 分钟 |
| fid_text | 600（文字） | PROTECT C 实测每秒 4.9 个 | 约 2 分钟 + 加载 0.5 分钟 |
| live_p | 600（读取） | EXP71 / EXP73 实测每秒 2.3–3.1 个 | 约 3.2–4.3 分钟 + 加载 |
| transfer | 41 + 120 + 41 + 120 = 322 | 同上 | 约 1.7–2.3 分钟 + 加载 |
| guard_science | 269 | EXP73 实测每秒 2.9 个 | 约 1.5 分钟 + 加载 |
| report | — | | 约 1 分钟 |
| **合计** | | | **约 11–14 分钟** |

  - 总上限 30 分钟，最后 120 秒留给报告；一次启动，不重试。
- **CPU 夹具**（scripts/livek/test_livek.py，10 项全过）：
  - START 绑定 PROTECT，换掉 report 的固定哈希就拒绝；K bundle 的组件与 EXP71 相同、身份与无 K bundle 不同；
  - live 输入就是 P 去掉表；
  - 只有不含读取题的面板可以复用（加入 science 会被拒绝）；
  - 文字保真：用登记概率能复现，概率偏 2e-5 就拒绝；
  - 用假读取器走完整链：全过 → 开发结果（science 是重算、六个面板标明复用、297 个温度请求有 K）；真图答错 → 迁移未达标；缺 science → incomplete；改一道题的 state → invalid；
  - 去掉千位分隔符的比较；
  - run.sh：顺序、保真失败时不跑后续、上限 1,800 / 120。
- 全部 tests 384 项通过，pyright 0 错误。scripts/protect 自己的夹具现在在 setup 报错，因为真实的 runs/protect-k 已经存在，它的 start 只允许新目录；这与 EXP72 的情况相同，与本次改动无关。
- 未实现 / 边界：
  - 发布用的 rebind 还没有传 k_rule（NOTE-01，等 K 候选进入发布流程时再做）；
  - 不生成也不打开确认集；
  - 对外只说「自动读数 + 确定性数值计算 + 模型决策」。
