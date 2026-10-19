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
