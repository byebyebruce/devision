# 完整事实文字 CE：数据、批计划与预写方案（DECISION-FACTS-PREP-02；CPU 准备，未训练）

日期：2026-10-19。落实 docs/experiments/2026-10-18-decision-facts-plan.md 的盘点。这一环节独立于 EXP62，不依赖 EXP62 通过。真实训练另审。

## 数据（`scripts/facts/build.py` → `data/facts/`；只用 EXP57 的 tokenizer / config，不加载权重）

- **来源**：只用 data/real_layout（生成器的图和标签）。不用 ScienceQA、dev_sqa、EXP59 或 EXP60 的任何东西。
- **训练集**：
  - train split 的 600 个温度族 + 600 个磁力族，按稳定哈希 `sha256("facts-v1:" + family)` 按轴分层，排除 EXP57 的 1,100 个 data/cmp 族；每族两个视图；
  - 共 2,400 行、9,600 题（温度每行：比较 + 4 道读数；磁力每行：比较 + 2 道距离读数）。
- **state 的构成**：行自己的 state_text + 本行读数题金标数值的表（exp55 接口：`reading.serialise / fmt`），磁力再加一行 "Also shown in the picture: Pair 1: … Pair 2: …"。
  - 大小取自生成器自己的 caption（captions_train.jsonl：two large magnets / one large magnet and one small magnet），不写任何力的关系。
- **选项与标签**：选项按 `random.Random(sha256(行 id + qid))` 固定换序，金标跟着选项走。标签就是生成器的标签（磁力一次只变一个因素：距离 / 大小 / 相等），不用任何强度公式补标签。
- **比较题金标**（规范类别；磁力里的 "weaker / smaller in Pair 1" 记为 P2 更强）：
  - 温度：A / B / same = 461 / 600 / 139（按轴：same mass 165 / 304，same speed 152 / 152，both differ 144 / 144，equal 139）；
  - 磁力：距离 P1 409 / P2 221，大小 P1 183 / P2 183，相等 204。
  - 磁力距离轴的 P1 偏多：「相等」族的第二个视图把 Pair 2 的距离加长，于是 Pair 1 更强。这一点只披露，不重新配平。
- **fit 切片**：训练集内按哈希分层固定 100 + 100 个完整族（G 条件）。
- **开发集 tune**：tune split 的 150 + 150 族（600 行，2,400 题），从不训练，分三种条件：
  - **G**：金标数字 + 金标大小；
  - **H**：EXP56 登记的实时表（自动数字）+ 金标大小。**混合 oracle，不是产品路径**；
  - **P**：EXP56 的实时表本身（完全自动）。现在没有自动的大小来源，所以磁力在 P 下没有大小事实；这就是今天的产品路径。
  - H 和 P 用的是 EXP56 已登记的 inputs_P（逐行核对 state = serialise(state_text, table)），没有新的模型请求。
- **隔离**：训练与 tune 的族 / case 交集 0 / 0，与 EXP57 训练族交集 0。
- **身份**：train 38649b91…，fit 295ee797…，tune_G f4a5c7ab…，tune_H 58e4147b…，tune_P 6b59b26b…；来源文件（questions / captions 的 train 与 tune、EXP56 inputs_P）的 SHA 记在 MANIFEST。

## 批计划（`data/facts/PLAN.json`，EXP57 的代码路径）

- 流程：read_decision_jsonl → rlcd._items（EXP57 的 tokenizer / config）→ random.seed(1) → plan_batches(8)。
- 结果：9,600 个 item，**1,200 次更新**，每批 8；4,800 个两视图单元，拆开 0；每题曝光恰好 1 次（温度 6,000，磁力 3,600）。
- 截断检查（EXP57 的序列预算）：train / tune_G / tune_H / tune_P 的 state 截断 0，选项不可区分 0；state 最长 104 个 token。

## 训练方案（待另审）

- **起点**：EXP57 冻结端点 962503fb / 09a5b447；text_only（与 EXP57 同一通路，image = None），只训决策 LoRA（r16）和决策头。视觉、读取组件冻结，不用 EXP61 的失败 bundle。
- **参数**：与 EXP57 相同。lr_lora 5e-5，lr_head 1e-5，warmup 100，seed 1，micro 8，1 个 epoch，固定末步，raw 温度 1 评测。

## 预写门槛（运行前固定，不因 EXP61 的结果降低）

- **fit**（G，训练内 100 + 100 族）：沿用 EXP57 原 fit 线。温度和磁力各自要求比较总体 ≥ 0.95，三类召回各 ≥ 0.90（温度 A / B / same；磁力 P1 / P2 / same）。
- **G-tune**：温度和磁力各自比较总体 ≥ 0.80，三类召回 ≥ 0.60，相对起点（EXP57）的 case 簇配对下界 > 0。磁力另报距离 / 大小 / 相等三轴；温度沿用 EXP57 的分轴。
- **P-tune**（温度）：比较 ≥ 0.65，相对起点的配对下界 > 0（EXP57 的线）。
- **H / P（磁力）**：只报告。H 不能写成 P 或产品迁移；P 下磁力的大小轴预期失败（没有大小事实），只作记录。
- **读数保留**：G all ≥ 0.98，P all ≥ 0.866，P_use（EXP55 线）。
- **保护**：exist / science 各 ≥ 起点 − 0.02（经 EXP57-R1-01 的端点 / 明细核对）。
- **分支**（按此顺序）：
  - fit 没学会：只约束这次的监督 / 预算；
  - fit 过而 G 不过：消费的泛化没有建立；
  - G 过而 P（温度）不过：读取与比较的耦合问题；
  - 保护或读数失败：只做保护修复；
  - 全过：下一步要另行许可，包括自动属性来源和实时 B1 验证。不是候选，不发布。

## 预算（硬上限 50 分钟，最后 120 秒报告）

| 阶段 | 依据（EXP57 实测） | 估计 |
|---|---|---|
| 加载、step 0 验证（fit 400 行） | EXP57 约 1 分钟 | 约 1.5 分钟 |
| 训练 1,200 次更新 | EXP57 每步约 1.0 秒 | 约 20 分钟 |
| 末尾验证、合并、保存、冻结检查 | EXP57 约 2 分钟 | 约 2 分钟 |
| decide：fit 400 行（新 + 起点）+ tune G / H / P 各 600 行（新），起点的 G / H / P 也要跑（各 600 行）；共约 4,400 个文字请求 | EXP57 4.7 个请求 / 秒 | 约 16 分钟 |
| exist / science | EXP57 约 5 分钟 | 约 5 分钟 |
| 报告 | | < 1 分钟 |
| **合计** | | **约 45 分钟** |

- 起点在 G / H / P 上的同题记录现在都没有（EXP55 / 56 用的是不同的数据），所以必须新跑。合计接近 50 分钟的上限。送审时可以选择：
  - 起点只跑 G 和 P，H 只报告新端点：约 41 分钟；
  - 或者把上限改为 60 分钟。
  - 这一项留给评审决定，不在这里自行缩减。

## CPU 夹具（`scripts/facts/test_build.py`，4 项，全过）

- 事实行只含 caption 的大小，没有任何关系词；规范类别的映射（weaker in Pair 1 → P2 等）。
- 选项换序后金标跟随选项。
- 训练 state = 行文字 + 金标表 +（磁力）事实行，逐行抽样复算；G / H / P 的差别正如定义（H / P 是 EXP56 的实时表，P 没有事实行，三者题目相同）。
- 隔离、行数 / 题数 / 族数、批计划计数、截断为 0、fit 族属于训练集。
- 这一环节还没有 runner：等方案和预算获批后再实现，沿用 EXP57 的 runner 结构。
