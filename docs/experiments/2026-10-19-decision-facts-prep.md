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

- 起点在新的 G / P 上没有同题记录，所以必须新跑。
- **评审裁定（DECISION-FACTS-IMPL-01）**：
  - 硬上限 50 分钟，最后 120 秒报告；
  - 新端点跑 fit / G / H / P，起点跑 fit / G / P，不跑起点 H：共 **3,800 个文字请求（15,200 题）**；
  - H 只报告新端点，不算新 − 旧；
  - 按实测估计约 43 分钟（起点不跑 H 省约 2.1 分钟）；
  - 优先级：训练 / 导出 → 全部主判定条件 → exist / science → 新端点 H → 收尾。H 没跑完时记为缺项，主判定与全链完整性分开记录。
- **轴统计的口径**：MANIFEST 里 `axes` 按族统计，用的是该族最后读到的那一行（末视图）的 axis，所以表里没有 equal 这一项。行级的 equal 题仍然存在（温度 139、磁力 204，见比较题金标），报告按每行的 axis 分层。数据不重抽。

## CPU 夹具（`scripts/facts/test_build.py`，4 项，全过）

- 事实行只含 caption 的大小，没有任何关系词；规范类别的映射（weaker in Pair 1 → P2 等）。
- 选项换序后金标跟随选项。
- 训练 state = 行文字 + 金标表 +（磁力）事实行，逐行抽样复算；G / H / P 的差别正如定义（H / P 是 EXP56 的实时表，P 没有事实行，三者题目相同）。
- 隔离、行数 / 题数 / 族数、批计划计数、截断为 0、fit 族属于训练集。
- runner 已实现（暂称 EXP63，见下一节）。

## EXP63 runner（DECISION-FACTS-IMPL-01；CPU，未运行）

- **文件**：`configs/exp63.yaml`、`scripts/exp63/{exp63.py, run.sh, test_exp63.py}`。没有改共享的 rlcd / reading / EXP57 代码，也没有改任何 EXP62 绑定文件；EXP57 的函数通过模块导入复用。
- **训练**：配置固定为 EXP57 端点起跑，data/facts/train.jsonl，val = fit.jsonl，text_only CE，LoRA r16 + 决策头，lr 5e-5 / 1e-5，warmup 100，seed 1，micro / eval batch 8，1 个 epoch（1,200 次更新），eval_every / save_every 0，amp off。start 时逐项核对这些参数。
- **START**（不加载模型）绑定：
  - EXP57 端点（FROZEN 的权重 / config 必须等于 962503fb / 09a5b447）、tokenizer 的树哈希、EXP57 已记录的 exist / science；
  - data/facts 的 5 个文件、MANIFEST（含来源哈希的复算）、PLAN（1,200 × 8、4,800 个完整单元、截断 0）；
  - exp49 questions、EXP54 inputs_P（read_right 集合）、YAML、代码哈希、请求数（3,800 / 15,200）。
- **freeze**：核对正好 1,200 次更新；导出与 **EXP57 的实际端点** 逐张量比较，冻结部分必须相同；然后写 FROZEN。
- **decide**：先新端点 fit / G / P，再起点 fit / G / P（起点重跑，因为这些文件的选项顺序是新的）。之后是 exist / science（devision-eval + freeze.py record），最后才是新端点 H。每个集合完成就登记，明细身份绑定集合、模型、输入 state 的哈希、raw、image = None。
- **报告**：
  - 比较按语义类别统计：温度 A / B / same，磁力 P1 / P2 / same（weaker in Pair 1 = P2）；按行的 axis 分层；整族两视图同时正确；
  - 新 − 起点按 case 簇配对；
  - 读数保留只算 1,800 道字段读数（e54.reading_records 排除比较题），G all ≥ 0.98、P all ≥ 0.866、P_use；
  - exist / science 用 EXP57-R1-01 / 02 的核对（端点路径、逐题完整、summary 由明细复算），起点是 EXP57 已登记的记录；
  - 门槛与分支：fit 优先，缺项给 incomplete（不当作通过或失败）；主判定完整性和全链完整性（含 H）分开报告；candidate_eligible 恒为 false。
- **CPU 夹具**：16 项全过。
  - 语义类别、按轴、两视图统计；
  - 10 个门槛 / 分支用例，含缺项；
  - 条件身份：同一份明细换成另一臂的模型、另一条件的输入、或被改动，都被拒绝；
  - START 绑定 EXP57 的实际端点（权重不符时不写任何文件）；
  - START 之后 facts 数据改动，报告给 invalid；
  - run.sh stub：顺序为 freeze → 主条件 → exist → science → H → 报告；主 decide 失败时直接 finalize，不跑 exist / science / H。
- **演练**：在真实文件上做了不加载模型的 START（/tmp），请求数 3,800、题数 15,200，缺项时报告给 incomplete；pipeline 的 dry-run 接受这份配置，没有创建任何目录。

## EXP63 结果（00:56:33–01:42:22，完整运行；预写分支：fit 未学会）

- **运行**：HEAD df10937（受审 829b7bf），START a482ef6a，report 3abc1710；全部阶段退出码 0，45.8 分钟；main_complete 和 chain_complete 都为 true。FROZEN 3e8443e9 / 70c302c4，export_check ok。
- **比较题**（新端点 vs 起点 EXP57，case 簇区间）：

| 条件 | 新 | 起点 | 配对差 |
|---|---|---|---|
| fit 温度 | 0.940（188 / 200） | 0.790 | +0.150 [+0.085, +0.215] |
| fit 磁力 | 0.995 | 0.520 | +0.475 |
| G 温度 | 0.940 | 0.803 | +0.137 [+0.094, +0.178] |
| G 磁力 | 0.997 | 0.467 | +0.530（22 簇） |
| P 温度 | 0.857 | 0.733 | +0.123 [+0.087, +0.161] |
| P 磁力（只报告） | 0.513 | 0.450 | +0.063 |
| H 磁力（只报告，混合 oracle） | 0.880 | — | — |

- **读数与保护**：G all 0.999，P all 0.887，P_use 通过；exist 696 → 697 / 736，science 346 → 346 / 577。
- **门槛**：fit_temp false（0.940 < 0.95，三类召回都 ≥ 0.90），其余全部 true。
- **分支**：fit not learned (fit_temp)。candidate_eligible = false。
- **事后分层**（评审 EXP63-RESULT-02，不改门槛）：
  - fit 的 12 个、G 的 18 个温度比较错误，四道读数都答对，所以问题在完整文字事实下的比较消费；
  - 「质量 / 速度反向、质量较大者胜」这一类只有 52 / 1,200 条训练题，fit 2 / 8、G 7 / 16。
- **磁力的边界**：
  - G 磁力 0.997 只说明给了完整事实时能消费；
  - P 磁力 0.513（大小轴 91 / 190、相等 3 / 25）是没有自动大小来源的结果；
  - H 0.880 用的是金标大小，是混合 oracle，**不是自动属性能力，也不是产品迁移**。
- 下一步：EXP65（同预算的普通追加 CE vs 规则加权对照）只做 CPU 准备，等另行许可。
