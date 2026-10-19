# EXP70 方案：显式给出组合量 K = m·v²，看冻结的 EXP68-X 能否完成比较（EXP70-PREP-01；CPU 准备，未运行）

日期：2026-10-19。依据 EXP68-RESULT-01：

- EXP68 换了新的组合数据后，冲突两类的偏向反了过来（质量胜 0.91，速度胜 0.49），合计没有可靠改善。
- 要区分的是：模型主要卡在「由两个对象的质量和速度形成组合量」，还是给出组合量以后仍然不会完成比较。
- 只做诊断，不训练。不再配平、加权同一种直接分类配方。candidate_eligible 恒为 false。

## 冻结与条件

- **模型**：EXP68-X 端点（runs/exp68-x，model 2e5b272e… / config 87f82281…，FROZEN.json），原始温度 1，普通 decide，没有图片（text_only，与训练和 EXP68 的评测相同）。
- **只有两个条件**：
  - raw：输入与 EXP68 完全相同（提示 + 四个数的表）；
  - K：同一输入，在表后加一个固定的块——一行标题，再按表的顺序（A 然后 B，从不排序）每个对象一行：

    ```
    Computed from the values above (mass times speed squared, no unit):
    Sample A: mass times speed squared = 66,567,600
    Sample B: mass times speed squared = 14,810,400
    ```

  - K 用精确整数运算，只从这一行 state 里的四个数计算；不读金标、选项顺序或答案；不写谁更大；不带单位，不称作真实的能量。数字格式与表一致（exp55 的 fmt，千位分隔）。
  - 题目和选项不变。不扫提示词、格式或其他算式。

## 数据（`scripts/exp70/build.py` → `data/exp70/`，只用 CPU）

| 文件 | 请求数 | 内容 |
|---|---|---|
| diag68_K.jsonl | 480 | EXP68 新 diag 的全部请求，加 K 块（raw 结果复用 EXP68 已登记的 x_diag68） |
| fid_raw.jsonl | 48 | 新 diag 里按 sha256("exp70-fid:" + id) 每类取 8 个，raw，重新 decide 作保真核对 |
| train_raw.jsonl | 120 | data/exp68/train.jsonl 里按 sha256("exp70-train:" + family) 每类取 10 个温度族（60 族 × 两视图），只留比较题 |
| train_K.jsonl | 120 | 同一训练切片，加 K 块 |
| **合计新请求** | **768** | |

- **核对**（全部行）：
  - state 里的 K 等于表中数值的精确计算；训练切片的表等于该行的金标读数；
  - K 的大小顺序就是金标（相等时为「neither」）；
  - 题目、选项、金标与来源逐字相同；
  - 只改金标或选项顺序时，K 块不变（夹具）。
- **截断**（EXP68-X 的 tokenizer / config，不加载权重）：四个文件都没有 state 被截断、选项都可区分；state 最长：raw 91 个 token，K 139 个 token。最大的 K 是 177,382,400。
- **身份**：diag68_K 065c11ea…，fid_raw c11ebabd…，train_raw 1e85f1aa…，train_K efb3c472…；来源 data/exp68 的 MANIFEST eed507d8… 及 train / diag。重建结果逐字节相同。
- 训练切片只用来检查 EXP68-X 在自己训练组合上的末点拟合，不把旧 facts 的 fit 当作本轮的训练 fit。

## 读出与预写判定（运行前冻结）

- **有效性**：
  - 文件已登记、身份绑定（EXP68-X 端点、START、输入哈希、原始温度、image = None），逐题唯一完整；
  - raw 的 48 个保真请求必须与 EXP68 已登记的 x_diag68 一致：预测完全相同，概率差 ≤ 1e-4；
  - 不一致给 invalid，缺项给 incomplete，都不下结论。
- **主要读出**（新 diag，480 个请求，K 与 raw 对比）：
  - 两个冲突类各自的准确率、两视图都对的比例、near / large 分层；
  - 冲突两类等权宏平均的 K − raw，以及每个冲突类的 K − raw，按组合（两视图一起）重抽，seed 0、2000 次；
  - 另报其余四类，作为 K 块是否干扰的描述。
- **训练切片**：raw / K 的六类准确率。如果 raw 在自己训练组合的冲突类上 < 0.90，就把「对训练组合拟合不足」列为替代解释。
- **分支**（按顺序）：
  1. 有效性不满足 → incomplete / invalid，没有结论；
  2. K − raw 冲突宏平均下界 > 0，且 K 条件下两个冲突类都 ≥ 0.80 → 外部给出的中间量能帮助决策。只支持这一点：不说明模型学会了乘法或平方，也不是发布能力；
  3. 下界 > 0，但有一个冲突类 < 0.80 → 部分帮助，记录哪一类仍不够；
  4. 区间包含 0 → 未定位。K 的格式没被利用、比较器不适配等替代解释都保留，不说明这种架构做不到；
  5. 上界 < 0 → 当前这种 K 块会干扰决策（可能超出训练分布）。这不构成反对中间表示的一般证据。
- 不按结果改门槛，不加其他格式或算式。

## 预算与编排（runner 在方案验收后再写）

- 一次加载 EXP68-X；768 个请求，按 EXP68 实测约 5.4 个 / 秒，约 2.5 分钟；加上加载和报告，约 4 分钟。上限 15 分钟，最后 120 秒报告。
- 单模型串行（与 EXP69 重跑等其他模型任务错开）；只运行一次，不扫参、不重试、不自动接后续。

## CPU 夹具（`scripts/exp70/test_exp70.py`，2 项，全过；pyright 0 错误）

- K 块只读 state：改金标或选项顺序不改变 K；格式和 A / B 顺序固定；金标被改时，K 顺序核对会拒绝。
- 生成的四个文件：请求数、每类数量、来源逐字相同、K 前缀、截断全部符合。

## 实现（EXP70-PLAN-ACCEPT-01 / IMPL-PREP-01 之后；未运行）

- **文件**：`scripts/exp70/exp70.py`、`scripts/exp70/run.sh`（硬上限 900 秒，最后 120 秒报告；标记 runs/exp70.started；与其他模型任务串行）。
- **START 绑定**：
  - EXP68 的运行（START 94c7632b… / report 7178c240… / registry fe7a5225… 及其登记的全部文件，包括 details_x_diag68.jsonl 605b0e2e…）；
  - EXP68-X 的冻结端点（FROZEN.json 和实际文件都是 2e5b272e… / 87f82281…）和 tokenizer；
  - data/exp68（沿用 bound_d68）；data/exp70（四个文件固定为已验收的哈希）；代码。
  - 每个阶段和报告都重核。
- **decide**（一次加载 X；普通 decide，image = None，原始温度 1）：
  - 先做 48 个 raw 保真请求，与 EXP68 已登记的 x_diag68 比较：预测相同，每个概率的差 ≤ 1e-4；结果写进 fidelity.json 并登记。不一致就立即停止，不发其他任何请求；
  - 之后依次是 diag68_K（480）、train_raw（120）、train_K（120）。每组做完立即写出并登记（沿用 e57.write_set 和 read_set 的身份与逐题核对）。
- **report**（不加载模型）：
  - 有效性：START、四组记录和复用的 raw 记录都经 read_set 核对；保真在报告里重算；缺文件给 incomplete，存在但核对不过给 invalid，都不下结论；
  - 统计：新 diag 上 raw 与 K 各类准确率、两视图都对、near / large（沿用 EXP68 的 diag68_summary）；K − raw 冲突宏平均和每类差值；训练切片 raw / K 的各类准确率，raw 冲突类 < 0.90 时标注拟合不足的替代解释；
  - 分支按方案。
- **统计口径**（评审 EXP70-NOTE-01）：区间是 95 % 百分位区间，来自对组合族的 2,000 次配对重抽（两视图一起抽），seed 0，即 EXP68 的 conflict_difference / r41.interval；单类差值用同一实现。门槛不变。
- **CPU 夹具**（`scripts/exp70/test_exp70_runner.py` 8 项，加上数据夹具共 10 项，全过；pyright 0 错误；不加载模型，用假 decider）：
  - START 重复启动被拒绝；data/exp70 改动后被拒绝；
  - 保真失败（概率偏 0.01）：只写出并登记 fid_raw 和 fidelity，其他请求一个都不发；报告给 invalid、没有 diag；
  - 三个分支：K 全对 → 「帮助」；K 与 raw 相同 → 「未定位」（差值为 0）；K 全错 → 「干扰」；保真最大差为 0，区间口径写明 95 %；
  - 缺 train_K → incomplete；train_K 登记后被改 → invalid；
  - run.sh stub：顺序 start → decide → report；decide 失败时 finalize；默认 900 秒、留 120 秒。
  - 夹具中发现的一点：equal 类的两种顺序表格文字相同、只差选项顺序，所以按 state 查找必须同时带上选项顺序（只影响假 decider，不影响 runner）。
- **说明**（评审 EXP70-NOTE-02）：报告调用 check_start(code=False)。所以报告重核的是运行证据、模型、数据和 tokenizer 的身份，不重核代码；代码文件的哈希在 decide 阶段（code=True）加载模型之前核对。只改说明，代码不变。

## EXP70 结果（05:58:27–05:59:28，完整运行，valid；预写分支：K 块干扰决策）

- **运行**：HEAD be188bf（相对已审的 49ab767 只多一个文档提交），START 77f9ab73，report 6e65cf13，registry a8347415；全部退出码 0，共 61 秒。
- **保真**：48 / 48 与 EXP68 登记一致，最大概率差 0.0。
- **新 diag**：K − raw 在全部六类上都是负的。冲突宏平均 −0.144 [−0.231, −0.056]；质量胜 0.85 → 0.70，速度胜 0.4875 → 0.35，「两者相等」1.00 → 0.30。两视图都对：质量胜 0.75 → 0.525，速度胜 0.325 → 0.10。
- **训练切片**：raw 速度胜只有 0.40（质量胜 0.90，其余 0.95–1.00）→ 拟合不足是替代解释；K 下各类也都变差。
- **分支**（预写）：当前这种 K 块干扰决策（可能超出训练分布），不构成反对中间表示的一般证据；不说明这种架构做不到。candidate_eligible = false。
- **观察**（不作结论）：连 raw 全对的「两者相等」都大幅下降，更像是格式没见过造成的，而不是 K 的信息有错。EXP68-X 对自己训练过的速度胜组合也只有 0.40，说明 EXP68 的末点在这一类上没有拟合好，解释 EXP68 时要一并考虑。
