# v0.4 读取链证据：确认来源与冻结协议、路由校准验证、reading_combination 接入（V04-READER-EVIDENCE-PREP-03；只用 CPU，不是模型许可）

日期：2026-10-19。依据 V04-READER-EVIDENCE-R2-ACCEPT-01 和盘点（docs/experiments/2026-10-19-v04-reader-evidence-inventory.md）。本文件只是设计，没有改任何运行链或绑定文件，没有加载权重。EXP57 的 fit 失败和组合资格仍待处理；EXP72 的结果不会自动给另一组权重资格；昂贵的全评（约 8.6 小时，按实际 controls 复算约 515 分钟）要等 EXP73 的保护结果之后再定。

## 1. 新读取链的独立确认：来源与冻结协议

### 1.1 来源可得性结论

- **本地真实来源**：没有。原始 ScienceQA 三个划分里的 876 道温度 / 磁力题全部已经暴露：train 533 题用于训练和开发，validation 182 题是每轮都评的 bench_lv_scienceqa，test 161 题是 dev_sqa（见盘点 B）。
- **外部真实来源**：目前找不到可以确认取得的。
  - ScienceQA 的题出自 IXL 的 K-12 题库，原图版权和获取方式都不明确；我们没有许可，也没有已知的公开导出。
  - 其他公开数据集（AI2D、TQA 等）的图解没有这两种「两罐粒子 + 数值表」「两对磁铁 + 距离标注」的格式，不能当作同任务的确认。
  - 所以**真实确认的缺口保留**：交付记录要明确写「新读取链没有在未见过的真实图上独立确认」。如果之后获得了来源（例如负责人取得授权的新题），再按 1.2 的协议执行。
- **合成泛化确认**（可做，但只能称合成确认，不能称未见真实来源确认）：由与开发无关的独立渲染器绘制。

### 1.2 冻结协议（真实来源和合成来源都适用）

1. **先冻结组合**：候选组合（决策端点 + reading_combination，见第 3 节）和全部规则锁定、登记哈希之后，才生成或打开确认集的答案。
2. **抽样规则先写**：
   - 温度 60 张、磁力 60 张（磁力里距离不同 / 大小不同 / 相同各占三分之一；温度按生成规则的六类各 10 张）；
   - 每张一道比较题，再加该图全部读数题；按 sha256(来源 id) 固定抽样，不按模型表现挑选。
   - 数量依据：120 张图在 0.8 左右的准确率下，95 % 区间半宽约 ±0.07，只够确认「大体上能用」，不足以比较小差异。
3. **去重与暴露隔离**：
   - 真实来源：对已有 876 道 ScienceQA 题和全部训练图做感知哈希（pHash，汉明距离 ≤ 6 判为重复）和题目文字去重；
   - 合成来源：数值组合与 data/real_layout 全部划分、data/exp68 / exp65 / exp70 的 canonical 组合不相交；
   - 确认集只登记哈希，不进入任何训练、开发、校准或方法选择。
4. **合成渲染器的独立性**：
   - 另写一个渲染器（不复用 real_layout 的 temp_render / force_render），使用不同的字体族（例如 DejaVu Sans / Liberation Serif）、画布尺寸、罐与磁铁的形状、标注位置和单位排版；
   - 参数在看任何模型成绩之前冻结并登记；生成后人工目视抽查 20 张，只看绘制是否正确、不看模型输出。
5. **读出**：通过实际评测入口（第 3 节）跑；同时跑配错图和选项倒序。报告准确率、看图收益（真图减错图）、倒序翻转率，温度和磁力分开，并逐题记录读数→表→state。
6. **不缩减原确认项**：confirm_property / confirm_chartqa（普通路径）全部保留；它们证明旧能力，不证明新读取链。

## 2. 路由路径的概率验证（与确认集严格隔离）

- **现状**：全量校准集 data/r13b/val.jsonl（3,892 题）全部是普通路径，路由路径（温度 / 磁力读取后决策）的概率从来没有被校准或验证过。3,892 题的校准不缩减，照原流程做。
- **角色隔离**：
  - 确认集（第 1 节）不参与任何温度拟合，也不用于选温度；
  - rl_confirm（600 行生成数据）保留它「同一生成器、未见过的数值组合」的确认角色，也不拿来拟温；
  - 新增的路由校准集与确认集、开发集按组合和图片两层不相交。
- **新增路由校准集**（生成数据；只是设计）：
  - 用 data/real_layout 生成器，以新的 seed 生成「rl_calib」划分：温度和磁力各 300 族、两视图，canonical 组合与已有全部划分以及确认集不相交；
  - 只用比较题，经实际评测入口走路由路径得到概率。
- **验证方案**：
  1. raw（温度 1）与 fitted（3,892 题拟合出的题型 / 选项数温度）两套概率，都在 rl_calib 上算 NLL、ECE（15 个等宽分箱）和可靠性曲线；
  2. 两折交叉：在 rl_calib 的一半上为路由路径单独拟合温度，在另一半上看 NLL / ECE 是否改善（族为单位分折）；
  3. 判定：如果 fitted 温度在路由路径上的 ECE 不差于 raw，就沿用同一套温度；否则把「路由路径另设温度」作为一项交付改动另行送审（需要在 config 里按路由分桶，decide 实际使用前必须验收）；
  4. 真实图上的概率只能用已暴露的 dev_sqa 作描述，不作校准依据。
- 预计请求：rl_calib 约 600 张图 × 2 条件（raw / fitted 只是后处理，实际只需 1 次前向）≈ 600 个请求，约 4–5 分钟。

## 3. reading_combination 接入发布身份和实际评测入口（最小设计）

### 3.1 现状（只读核对）

- 发布评测的实际入口是 devision-eval（src/devision/train/cli.py eval_main → Decider.load），只走普通 decide。devision.load 和 devision-serve 也只加载 Decider。也就是说，**当前发布流程根本不经过读取链**：温度 / 磁力题在发布评测里会被当作普通题。
- check_eval.py 的 IDENTITY.json 是 tree_hashes(checkpoint)，即 checkpoint 目录下的全部文件。
- bundle 的 reading.json 已经绑定了 policy / grid / templates / domains / corrector 规则和组件、决策端点的 sha256（评审已指出）；代码文件不在其中。

### 3.2 设计

1. **打包**：把读取组件作为 checkpoint 的子目录 `<checkpoint>/reading/`（reading.json、selector.pt、head.pt、corrector.pt），由 write_bundle 生成。这样 check_eval 的 tree_hashes(checkpoint) 不用改就会把整个组合纳入 IDENTITY。同一份权重的 off / on 是两个不同的 checkpoint 目录（不同的 reading/），筛查和全评证据无法互借。
2. **加载**：新增一个入口 `devision.load(path, reading="auto")`：
   - checkpoint 里有 reading/ 时，返回 Reading.load(path, path/reading)（经验证的入口：先核 reading.json 里的全部 sha256 和规则，再加载 Decider 和组件）；
   - 没有 reading/ 时，与现在完全一样返回 Decider；
   - reading="off" 可以显式关闭（只用于对照）。
   - devision-serve 和 devision-eval 都通过它加载，所以 API、评测、发布用的是同一个对象。
3. **评测入口**：evaluate_records 对每个请求调用这个对象的 decide：
   - 普通题 → Reading.decide 直接转给 Decider（已有行为，夹具已证明与 Decider 相同）；
   - 路由题 → 读取实际展示的图（配错图对照时就是配错的那张图）→ 表 → 普通 decide；
   - 选项倒序 → 只改 criteria 的顺序，路由按题目文字，读数不受影响。
   - 汇总的 run 字段新增 `reading`: {reading.json 的 sha256, 是否带校正器}；check_eval 的内容检查要求它等于 IDENTITY 里 reading/reading.json 的哈希；没有 reading/ 的 checkpoint 必须记为 null。
4. **代码身份**：
   - 规则已在 reading.json 里；代码（src/devision/reading.py、image.py、infer.py、train/evaluate.py）由发布 wheel 的哈希绑定：check_eval freeze 时额外记录当前安装的 devision 版本和 wheel（或源码树）哈希，seal / check 时核对一致；
   - 评测必须在冻结记录的那个安装上跑。
5. **安装核验**（scripts/release/hf.py verify 的扩展）：
   - 在干净的虚拟环境里从 wheel 安装，用发布的 checkpoint（含 reading/）跑固定的 smoke：普通题 2 道、温度 1 道、磁力 1 道，再加一道配错图；
   - 要求读数字符串、表、state 和答案概率与源 checkpoint 在 CPU 上的结果一致；
   - 另核 `reading="off"` 时与 Decider 完全相同。
- **在哪一步加载和验证**：
  - Reading.load 在加载 Decider 之前核对 reading.json 绑定的全部哈希和规则；
  - check_eval freeze 记录 checkpoint 树（含 reading/）和安装哈希；
  - 每个集合跑完后 seal 核对；
  - check 在发布前做最终核对；
  - verify 在干净安装里复现。
- 以上都还没有实现。实现属于发布代码改动（src/devision/__init__.py、train/cli.py、train/evaluate.py、scripts/release/*），需要单独送审，并且要在昂贵全评之前完成，否则全评测到的不是读取链。

## 4. 时间与资格

- 原定 2026-10-18 的期限已经错过，不改历史承诺。约 14–16 小时只是带前提的资源估算，不是新的交付保证，也不包括获取真实确认来源的等待。
- 顺序建议：EXP73 保护结果 → 裁定候选组合和 EXP57 fit 失败的处理 → 实现并验收 3.2 → 路由校准验证（第 2 节）→ 全评 → 合成确认（第 1 节）→ 打包与安装核验。真实确认缺口在交付记录中写明。
