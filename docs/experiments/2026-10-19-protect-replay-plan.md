# PROTECT：单臂混合回放恢复（PROTECT-PREP-01；CPU 准备，未训练，不是运行许可）

日期：2026-10-19。依据 EXP73-RESULT-01：

- EXP71-on 在 science（−3.64 点 [−7.01, −0.34]）和 flip（−4.73 点 [−6.59, −2.78]）上有保护代价。
- 损失主要在普通路径：science 普通题的 557 道和 EXP57 的预测逐题相同；flip 一题都不走读取链。
- 退化沿训练链累积：science 普通题 v0.3 364 → EXP48-X 348 → EXP50-T 345 → EXP57 338。EXP57 起只做纯文字训练，没有带图回放。

本实验是一次工程恢复：把新能力的保持和旧能力的恢复放进同一个训练目标。它不是对「回放机制」的单因素归因，也不是候选。

## 起点（研究提案，不是晋升）

- EXP72-K 的冻结末点 runs/exp72-k/stage2：model.safetensors cbd1e7ac…3782、config.json 4f9ccf48…0e23，绑定 EXP72 的 START d61b3080… / report 7b719fdf… / registry f986172…。正式训练前由 runner 对回这些哈希，并做起点保真：从 EXP72 已登记的 details_k_slice 取前 48 题重跑，要求预测相同、概率差 ≤ 1e-4。
- 保留它学到的 K 输入契约：K 只由 state 里自己的表算出（scripts/exp70/build.py 的 k_block），不读 gold、选项或题号。
- 这一轮不改选择器、字符头、校正器和版式。

## 数据（scripts/protect/build.py → data/protect/，已生成）

- **新任务**：data/exp72/train_K.jsonl 原样保留，2,800 行、3,800 题，每行都标 `"input": "text"`。
  - 其中 200 行（读数 / 磁力回放族）的 image 字段写的是 real_layout 图片，但 EXP72 是在全局 text_only 下训练的，这些图从来没有加载过。所以这里也按文字评分，不改变新任务的输入。
- **回放**：从 v0.3 自己的训练集 data/r14b/train.jsonl（81,875 题）里抽带真实图片的题，每行标 `"input": "picture"`。
  - **隔离**：删掉曝光过的图片及其同族。曝光文件包括配置里一切非训练用途的 JSONL（val / eval / sets / history / calibration_fit / extras），以及 data/exp*、data/facts、data/real_layout、data/v04_*、data/lv_bench 下所有非 train 开头的文件，共 163 个。同族包括：镜像副本（去掉 `:flip` 后缀的基础 id）、COCO / VG 双胞胎（data/v5/image_identity.json）、同名文件。只按文件名命中的仅 4 题（general），其余都按 id 命中。
  - **配额**（预先固定，见脚本中的 ALLOC，不看任何题目）：science、mirror、计数和 vsr 有下限，分别对应七个保护面板的能力；其余按 v0.3 原混合比例。按「sha256(单元键)」排序抽取，单元整体保留：镜像对（同一张 COCO 图的原图加翻转图）或同组的题一起进出。不从任何保护面板的错题里挑。

| 块 | 隔离后可用（题 / 单元） | 隔离删除 | 配额 | 实际抽取 | 主要来源 |
|---|---|---|---|---|---|
| science | 5,314 / 4,534 | 2,561 | 1,600 | 1,600 | scienceqa 1,035、ai2d 290、tqa 275 |
| mirror（完整镜像对） | 780 / 383（另有缺一半的单元已删） | 1,220 | 1,000 | **780（390 对）** | coco-relation 462、coco-position 318 |
| count_balanced | 8,600 / 8,388 | 3,400 | 800 | 800 | tallyqa、objects365 |
| count_original | 4,312 / 4,273 | 1,688 | 400 | 400 | tallyqa、objects365 |
| pixmo | 6,421 / 4,074 | 5,579 | 600 | 600 | pixmo_points |
| existence | 7,105 / 6,834 | 2,895 | 900 | 900 | objects365 860、coco-exist 40 |
| general | 6,055 / 5,698 | 3,943 | 900 | 900 | aokvqa、gqa-v2、vqav2-yesno 等 |
| ext | 13,716 / 11,105 | 7,286 | 1,200 | 1,200 | 实际 10 个来源，各 103–151 题（按块哈希抽，没有逐来源等额约束；保护评测仍用 13 来源宏平均） |
| vsr | 519 / 493 | 481 | 200 | 200 | vsr |
| **合计** | | | 7,600 | **7,380** | |

- **数量不足，如实报告**：完整镜像对只有 390 对（780 题），比配额少 220 题。我没有事后从别的块补，所以回放是 7,380 题，新任务与回放约为 1 : 1.94（原提案约 1 : 2）。
- 身份：replay.jsonl b399a457…、train_mix.jsonl 467e4014…、MANIFEST.json aa3f16ec…；输入文件的哈希、曝光文件清单和每块的数量都记在 MANIFEST 里。
- **批计划**（每批 8 题；同一批只有一种输入；单元整体）：文字 475 批（没有不满的批）+ 图片 923 批（1 批不满）= **1,398 次更新**。

## 两条前向分开（rlcd mixed_input；已实现，tiny 已验证）

- 新增配置 `mixed_input`：每行都必须带 `"input"`，缺了就拒绝。
  - "text" 行完全等同 text_only：image=None，即使行里写了图片也不加载；
  - "picture" 行走原来的图片路径；
  - 不会用空白图代替无图。
- plan_mixed：在两种输入上分别跑 plan_batches（同一 group + qid 的单元整体），再把两边的批一起打乱；一批里只有一种输入。
- 与 text_only 一样只训练 LoRA 和决策头。视觉塔、投影层、条件模块和读取头全部冻结；图片批原样经过它们。
- 验证（_val_logits mixed）：每题按自己的输入评分，顺序保持不变。
- 夹具 tests/train/test_mixed_input.py（6 项）：
  - 文字行等于 text_only，也等于 decide(image=None)；图片行等于用它自己的图 decide；
  - 一批里不混两种输入，单元整体；
  - 一次更新只改 LoRA 和头，视觉、投影层、条件模块逐位不变；
  - 只有文字的数据完全不调用视觉；
  - 缺 input 的行，以及 mixed_input 和 text_only 同时打开，都会被拒绝。
- 全部测试 309 项通过，pyright 0 错误。

## 训练配方（运行前固定）

- 从 EXP72-K 末点开始；与 EXP72-K 相同：新的 LoRA r16（B 零初始化，所以从起点函数开始）、CE、lr_lora 5e-5、lr_head 1e-5、warmup 100、每批 8 题、1 个 epoch、seed 1、amp off；mixed_input: true。
- 训练中的 val 用 EXP72 的 slice_K（240 题，标 text）。训练末尾的拟温只作记录；所有读出都用原始温度 1，与 EXP72 / EXP73 相同。
- **末点固定为最后一步**，不挑面板最好的步，不做中途评测。

## 终点读出与门槛（运行前冻结；缺任何一项都是 incomplete，不能算通过）

新端点和 EXP71 的三个读取组件重新绑定成一个新组合（write_bundle：组件原样、决策端点换成新的），身份是新的。

**A. 旧能力保护（相对固定的 v0.3，沿用 EXP73 / EXP58 的规则）**：七个面板 exist / science / ext（13 来源宏平均）/ count_wide / count_narrow / flip（原图和翻转图同簇）/ vsr，经新组合的 Reading.decide 作答。
- 门槛：没有任何面板出现代价（差值 ≤ −3 点且区间上界 < 0）。
- science 的读取题和普通题分开描述。
- 同时对 EXP73 给出配对差（描述，不作门槛）。

**B. 新能力保留——给定正确表（沿用 EXP72 的 K 臂线）**：
- 训练切片六类各 ≥ 0.90；
- 新诊断（data/exp70/diag68_K.jsonl）两个冲突类各 ≥ 0.80；
- G 的 1,800 道数值读数 all ≥ 0.98。
- 「两者相等」类、两视图都对、旧 fit 单独报告，不能用均值顶替失败类。

**C. 新能力保留——自动表 P 与旧读数门槛（沿用 DECISION-FACTS / EXP57 的线）**：data/facts/tune_P.jsonl 是 EXP56 已登记的实时表，K 由表计算。
- 门槛：P（温度）比较 ≥ 0.65；读数保留 P all ≥ 0.866；EXP55 的 P_use 为 True。
- K 的规则：表里四个值都可读且都是整数时才加 K 块；任何一个是 unreadable 或非整数时不加 K 块，题照常计入。这是产品里实时读取会遇到的情况，不加 K 块的比例另外报告。
- 起点 EXP72-K 也跑同一份 P（600 个请求），给出新端点减起点的配对差（描述）。
- **区分**：B 证明的是「给定正确表时能力保持」；C 用的是 EXP56 那一版读取器的实时表，不是 EXP71（带校正器）的组合。所以 C 通过也不等于新的实时组合达标；EXP71 旧端点的迁移证据不能借给新端点。

**分支**：
0. 有缺失或无效 → 没有结论，已算出的指标照常报告；
1. A 有任何代价 → 恢复失败，不做任何后续；
2. A 通过，但 B 或 C 有门槛不过 → 以新能力换旧能力，记录是哪一项不过，不是候选，不追加预算；
3. A、B、C 全部通过 → 工程恢复成功（不作机制归因）。下一步按顺序、分别申请：新组合在 dev_sqa 上的实时迁移（相对 v0.3 的原迁移目标）→ 独立确认 → 校准 → 全评 → 安装与发布核验。新组合有新身份，旧组合的校准和安装成绩都不沿用；29 个集合的职责和真实确认来源的缺口都保持不变。

## 预算（待定的估算；硬截止在许可时定）

| 阶段 | 依据 | 估计 |
|---|---|---|
| start + 起点保真（48 题） | EXP72 | 约 1 分钟 |
| 训练：文字 475 批 | EXP72-K 实测每秒 0.713 次更新 | 约 11 分钟 |
| 训练：图片 923 批 | 评审给的每次更新 1.0–2.4 秒（带局部视图；**本配方还没有实测**） | 约 15–37 分钟 |
| 加载与首末 val | EXP68 | 约 3 分钟 |
| 重新绑定 bundle、冻结 | | < 1 分钟 |
| B：训练切片 240 + 诊断 480 + G 600 + 旧 fit 400 | EXP72 decide | 约 7 分钟 |
| C：P 600（新端点）+ 600（起点） | EXP57 每秒 4.7 个请求 | 约 4.5 分钟 |
| A：七个面板 5,569 个请求 | EXP73 实测 30.6 分钟 | 约 31 分钟 |
| 报告 | | 约 2 分钟 |
| **合计** | | **约 75–97 分钟** |

- 建议硬截止 **120 分钟**，最后 180 秒留给报告；一次启动，不重试。
- 优先级：起点保真 → 训练 → 冻结 → B → C → A → 只报告的项（旧 fit、对 EXP73 的差）。到限缺了必要项就判 incomplete，不删集合、不改子集。
- 图片批每次更新的耗时是最大的不确定项：如果首窗超过 2.4 秒，按 watch 的 ETA 判断，到限就判 incomplete，不延长。

## 实现步骤（方案验收后再做，另送实现验收）

1. 把 v04-reading-entry 合并进本分支，以获得带校正器的读取代码；已用 git merge-tree 试过，没有冲突。
2. configs/protect-k.yaml：init 指向 EXP72-K，data 为 data/protect/train_mix.jsonl，参数如上。
3. scripts/protect/protect.py 与 run.sh：沿用 EXP72 / EXP73 的 start / registry / watch / report 框架。
   - 阶段：start → fid → train → freeze（重新绑定 bundle）→ B → C（新端点和起点）→ A → report。
   - A 复用 EXP73 的 run_set 和代价规则；B 复用 EXP72 的读出；C 复用 DECISION-FACTS / EXP57 的 P 门槛计算。
4. CPU 夹具：假读取器和假决策器走分支 0–3；混合输入的批计划等于 MANIFEST 记录的计划；run.sh 的顺序、上限和到限行为。

## 第 1 轮修正（PROTECT-R1-01、NOTE-01）

- **镜像对被拆批（R1-01）**：属实。r14b 里原图的 group 是 `coco:<id>`，翻转图是 `coco:<id>:flip`，plan_batches 按 (group, qid) 装批，所以第一次生成的 data/protect 在 seed 1 的真实计划里，383 个镜像单元全部被拆开。
  - 修正：重新生成到 **data/protect2/**（data/protect 原样保留，r14b 不动）。每个镜像单元的各行共用训练 group `protect-mirror:<基础图>`，原 group 记在 source_group 里。题目、金标和图片都不变：去掉 group / source_group 后，两版 replay 与 train_mix 逐行相同，顺序也相同。
  - 单元数要更正：780 道镜像题是 **383 个图片单元**，其中 376 个是「一题原图 + 一题翻转」，7 个是「两题原图 + 两题翻转」，合计 390 个「原图—翻转」题对。之前写的「390 对」指的是题对，不是图片单元。
  - build.py 现在直接按训练的方式抽计划（random.seed(1) → rlcd.plan_mixed，用 train_mix 文件顺序里的题），并检查：每题恰好一次、一批里不混输入、单元不超过一批（超过就拒绝，不让 plan_batches 悄悄切开）、每个镜像单元都在同一批（按基础图检查，不看 group）。
  - 结果记在 MANIFEST.plan：文字 475 批 + 图片 923 批 = **1,398 次更新**（不变）；最大单元 6 题；383 / 383 个镜像单元都在同一批；plan_sha256 2f7314d4…。
  - 夹具 scripts/protect/test_build.py：第一版数据被检查拒绝（拆开 383 个）；新版全部同批，并与 MANIFEST 一致；超过一批的单元被拒绝。tests/train/test_mixed_input.py 新增一项：训练实际抽到的就是预先算好的计划。
  - 身份：replay.jsonl d86c86f8…、train_mix.jsonl 7ae73f11…、MANIFEST.json 918fec57…。重跑脚本得到逐字节相同的文件。
- **NOTE-01**：
  - ext 的措辞改成实际的 10 个来源（见上表）；
  - MANIFEST 的每个曝光文件都附上 sha256，runner 的 START 要绑定这份清单，并在加载前逐个核对。
