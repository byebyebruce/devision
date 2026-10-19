# EXP74 方案：在当前决策分支上补学面板绑定与读数（PANEL-LEARN-PREP-01；CPU 准备完成，未运行模型）

日期：2026-10-19。评审 V04-K-REPORT-ACCEPT-01 / PANEL-LINEAGE-01 / PANEL-LEARN-PREP-01 选定这条路线；本方案只放行了 CPU 准备，模型运行需要另行许可。不算一轮，不产出可采用的模型。worktree `devision-panel-learn`，分支 `panel-learn-prep`（从 v04-materials 332f303 拉出）。

## 1. 谱系（纠正此前"在链上丢失"的说法）

当前分支**从来没有**具备 dev16 读数 / 比较能力；能力只在另一条实验分支上被证明可得。下面的分数都来自已有的逐题记录（`*/dev16.details.jsonl`），同一套 1,200 题，读数 = readA + readB（800），比较 = cmp（400）。本次没有加载任何模型。

| 节点 | 父节点（run.json 的 init） | dev16 读数 | dev16 比较 | 来源 |
|---|---|---|---|---|
| r15-vision（v0.3） | — | 277/800 = 0.346 | 172/400 = 0.430 | exp22-fit step0 |
| **EXP26 分支** | | | | |
| exp24-align | r15 | 258/800 = 0.323 | 169/400 = 0.422 | exp24-fit / exp25 step0 |
| exp26-align | exp24-align | 329/800 = 0.411 | 171/400 = 0.427 | exp26-read-compare step0 |
| exp26-read-compare 5,000 步 | exp26-align | **738/800 = 0.922** | **380/400 = 0.950** | monitor/step5000（原规则 pass） |
| **当前分支** | | | | |
| exp30-align-replay | r15 | 未测 | 未测 | |
| exp32-align | exp30 | 未测 | 未测 | |
| （旁支）exp32-fit = EXP32-B | exp32-align | 293/800 = 0.366 | 166/400 = 0.415 | 旁支终点，不在当前初始化链上 |
| exp37-l | exp32-align | 287/800 = 0.359 | 167/400 = 0.417 | exp40-m step0 |
| exp40-m | exp37-l | 267/800 = 0.334 | 166/400 = 0.415 | exp40-m eval |
| （旁支）exp40-o | exp37-l | 284/800 = 0.355 | 172/400 = 0.430 | |
| exp45-g → exp48-x → exp50-t → exp57 → exp63 → exp68-x → exp72-k → protect-k | 依次 | 未测 | 未测 | |
| v04-k（protect-k 权重 + 读数组件） | protect-k | 294/800 = 0.3675 | 166/400 = 0.415 | 全评已封存 extra_dev16 |

结论：当前链上测过的节点都在 0.33–0.37 / 0.415–0.43 之间，和 r15 基线一样。exp30 的对齐混合里有 20,000 条 exp24 读字 caption，但**从来没有** EXP26 的面板绑定对齐（`data/exp26/align_*`），也没有复现 EXP26 那种充分的读数 / 比较训练配方。当前链并非完全没有见过合成监督：EXP40-M 混入了 1,920 道 exp25 图题和 484 道文字题（旁支 EXP32-B 有 2,401 道合成题）（评审 NOTE-02 更正）。所以准确的说法是"项目曾在 EXP26 分支证明可学，当前分支至今没有学到"，不是"后来遗忘"。dev16 是没有参与训练的族，但在开发中反复监测过，不是 held-in，也不是独立确认。

## 2. 要回答的问题

在当前决策分支（冻结候选的决策权重 185b9496… 的副本）上，冻结 SigLIP 和全部读数组件，只更新投影层 / LoRA / 决策头，EXP26 的"面板绑定对齐 → 读数 + 比较决策"能不能在一个固定的训练族切片上学会，同时温度 / 磁力的文字消费和旧能力不坏？先回答"在训练族上能不能学会"，学会了才谈未见族和发布。

## 3. 设计（两关，各自单独申请运行许可）

**起点**：`runs/v04-k/cand`，只读。Decider.load 读取它的决策权重，没有 reading/ 也照样加载；产物写到 `runs/exp74-panel-d1/`、`runs/exp74-panel-d2/`。旧候选、包、IDENTITY、RESULTS 和确认产物都不动。不把 EXP26 的权重换进来。

### D1：面板绑定对齐（`configs/exp74-panel-d1.yaml`）

- 与 `configs/exp26-align.yaml` 只差：init 换成冻结候选，遍数 2 → 1（预算），eval_every 500 → 625。数据（`data/exp26/align_train.jsonl` 20,000 张、`align_val.jsonl` 1,000 张）、batch 16、投影层 lr 2e-5、LoRA r16 / alpha 32 / lr 1e-4、warmup 100、mask 0.5、caption 长度 40、seed 0 都不变。说明文字总是 A 先、不管 A 画在哪边，面板的位置、顺序和数值都随机。
- **更新的参数**：投影层，以及新加在 ModernBERT 上的 LoRA（保存时合并进 decision.encoder）。SigLIP、question conditioning、局部视图几何、决策头都不训练（夹具已核：只有 projector 和 decision.encoder 变化）。读数组件（selector / 字符头 / corrector）是 reading/ 里的独立文件，不参与训练。
- **兼容性修改（唯一的代码改动）**：原来的 align.py 把局部视图打包后的多通道张量直接送进 SigLIP（tiny 夹具 local_views=4 时是 16 通道，真实候选 local_views=12 时是 40 通道），在带局部视图的 checkpoint 上会报错（夹具复现了这个错误）。改为 `_full_picture` 先取出整图，与 `model.visual_inputs` 投影成视觉 token 的是同一张图（夹具核对两者逐位相等；不带局部视图的模型不变）。这样 caption 对齐只看整图；局部视图只供 conditioning 使用，而 caption 对齐本来就不经过 conditioning。
- **不使用 align 的 decision_data**：它和 micro_batch 互斥，而且总是加载图片，文字行不会按 image=None 打分（夹具已核）。D1 只做 caption 对齐，不混决策监督；保护交给 D2 的回放。不用空白图代替 image=None。
- **门槛**（`scripts/exp24/gate.py`，原样复用）：1,000 张留出面板图上，真图减换图的混合补词准确率差 ≥ 0.30 才进入 D2，否则停下报告（exit 10）。EXP26 两遍之后是 0.347；这里只训一遍，可能达不到，达不到也如实报告。

### D2：读数 → 决策，带回放保护（`configs/exp74-panel-d2.yaml`）

- 起点是 D1 的输出；`mixed_input`、CE，与 PROTECT-K 同一条已用过的路径：图片行走图片，文字 / K 行按 image=None 打分。
- **更新的参数**：新 LoRA（r16，lr 5e-5）和决策头（lr 1e-5）。投影层、conditioning、SigLIP、读数组件都冻结（夹具已核：只有 decision.encoder 变化）。**1 遍**（评审 PANEL-D2-R1-01 裁定，原来写的 2 遍），micro_batch 8，warmup 100，seed 1，amp 关。
- **数据** `data/panel_learn/train.jsonl`（`scripts/panel_learn/build.py`，seed 4，MANIFEST 已写，sha 7c9dbb97…）：共 4,200 行、8,430 题。
  - 面板读数 / 比较：exp25 的 1,000 个训练族，每族 2 个数值视图共 2,000 行，必定包含 fit 的 200 个族；
  - 文字比较：600 行。exp25 的文字比较行里有 424 行的擦除底图与 dev_syn_erased 的图逐字节相同（86 张），**已按图片身份全部排除**；EXP26 当年的训练里包含这些图，在这里如实记录；
  - 回放：PROTECT 的 train_mix 1,600 行，其中图片 1,000 行（按来源比例）、文字 / K 600 行（温度 557 / 磁力 43，按原比例）。
  - 面板、文字、回放约为 48 : 14 : 38（按行）。
- **审计**（build.py 有任何一项不通过就拒绝写出）：
  - 训练图片 3,440 张，与 28,243 张留出图片逐字节相同的为 0。留出图片包括 dev16、dev_syn、dev_syn_erased、三套冻结确认（property、ChartQA、K 合成确认）及其错图文件，以及全评 29 个集合。
  - 与 dev16 族的交集为 0；每一行都有 input 和 group。
  - D1 的 21,000 张图与留出图片相同的也为 0；align 的训练和验证族不重叠。
  - 没有混入新确认集、ChartQA、property 或全评测试题。
- **监测**（step 0、每遍结束、final，都写逐题明细；全部按 input 字段打分）：
  - val = fit（`data/panel_learn/fit.jsonl`，200 个训练族，1,200 题）；
  - dev16（1,200 题，未见族）；erased（600 题，擦除对照）；
  - k_text（PROTECT 的 K 文字切片 240 题）；
  - rs_*：冻结读数步骤在封存全评里给出的真实 ScienceQA 状态，温度 41、磁力 120 及两者的错图（`scripts/panel_learn/read_states.py`，每个文件都核了封存 RESULTS 的 SHA）。读数组件冻结，这正是新决策模型将收到的输入，所以可以离线检查真实迁移和 P/K 文字消费，不需要重新绑定读数包。
- **之后**：exist、science 两个保护面板，用 devision-eval 走普通图片路径（pipeline 的 evaluate 部分）。

### 一致性前置检查（D1 运行的第一步）

`scripts/panel_learn/consistency.py`：冻结候选的决策权重（plain Decider）对 322 个封存状态做 decide(state, q, image=None)，每题的预测必须等于封存全评的预测；不相等就判 invalid 并停止，说明 rs_* 不是这条决策路径。

## 4. 判定（运行前写下，`scripts/panel_learn/report.py`）

- **D1**：差值 ≥ 0.30 进入 D2；否则停下，报告"在 1 遍预算下绑定对齐没有达到 EXP26 的门槛"，不自动加遍数。
- **D2 证据核对在先**（PANEL-D2-R1-02，report.py 的 d2_report）。先核对以下各项，再判结局；已完成的部分照样报告：
  - **invalid**：run.json 的 stages 不是 D2 配置；训练数据不是 MANIFEST 里的那份；更新次数不等于 plan_mixed 规划的次数（1,054）；终点监测的题键有重复，或与数据文件不完全相同；选项、gold 或 correct 与数据不符；探针不属于这次 D2 端点（checkpoint 路径、数据 sha 不符，或者早于保存的权重）。
  - **incomplete**：没有 run.json；没有 train_report（训练没有完成）；最后一次更新时没有评估；没有 final 监测；终点缺某个监测文件；缺 exist / science 探针。
  - 不再取"最大 step 目录"作为终点：终点是 train_report 里最后一次评估的步数，必须等于实际更新次数。缺探针时判 incomplete，不会因为缺证据而判成 B。
  - 逐个集合先判有效性（PANEL-D2-R2-01）：缺文件判 incomplete；读不出（截断）、空文件、题键或内容不符判 invalid。只有有效且有分母的集合才进入统计，其他有效集合的成绩照常报告；没有 run 目录时也写出报告。
  - 合并后的端点（PANEL-D2-R2-02）：final/ 下每个监测集合都要完整且内容正确，并且由逐题记录重算的准确率等于 train_report.final 里对应的值；空的 final 目录、缺文件或数值不符都不能证明端点完整。诊断的主读数仍是最后一遍结束时的监测（训练端点），final 用来核对保存的合并端点。
  - 探针准确率从核过的明细重算；与 summary 的 n 或 accuracy_all 不符就判 invalid，不用 summary 覆盖明细。
  - 只有全部证据都完整有效时，才按下面的规则判 A / B / C / D。
- **D2**（看这一遍结束时的监测）：
  - 有效性：擦除对照的读数 < 0.50 且比较 < 0.55（delivery.py 的规则），否则 invalid。
  - fit：readA / readB 和比较按 exp21 readout 的 classify 判定（pass = 点估计 ≥ 0.80 且下界 ≥ 0.75；fail = 点估计 < 0.60 且上界 < 0.65；其余为阈值附近），按族重采样。
  - dev16：同一规则，只报告（开发监测；控制无效时作废）。
  - 保护（kept）：k_text ≥ 0.95；rs_temp ≥ 39/41；rs_force ≥ 0.458（封存值 0.508 − 0.05）；exist、science 不低于封存候选 3 个点以上。
  - 结局：invalid；A = fit 学会且保护保持；B = fit 学会但保护被破坏；C = fit 没学会（有一项为 fail）；D = 阈值附近。**任何结局都不自动开始下一步**。A 之后才另写方案：扩大族数 / 步数、未见族、七个保护面板的筛查、新的冻结与校准；不安排第二次九小时全评。
- 结局的含义边界：A 只说明"在当前分支、这个预算下，训练族上能学会"，不说明未见族、真实 ScienceQA 或发布达标；C 只说明"在冻结 SigLIP、只更新投影层 / LoRA / 头、这个预算下没学会"，不说明这条路线一定不可学。

## 5. 资源和截止（估计；运行时用实测更新）

- D1：1,250 步 × batch 16。只看整图，接近 EXP26 的 1.85 s/步；但读图时仍会生成局部视图（CPU），估计 2.0–2.5 s/步，共 42–52 分钟；加上 3 次 align_val 评估（各约 2 分钟）和一致性检查（约 2 分钟），约 50–60 分钟。**硬上限 75 分钟**，最后 120 s 留给门槛报告。
- D2（PANEL-D2-R1-01 更正）：原来按 4,200 行 × 2 遍 / 8 估算约 1,050 次更新是错的，rlcd 按题展开（`_items`），plan_mixed 按题安排。用实际题单和 rlcd 自己的 plan_mixed 重算（`scripts/panel_learn/plan.py`，seed 1，micro 8，不构建模型）：
  - 8,430 题，其中图片 7,600、文字 830；5,277 个单元，最大单元 2 题；**每遍 1,054 次更新**，2 遍是 2,108。按 2.6–3.0 s/次，2 遍纯训练就要 91–105 分钟，装不进 95 分钟。
  - 改为 **1 遍 1,054 次更新**：训练 45.7–52.7 分钟；监测 3 轮（step 0、这一遍结束、final），每轮 3,000 张图片题和 562 道文字题，按每题约 0.11 s / 0.02 s 估计约 5.7 分钟；exist / science 探针约 5 分钟；加载约 1.5 分钟。合计约 **69–76 分钟**，硬上限仍是 95 分钟，最后 120 s 留给报告。必要的监测和终点读出都保留。
  - 第 1 遍各来源的曝光（包含该来源的更新次数）：exp21-syn 768、exp21-syn-text 219、objects365 118、real-layout-temp 104、scienceqa 100、tallyqa 78、real-layout-force 75，其余见 plan.py 的输出。
  - 结论的范围：1 遍是预算内的诊断。1 遍没学会，不能外推成"充分监督下也学不会"。
- 两关合计约 2–2.5 小时，超过评审建议的约 90 分钟，因此拆成两次许可，中间以 D1 门槛分开。删掉必要的对照不能换来更短的时间。
- 设备 MPS，一次只跑一个模型；`scripts/panel_learn/run.sh` 用 EXP74_PHASE=d1 或 d2 启动，每个阶段只启动一次（`runs/exp74-<阶段>.started`），没有续跑或重试；启动前用 ps 完整命令行检查没有其他模型任务。

## 6. 实施和验证（CPU，已完成）

- 代码：`src/devision/train/align.py` 增加 `_full_picture`（唯一的产品外训练代码改动，不涉及推理模块）。
- 数据脚本：`scripts/panel_learn/build.py`、`eval_sets.py`、`read_states.py`；运行脚本：`consistency.py`、`report.py`、`run.sh`；配置：`configs/exp74-panel-d1.yaml`、`exp74-panel-d2.yaml`（两个 --dry-run 都通过）。
- 生成的数据：`data/panel_learn/` 下的 train、fit、dev16、erased、read_states_*，以及三个 MANIFEST。
- 夹具：
  - `tests/train/test_panel_learn_path.py`（4 项）：在带局部视图和 conditioning 的 tiny checkpoint 上，D1 align 能运行，只改投影层和 encoder；D2 mixed_input 能运行，只改 encoder 和头；align 的决策监督与 micro_batch 互斥，并且会去打开文字行的图；align 的视觉 token 与 visual_inputs 逐位相等。
  - `scripts/panel_learn/test_panel_learn.py`（8 项）：D1 门槛的退出码；D2 的 A / B（两种保护失败）/ C / invalid 结局；runner 的顺序、每阶段只启动一次、D1 门槛不过时拒绝 D2、非法阶段、上限值。
- 已有的 align / mixed 相关测试 23 项通过；pyright 0 错误。

## 7. 已知限制

- 单种子；D1 和 D2 都只训 1 遍（EXP26 是 2 遍 + 2 遍）。
- 保护只覆盖部分集合：K 文字切片、离线封存读数状态（只检验新决策模型如何使用这些状态，不是实时组合）、exist、science。A 之后仍需要实时组合、七个保护面板和新的冻结与校准，才能说真实交付保持。投影层在 D1 中被修改，这可能影响所有带图题，D2 的回放和 exist / science 探针只能部分覆盖；七个面板和全评留到 A 之后的筛查。
- 面板图的画法与 dev16 相同（exp21），学会了也不代表真实 ScienceQA 或其他版式会提高。
- fit 门槛用的是训练族：通过只代表"学得会"，不代表泛化。
