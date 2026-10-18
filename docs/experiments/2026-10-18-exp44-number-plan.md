# exp44 方案：按问题重建数字，与决策 CE 联合训练（单 J 臂；CPU 准备完成，未运行，等评论家的模型许可）

日期：2026-10-18。依据：评论家 EXP44-PREP-01..04（EXP43 证据模式 2）。

## 问题

EXP43 显示，同一个 EXP42-C 能从文字转录中按对象取值（tune 两对象同时对 0.943），但看图时无论用名称还是左右指代，两对象同时对都只有约 0.01。

- 以前的做法：EXP36 / 37 是在整图说明文字里补全所有数字；EXP42 只有候选评分的 CE。
- 本实验的新变量：在普通决策 CE 训练的同时，直接要求共享编码器**按当前问题重建被问的那个数字**，不再只靠候选评分间接传递信号。

## 设置（J 臂；纯 CE 对照直接复用 EXP42-C，不重训）

- **起点**：与 EXP42-C 相同，`runs/exp40-m/stage2`（`3fb4232e…` / `74e32e7c…`）。不是从 C 再多训 576 步。
- **数据**：`data/exp42/train.jsonl`（`27e57372…`），144 个单元 × 8 题 × 重复 4 次 = 4,608 次呈现、576 次更新。题、选项、gold、图片和批计划都不变（PLAN.json 不变：576 个满 8 的批，每批一个单元）。
- **配置**：`configs/exp44-j.yaml`，与 `configs/exp42-c.yaml` 只差 name、aux_weight 1.0、aux_slots 8。
  - objective ce，seed 1，参数组、LoRA、学习率、warmup、micro / eval batch 8、视觉塔顶 4 层、val_sub、监测和保存设置都相同。
  - 单种子、固定末点，不挑最好的一步。
- **数字重建分支**（`src/devision/train/rlcd.py`）：
  - 输入：`[CLS] state_text [SEP] 问题 [SEP] [MASK]×8 [SEP]`，同一张图。
    - 没有候选项、gold 数字、整图说明、转录、对象框或左右提示；也不注入目标单位或数值。
    - 8 个槽对所有题都相同，输入和 attention 与答案长度无关（夹具用真实 tokenizer 核过：7 种不同长度的 gold 得到完全相同的输入）。
  - 目标：gold 的完整数字（保留原来的小数和千位格式，按"前面有一个空格"的固定方式切 token）加 [SEP] 结束符，之后的槽 label 为 −100。
    - 缺数字、gold 不是单一选项、超出 8 槽或输入超过 512 都直接拒绝，不静默跳过。
  - 头：复用 align 阶段的 ModernBERT MLM 头（`align._MLMHead`，与当前冻结的词嵌入绑定），冻结、不进优化器，只在 8 个槽上算词表 logits。
    - 来源是本地缓存，不下载新模型。文件 sha256 `44510fec…`；三个张量的 sha256：head.dense.weight `e8efed1d…`、head.norm.weight `140da6a0…`、decoder.bias `328853aa…`（`data/exp44/CHECK.json`）。
    - 辅助头不进入产品推理，checkpoint 里没有它。
  - loss = 普通 CE + 1.0 × 数字 loss。数字 loss 先在每题的数字 token 和结束符上取均值，再在题间取均值。
  - 日志分别记录 `train/nll`（gold CE）、`train/number_loss`、`train/number_accuracy`（整串精确正确，只看目标槽）。新的 total loss 不能和 EXP42 的 NLL 直接比较。
- **执行方式**：
  - 同一有效批先做普通 CE 前向和反向，释放计算图，再做辅助前向和反向。两次前向用的是同一份尚未更新的权重。
  - 一次 zero_grad、一次总梯度 clip、一次 optimizer / scheduler step。
  - CE 的有限性检查沿用原有的（在反向之前）；数字 loss 非有限时，在它反向之前报错，不做更新。
- **默认路径**：aux_weight = 0 时不加载头，不构建任何东西，训练与以前完全相同。
  - 启用时，头和题表在设种子之前构建（与 teacher 相同的位置），所以 LoRA 初始化、批序和 CE 噪声序列不变。
  - aux_weight / aux_slots / aux_head 写进 TrainConfig、resume 键和 train_report（含头的身份）。
  - 限制（评论家 EXP44-NOTE-01）：resume 只比较 aux_head 的仓库名，resume.pt 里没有存头张量的身份，所以同一仓库下实际头张量变了，恢复时**不会**被拒绝。本链是新目录、启动时核对实际缓存文件和张量、不重试，所以不受影响；以后若要用辅助 resume，需要先保存并比较头身份。

## 全量检查与例子（`scripts/exp44/check.py` → `data/exp44/CHECK.json`、`runs/exp44-preview/examples.md`）

- 训练集 4,608 题，fit_O / fit_L 各 576 题，rl_tune / rl_tune_L 读数题各 1,800 题：没有一题被拒绝。
  - 目标跨度（数字 token + 结束符）为 2–5 个 token，最大 5 ≤ 8。
  - 格式包括 9 / 99 / 999 / 9,999 / 9.9 / 99.9 / 9.99。
- 12 个例子按固定规则选取：rl_tune 每个字段按 id 取前两张异值图，每张两个对象。覆盖：
  - 整数（11 / 36）、长速度（1,270 / 1,370）、小数（21.5 / 24.7 / 2.75 / 3.75）；
  - 不同 gold 长度（2–4 个 token）；
  - 孪生图（rl00193 的 v0 和 v1）。
  - 每例列出辅助输入（含 8 个 MASK 的位置）、目标 token 和普通 CE 输入。CE 输入照常含候选项；辅助输入不含候选。

## 读出（`scripts/exp44/report.py`；运行前写定）

- **主证据**：正常 decide、原图、原名称问题、温度 1。
  - J 的 fit_O / fit_L / rl_tune / rl_tune_L；M（`runs/exp42-start`）和 C（`runs/exp42-c`）复用 EXP42 的冻结记录。
  - 统计沿用 EXP42 的四项和 case 簇。报告 J − M、J − C；O / L、fit / tune、三个字段分开。
  - exist / science 只报告风险。
- **数字读出**（`scripts/exp44/aux_readout.py`）：用冻结 MLM 头和固定 8 个 MASK，不给候选。
  - 对象：J 的四组；M 的 fit_O / rl_tune 优先级最低，缺了就明示 missing。
  - 截取：取到第一个 [SEP] 为止；没有结束符或不是数字记为无效，不按候选纠正。
  - 报告：整串精确正确（逐字段）、两对象、孪生、同答、有效率。
  - **依图检查**：只看数值改变的孪生，且两视图的辅助输入完全相同（只差图片），把另一视图的预测按本题 gold 打分作为"错图"。按字段报告"真图 − 错图"和 case 簇区间。
    - 本数据中两视图的 state_text 和问题相同，所以辅助输入相同；输入不同的孪生会被跳过并计数。
    - 依图阳性不等于证明了 OCR：粒子大小和间隙也随图变化。
- **数字读出的逐题核对**（评论家 EXP44-R1-01 / 02）：
  - 每条记录的 gold_number 必须等于从源题 gold 重建的数字；image_used 等于源图；aux_input_sha256 等于用该 checkpoint 的本地 tokenizer 按训练同一规则（rlcd.aux_example）重建的输入 hash（只加载 tokenizer）。
  - valid 按读出的解析规则重算，并与明细一致；correct = valid 且预测等于 gold。
  - 有效率按 valid 计，不是按"有预测串"计。
  - 不一致就拒绝，之后才做两对象、孪生和依图的汇总。
- **身份**（不满足就拒绝）：
  - M、C 仍是冻结身份；
  - J 的 START / run.json / train_report 都写明 objective ce、aux 1.0 / 8，起点和数据 sha 正确，头的身份等于 CHECK.json；
  - J 的每个文件都记录在冻结身份下；
  - 摘要中的 checkpoint / 温度 / control / data sha 都正确；逐题无漏无重。
  - 缺项记为 incomplete，不补 0。
- **条件**（预写；算力投入条件，不是发布条件；J 的四个主决策集缺任何一个，两个条件都记为 incomplete，已读的各集数值和区间保留，评论家 EXP44-R1-03）：
  - **进步**：原图 tune 异值宏平均 ≥ 0.60；两对象 ≥ 0.20 且 J − M 下界 > 0；孪生 ≥ 0.20 且 J − M 下界 > 0。三条同时满足才讨论扩大训练。
  - **归因于数字 loss**：原图 tune 两对象 J − C ≥ 0.10 且下界 > 0。
  - 辅助读数、fit 或 L 单独好转，只用于定位下一环，不自动接长训。如果辅助头读对了但决策仍失败，写作"共享编码器训练后能读出数字，但读数到候选评分仍未打通"，不说"视觉已修复"。
  - 主条件不满足就停止这个短配方，不加步、不换种子、不扫权重。

## 运行与预算（`scripts/exp44/chain.sh`）

- **启动前检查**：
  - 没有其他模型进程；
  - M、C 冻结检查通过，M 的 sha 正确；
  - 4 份 exp42 数据的 sha 正确；
  - 本地缓存的 MLM 头身份等于 CHECK.json；
  - 运行目录原本不存在。
- **截止**：启动后 75 分钟，最后 120 秒留给报告。各步用 run_capped，到点 TERM，15 秒后 KILL。不重试。
- **顺序**：
  1. J 训练（受 watch 监控）→ 冻结；
  2. J 的 4 个决策集（温度 1、每请求最多 8 题）；
  3. J 的 4 个数字读出集；
  4. exist / science；
  5. M 的 fit_O / rl_tune 数字读出（最低优先级）；
  6. 报告。
- **watch**（`scripts/exp44/watch.py`）：
  - 首窗或相邻两个窗口 > 5 秒 / 更新 → 停止（本轮每次更新两次前向）；
  - MPS > 16 GB 且可用内存 < 2 GB → 停止；
  - 停滞上限：首窗前 20 分钟、第 500 窗口之后 20 分钟、其余 10 分钟。
- **预算估计**：参考 EXP42-C 实测 1.39–1.41 秒 / 更新（一次前向和反向），以及 EXP42 的评测吞吐。

| 部分 | 估计 |
|---|---|
| J 训练：加载约 1 + 3 次 val_sub 约 5 + 576 × 约 2.4 秒（CE 1.4 + 辅助约 1.0；辅助序列 54–62 token，比 CE 输入短）约 23 + 保存约 1 | 约 30 分钟 |
| J 决策 4 集（EXP42 中约 6 分钟） | 约 6 分钟 |
| J 数字读出 4 集（4,752 题，只前向，8 题一批）+ 4 次加载 | 约 5 分钟 |
| exist / science | 约 1.5 分钟 |
| M 数字读出 2 集（2,376 题） | 约 2.5 分钟 |
| 报告 | 约 1 分钟 |
| 合计 | 约 46 分钟（上限 75，余量约 29 分钟） |

- 第 100 步出来后，按实测重算并回报。

## CPU 验证

- `tests/train/test_number_aux.py`，5 项：
  1. 用真实 tokenizer 核对输入与答案无关：7 种 gold 输入相同，8 个 MASK，输入里没有数字；目标长度不同、以 [SEP] 结尾；非数字、超槽、非单一 gold 都拒绝。
  2. 关闭时与以前的运行相同，即使写了 aux_head 也不加载。
  3. 开启时批次顺序和第一步损失与关闭时相同，之后不同；report 记录 aux；保存的 checkpoint 没有辅助头参数，加载后 decide 正常。
  4. 数字 loss 非有限时在更新前报错，不留 model.safetensors。
  5. 同一批"两次 backward"的梯度等于"联合 loss 一次 backward"（atol 1e-6）；梯度到达 projector；头被冻结。
- 已有的 test_objective / test_distill / test_resume 照常通过（合计 21 项）。
- `scripts/exp44/test_exp44.py`，10 项：数字读出在 tiny checkpoint 上端到端可运行；依图只在输入相同的孪生上计算；J 的身份拒绝错误的权重 / objective / 头；缺 J 时为 incomplete，C 的读数复现 EXP42；有效率按解析规则计（合法数字 / 有 SEP 的非数字 / 无结束符：6 题中 3 题有效）；改 gold_number、image_used、输入 hash 或谎报 valid，都拒绝；rl_tune 过线但缺 rl_tune_L 时两个条件都是 incomplete。
- 报告冒烟：真实的 M、C 记录复现 EXP42 读数（rl_tune 两对象 M 0.008、C 0.013）。
- chain 演练（假 uv）：启动检查（含头身份）通过；J 的 11 项按序 completed，冻结记录 10 个集合；报告在假摘要处拒绝（预期）。
- pyright 0。没有运行真模型。
