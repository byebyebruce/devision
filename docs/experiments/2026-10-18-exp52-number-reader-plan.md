# exp52 方案：共享局部数值读出与无 oracle 的普通决策（N / Z 两臂；CPU 准备完成，未运行，等评论家的模型许可）

日期：2026-10-18。依据：评论家 EXP52-PREP-01（EXP51-RESULT-01 验收之后）。

## 问题

EXP51 表明：在给定字段区域时，EXP50-T 的原始局部 K/V 能读出原图数字（rl_tune 整串 0.827，T − C +0.160），但 EXP50 的普通决策没用上。本实验检验：把数字读出放到普通选项评分实际读取的同一组 token 上，并给这组 token 直接的数字监督，能否让**普通 decide**（无 oracle）在原图上进步；以及进步是否来自数字监督本身（N − Z）。

## 实现（`src/devision/model.py`、`src/devision/train/rlcd.py`；默认关闭）

- **`ModelConfig.number_reader`**：默认 False，旧 checkpoint 不受影响（夹具：去掉该字段的旧 config 加载后答案逐位相同）。
- **`NumberReader`**：
  - 5 个可学习槽，宽度 256；
  - 第一步，槽对问题自身的逐个 token 做注意力：用 `question_mask` 的范围（[CLS] 之后、第一个选项标记前的 [SEP] 之前），不取平均，选项不在这个范围里；
  - 第二步，问题条件化后的槽对视觉 K/V 做注意力：EXP50-T 的全图 + 12 视图 SigLIP patch（3,328 个），每个 patch 加上它在整张原图中的几何（visual_inputs 原有的 patch_geom；不是相对字段框的 oracle 几何）；
  - 得到的内容 token 同时走两路：
    - `chars`：每槽一个字符，类别为数字、"."、EOS，只用于训练损失和诊断；
    - `to_text`：映射到 ModernBERT 宽度，接在 64 个视觉 token 之后（视觉序列 64 + 5 = 69），普通选项评分直接读取。
  - 内容由图片和问题文本决定；不读框、field / object 标签、qid、gold、候选或答案长度。
  - 与 EXP51 头的关系：沿用其宽度 256、5 个槽和字符表；但查询由问题 token 条件化，几何是全图几何，参数从头初始化，不复用 EXP51 的 oracle 读出头。
- **前向**：`VisionDecisionModel.forward` 中，reader 读取的是条件模块改写 visual 之前的 patches；内容 token 拼到 visual 之后再进入 ModernBERT。marker 偏移随 visual 长度自动加上。decide、训练、验证和读出脚本走同一个前向，没有 detach，也没有训练专用的旁路。
  - 诊断开关 `reader.keep_chars`（保留字符 logits）和 `reader.off`（不拼接内容 token，只用于固定移除检查），都不存进 checkpoint，默认不影响 decide。
  - decide 的 usage 计数包含这 5 个 token。
- **训练**（`TrainConfig`）：
  - 新增 `number_reader`、`number_file`、`number_weight`、`lr_reader`。
  - reader 在批次种子之前，用固定种子 seed + 52（fork_rng）初始化，所以两臂逐张量相同。
  - 字符损失：严格整串字符 CE，包含 EOS，每题先对有效槽取平均，再对本批有目标的题取平均。两臂都计算（用于监测），只乘以各自的权重。
  - reader 单独一个参数组（lr 5e-4，与其他组共享预热和余弦）；SigLIP 全冻结。
  - 与 grounding、MLM 数字、蒸馏损失同时使用时拒绝。
  - train_report 记录 number（权重、文件 sha、lr、init_seed、参数量）。

## 数据与批计划

- **数字侧表**（`scripts/exp52/build.py` → `data/exp52/numbers.jsonl`，`a2f354b3…`）：EXP42 训练集全部 4,608 题（都是读值题）的金标数字，去掉千位逗号和单位，长度 1–4 个字符。只作训练目标，前向不读它（夹具：选项、qid、gold 改变时 reader 输入不变）。
- **训练**：EXP42 O+L 训练集，576 次更新，micro batch 8，每批两个完整四题组。
  - 批计划与 EXP50 的 `data/exp50/PLAN.json`（`dd276177…`）相同：EXP50-T 与 EXP48-X 的 tokenizer 文件哈希相同（`e1ec4ab9…`），config 只差局部视图字段。
  - val 用 val_quarter。
  - CE + interaction 1.0、lr / warmup / seed 都与 EXP50 相同；终点固定在第 576 步，不按 tune 选点。
- **两臂**：`configs/exp52-{n,z}.yaml` 只差 name 和 number_weight（1.0 / 0.0），都从 EXP50-T 的冻结终点（`1a8e5439…` / `99c6246e…`）开始。对照叫 Z（权重 0），避免与 EXP50 的 C（降细节）混淆。

## 读出与预写判据（`scripts/exp52/report.py`）

- **普通决策**：devision-eval（raw 温度 1，分组 8）。集合为 fit_O / fit_L / rl_tune / rl_tune_L，基线 B = EXP50-T 已冻结的同题记录。统计用 exp42 的口径和 EXP48 的严格四题。
  1. **progress_A**（A ∈ {N, Z}，对应 EXP44 的条件）：原图 rl_tune 异值 ≥ 0.60、两对象 ≥ 0.20、孪生 ≥ 0.20；两对象和孪生相对 B 的 case 簇配对差下界都 > 0。四题、同答、各字段、fit 和 L 单独报告，不能替代原图。
  2. **attribution**：N − Z 原图两对象 ≥ +0.10，下界 > 0，且 progress_N 成立。Z 自己的进步作为共同结构的正证据保留，不归于数字监督。
- **数字读出**（`scripts/exp52/readout.py`，无 oracle）：
  - 在 fit_O 和 rl_tune 的读值题上做一次普通前向，同时得到选项 logits 和同一组内容的字符串；
  - 供体 = 同一问题在值改变孪生图上的预测；
  - 统计用 EXP51 的口径（整串、字段、两对象、孪生、真减供体、跟随供体），EXP51 的投资门槛只作参考，并注明没有 oracle、难度不同；
  - 数字阳性但决策没进步，记为"到决策的使用尚未打通"；数字没学好，不重新宣布局部像素不可读。
- **内容使用检查**：同一次读出里，再做一次不拼接内容 token 的前向（固定移除），报告 rl_tune / fit_O 上有无内容时的准确率和预测改变的比例，只作依赖性证据。
- **迁移和风险**：ScienceQA temp / force 的真图和固定错图切片（41 + 120 题），N、Z 和 EXP50-T（runs/exp52-base，只读其冻结权重，不改它的目录），报告准确率和看图收益；exist / science 只报告，不宣称七面板保护。
- 缺项记 incomplete，不补判。

## 预算（估计；以首窗和实测为准）

基准是 EXP50-T 的实测：训练 26.5 分钟（2.31–2.37 秒 / 更新）；fit_O 1:53、fit_L 1:52、rl_tune 5:08、rl_tune_L 5:09；exist 3:16、science 1:51。

| 顺序 | 项目 | 估计 |
|---|---|---|
| 1 | N 训练（约 2.45 秒 / 更新，reader 多两个小注意力）+ 冻结 | 27.8 分钟 |
| 1 | N 四个主集合 | 14.0 分钟 |
| 2 | Z 训练 + 四个主集合 | 41.8 分钟 |
| 3 | 数字读出和使用检查：每臂 880 张图，每张一次视觉编码和两次 ModernBERT 前向 | 2 × 约 8 分钟 |
| 4 | ScienceQA 迁移：322 次请求，N / Z / EXP50-T 各一次 | 3 × 约 2.8 分钟 |
| 5 | exist / science：N、Z | 2 × 5.1 分钟 |
| 6 | 报告 | 约 1 分钟 |
| | **合计** | **约 119 分钟** |

- **截止**：启动 + 120 分钟，最后 120 秒留给报告。判据所需的两臂四个主集合约 84 分钟完成，加上两臂读出约 100 分钟。最可能被截掉的是最后的 Z 风险集，按缺项报告，不靠截掉主对照兑现预算。若评审希望风险集也确定能完成，需要把上限定为 130 分钟。
- **watch**：首窗或相邻两个窗口 > 2.7 秒 / 更新就停；内存规则（MPS > 16 GB 且可用 < 2 GB）；停滞上限 15 / 15 / 10 分钟。
- **内存**：EXP50-T 训练时 MPS 9.75 GB。reader 新增参数 1,321,228 个，激活很小（8 题 × 5 槽 × 3,328 键）。预计峰值 ≤ 11 GB。
- **磁盘**：每臂 checkpoint 约 2 GB。
- **不加跨集合缓存**：读出脚本在一次前向里同时得到决策和数字诊断。

## CPU 验证

- `tests/train/test_number_reader.py`（7 项）：
  - 默认关闭，旧 checkpoint 不变；
  - reader 的输入只有图片和问题：选项排列 / 数量、qid 不改变字符 logits，换问题会改变；
  - decide 调用 reader 一次，ModernBERT 收到 64 + 5 个视觉 token；同图多问与单问答案一致；内容 token 改变选项 logits；保存再加载后 decide 一致；
  - 决策 CE 和字符 CE 的梯度都到达共享的 reader 主干；
  - 字符目标和损失的边界情况；
  - 两臂起始张量和第一步 loss 相同，之后不同；权重 0 的臂与没有侧表的运行逐步相同；
  - 坏侧表和不兼容的设置被拒绝。
- `scripts/exp52/test_exp52.py`（2 项）：侧表逐题覆盖训练集且等于金标；读出脚本在 tiny 局部 reader checkpoint 和真实图片上运行，供体等于孪生自己的预测，有无内容时的 logits 不同。
- 全仓 `uv run pytest -q` 276 项通过；exp50 / 51 / 52 脚本测试 30 项通过；pyright 0 错误。
- 无模型的 `scripts/exp52/report.py` 已复现 B（EXP50-T）的读出：rl_tune 两对象 0.056、孪生 0.055。

## 发布关键路径（如实更新）

- EXP52 若某臂 progress 成立（最早约 19:30 获知），仍须：候选训练与迁移 / 七保护 / dev16 筛查（时长未定）→ r13b/val 3,892 题全量校准（约 20 分钟）→ 冻结 → 29 集完整评测（12 视图约 7.7–9.0 小时，reader 再加少量）→ 确认集、概率质量、CPU / API、构建和发布核验（1–1.5 小时）。
- 即使候选训练用时为 0，最早也约在次日 05:00–06:30 结束。**按现有本机串行路线，今天 23:59 之前不能完成完整发布。** 发布证据不删减，不以旧权重改名代替；v0.3 继续保留。

## 状态

CPU 准备完成，**未运行**：没有加载真实权重，没有训练或评测。等评论家的 EXP52-RUN-01。
