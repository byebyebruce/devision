# deVision 与 laya-vision 的差距

laya-vision 是目前唯一公开的、能看图的 System One（Jev 式）模型，也是本项目最直接的对照。本文对比两者的指标、结构和训练，并归纳差距的来源。laya-vision 的数据检索于 2026-10-01；deVision 一方用的是发布的 checkpoint `stage2-cocoqa`。

## 结论

**2026-10-03 更新（同一批题逐题配对，第 2 轮 `runs/v2-align-lora`，详见下一节）**：VQAv2 是非题打平（+0.4 个点，区间 ±2）；POPE 三档都不低于对方公开分；A-OKVQA 低 7 个点、ScienceQA 低 35 个点，这两类题我们没训练过、对方训练过；左右关系仍在随机水平。下面五条是 2026-10-01 用发布 checkpoint 和不同题目得出的旧结论，保留作对照。

- **"图里有没有某物"已经追平**：POPE adversarial 0.773 对 0.777。
- **日常是非题还差约 7 个点**：VQAv2 yes/no 0.648 对 0.715（历史最好的 `two-stage-mac` 是 0.688，差 3 个点）。
- **空间关系差距最大**：我们的左右判断在随机水平，laya-vision 在 VSR 上是 0.875。
- **校准持平**：ECE 都在 0.02–0.05；我们在 POPE 上偏差（0.098），因为温度没在这类题上拟合。
- 差距主要来自：我们的语言模型从没见过图、连接层是随机初始化的；以及训练数据少一个数量级、题型窄。

## 同一批题上的逐题对比（2026-10-03，2026-10-04 加 v5）

v5（`runs/v5-data3/stage2`）加入了 A-OKVQA / ScienceQA / AI2D / TQA 的训练集：A-OKVQA 追平（+2.4，区间含 0），ScienceQA 差距从 34 个点缩到 22.6 个点，但 v5 在 ScienceQA 上配错图仍有 0.576，分数主要来自文字而不是看图。

对象：第 2 轮 `runs/v2-align-lora/stage2b`（对齐阶段加 LoRA，非发布 checkpoint）vs laya-vision 201M 公开的逐题预测。评测集按他们的题重建（`scripts/data/lv_bench.py`，标签逐题核对一致；VQAv2 两边都去掉 113 道标注者五五开的题），配置 `configs/v2-align-lora-lvbench.yaml`。差值 = 我们 − 他们，95% 区间按图片配对重采样（`devision-compare`，结果在 `runs/v2-align-lora/eval/compare/`）。"未见过" = 去掉我们训练用过的图片（A-OKVQA 按感知哈希，近似）。

| 集合 | 题数 | laya-vision | v2 | v2 差值 [95% 区间] | v3 | v3 差值 [95% 区间] | v5 | v5 差值 [95% 区间] |
|---|---|---|---|---|---|---|---|---|
| VQAv2 yes/no，全部 | 4,887 | 0.717 | 0.721 | +0.004 [−0.013, +0.020] | 0.728 | +0.011 [−0.004, +0.026] | 0.736 | +0.019 [+0.002, +0.034] |
| VQAv2 yes/no，未见过 | 4,625 | 0.718 | 0.718 | +0.000 [−0.015, +0.017] | 0.727 | +0.010 [−0.006, +0.025] | 0.735 | +0.018 [+0.001, +0.034] |
| A-OKVQA，全部 | 1,138 | 0.598 | 0.527 | −0.071 [−0.105, −0.035] | 0.561 | −0.038 [−0.072, −0.004] | 0.622 | +0.024 [−0.007, +0.056] |
| A-OKVQA，未见过 | 1,089 | 0.592 | 0.527 | −0.065 [−0.099, −0.029] | 0.559 | −0.033 [−0.069, +0.002] | 0.622 | +0.029 [−0.004, +0.063] |
| ScienceQA（全部都未见过） | 2,097 | 0.824 | 0.471 | −0.353 [−0.379, −0.328] | 0.485 | −0.340 [−0.364, −0.314] | 0.598 | −0.226 [−0.251, −0.200] |

v3 对 v2（同题配对，`runs/v3-data2/eval/compare/v2_vs_v3.bench_lv_*.json`）：VQAv2 是非 +0.7 [−0.4, +1.9]，A-OKVQA +3.3 [+0.9, +5.9]，ScienceQA +1.4 [−0.3, +3.1]。v3 在这四个集合上都是 heldout（v2 的 VQAv2 / POPE 是 monitoring）。

POPE（9,000 题）他们只公开总分，不能配对；我们的区间只反映我们这一侧的抽样：

| POPE | laya-vision | v2 [95% 区间] | v2 precision / recall / yes 比例 | v3 | v3 precision / recall / yes 比例 | v5 | v5 precision / recall / yes 比例 |
|---|---|---|---|---|---|---|---|
| random | 0.836 | 0.842 [0.830, 0.853] | 0.961 / 0.713 / 0.37 | 0.850 | 0.970 / 0.721 / 0.37 | 0.857 | 0.973 / 0.735 / 0.38 |
| popular | 0.819 | 0.835 [0.823, 0.847] | 0.944 / 0.713 / 0.38 | 0.836 | 0.936 / 0.721 / 0.39 | 0.844 | 0.940 / 0.735 / 0.39 |
| adversarial | 0.777 | 0.805 [0.793, 0.817] | 0.875 / 0.713 / 0.41 | 0.805 | 0.866 / 0.721 / 0.42 | 0.811 | 0.868 / 0.733 / 0.42 |

去掉训练用过图片后（8,604 题）三档分别为 0.842 / 0.836 / 0.805，不变。

解读与限制：

- **VQAv2 是非题打平**，区间在 ±2 个点内。
- **POPE 三档都不低于对方**，popular / adversarial 高 1.6 / 2.8 个点；但我们偏向答 "没有"（recall 0.71），对方的 precision / recall 没有公开，比不了偏向。对这个模型 POPE 是 monitoring（训练中监测过 300 题的旧 POPE 与它共享图片），不是严格的未见集。
- **A-OKVQA 低约 7 个点，ScienceQA 低 35 个点（我们 0.471，按选项数加权的随机水平 0.357；A-OKVQA 随机 0.25）**：这两类题我们完全没训练过（常识推理、教科书插图），而 laya-vision 的训练数据（The Cauldron）含这两个数据集的训练集。这是题型覆盖的差距，不是同条件下的能力比较。
- 两边都没训练过的空间关系（VSR）仍没有同题对比；我们在自己的测试集上左右仍是随机水平（见实验记录）。
- 发布的 `stage2-cocoqa` 没有在这批题上评测过，上表不能当作 0.1 的结果。

## 指标（2026-10-01，发布 checkpoint 与对方公开数字，题目不同）

| 能力 | deVision（`stage2-cocoqa`） | laya-vision 201M | laya-vision ModernVBERT-250M | 差距 |
|---|---|---|---|---|
| POPE adversarial | **0.773**（300 题），ECE 0.098 | 0.777（3,000 题），ECE 0.048 | — | 持平，差别在噪声内 |
| POPE popular / random | 未测 | 0.819 / 0.836 | — | — |
| VQAv2 yes/no | 0.648（1,000 题），ECE 0.025 | 0.715（5,000 题），ECE 0.065 | 0.718（5,000 题） | 约 7 个点 |
| 空间关系 | 左右在随机水平（COCO 题 0.49，GQA 0.47–0.62） | VSR 0.875（160 题） | — | 最大 |
| GQA testdev（是非 + 二选一） | 0.635（1,000 题） | 未报告 | — | — |
| A-OKVQA（四选一） | 未测 | 0.598 | 0.652 | 未知 |
| ScienceQA（图片子集） | 未测 | 0.822 | 0.790 | 未知 |
| 整体 ECE | 0.025–0.054（POPE 0.098） | 0.041（34 个集合） | 0.022（22 个集合，校准后） | 持平 |

换成 100 道题（瞎猜对 50 道）：

| 题型 | laya-vision 比瞎猜多对 | 我们比瞎猜多对 |
|---|---|---|
| 有没有某物 | 28 | 27 |
| 日常是非题 | 21–22 | 15（历史最好 19） |
| X 在 Y 的左边还是右边 | 约 37 | 0 |

**不能直接比的地方：**

- 我们的 POPE 只有 300 题（标准误约 2.4%），laya-vision 是 3,000 题。
- 两边的 VQAv2 都是从官方 val 按图重切，但抽到的题不同；我们的正负各半。
- VSR 只有 160 题，来自 Cauldron 留出集，和训练图是否完全分开不清楚。
- 差 3 个点以内都不算显著。

## 结构

两边的骨架一样：冻结的视觉塔 → 连接层 → 语言模型 → Laya 决策头（2 层双向 Transformer + 打分器），损失（RLCD：带噪 logit 的 proper scoring rule 策略梯度 + soft CE）和按题型拟合温度也一样。差别在中间两段。

| | deVision | laya-vision 201M | laya-vision ModernVBERT-250M |
|---|---|---|---|
| 视觉塔 | SigLIP2-B/16，256 px，冻结 | SigLIP-B/16，512 px，冻结 | SigLIP2，512 px，冻结 |
| 压缩 | 2×2 → 64 个 token | 4×4 → 64 个 token | 4×4 → 64 个 token |
| **连接层** | 两层 MLP，**随机初始化**，靠 caption 完形填空从零对齐 | SmolVLM 自带，**已预训练** | ModernVBERT 自带，**已预训练** |
| **语言模型** | ModernBERT-large（3.95 亿，28 层，1024 维，双向），**只在纯文本上预训练过** | SmolLM2（裁到 20 层，576 维，因果），**预训练时已在看图** | Ettin-150M（ModernBERT 结构，22 层，768 维，双向），**预训练时已在看图**（MLM 对齐） |
| **语言模型怎么训** | 主干冻结，**只训 LoRA**（r=16，约 720 万参数） | 连接层和语言模型**全量训练** | 文本编码器和连接层**全量训练** |
| 选项读出 | 每个选项的 `[MASK]`，双向注意力，无顺序偏差 | 选项行末的 `\n`，因果注意力，靠打乱顺序 / 多排列平均缓解 | 每个选项的 `[MASK]`，双向 |
| 参数 | 5.18 亿 | 2.01 亿 | 2.67 亿 |
| 权重文件 | 2.07 GB（fp32） | — | — |
| 延迟 | 约 165–190 ms（Mac CPU） | 40.8 ms（L4 GPU） | 32 ms（L4 GPU，bf16） |

我们的"脑子"更大（3.95 亿参数的语言模型），但"眼睛"是临时接上的；对方的语言模型小，但和视觉部分在预训练时就一起学过看图。

## 训练

| | deVision | laya-vision 201M | laya-vision ModernVBERT-250M |
|---|---|---|---|
| 对齐 | COCO train2014 caption 82,583 张图，只训投影层，约 4 小时 | 不需要（SmolVLM 预训练） | 不需要（ModernVBERT 预训练） |
| 决策题 | 约 18 万题：GQA 6 万、VQAv2 是非 4 万、COCO 实例框出题 5.7 万 + 回放 | 182 万条（含游戏帧、Cauldron、VLFeedback、AVA 等） | Cauldron 19 个子集 26.99 万题 |
| 题型 | 是非、二选一 | 是非、多选、打分 | 是非、多选（2–8 选） |
| 硬件 / 时长 | Mac MPS，约 16 小时 | 1 张 H100，2 小时 | 1 张 A100，68 分钟 |

## 差距的来源

1. **语言模型没见过图，连接层从零训。** 本项目的实验显示，caption 对齐数据从 3 万张加到 8.3 万张，补词 nll 只降 0.06，阶段 1 已饱和；对方直接用预训练好的视觉-语言连接，不需要这一步。
2. **按文字找物体学不会。** 线性探针显示视觉 token 里"什么颜色的东西在哪边"完整可读（1.000），但用题里的词挑出对应物体这一步，在只训投影层、LoRA（r=16 / r=64）、全量放开 ModernBERT、样本加到 4 倍、改用带方位的完形填空时都学不会。这是空间题差距的直接原因（见 `../experiments/2026-09-30-rlcd-plateau.md`）。
3. **数据少一个数量级，题型窄。** 对方有 VSR、Visual7W、TallyQA 等专门的空间、指代、计数题；我们只有 GQA、VQAv2 和自己从 COCO 框出的题。VQAv2 的差距主要来自这里。
4. **分辨率。** 对方 512 px，每个视觉 token 覆盖更多原图细节；我们 256 px。线性探针显示位置信息并没有丢，所以这不是目前的主要瓶颈，但对小物体和细节类问题有影响。

我们占优的地方：双向注意力，选项之间天然互相可见，没有顺序偏差；语言模型更大。

## 可以怎么缩小差距（不换骨干）

| 方向 | 针对 | 成本 |
|---|---|---|
| 把 The Cauldron 的闭式题（约 27 万，含 VSR、Visual7W、TallyQA）转换进来，参考 laya-vision 的 `cauldron.py` | 日常是非题、题型覆盖 | 中 |
| 温度拟合时加入 POPE 类的题 | POPE 校准 | 低 |
| 更大规模的绑定训练（5 万张以上合成图，放开 ModernBERT，更长训练） | 空间关系 | 高，不保证有效 |
| 在决策头里加"选项标记对视觉 token 的交叉注意力" | 空间关系 | 中，要先改 spec |
| 在同一批集合上评测（POPE 完整 3,000 题、A-OKVQA、VSR、ScienceQA） | 让对比更公平 | 低 |

换成 ModernVBERT 这类预训练过的视觉-语言编码器最可能一步缩小差距，但这属于换骨干，当前不做。

## 来源

- [laya-vision 201M 评测页](https://r33drichards.github.io/laya-vision/reference/evals/laya-vision-201m/)
- [thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision)、[thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m)
- [thaitea/laya-vision-modernvbert-250m](https://huggingface.co/thaitea/laya-vision-modernvbert-250m)
- [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision)，结构说明见 `site-docs/concepts/architecture.md`
- [ModernVBERT/modernvbert](https://huggingface.co/ModernVBERT/modernvbert)（arXiv 2510.01149）
- [systemonemodels.org](https://systemonemodels.org/)：收录的 System One 模型里其余都只接受文本
- deVision 的数字：`runs/v1-release-0.1/eval/*.json`（第 1 轮）、`runs/v2-align-lora/eval/`（第 2 轮）、`runs/analysis/`，过程见 `../experiments/2026-09-30-rlcd-plateau.md`
