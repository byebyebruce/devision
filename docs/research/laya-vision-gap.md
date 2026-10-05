# deVision 与 laya-vision 的差距

> 2026-10-04 架构复核：当前 201M checkpoint 使用选项区双向注意力；双向编码也不保证选项换序不变。旧文中的“无顺序偏差”和“对齐只训投影层”等描述不能作为当前结论。历史内容保留，更新口径见文末“架构复核”。

> 2026-10-04 19:42:19 +08:00 数据口径复核：201M 的“182 万”是累计训练抽样次数，其中 45% 为游戏帧，不能与我们去重后的问答数直接比较；ModernVBERT 分支报告的是处理后的约 26.99 万训练题。旧文“少一个数量级”不再作为当前结论；当前 data-v4 通用混合为 168,836 题，A-OKVQA、ScienceQA、AI2D、TQA 均已接入。固定源码的 19 个 Cauldron 子集不含 TallyQA，它是另行建议的复杂计数来源。历史描述保留，来源、版本及适配边界见[数据扩充复核](data-expansion-laya-sources-2026-10-04.md)。

laya-vision 是目前唯一公开的、能看图的 System One（Jev 式）模型，也是本项目最直接的对照。本文对比两者的指标、结构和训练，并归纳差距的来源。laya-vision 的数据检索于 2026-10-01；deVision 一方用的是发布的 checkpoint `stage2-cocoqa`。

## 结论

**2026-10-03 更新（同一批题逐题配对，第 2 轮 `runs/v2-align-lora`，详见下一节）**：VQAv2 是非题打平（+0.4 个点，区间 ±2）；POPE 三档都不低于对方公开分；A-OKVQA 低 7 个点、ScienceQA 低 35 个点，这两类题我们没训练过、对方训练过；左右关系仍在随机水平。下面五条是 2026-10-01 用发布 checkpoint 和不同题目得出的旧结论，保留作对照。

- **"图里有没有某物"已经追平**：POPE adversarial 0.773 对 0.777。
- **日常是非题还差约 7 个点**：VQAv2 yes/no 0.648 对 0.715（历史最好的 `two-stage-mac` 是 0.688，差 3 个点）。
- **空间关系差距最大**：我们的左右判断在随机水平，laya-vision 在 VSR 上是 0.875。
- **校准持平**：ECE 都在 0.02–0.05；我们在 POPE 上偏差（0.098），因为温度没在这类题上拟合。
- 差距主要来自：我们的语言模型从没见过图、连接层是随机初始化的；以及训练数据少一个数量级、题型窄。

## 同一批题上的逐题对比（2026-10-03，2026-10-04 加 v5、v6、v7，2026-10-05 加 v8）

v8（`runs/v8-vsr-v7w/stage2`，从 v7 接着训，加 VSR 与 Visual7W telling）：ScienceQA 0.709 → 0.740，差距缩到 8.4 个点；VQAv2 是非、A-OKVQA 持平；POPE 三档比 v7 低约 1 个点，仍高于对方公开分。
v9b（`runs/v9b-lr-replay/stage2`，从 v8 接着训 2 遍，真实照片左右题加回放，是否取代 v8 待定）：与 laya-vision 逐题配对，VQAv2 是非 +0.9 [−0.6, +2.4]、A-OKVQA +1.5、ScienceQA −8.2 [−10.3, −6.2]，与 v8 相近；POPE 三档 86.6 / 84.8 / 81.1，比 v8 高约 1 个点。左右关系第一次在留出镜像集上学会，位置题同时对 23% → 53%（训练日志 v9b 一节）。
ScienceQA 剩下的差距几乎都在必须看图的题上：同题面不同答案的 393 题对方 72.5%、v8 50.4%，自然科学部分 70.0% 对 44.9%——对方确实在看示意图，我们在这类题上接近随机。每轮的详细比较由 `scripts/lv_report.py` 生成到 `runs/<轮次>/eval/compare/laya_vision.md`。

v7（`runs/v7-sqa/stage2`，从 v6 接着用 data-v4 的 ScienceQA 训一遍）：ScienceQA 0.584 → 0.709，差距从 24.0 缩到 11.5 个点；配错图 0.642，看图的增益从 1.5 升到 6.7 个点，但分数仍以文字为主（涨幅大头在社会科学的模板题，见 `training-log.md` v7 一节）。其余三项与 v6 持平。

v5（`runs/v5-data3/stage2`）加入了 A-OKVQA / ScienceQA / AI2D / TQA 的训练集：A-OKVQA 追平（+2.4，区间含 0），ScienceQA 差距从 34 个点缩到 22.6 个点，但 v5 在 ScienceQA 上配错图仍有 0.576，分数主要来自文字而不是看图。

对象：第 2 轮 `runs/v2-align-lora/stage2b`（对齐阶段加 LoRA，非发布 checkpoint）vs laya-vision 201M 公开的逐题预测。评测集按他们的题重建（`scripts/data/lv_bench.py`，标签逐题核对一致；VQAv2 两边都去掉 113 道标注者五五开的题），配置 `configs/v2-align-lora-lvbench.yaml`。差值 = 我们 − 他们，95% 区间按图片配对重采样（`devision-compare`，结果在 `runs/v2-align-lora/eval/compare/`）。"未见过" = 去掉我们训练用过的图片（A-OKVQA 按感知哈希，近似）。

| 集合 | 题数 | laya-vision | v2 | v2 差值 [95% 区间] | v3 | v3 差值 [95% 区间] | v5 | v5 差值 [95% 区间] | v6 | v6 差值 [95% 区间] | v7 | v7 差值 [95% 区间] | v8 | v8 差值 [95% 区间] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VQAv2 yes/no，全部 | 4,887 | 0.717 | 0.721 | +0.004 [−0.013, +0.020] | 0.728 | +0.011 [−0.004, +0.026] | 0.736 | +0.019 [+0.002, +0.034] | 0.736 | +0.019 [+0.002, +0.034] | 0.733 | +0.015 [−0.001, +0.030] | 0.733 | +0.015 [−0.001, +0.030] |
| VQAv2 yes/no，未见过 | 4,625 | 0.718 | 0.718 | +0.000 [−0.015, +0.017] | 0.727 | +0.010 [−0.006, +0.025] | 0.735 | +0.018 [+0.001, +0.034] | 0.735 | +0.018 [+0.001, +0.035] | 0.732 | +0.014 [−0.002, +0.030] | 0.733 | +0.015 [−0.001, +0.032] |
| A-OKVQA，全部 | 1,138 | 0.598 | 0.527 | −0.071 [−0.105, −0.035] | 0.561 | −0.038 [−0.072, −0.004] | 0.622 | +0.024 [−0.007, +0.056] | 0.622 | +0.024 [−0.008, +0.057] | 0.625 | +0.026 [−0.005, +0.059] | 0.613 | +0.015 [−0.018, +0.048] |
| A-OKVQA，未见过 | 1,089 | 0.592 | 0.527 | −0.065 [−0.099, −0.029] | 0.559 | −0.033 [−0.069, +0.002] | 0.622 | +0.029 [−0.004, +0.063] | 0.622 | +0.029 [−0.004, +0.064] | 0.625 | +0.033 [+0.000, +0.065] | 0.613 | +0.021 [−0.010, +0.054] |
| ScienceQA（全部都未见过） | 2,097 | 0.824 | 0.471 | −0.353 [−0.379, −0.328] | 0.485 | −0.340 [−0.364, −0.314] | 0.598 | −0.226 [−0.251, −0.200] | 0.584 | −0.240 [−0.266, −0.215] | 0.709 | −0.115 [−0.137, −0.093] | 0.740 | −0.084 [−0.105, −0.062] |

v3 对 v2（同题配对，`runs/v3-data2/eval/compare/v2_vs_v3.bench_lv_*.json`）：VQAv2 是非 +0.7 [−0.4, +1.9]，A-OKVQA +3.3 [+0.9, +5.9]，ScienceQA +1.4 [−0.3, +3.1]。v3 在这四个集合上都是 heldout（v2 的 VQAv2 / POPE 是 monitoring）。

POPE（9,000 题）他们只公开总分，不能配对；我们的区间只反映我们这一侧的抽样：

| POPE | laya-vision | v2 [95% 区间] | v2 precision / recall / yes 比例 | v3 | v3 precision / recall / yes 比例 | v5 | v5 precision / recall / yes 比例 | v6 | v6 precision / recall | v7 | v7 precision / recall | v8 | v8 precision / recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.836 | 0.842 [0.830, 0.853] | 0.961 / 0.713 / 0.37 | 0.850 | 0.970 / 0.721 / 0.37 | 0.857 | 0.973 / 0.735 / 0.38 | 0.856 | 0.975 / 0.731 | 0.865 | 0.962 / 0.760 | 0.852 | 0.968 / 0.729 |
| popular | 0.819 | 0.835 [0.823, 0.847] | 0.944 / 0.713 / 0.38 | 0.836 | 0.936 / 0.721 / 0.39 | 0.844 | 0.940 / 0.735 / 0.39 | 0.838 | 0.930 / 0.731 | 0.842 | 0.910 / 0.760 | 0.836 | 0.929 / 0.729 |
| adversarial | 0.777 | 0.805 [0.793, 0.817] | 0.875 / 0.713 / 0.41 | 0.805 | 0.866 / 0.721 / 0.42 | 0.811 | 0.868 / 0.733 / 0.42 | 0.808 | 0.864 / 0.730 | 0.810 | 0.846 / 0.759 | 0.808 | 0.867 / 0.728 |

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

## 架构复核（2026-10-04）

记录时间：2026-10-04 10:15:21 +08:00。本地代码 `7a8c459`，性能参考已有 `runs/v5-data3/stage2` 产物；没有运行新评测、训练或加载模型。外部代码核对到 [`9e1e241`](https://github.com/r33drichards/laya-vision/tree/9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c)，201M 权重仓库 revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`，ModernVBERT 分支 revision `4081dc6e00284a0d7da2071baedb9f8bf2f9d451`。仅读取远端文档、源码及配置，没有下载权重。

### 已核实的区别

| 维度 | deVision 当前架构 | Laya Vision 201M | Laya Vision ModernVBERT 分支 |
|---|---|---|---|
| 初始化 | 英文 Laya 的 ModernBERT-large 和决策头 + 单独预训练的 SigLIP2 + 新建投影层 | 已有图文预训练的 SmolVLM；当前版本继承前一版视觉决策 checkpoint 并裁层继续训练 | 已有图文预训练的 ModernVBERT + 为其维度训练的决策头 |
| 文本网络 | 28 层、1024 维、双向，约 395M | SmolLM2 裁为 20 层、576 维；因果主干，选项区双向 | 22 层、768 维、双向，约 150M |
| 视觉输入 | SigLIP2，256×256；256 patch → 2×2 合并 → 64 token | SigLIP，512×512；1024 patch → 4×4 合并 → 64 token | SigLIP2，512×512；1024 patch → 4×4 合并 → 64 token |
| 决策读出 | 2 层双向 Transformer + scorer；每个候选项前的 `[MASK]` | 同类决策头；候选项行末的换行 token | 同类决策头；`[MASK]` |
| 训练配方 | 当前 scratch 对齐训投影层 + LoRA；决策阶段训投影层 + 决策头 + LoRA；视觉塔冻结 | 视觉塔冻结，文本主干和连接层全量训练 | 视觉塔冻结，文本主干和连接层全量训练 |
| 参数量 | 约 518M | 约 201M | 约 267M（名称仍是 250M） |

本地依据：`src/devision/model/network.py:54`、`:92`、`:131`；`src/devision/train/align.py:143`；`src/devision/train/rlcd.py:326`；`configs/scratch.yaml`。518M 由已有 checkpoint 的 safetensors 头部 shape 统计复核（约数包含少量 buffer 和未使用的 pooling / act head），没有读取权重张量。外部依据：[201M 模型卡](https://huggingface.co/thaitea/laya-vision/tree/f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc)、[201M 配置](https://huggingface.co/thaitea/laya-vision/blob/f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc/vlm_agent_config.json)、[ModernVBERT 模型卡](https://huggingface.co/thaitea/laya-vision-modernvbert-250m/tree/4081dc6e00284a0d7da2071baedb9f8bf2f9d451)、[决策头与训练模式源码](https://github.com/r33drichards/laya-vision/blob/9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c/laya/vlm.py#L511)。同类决策头不代表直接复制英文 Laya 的头权重；三者 hidden size 不同。

### 对旧结论的修订

- **选项可见性**：201M 配置明确为 `option_attention: block`。选项区互相可见，顶部决策头也双向；“只能看前面的选项”只适用于 causal 配置的主干。双向结构本身也不消除位置、措辞和训练分布造成的顺序敏感性，需换序对照验证。
- **预训练起点**：我们继承的是文本决策能力，图文融合仍需自己训练；对方继承的是多模态融合能力。当前 deVision 对齐也训练 LoRA，不能继续称为“只训投影层”。这能解释训练难点的差异，但不能单独证明某项分数差距由架构导致。
- **空间与细节**：512 输入提供更多像素和 patch，两边送入文本网络的图像 token 数仍是 64，且均为 8×8 网格。提高输入分辨率有利于保留细节，不等于增加输出空间格点，更不保证文字与区域绑定成功。SigLIP2 的代际更新和更大的文本主干也不能单独证明最终效果更好。
- **资源**：我们的 LoRA 降低了相对自身全量微调的梯度及优化器开销，不能据此断言总训练内存比 201M 更低；518M 的参数存储更大，反向传播仍有激活开销。对方 512 输入又增加视觉侧计算，不能用参数比例直接推算延迟。既有 Mac CPU 与 L4 GPU 数字不是同条件速度比较。
- **结果归因**：已存逐题比较中，v5 对 201M 的 VQAv2 是非差值 +1.86 个点（95% 区间 +0.22 到 +3.41），A-OKVQA +2.37（−0.70 到 +5.62），ScienceQA −22.56（−25.08 到 −20.03）。A-OKVQA 尚无明确领先证据，也不是等效性结论。v5 ScienceQA 真图 0.5985、错配图 0.5756；正确图像带来的增益有限。数据、预训练、分辨率和训练预算不同，不能将这些差异单独归因于架构。依据：`runs/v5-data3/eval/compare/bench_lv_*.all.json` 与 `bench_lv_scienceqa{,.mismatched}.json`。

### 判断与可选验证

保留现有架构继续验证已计划的数据改进有依据；暂无同条件证据要求立即换骨干。若后续由负责人安排架构实验，ModernVBERT 是更接近当前双向 `[MASK]` 决策路径的对照：先固定一批未参与调参的绑定/镜像与插图题，在同一 CPU、精度、线程、batch 和预热设置下比较真实图、错配图、换序结果及延迟/峰值内存，再决定是否开展同数据训练。首轮只回答“现成 checkpoint 的效果与成本是否值得继续研究”，不能证明架构因果优劣。由执行者实施，需另行下载权重与运行推理；成本取决于题数和设备，先用少量请求估时后设置预算。本次未执行这些建议。
