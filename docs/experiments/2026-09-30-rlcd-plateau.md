# 实验记录：直接 RLCD 微调停在 ln2（2026-09-30）

## 结论

从 Laya + SigLIP2 + 随机初始化的投影层出发，直接在 GQA/VQAv2 决策题上做 RLCD 微调，模型学不到东西：训练 nll 很快停在 ln2 ≈ 0.693，即每题都输出 50/50，val 准确率在随机水平。加 warmup、加数据、加步数都没用。瓶颈是投影层没有学会把视觉特征对齐到 ModernBERT，下一步是在任务微调前加投影层对齐阶段。

## 运行

所有运行都在 Apple M4（24 GB）上、`--device mps`，超参都是默认值：lr 投影层 1e-3 / 决策头 1e-4 / LoRA 2e-4，LoRA r=16，micro_batch 8，group_size 4，64 个视觉 token，余弦调度，无 warmup。val 都是 `data/val.jsonl`（200 题，GQA）。

| 运行 | 训练数据 | 步数 | 训练 nll | val 准确率 |
|---|---|---|---|---|
| `mvp-mps` | 450 题 | 约 112 | 0.69 附近 | 0.50 |
| `small-2k` | `train_small.jsonl`，2000 题 | 73/500（进程中断，原因不明） | 0.69 | — |
| 对照：无 warmup | `train_small.jsonl` | 200 | 0.694 | 0.535 |
| 对照：100 步 warmup | `train_small.jsonl` | 200 | 0.696 | 0.520 |
| `full-1ep` | `train.jsonl`，19998 题 | 2500（1 epoch） | 0.693 | 0.530 |

`full-1ep` 训练 nll 每 500 步的均值：0.697 / 0.694 / 0.694 / 0.693 / 0.694。拟合出的温度是 [1.0, 1.0, 1.0]。checkpoint 在 `runs/full-1ep`，SwanLab run `full-1ep`。

## 诊断

用真实模型、在 `train_small.jsonl` 的前 32 题上做的检查：

- **训练流程没有 bug**：换图后 true−false 的 logit 差平均变化约 1.6，图片确实在影响输出；投影层、决策头、LoRA 三组参数都有梯度；32 题反复训练 60 步能记住（nll 0.15，训练集准确率 0.91）。
- **Laya 不看图时是自信地瞎猜**：logit 差在 ±3 左右，16 题准确率约 0.4，nll 比 0.693 还高。
- **退化是最优解，不是被打坏**：约 5–75 步内，所有题的 logit 差缩到 ±0.05。模型读不出图像信息时，把输出收成 50/50 就是 nll 最低的解。warmup 只把退化推迟了约 25 步。
- **第 0 步投影层梯度范数是 1432**，决策头 13，LoRA 66。随机初始化的投影层一开始输出的视觉 token 基本是噪声。

## 判断

随机初始化的投影层要学会把 SigLIP 的特征变成 ModernBERT 能用的 token。每道决策题只提供 1 bit 的监督信号，太弱，2 万题一个 epoch 也不够。常见的视觉语言模型会先用大量图文对单独训练投影层（对齐阶段），再做任务微调，我们跳过了这一步。

## 追加：只训投影层的图文匹配也学不动

想先用图文匹配题（noul："Does this description match the image? {caption}"）单独对齐投影层。数据是本地 9129 张 COCO train2014 图的 caption，正例用本图的 caption，负例用随机另一张图的 caption，训练 18058 题、val 200 题。冻结其余全部参数，只训投影层（lr 1e-3，50 步 warmup）：

- 第 0 步 val 准确率 0.475，训练 nll 0.85。
- 约 100 步后 nll 又停在 0.69，250 步时 val 准确率 0.500。实验提前停止。

这道题本身很容易，所以又做了两个对照，都在同一份 align_val（200 题）上：

| 对照 | 结果 |
|---|---|
| SigLIP2 自带文本塔零样本打分（用我们的 letterbox 预处理） | 以中位数为阈值准确率 0.99；正例平均分 −0.3，负例 −20.5 |
| 冻结的 Laya，不给图，把这张图的另一条 caption 当文字 state | 以中位数为阈值准确率 0.93；logit 差正例平均 −1.1，负例 −14.6 |

结论更新：

- SigLIP 的特征和预处理没问题，信息就在视觉特征里；SigLIP 的权重也完整加载了（missing keys 为 0）。
- 冻结的 Laya 读得懂"文字形式的图片内容"，会用它做判断。
- 卡住的是中间这段：一题一个是/否信号，经过冻结的 24 层 ModernBERT 反传，太弱，教不会随机初始化的投影层产出 ModernBERT 能读的 token。所以换成图文匹配题也没用，对齐阶段需要更密集的监督。

## 追加：带图完形填空对齐有效

对齐目标改成带图的完形填空（masked caption）：输入 `[CLS] <64 个视觉 token> <遮掉 50% 词的 COCO caption> [SEP]`，用 ModernBERT-large 原版的 MLM 预测头（`answerdotai/ModernBERT-large` 的 `head.*` 和 `decoder.bias`，decoder 权重与 Laya 编码器的词嵌入绑定）预测被遮的词。冻结其余全部参数，只训投影层：lr 1e-3，50 步 warmup，batch 16，共 800 步。训练图片同上面 align_train，评测用 align_val 的 100 张图。

val 上被遮词的准确率 / nll。"错配"指每张图配上另一张图的 caption：

| 步数 | 真实图片 | 错配图片 |
|---|---|---|
| 0（不给图） | 0.164 / 6.03 | — |
| 0 | 0.126 / 6.53 | 0.118 / 6.67 |
| 200 | 0.372 / 3.56 | 0.364 / 3.66 |
| 400 | 0.395 / 3.36 | 0.355 / 3.67 |
| 600 | 0.390 / 3.22 | 0.353 / 3.88 |
| 800 | 0.423 / 3.18 | 0.339 / 3.95 |

真实图片越来越好，错配图片越来越差，说明投影层开始传递图像信息了。

用这个投影层初始化，再做决策题微调（投影层 lr 1e-4、决策头 1e-4、LoRA 2e-4，50 步 warmup，400 步）：

| 任务 | 训练 nll | val 准确率 |
|---|---|---|
| GQA/VQAv2（`train_small.jsonl` → `val.jsonl`） | 一直 0.69–0.70 | 0.490 |
| 图文匹配（align_train → align_val） | 第 50 步 0.57，第 200 步 0.39，第 400 步 0.42 | **0.875** |

结论：

- 对齐后决策头能用上视觉信息。同一道图文匹配题，不做对齐时停在 0.69、val 0.50，对齐后 val 0.875。这是第一次训练 nll 明显低于 ln2。
- GQA/VQAv2 仍学不动。800 步对齐（约 1.3 万个图文对）太浅，还不够回答属性、空间关系这类细节问题。
- 下一步：把对齐阶段做进训练代码（先改 spec），放到 A100 上用全部 COCO caption（约 41 万条）做更长的对齐，再做 GQA/VQAv2 微调。

## 追加：两阶段完整跑通（Mac MPS，2026-10-01）

流水线 `runs/logs/pipeline.sh`，SwanLab run `align-mac`、`two-stage-mac`。

阶段 1 `devision-align`：COCO caption 29,800 张训练 / 200 张 val，batch 16，3 个 epoch 共 5,586 步，约 2 小时 20 分。val 上被遮词的准确率 / nll：

| 步数 | 真实图片 | 错配图片 |
|---|---|---|
| 0 | 0.137 / 7.07 | 0.122 / 7.09 |
| 500 | 0.407 / 3.16 | 0.360 / 3.96 |
| 1000 | 0.451 / 2.91 | 0.356 / 4.13 |
| 2500 | 0.487 / 2.60 | 0.356 / 4.20 |
| 5586 | 0.497 / 2.49 | 0.351 / 4.30 |

第 2,500 步后基本饱和。

阶段 2 `devision-train --init runs/align-mac --lr-new 1e-4`：`train.jsonl`（19,998 题），micro batch 8，4 个 epoch 共 10,000 步，约 3 小时。

| epoch | 训练 nll（每 50 步采样的均值） | val 准确率 |
|---|---|---|
| 1 | 0.662 | 0.530 |
| 2 | 0.619 | **0.620** |
| 3 | 0.509 | 0.615 |
| 4 | 0.421 | 0.595 |

拟合温度 choice 5.0（上限）、noul 3.15。最终评测：

| 集合 | 两阶段 `two-stage-mac` | 只做阶段 2 `full-1ep` |
|---|---|---|
| val（200 题） | 0.595（noul 0.627，choice 0.530），ECE 0.055 | 0.530 |
| POPE（300 题） | **0.763**，ECE 0.079 | 0.500 |

结论：

- 对齐后 GQA/VQAv2 能学动了：训练 nll 第一次持续低于 ln2，POPE 从 0.50 到 0.76。
- 第 2 个 epoch 之后训练 nll 继续降、val 不涨反降，开始过拟合；choice 题仍在随机水平（0.53），温度撞到上限 5.0，说明 choice 的 logit 过度自信且没学到东西。
- 下一步（仍在 Mac 上）：用全部 COCO caption（82,783 张图、41 万条）对齐 2 个 epoch，val 沿用同样 200 张图以便对比；阶段 2 只跑 2 个 epoch。之后再扩大阶段 2 数据量；choice 题单独看数据和损失。

## 追加：全量 COCO caption 对齐（Mac MPS，2026-10-01）

流水线 `runs/logs/pipeline-full.sh`，SwanLab run `align-full-mac`、`two-stage-full-mac`（训练中每 500 步在 val 和 POPE 上评测）。

阶段 1：COCO train2014 全部 82,583 张图（剔除评测图和 align val 的 200 张），batch 16，2 个 epoch 共 10,322 步，约 4 小时 10 分。终点 val 补词准确率 / nll：真实图片 0.501 / 2.43，错配图片 0.361，差 0.140（`align-mac`：0.497 / 2.49，0.351，差 0.146）。数据多 2.8 倍，nll 只降 0.06。

阶段 2：`--epochs 2`，5,000 步，约 1 小时 35 分。未套温度的评测：

| 步数 | val | noul | choice | val nll | POPE | POPE nll |
|---|---|---|---|---|---|---|
| 0 | 0.495 | 0.500 | 0.485 | 0.984 | 0.553 | 1.078 |
| 1000 | 0.615 | 0.612 | 0.621 | 0.654 | 0.633 | 0.653 |
| 2000 | 0.570 | 0.575 | 0.561 | 0.682 | 0.587 | 0.652 |
| 3000 | 0.570 | 0.612 | 0.485 | 0.708 | 0.713 | 0.596 |
| 4000 | 0.605 | 0.619 | 0.576 | 0.666 | 0.743 | 0.546 |
| 5000 | 0.610 | 0.634 | 0.561 | 0.682 | 0.740 | 0.536 |

拟合温度 choice 2.59、noul 1.77。最终（经 `decide`，和训练中 `final/` 一致）：val 0.610（noul 0.634，choice 0.561），ECE 0.075；POPE 0.740，ECE 0.110。

结论：

- 和 `two-stage-mac`（val 0.595 / choice 0.530 / POPE 0.763）的差别都在噪声内：val 200 题标准误约 3.5%，其中 choice 只有 66 题（约 6%），POPE 300 题约 2.5%。更多对齐数据没有带来可测的下游提升。
- 第 1 个 epoch 的训练 nll 相当于没见过的题上的 nll，两轮都在 0.68–0.70，接近 ln2：GQA/VQAv2 题几乎没有学到可泛化的东西。第 2 个 epoch 起训练 nll 下降而 val 不动，是在记题；`two-stage-mac` 第 3、4 个 epoch 降到 0.40 也是这样。
- POPE（有没有某物）能学会，仍在上升（第 4,500 步 0.753），阶段 2 可能还没训够。
- 温度在 val（全是 GQA）上拟合，套到 POPE 上 ECE 从 0.075 变差到 0.110：val 和 POPE 的分布不同。

## 追加：更大的评测集与按题型诊断（2026-10-01）

- 发现 COCO/VG 跨数据集泄漏：`train.jsonl` 有 10 题在评测图上（5 张，GQA 训练图里的 COCO val2014 图与 POPE 重合）。原因是剔除只比 `vg:`/`coco:` 原 id；现按 VG `image_data.json` 两边对齐，已从 train 剔除。
- 新评测集：`val_gqa`（GQA testdev 1000 题，346 张图，yes/no/choice 各约 1/3）、`val_vqav2`（VQAv2 val2014 yes/no 1000 题，980 张图，正负各半）。1000 题标准误约 1.6%。

| checkpoint | val_gqa（noul / choice） | val_vqav2 | ECE（gqa / vqav2） |
|---|---|---|---|
| `two-stage-mac` | 0.590（0.612 / 0.547） | **0.688** | 0.051 / 0.042 |
| `two-stage-full-mac` | 0.602（0.606 / 0.595） | 0.608 | 0.032 / 0.039 |

VQAv2 上 `two-stage-mac` 明显更好（+8 个点，约 5 个标准误），它的阶段 2 跑了 4 个 epoch，是本轮的两倍；GQA 上两者持平。

按 GQA 题型（`runs/analysis/gqa_by_type.py`，结果在 `runs/analysis/gqa_by_type.txt`）：

- **空间位置题在随机水平且塌缩到一个答案**：left/right 0.53（`two-stage-full-mac` 70% 答 right），top/bottom 0.47（77% 答 bottom），relChooser 0.53，positionChoose 0.50。训练 choice 题里 64% 是这类题。
- 属性选择（chooseAttr）0.53–0.60，类别选择 0.70–0.74。
- noul 上偏高的子类（twoSameMaterialC 0.90、relVerifyCo 0.80）几乎都是同一答案占 70–90%，是先验，不是看图。
- 推测：视觉 token 按光栅顺序排成 1D 交给 ModernBERT（1D RoPE），投影层没有显式 2D 位置；caption 对齐也很少要求左右/上下，所以空间信息没被训练出来。

阶段 2 扩大到 10 万题（`data/train_100k.jsonl`：GQA 6 万、VQAv2 4 万，同样剔除全部评测图），1 个 epoch，SwanLab run `stage2-100k`。

## 追加：10 万题阶段 2（2026-10-01）

数据 `data/train_100k.jsonl`（GQA 6 万 + VQAv2 4 万），从 `align-full-mac` 开始，1 个 epoch（12,500 步），`--val data/val_mix.jsonl --eval pope=data/pope.jsonl`。

**默认学习率那次塌缩了**（`stage2-100k`，日志 `runs/logs/stage2-100k.log`）：第 1,000 步 gqa 0.533 / vqav2 0.585 / POPE 0.597，第 2,000 和 3,000 步三个集合的 nll 都正好 0.693、准确率 0.50，即对每题输出 50/50；第 1,000 步前后单个 batch 的 loss 在 0.23–0.86 之间跳。原因：阶段 2 没有预热，余弦按 12,500 步衰减，第 2,000 步学习率还在峰值的 94%（`two-stage-full-mac` 共 5,000 步，同一步约 65%，它在 2,000–2,500 步 val nll 也一度回到 0.68–0.70）。另外 σ 按 epoch 插值，单 epoch 时恒为 0.4。已加线性预热（`--warmup`，默认 200）并改为按步插值 σ（commit `90f287e`），在第 3,250 步停掉。

**`stage2-100k-lowlr`**：`--lr-new 5e-5 --lr-head 5e-5 --lr-lora 1e-4 --warmup 500`，约 4 小时 50 分（含每 500 步评测），没有塌缩。

| 步数 | val_gqa（noul / choice） | gqa nll | val_vqav2 | vqav2 nll | POPE |
|---|---|---|---|---|---|
| 0 | 0.514（0.498 / 0.546） | 0.998 | 0.511 | 1.180 | 0.553 |
| 2500 | 0.548（0.564 / 0.516） | 0.676 | 0.605 | 0.658 | 0.643 |
| 5000 | 0.561（0.582 / 0.519） | 0.677 | 0.624 | 0.654 | 0.710 |
| 7500 | 0.617（0.643 / 0.565） | 0.661 | 0.615 | 0.651 | 0.700 |
| 10000 | 0.606（0.618 / 0.583） | 0.655 | 0.637 | 0.639 | 0.700 |
| 12500 | 0.607（0.621 / 0.580） | 0.649 | 0.634 | 0.638 | 0.713 |

训练 nll（每题只见一次，等于新题上的 nll），每 2,500 步平均：0.685、0.686、0.672、0.590、0.605。

拟合温度 choice 2.26、noul 1.40。最终经 `decide`：

| checkpoint | val_gqa（noul / choice） | val_vqav2 | POPE | ECE（gqa / vqav2 / pope） |
|---|---|---|---|---|
| `stage2-100k-lowlr` | **0.607**（0.621 / 0.580） | 0.634 | 0.713 | **0.034 / 0.032 / 0.054** |
| `two-stage-mac` | 0.590（0.612 / 0.547） | **0.688** | **0.763** | 0.051 / 0.042 / 0.079 |
| `two-stage-full-mac` | 0.602（0.606 / 0.595） | 0.608 | 0.740 | 0.032 / 0.039 / 0.110 |

GQA 按题型（`runs/analysis/gqa_by_type-stage2-100k-lowlr.txt`）：left/right 0.40（77% 答 left），top/bottom 0.50（73% 答 bottom），to the left of / to the right of 0.50，positionChoose 0.45，relChooser 0.51。属性选择 0.60，类别选择 0.70。

结论：

- 学习率是这条流水线的稳定性问题：没有预热时长训练会塌缩，修正后能稳定训完。
- 后 40% 的训练 nll 在 0.59–0.61，第一次明显低于 ln2，说明 10 万题里确实学到了能泛化的东西；但 val nll 只到 0.64–0.65，准确率 GQA 持平、VQAv2 和 POPE 反而比 `two-stage-mac` 低约 5 个点。数据量 5 倍，准确率没有突破。
- 校准是三轮里最好的（ECE 0.03–0.05，温度最接近 1）。
- **空间位置题仍在随机水平，而且塌缩到一个答案**：加数据没有用。
- 对照（检索于 2026-10-01）：唯一公开的视觉 System One 模型 laya-vision。201M 版（SmolVLM-256M 骨干，512 px）POPE adversarial 0.777、VQAv2 yes/no 0.715、VSR 0.875（160 题），ECE 0.041；ModernVBERT-250M 版（双向，`[MASK]` 读出，结构与本项目一致）VQAv2 yes/no 0.718，在 A100 上训 68 分钟。它们与本项目的主要差别是语言模型预训练时已看过图、连接层是预训练的且全量训练，而不是随机投影层加冻结的纯文本 ModernBERT。

下一步：把骨干换成 ModernVBERT（MIT，transformers 原生支持，本机 CPU 前向 255 ms，现模型 186 ms），去掉阶段 1；需要先改 spec，并把输入从 256 px 改为 512 px。

## 追加：空间信息丢在哪，以及 COCO 物体框出题（2026-10-01～02）

**探针**（`runs/analysis/spatial_probe.py`，结果 `spatial_probe.txt`）：合成图（随机底色 + 一个圆，明确在左/右半边、上/下半边），对三处表示做线性探针。`stage2-100k-lowlr` 上左右 / 上下的留出准确率：SigLIP2 特征 1.000 / 1.000，投影层输出 1.000 / 1.000，ModernBERT 在选项标记位的输出 0.976 / 0.998；`decide` 实际回答 0.498（100% 答 left）。位置信息一路到达决策头的输入，没有丢。

**可学性**（`runs/analysis/spatial_learnable.py`）：同样的合成题 2,000 张图 × 左右/上下各一问，从 `stage2-100k-lowlr` 训 1 epoch（500 步）：准确率 0.495 → 0.85（100 步）→ 0.97（200 步）→ 0.99。结构和训练流程能学会绝对位置；但 val_gqa 不变（0.607 → 0.607）。

**COCO 物体框出题**（`scripts/data/cocoqa.py`，仓库外，带测试）：从 `instances_*2014.json` 生成四类题，只问图中唯一实例的类别，位置只在间隔明确时出题：exist（noul；负例一半取共现最多的缺席类别，即 POPE adversarial 的做法）、absolute（左右或上下）、relative（choice 或 noul，左右或上下）、size（面积比 ≥ 2）。训练集 57,197 题（train2014，剔除全部评测图），评测集 `val_cocoqa` 1,000 题（val2014，四类各 250）。抽 8 题对图核对无误。

`stage2-cocoqa`：从 `stage2-100k-lowlr` 继续，数据 57k COCO 题 + 从 `train_100k` 回放 23k，1 epoch（10,000 步，约 3 小时 40 分），学习率同 lowlr，预热 500。没有塌缩，旧集合没有遗忘。

| 步数 | val_gqa（noul / choice） | val_vqav2 | POPE | val_cocoqa（noul / choice） |
|---|---|---|---|---|
| 0 | 0.607（0.621 / 0.580） | 0.634 | 0.713 | 0.626（0.698 / 0.575） |
| 1000 | 0.606（0.612 / 0.595） | 0.628 | 0.753 | 0.725 |
| 3000 | 0.627（0.643 / 0.595） | 0.627 | 0.773 | 0.700 |
| 6000 | 0.622（0.634 / 0.598） | 0.630 | 0.760 | 0.773 |
| 8000 | 0.638（0.642 / 0.631） | 0.639 | 0.777 | 0.766 |
| 10000 | 0.635（0.639 / 0.628） | 0.648 | 0.773 | 0.761 |

拟合温度 choice 2.72、noul 1.46。最终经 `decide`：val_gqa **0.635**（noul 0.639，choice 0.628），val_vqav2 **0.648**，POPE **0.773**，val_cocoqa 0.761；ECE 0.042 / 0.025 / 0.098 / 0.054。POPE 首次超过 `two-stage-mac`（0.763），和 laya-vision 201M（0.777，3,000 题）持平；GQA choice 第一次明显离开随机水平。POPE 的 ECE 变差（0.098），温度只在 val_mix 上拟合。

val_cocoqa 分题型（`runs/analysis/cocoqa_by_kind.py`），以及只看类别、不看图的先验基线（用训练集里每个类别的多数答案）：

| 题型 | `stage2-100k-lowlr` | `stage2-cocoqa` | 类别先验 |
|---|---|---|---|
| exist | 0.828 | **0.920** | — |
| absolute 左右（n=110） | 0.491 | 0.491 | 0.445 |
| absolute 上下（n=140） | 0.671 | 0.814 | 0.757 |
| relative 左右（n=49） | 0.469 | 0.388 | — |
| relative 上下（n=34） | 0.588 | 0.765 | — |
| size | 0.576 | 0.908 | 0.844 |

GQA 空间题：to the left of / to the right of 0.50 → 0.62（71% 答 right），left/right 0.40 → 0.47，bottom/top 0.50 → 0.67，positionChoose 0.45 → 0.57，relChooser 0.51 → 0.60。

结论：

- 有没有某物学得最好（exist 0.92，POPE 0.773）。
- **左右完全没学会**，即使标签干净：absolute 左右 0.49、relative 左右 0.39。上下和大小的提升大部分是类别先验（插座在下、时钟在上；冰箱比杯子大），比先验只高约 6 个点。
- 合成图上一个圆的左右 200 步就能学会，COCO 上却学不会：差别在于 COCO 题要先在多个物体里找到文字说的那个（"the dog"），再读它的位置。模型知道图里**有什么**，但不知道**文字说的那个东西在哪**——缺的是文字到图像区域的绑定。

## 追加：文字到区域的绑定（2026-10-02）

合成双物体图（`runs/analysis/binding_learnable.py`）：灰色噪声底上两个不同颜色的圆或方块，一个在左半边、一个在右半边，问 "Is the red square on the left or on the right?"。和单物体题不同，答案要靠题里的颜色和形状挑出物体。抽 4 张对图核对，标签无误（`runs/analysis/binding_check.jpg`）。

从 `stage2-cocoqa` 继续训练，留出 500 张评测（`runs/analysis/binding_learnable.txt`）：

| 设置 | 训练图 / 步数 | 准确率 |
|---|---|---|
| LoRA r=16（默认） | 2,500 / 626 | 0.54（从第 300 步起恒定） |
| LoRA r=64，lr_lora 4e-4 | 2,500 / 626 | 0.54 |
| 放开 ModernBERT 全部 28 层，lr 2e-5 | 2,500 / 626 | 0.54 |
| 放开全部 28 层，lr 5e-5，head 2e-4 | 2,500 / 626 | 0.50 |
| LoRA r=16 | 10,000 / 2,500 | 0.48–0.52，始终在随机水平 |

探针（`runs/analysis/binding_probe.py`）：视觉 token 上 "红色物体在哪边" / "蓝色物体在哪边" 线性可读 1.000 / 1.000；ModernBERT 在选项标记位的输出上，题目所问物体的方向只有 0.584。

结论：视觉侧有完整的 "什么颜色的东西在哪边"，但阶段 2 的决策训练学不会用题里的词去挑出对应的物体；放开 ModernBERT 全部参数、样本加到 4 倍都不行，所以不是 LoRA 容量问题。单物体题 200 步就能学会，说明卡在 "按文字选物体" 这一步。新增 `--unfreeze-top` / `--lr-encoder`（commit `f2da143`）。

带方位的完形填空（`runs/analysis/binding_captions.py`）：给同一批双物体图配描述（"a red square on the left and a yellow circle on the right"、"the red square is to the left of the yellow circle" 等 5 种），从 `stage2-cocoqa` 起用 `devision-align` 只训投影层 2 个 epoch（1,250 步）。补词准确率 0.35 → 0.915（错配图片 0.777，差 0.14）。但专门测方位词——"the X is to the [MASK] of the Y"（训练用过的句式）里 left/right 的打分——只有 0.473（`stage2-cocoqa` 0.500）。之后再训双物体决策题 2,500 步，仍为 0.48。

结论：补词准确率高来自颜色、形状和句式这些不需要绑定的词；"文字说的那个物体在哪边" 在完形填空里同样学不会。目前在这个结构上，投影层单训（阶段 1）、LoRA 或全量放开 ModernBERT（阶段 2）、4 倍样本，都没能学会按文字挑物体。
