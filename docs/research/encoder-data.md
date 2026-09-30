# 视觉 Jev 式决策模型：256px 编码器、自然图像数据与 laya-vision 调研

- 调研日期：2026-09-30
- 场景：单张自然图像，统一 resize 到 256×256（不切片、不用 NaFlex），问题只用英文，形如“图里有狗吗”→ `noul`、“人是站着还是坐着”→ `choice`。推理在服务器 CPU 上跑，训练用 1×A100，MVP 规模 1 万条样本。
- 来源以论文、HF 模型卡/数据卡、官方仓库为准，每条事实后面附链接。标“（估算）”的是我按 FLOPs 推算的，标“（本机实测）”的是我在本机跑的合成基准，两者都不是官方数字。
- 原先的 GUI 截图方向已经放弃，本文只保留和自然图像有关的内容。

---

## 0. 结论

1. **首选编码器：SigLIP2-B/16-256**（86M，Apache-2.0，256 token，ImageNet 零样本 79.1%）。追求精度、CPU 预算也够的话，换 **SigLIP2-So400m/16-256**（83.4%，但 FLOPs 约是 B 的 4.8 倍）。PE-Core-S/B 可以作为 Apache-2.0 的备选；EUPE 是非商用许可，DINOv3 必须标注 "Built with DINOv3"，C-RADIOv4 没有 B 级尺寸，这三者都只适合做对照。
2. **CPU 延迟（本机实测，Apple M4，fp32，随机权重，batch 1）**：ViT-S/16@256 约 16–21 ms，ViT-B/16@256 约 48–61 ms，So400m/16@256 约 184–231 ms，ViT-L/14@252 约 205–256 ms。在服务器 x86 CPU 上预计是同一个量级，但没有实测（见第 1.3 节）。
3. **不要把 SmolVLM/Idefics3 的 pixel-shuffle ×4 用在 256px 上**：256px 图像只剩 16 个视觉 token（4×4 网格）。laya-vision 在 Atari 上实测，把分辨率从 512 降到 256 后，帧准确率从 0.596 降到 0.554，中位游戏得分从 0.20 降到 0.03（[来源](https://github.com/r33drichards/laya-vision/blob/main/site-docs/reference/results/game-training.md)）。256px 输入应该用原生 256 训练的编码器，并且保留 256 个 token，最多做 ×2 合并到 64 个。
4. **laya-vision 可以直接复用**：它有完整的训练、数据转换、温度校准和评测代码，代码是 Apache-2.0，已经支持 `image_size=256`，训练数据里也有 VQAv2 yes/no、A-OKVQA 等自然图像集。它的已发布权重是 CC BY-NC-SA 4.0，不能商用。
5. **官方 Laya 的 RLCD 训练代码已经公开**：有 Kaggle 2×T4 notebook 和单卡脚本 `research/scripts/finetune_single_device.py`，Apache-2.0。

---

## 1. 256px 固定分辨率下的视觉编码器

### 1.1 对比表

token 数按 patch 网格算，不含 CLS/register。GFLOPs 由 `torch.utils.flop_counter` 在同构 ViT 上统计（1 次乘加记 2 FLOPs），只算视觉塔。

| 编码器 | 视觉塔参数 | 原生分辨率 | 256px 下 token 数 | GFLOPs @256（本机实测） | 零样本 / 线性探测 | VQA 相关结果 | 许可 |
|---|---|---|---|---|---|---|---|
| SigLIP2-B/16-256 | 86M（[论文](https://arxiv.org/html/2502.14786)） | 256 | 256 | 43.8 | IN-1k 零样本 79.1%（[论文 Table 1](https://arxiv.org/html/2502.14786)） | 论文没有报告 B 尺寸的 VLM 迁移结果 | Apache-2.0（[HF](https://huggingface.co/google/siglip2-base-patch16-256)） |
| SigLIP2-L/16-256（参照） | 303M（[论文](https://arxiv.org/html/2502.14786)） | 256 | 256 | ≈160（估算） | IN-1k 82.5%（[论文 Table 1](https://arxiv.org/html/2502.14786)） | PaliGemma 迁移：VQAv2 82.1、GQA 66.1、OKVQA 63.3、AOKVQA-MC 77.6、TallyQA-simple 82.0、CountBenchQA 82.2、VizWiz 76.0（[论文 Table 6](https://arxiv.org/html/2502.14786)） | Apache-2.0 |
| SigLIP2-So400m/16-256 | 400M（[论文](https://arxiv.org/html/2502.14786)） | 256 | 256 | 210.9 | IN-1k 83.4%（[论文 Table 1](https://arxiv.org/html/2502.14786)） | So400m/14@224（同样 256 token）：VQAv2 82.8、GQA 65.7、AOKVQA-MC 80.5、TallyQA-simple 82.2、CountBenchQA 84.7（[论文 Table 6](https://arxiv.org/html/2502.14786)） | Apache-2.0 |
| PE-Core-S16-384 | ViT-S 级（参数量未核实） | 384 | 256 | 11.0（按 ViT-S/16 统计） | IN-1k 72.7%（[repo](https://github.com/facebookresearch/perception_models)） | 无 | Apache-2.0（[HF](https://huggingface.co/facebook/PE-Core-S16-384)） |
| PE-Core-B16-224 | 0.09B（[HF](https://huggingface.co/facebook/PE-Core-B16-224)） | 224 | 256（需要插值位置编码） | 43.8 | IN-1k 78.4%（@224） | EUPE 论文中 PEcore-B 的数字：TextVQA 50.8、GQA 65.6（[EUPE 论文](https://arxiv.org/html/2603.22387)） | Apache-2.0 |
| PE-Core-L14-336 | 0.32B（[HF](https://huggingface.co/facebook/PE-Core-S16-384)） | 336 | 324（用 252px；256 不能被 14 整除） | 196.1 | IN-1k 83.5%（@336） | 无 | Apache-2.0 |
| EUPE-ViT-S / ViT-B | 21M / 86M（[repo](https://github.com/facebookresearch/EUPE)） | 224 起，可用 16 的倍数（[HF](https://huggingface.co/facebook/EUPE-ViT-B)） | 256 | 11.0 / 43.8 | ViT-B：卡片写 IN-1k 零样本 79.7%、k-NN 84.1%（[HF](https://huggingface.co/facebook/EUPE-ViT-B)） | ViT-B：GQA 67.3（SigLIP2-B 65.2）、RealWorldQA 55.5（SigLIP2-B 52.5）、TextVQA 50.4（SigLIP2-B 51.6）（[论文](https://arxiv.org/html/2603.22387)） | **FAIR Research License，非商用**（[HF](https://huggingface.co/facebook/EUPE-ViT-B)） |
| C-RADIOv4-SO400M / H | 412M / 631M（[技术报告](https://arxiv.org/html/2601.17237)） | 多分辨率训练，低分辨率档为 128–432（[报告](https://arxiv.org/html/2601.17237)） | 256（另有 summary/register token） | ≈211 / ≈320（估算） | 零样本 IN-1k 82.01 / 83.09（[HF](https://huggingface.co/nvidia/C-RADIOv4-H)） | 未报告 VQA 类结果 | NVIDIA Open Model License，可商用（[HF](https://huggingface.co/nvidia/C-RADIOv4-H)） |
| SmolVLM-256M 视觉塔（SigLIP-B/16-512） | 93M（[SmolVLM 论文](https://arxiv.org/html/2504.05299)） | 512 | 256 个 patch，经 pixel-shuffle ×4 后**只剩 16 个**（[laya-vision](https://github.com/r33drichards/laya-vision/blob/main/laya/preprocess.py)） | 43.8（@512 为 175.2） | — | 整个 SmolVLM-256M：OCRBench 52.6、TextVQA 50.2、ScienceQA 73.8（@512 切片）（[论文](https://arxiv.org/html/2504.05299)） | Apache-2.0 |
| DINOv3 ViT-S / S+ / B / L | 21M / 29M / 86M / 300M，patch 16，4 个 register（[repo](https://github.com/facebookresearch/dinov3)，[HF](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)） | 训练分辨率可变 | 256（另加 1 CLS + 4 reg） | 11.0 / — / 43.8 / ≈160（估算） | IN-ReaL 线性探测 87.0 / 88.0 / 89.3 / 90.2（[HF](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)） | 纯视觉预训练，没有和文本对齐 | DINOv3 License：可商用，但须标注 "Built with DINOv3"，并禁止军事等用途（[许可](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)） |

补充：

- SigLIP2 固定分辨率版的做法是在训练进度 95% 时，把序列长度 256 的 checkpoint 的位置编码 resize 到目标长度后继续训练（[论文 §2.4.1](https://arxiv.org/html/2502.14786)）。因此 `-256` 版本在 256 px 上是原生训练的，不需要插值。
- 在 256 token、同为 L 尺寸的设定下，SigLIP2 的 VQAv2、GQA、OKVQA、TallyQA 迁移结果都高于 SigLIP 1，只有 AOKVQA-MC 略低（78.3→77.6）（[论文 Table 6](https://arxiv.org/html/2502.14786)）。
- EUPE 在 GQA 和 RealWorldQA 上比同尺寸的 SigLIP2-B、PE-B 高 1.7–3 分，但许可是非商用（[论文](https://arxiv.org/html/2603.22387)，[HF](https://huggingface.co/facebook/EUPE-ViT-B)）。
- ModernVBERT 用的也是 SigLIP2-B/16-512（86M）加 pixel-shuffle ×4，每个 512 切片 64 token，后接双向的 ModernBERT（[论文](https://arxiv.org/html/2510.01149)）。论文报告在文档检索任务上，双向注意力比因果注意力高 10.6 nDCG@5（[论文](https://arxiv.org/html/2510.01149)），这支持 Laya 式的 `[MASK]` 读出方案。

### 1.2 CPU 延迟

**官方或第三方报告的数字**

- EUPE ViT-B 在 iPhone 15 Pro CPU 上：256×256 为 55.2 ms，512×512 为 305.2 ms（[EUPE 论文](https://arxiv.org/html/2603.22387)）。
- laya-vision 整个模型（SmolVLM-256M，fp32，4 vCPU 的 Modal 容器，96×96 图像，3 个问题）：完整路径 1316 ms，开启前缀缓存后 1182 ms（[architecture.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/architecture.md)）。官方 HF Space 在免费 CPU 上约 1–3 s 一张图（[space/app.py](https://github.com/r33drichards/laya-vision/blob/main/space/app.py)）。
- 量化要谨慎：laya-vision 的动态 int8 量化让概率最多偏移 0.62，9 个验证问题里有 4 个 top 答案发生翻转（[what-didnt-work](https://github.com/r33drichards/laya-vision/blob/main/site-docs/reference/results/what-didnt-work.md)）。

**本机实测**

- 条件：Apple M4（10 核），PyTorch 2.14 CPU，fp32，timm 同构 ViT，随机权重，batch 1，取 10 次的中位数。
- 这组数字只反映视觉塔本身，不含文本编码器和决策头。

| 结构（对应编码器） | 参数 | token | GFLOPs | 1 线程 | 4 线程 |
|---|---|---|---|---|---|
| ViT-S/16@256（PE-S、EUPE-S、DINOv3-S） | 21.7M | 256 | 11.0 | 21.1 ms | 15.9 ms |
| ViT-B/16@256（SigLIP2-B、PE-B、EUPE-B） | 85.8M | 256 | 43.8 | 61.3 ms | 47.7 ms |
| ViT-B/16@512（SmolVLM/ModernVBERT 默认） | 85.8M | 1024 | 175.2 | 289.2 ms | 189.1 ms |
| ViT-L/14@252（PE-L） | 303.2M | 324 | 196.1 | 255.8 ms | 205.1 ms |
| ViT-So400m/16@256（SigLIP2-So400m、C-RADIOv4-SO400M） | 412.6M | 256 | 210.9 | 230.9 ms | 184.4 ms |
| ViT-B/16@256，batch 8 | — | — | — | — | 38.9 ms/张 |

**换算到服务器 CPU（估算）**

- 延迟大致和 GFLOPs 成正比。按服务器 x86 CPU 上 fp32 PyTorch/ONNX 实际吞吐 0.3–1 TFLOPS 估算，ViT-B/16@256 约 45–150 ms，So400m 约 0.2–0.7 s。
- 有 AMX/bf16 的 Xeon 用 OpenVINO 或 IPEX 通常还能再快一些。以上都没有实测。
- 文本侧可以按同样的方法估：以 ModernBERT-150M 为例，约 200 token 时约 2×150M×200 ≈ 60 GFLOPs（估算），和 ViT-B 视觉塔的开销同一量级。
- 部署时要做**视觉特征缓存**：同一张图的多个问题只编码一次图像。laya-vision 的因果分支已经实现了这一点（[architecture.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/architecture.md)）。

### 1.3 选型建议

- **token 数**：SigLIP2-B/16-256 的 256 个 token 直接送进文本编码器（ModernBERT 类），或用 ×2 pixel-shuffle 压到 64 个。不要用 ×4 压到 16 个：laya-vision 在 256 px、16 token 下，需要精确定位小目标的 Atari 游戏几乎全部失效（Boxing 0.57→0.03，Qbert 0.31→0.03），而只需要判断“车道是否空”的 Freeway 基本没受影响（0.75→0.65）（[game-training.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/reference/results/game-training.md)）。“是否存在某物、姿态、计数”这类问题属于前者。
- **骨干**：直接改 laya-vision 的 ModernVBERT 分支最省事。它已经是 SigLIP2 + 双向 ModernBERT + `[MASK]` 读出，只是默认 512 px。注意 ModernVBERT 的视觉塔是 512 版，在 256 px 下需要位置编码插值，属于分布外输入。另一种做法是把视觉塔换成 `siglip2-base-patch16-256`，但连接层需要重新对齐。
- **预处理**：训练和推理必须走同一条预处理路径。仅仅换 resize 滤波器，就让 Atari 中位分从 0.201 降到 0.165（[game-training.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/reference/results/game-training.md)）。

---

## 2. 可以转成 noul / choice 的公开自然图像数据

### 2.1 训练候选

「规则可转」表示不用 LLM，只靠规则就能生成题目。

| 数据集 | 规模 | 许可 | 规则可转 | 映射方式 |
|---|---|---|---|---|
| **GQA**（balanced） | 11.3 万张图；2200 万题，其中 balanced 版 170 万（[论文](https://arxiv.org/abs/1902.09506)） | 官网没有写许可；图像来自 COCO 和 Flickr（[下载页](https://cs.stanford.edu/people/dorarad/gqa/download.html)） | 是，最适合规则转换 | 结构类型：`verify`/`logical` → noul；`choose`（如 “Is it red or blue?”）→ 二选一 choice，两个选项可直接从 `semantic` 程序中取出；`compare` → choice 或 noul。用 `isBalanced` 过滤以压低答案先验（[论文](https://arxiv.org/abs/1902.09506)，[下载页](https://cs.stanford.edu/people/dorarad/gqa/download.html)） |
| **VQAv2** | train 443,757 题，val 214,354 题，每题 10 个答案，约 20 万张 COCO 图（[下载页](https://visualqa.org/download.html)，[论文](https://arxiv.org/abs/1612.00837)） | 标注 CC BY 4.0；图像遵守 Flickr ToU（[terms](https://visualqa.org/terms.html)） | 是 | `answer_type=yes/no` → noul；`number` → 按计数分桶，出 choice 或 score；`other` → choice，正确项取 `multiple_choice_answer`，干扰项从同一 `question_type` 的高频答案中采样。VQA v1 的多选版本就是这样设计的，共 18 个候选（[v1 论文](https://arxiv.org/abs/1505.00468)）。v1 中 “Is there a clock” 有 98% 的答案是 yes，所以要成对采样 v2 的互补图像对来消除偏置（[v2 论文](https://arxiv.org/abs/1612.00837)）。注意 Cauldron 版本丢掉了 `answer_type`，要做规则转换得用官方 JSON |
| **Open Images V7** | 约 900 万张图；人工验证的图像级标签：train 5880 万条，其中正例 2110 万、负例 3760 万；另有 1600 万个框（600 类）和 train 317 万条关系标注（1466 种关系，含 “woman is jumping” 这类动作）（[facts](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)） | 标注 CC BY 4.0；图像标为 CC BY 2.0，官方提醒逐张核对（[facts](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)） | 是，商用最友好 | 已验证正例 → noul yes；已验证负例 → noul no。这些负例是人工确认的，不是随机采的，质量高于 POPE 式采样。机器生成的标签误报率高，不要用。关系和动作三元组可以出成 choice，例如 “站/坐/跳”；框数可以做计数 score |
| **Visual Genome** | 108,077 张图；170 万 QA；380 万个物体；280 万个属性（[官网](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)） | CC BY 4.0（[HF 卡](https://huggingface.co/datasets/ranjaykrishna/visual_genome)）；图像是 YFCC 与 COCO 的交集（[论文](https://arxiv.org/abs/1602.07332)） | 是 | “X 是否具有属性 Y” → noul，负例从同一属性类别里取；关系三元组 → noul；交换主语和宾语可以构造 hard negative 的 choice |
| **COCO 标签 + POPE 式构造** | 2017 版 train 11.8 万张、val 5 千张（[COCO](https://cocodataset.org/#download)） | 标注 CC BY 4.0；图像须遵守 Flickr ToU（[terms](https://cocodataset.org/#termsofuse)） | 是 | 用模板 “Is there a {} in the image?”，图中出现的类别出 yes 题，未出现的出 no 题。负例有三种采法：random（随机）、popular（全库高频类别）、adversarial（与图中物体高共现的类别），难度依次上升（[POPE 论文](https://arxiv.org/abs/2305.10355)，[repo](https://github.com/RUCAIBox/POPE)）。缺点是 COCO 的标注不穷尽，漏标会让 no 题带噪 |
| **TallyQA** | 287,907 道计数题，其中 train 249,318 题，覆盖 132,981 张图（[论文](https://arxiv.org/abs/1810.12440)） | repo 为 Apache-2.0（[repo](https://github.com/manoja328/TallyQA_dataset)）；图像来自 COCO 和 VG | 是 | 计数分成 0/1/2/3/4/5+ 几档，出 score 或 choice；complex 子集里有答案为 0 的题，可以当 hard negative |
| **VSR** | 10,972 条“描述-图像”对，标 True/False，涉及 66 种空间关系（[论文](https://arxiv.org/abs/2205.00363)） | repo 为 Apache-2.0，HF 卡写 CC BY 4.0（[repo](https://github.com/cambridgeltl/visual-spatial-reasoning)）；图像来自 COCO 2017 | 是 | 直接当 noul |
| **A-OKVQA** | 24,903 题，划分为 17.1K/1.1K/6.7K；自带 4 选 1 的 `choices`（[论文](https://arxiv.org/abs/2206.01718)，[repo](https://github.com/allenai/aokvqa)） | 代码 Apache-2.0；数据本身的许可未单独说明 | 已经是选择题 | 直接用作 choice。偏重常识推理，比“简单视觉问题”难 |
| **VizWiz-VQA** | train 20,523 对，val 4,319 对（[官网](https://vizwiz.org/tasks-and-datasets/vqa/)） | CC BY 4.0（[官网](https://vizwiz.org/tasks-and-datasets/vqa/)） | 是 | “这个问题能否根据图像回答” → noul；yes/no 类问题 → noul。图像由盲人拍摄，画质常常很差 |
| **The Cauldron** 相关子集 | vqav2、aokvqa、tallyqa、vsr、cocoqa、visual7w、okvqa 等（[HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)） | 沿用各子集自己的许可（[HF](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)） | 部分可以 | laya-vision 的 `cauldron.py` 可以直接复用（见第 3 节）。但 Cauldron 去掉了 `answer_type`、GQA 类型等元数据 |
| **V-COCO / HICO-DET**（人物动作） | V-COCO：1 万张 COCO 图中的 1.6 万人，26 种动作（[论文](https://arxiv.org/abs/1505.04474)）；HICO-DET：600 类人-物交互（[论文](https://arxiv.org/abs/1702.05448)） | V-COCO 的 repo 为 MIT；HICO-DET 未核实 | 是 | 图中只有一个人时，可以对动作或姿态出 choice；多人时需要裁剪或画框指定是哪个人 |
| OK-VQA | 14,055 题（[论文](https://arxiv.org/abs/1906.00067)） | 未找到许可 | 否 | 开放式问答，依赖外部知识，不建议用 |
| Objects365 | 365 类，200 万张图（[官网](https://www.objects365.org/overview.html)） | **仅限学术用途**，图像不得再分发（[下载页](https://www.objects365.org/download.html)） | 是 | 能生成存在性 noul，但许可不适合商用 |
| CLEVR | train 7 万张图 / 70 万题（[官网](https://cs.stanford.edu/people/jcjohns/clevr/)） | CC BY 4.0 | 是 | 合成场景，不属于自然图像，不建议用 |

### 2.2 只用于评测（不要拿来训练）

- POPE（random / popular / adversarial 三个子集，500 张 COCO val2014 图，每张 3 道 yes 题和 3 道 no 题）（[论文](https://arxiv.org/abs/2305.10355)）。
- NaturalBench 英文部分：1 万条人工验证样本，同一个问题配两张答案相反的图，能抵抗答案先验，卡片写 Apache-2.0（[HF](https://huggingface.co/datasets/BaiqiL/NaturalBench)，[论文](https://arxiv.org/abs/2410.14669)）。
- MME：每张图配两道题，一道答 yes、一道答 no（[论文](https://arxiv.org/abs/2306.13394)）。
- BLINK：3,807 道选择题，含计数、相对深度、空间关系等任务，Apache-2.0（[HF](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)）。
- SugarCrepe：二选一描述匹配，MIT（[repo](https://github.com/RAIVNLab/sugar-crepe)）。
- CountBench：540 张图（[论文](https://arxiv.org/abs/2302.12066)）。
- Winoground：图像来自 Getty，仅限研究（[论文](https://arxiv.org/abs/2204.03162)）。

### 2.3 泄漏：按 image id 去重

- COCO 2014 的图：VQAv2、OK-VQA、POPE（val2014）。
- COCO 2017 的图：A-OKVQA、VSR、SugarCrepe、LVIS。COCO 2017 的 train 包含了大部分 val2014，所以 POPE 的评测图可能出现在基于 COCO 2017 的训练集里（[COCO](https://cocodataset.org/#download)）。
- VG 的图是 COCO 与 YFCC 的交集（[论文](https://arxiv.org/abs/1602.07332)）；GQA 用的是 VG 的图；TallyQA 的 train 混合了 COCO 和 VG 的图（[论文](https://arxiv.org/abs/1810.12440)）。
- 因此所有评测图（POPE、SugarCrepe、A-OKVQA val、VSR test）都要按 image id，从每一个训练源里剔除。

### 2.4 1 万条 MVP 的配比建议（全部规则生成，不用 LLM）

1. **GQA balanced，约 3.5k**：`verify`/`logical` → noul，`choose` → 二选一 choice。许可不明，暂时按研究用途对待。
2. **VQAv2，约 3k**：yes/no 题取互补图像对，保证 yes 和 no 各占一半；`other` 题配同类干扰项出 choice；`number` 题出 score。
3. **Open Images V7 已验证标签，约 2k**：存在性 noul，负例从已验证负例中按“高频”和“共现”两种方式挑；关系和动作三元组出 choice。
4. **TallyQA，约 1k**：计数 score。
5. **VSR，约 0.5k**：空间关系 noul。

- 每个 noul 子集都要把 yes/no 比例控制在 50/50，并且按问题模板分层抽样，避免模型只学到答案先验。
- **商用注意**：COCO、VG、GQA、VQAv2、TallyQA、VSR 的图像最终都要遵守 Flickr ToU 和每张图各自的许可。Objects365 和 Winoground 不能商用。要做可商用的版本，应以 Open Images 为主，并逐张核对图像许可（[Open Images](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)）。

---

## 3. laya-vision 仓库与官方 Laya 训练代码

### 3.1 laya-vision（r33drichards/laya-vision，commit `d4075b0`，2026-09-25）

- **定位**：它是 Laya 的独立 fork，把 ModernBERT 文本编码器换成小型 VLM；`predict(state, questions)` 接口、RLCD 目标和温度校准保持不变。作者声明与 Convai 无关（[README](https://github.com/r33drichards/laya-vision)）。
- **三种骨干**（[architecture.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/architecture.md)）：
  - SmolVLM-256M：因果注意力，在选项末尾的 `\n` 处读出，可以选 block 注意力让选项之间互相可见。
  - ModernVBERT-250M：双向注意力，在 `[MASK]` 处读出，没有选项顺序偏差。
  - 决策头：2 层双向 TransformerEncoder，接一个 scorer 给每个选项出 logit，另有 act_head 做 decide/escalate 判断。
- **训练代码**：`laya/vlm_train.py`（885 行）。
  - 损失函数 `vlm_loss` 由三部分组成：带噪 logit 的 proper scoring rule 策略梯度（组大小 4，σ=0.3，奖励为 log score 加 0.75×spherical，score 题再减 RPS）、`w_ce` 权重的 soft CE，以及 act head 的期望效用项。
  - 训练结束后用 LBFGS 拟合每种题型的温度（[vlm_train.py](https://github.com/r33drichards/laya-vision/blob/main/laya/vlm_train.py)）。
  - 冻结策略可选只训 head、训最后 N 层或全部，视觉塔默认冻结（[architecture.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/architecture.md)）。
  - 训练和评测任务通过 Modal 调度（`modal_app.py`）。
- **数据格式**：每行一个 JSONL 记录，形如 `{"id","image","state_text","question":{type,instructions,criteria},"label"}`，可选 `target`（soft 分布）（[vlm_train.py `jsonl_example`](https://github.com/r33drichards/laya-vision/blob/main/laya/vlm_train.py)）。
- **数据转换管线**：
  - `laya/cauldron.py` 从 The Cauldron 中选 19 个闭式答案子集，按规则转换：带字母选项和 Options 列表的题转成 `choice`，“Answer yes or no” 的题转成 `noul`。数字、描述和自由文本答案一律跳过，共得到约 27 万个问题（[cauldron.py](https://github.com/r33drichards/laya-vision/blob/main/laya/cauldron.py)，[data.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/data.md)）。
  - 另外还有评分数据（`rubric.py`：VLFeedback、AVA、RichHF-18K、CrisisMMD）和评测集（`evalsets.py`：POPE、VizWiz、CIFAR-10H、FER+、KonIQ、EvalMuse）（[data.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/data.md)）。
  - 最早的三个数据集（A-OKVQA、ScienceQA、VQAv2 yes/no）的准备代码在 `siglip-projector-experiment` 分支上（[data.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/data.md)）。
  - 数据不含 GUI 截图。
- **分辨率处理**：
  - 默认沿用 Idefics3 的做法：每张图一个 512 px 切片，patch 16，pixel-shuffle ×4，每张图 64 个 token（[architecture.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/architecture.md)）。
  - `ImagePrep(image_size=256)` 可以直接切到 256 px，这时每张图 16 个 token。它还把 HF processor 的两跳 LANCZOS 合成为两次矩阵乘，CPU 预处理从 33.6 ms/帧降到 0.14 ms/帧（[preprocess.py](https://github.com/r33drichards/laya-vision/blob/main/laya/preprocess.py)，[game-training.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/reference/results/game-training.md)）。
  - 切片实验：1024 px 切片带来约 +1.1 分和 +35% 延迟；2048 px 没有增益，延迟变为 2.5 倍，所以默认不切片（[split-bench.md](https://github.com/r33drichards/laya-vision/blob/main/site-docs/reference/results/split-bench.md)）。
- **已有结果**：推荐的 201M checkpoint 在 A-OKVQA 上 59.8%，ScienceQA 82.4%，VQAv2 yes/no 71.4%；34 个验证集、共 59,427 题的总体准确率 69.1%，ECE 0.041（[README](https://github.com/r33drichards/laya-vision)）。
- **许可**：代码 Apache-2.0；权重 CC BY-NC-SA 4.0，因为训练数据包含 ScienceQA 和 CrisisMMD（[README](https://github.com/r33drichards/laya-vision)）。商用就必须自己训练，并避开非商用数据。
- **可复用性评价**：可以直接复用的有损失函数、决策头、序列构造、温度校准、JSONL 格式、Cauldron→noul/choice 转换器和评测脚本。需要改的有两处：
  - 训练入口绑定 Modal，本地单卡要自己写启动脚本。
  - 在 256 px 下每张图只有 16 个 token，需要调整 pixel-shuffle 比例，或者换用原生 256 的视觉塔。

### 3.2 官方 Laya（NandhaKishorM/laya，Convai Innovations）

- **RLCD 训练代码已公开**，许可 Apache-2.0：
  - Kaggle 2×T4 notebook `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` 覆盖完整流程：构造数据、RLCD 训练、温度校准、评测、推送到 Hub（[finetune.md](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md)）。
  - 单设备脚本 `research/scripts/finetune_single_device.py`，CPU 和 GPU 都能跑（[脚本](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/finetune_single_device.py)）。
- **RLCD 配方**（[finetune.md](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md)）：
  - 目标是 gold 概率分布（soft target）。
  - 策略梯度部分是 GRPO 风格：每条样本采 4 组带噪 logit，噪声 σ 从 0.4 退火到 0.1，奖励为 spherical 0.75 加 RPS 1.0。
  - 同时加上全权重的 soft CE。
  - 超参：4 个 epoch，有效 batch 64，编码器学习率 2.5e-5，head 学习率 1e-4，`max_len` 1024，`head_max_len` 256。
- 官方 Laya 只处理文本，没有视觉分支（[README](https://github.com/NandhaKishorM/laya)）。

---

## 未核实

- PE-Core-S16-384 和 T16-384 的准确参数量：HF 卡片只列了 B/L/G，S/T 的 ImageNet 数字来自 repo 表格。表中的 GFLOPs 按标准 ViT-S/16 结构统计。
- EUPE 卡片上“ImageNet 零样本 79.7%”具体用什么文本塔测得：EUPE 本身不是 CLIP 式模型，这个数字的测法没有核实。
- C-RADIOv4 在 256 px 下的质量：官方只给出 1024 px 左右的零样本数字，低分辨率下表现未核实。此外 C-RADIOv4 没有 B 级尺寸；C-RADIOv3-B 的许可未核实。
- 所有服务器 CPU 延迟都是按 FLOPs 估算的，没有找到 SigLIP2、PE、EUPE 在 x86 服务器、ONNX 或 OpenVINO 下的官方基准。本机数字来自 Apple M4 和随机权重的同构 ViT，注意力实现和真实 checkpoint 有差异（如 SigLIP 的 MAP head、DINOv3 的 RoPE 和 register）。
- SigLIP2-B/16 在 VQA 迁移上的表现：论文只给了 L 和 So400m 的 PaliGemma 结果，B 尺寸没有给。
- laya-vision 的 ModernVBERT 分支在 256 px（位置编码插值）下的精度没有实验数据。
- 数据许可：GQA 官网没有许可声明；OK-VQA、A-OKVQA（数据本身，不是代码）、HICO-DET、MME、LVIS（GitHub 显示 NOASSERTION）、KonIQ-10k 都没找到明确的数据许可。
- COCO 图像中 NC 类 Flickr 许可占多大比例未核实（标注包太大，没有下载统计）。
- VG 元数据里是否带 `coco_id` 字段、能不能直接用来和 COCO 去重，未核实。
- GQA 的属性是否覆盖 standing / sitting 这类姿态，未核实。
- NaturalBench 中纯英文样本的具体数量未核实。
- 第 2 节的数据集规模和许可由子代理查阅一手来源得到。我只复核了 Open Images 的数字，其余各项虽然附了来源链接，但没有逐项核对原文。
