# EmbeddingGemma 2 能否替代 deVision 的 SigLIP2

记录时间：2026-10-09 12:30:36 +08:00。观察日期：2026-10-09。

范围：核对 Google 官方发布材料、Google 的 Hugging Face 模型卡与小型配置文件，以及 Transformers 官方实现；不下载权重、不加载模型、不运行评测或训练。下文的张量行为来自源码静态阅读，分辨率表来自源码公式的数学推导，均不冒充运行结果。deVision 的适配判断另列于后续补充部分。

核查版本：`google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`；Transformers commit `4cc2aa84301c9aa210b5513dab9fefae03981f6e`。远端 `main` 会变化，复现应固定这两个版本。

## 当前判断

**EmbeddingGemma 2 是值得评估的新图文编码主干，但现有官方证据不足以证明其单独视觉塔能够改善 deVision 的区域绑定。** 740M 是文本、视觉、音频三部分的合计；保留完整图文路径约为 440M，只取视觉部分约为 170M，两种用法的能力与接口不能混为一谈。官方图像检索成绩对应经过共享文本主干的完整 embedding 路径，不是独立视觉塔的成绩。[发布公告][launch]、[开发指南][guide]

## 1. 模型身份与架构

Google 于 **2026-10-06** 发布 `google/embeddinggemma-2`，基于 Gemma 4，Apache 2.0。官方近似参数拆分为文本 270M（主干 130M、embedder 140M）、视觉 170M、音频 300M。它区别于此前的纯文本 [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m)，也区别于云端 [Gemini Embedding API](https://ai.google.dev/gemini-api/docs/embeddings)；不能从后两者借用训练配方或效果结论。[发布公告][launch]、[模型卡][card]

| 组件 | 固定 checkpoint 配置中的事实 |
|---|---|
| 视觉塔 | `model_type=gemma4_vision`，映射到 `Gemma4VisionModel`；16 层、宽度 768、FFN 3072、12 个 attention heads；patch 为 16×16 |
| 视觉位置与池化 | learned x/y 位置嵌入加 axial 2D RoPE；`pooling_kernel_size=3`；`standardize=false`；`use_clipped_linears=false` |
| 图文连接 | 视觉输出先作无可学习缩放的 RMS normalization，再线性投影 768→512 |
| 共享文本主干 | `embedding_gemma2_text`；24 层、宽度 512、FFN 2048、词表 262,144；5 个局部层后接 1 个全局层；双向注意力，无生成式 KV cache |
| 文本输出 | 每个 token 的最终表示经线性层 512→768；Sentence Transformers 再均值池化、L2 normalize |

架构数值来自 [checkpoint config][config]；执行路径来自 [EmbeddingGemma2Model][embedding-code] 与 [Gemma4VisionModel][vision-code]。这证明它采用 Gemma 4 视觉实现；**未查到证据证明这 170M 权重直接等同于某个 SigLIP2 checkpoint，或与某个 Gemma 4 生成模型的视觉权重完全相同**。架构继承不等于权重身份相同。

文本 `sliding_window=512` 与模型卡“1024”不宜直接判为矛盾：实现明确将 512 定义为双向窗口半径，允许 `abs(q-k)<=512`。配置中的 `max_position_embeddings=262144` 也不能替代官方支持的 **8,192-token 输入上下文**。[文本配置说明][text-config]、[模型卡][card]

## 2. 视觉 token 预算并非 raw patch 数

[processor_config.json][processor-config] 默认图像 `max_soft_tokens=280`、patch16、pooling3；图像预处理执行按长宽比缩放，目标高宽向下取整至 **48 像素的倍数**，然后 patchify 并补齐。张量 `pixel_values` 是 `[batch,max_patches,768]`，最后的 768 是 `16×16×3` 个像素值；`image_position_ids` 是 `[batch,max_patches,2]`，padding 坐标为 `(-1,-1)`。它不是普通 `[B,3,H,W]` 图像张量。最大 raw patch 数为 `9×max_soft_tokens`。[图像预处理实现 L33–84、L219–274][image-code]

源码支持的预算为 70、140、280、560、1120。以下仅是**正方形输入且开启默认 resize 时的静态推导**，有效 token 数可能少于预算；长方形的实际网格另由长宽比决定。

| soft-token 上限 | 补齐后的 raw patch 数 | 正方形缩放分辨率 | 有效 raw 网格及数量 | 3×3 池化后的有效网格及数量 |
|---:|---:|---:|---:|---:|
| 70 | 630 | 384×384 | 24×24 = 576 | 8×8 = 64 |
| 140 | 1,260 | 528×528 | 33×33 = 1,089 | 11×11 = 121 |
| 280（默认） | 2,520 | 768×768 | 48×48 = 2,304 | 16×16 = 256 |
| 560 | 5,040 | 1,104×1,104 | 69×69 = 4,761 | 23×23 = 529 |
| 1120 | 10,080 | 1,584×1,584 | 99×99 = 9,801 | 33×33 = 1,089 |

推导公式为 `side=48×floor(sqrt(9×budget×16²)/48)`，随后分别除以 16 和 48 得到 raw/soft 网格边长。**这不是实测分辨率扫描或延迟数据。** 注意视觉 Transformer 先处理 raw patch，之后才池化，不能把“默认 280 个 soft token”理解为只计算 280 个 patch；更不能仅根据参数量比例估计 CPU 时延。[图像预处理实现][image-code]、[视觉 forward L2014–2034][vision-code]

## 3. 可提取哪些中间表示

| 提取位置 | 静态源码显示的语义 | 使用限制 |
|---|---|---|
| `vision_tower.encoder` 的层输出 | 带空间坐标的 raw-patch 序列，宽度 768；含 padding | 需要正确传入 patch embedder、坐标和 mask；不是完整检索 embedding |
| `vision_tower(...).last_hidden_state` | raw 表示按坐标 3×3 平均、乘 `sqrt(768)`，去除 padding 后平铺，形状为 `[全部图片的有效soft-token总数,768]` | 不是 `[B,N,768]`；需凭每图有效数拆分，长方形不能强行假设正方网格 |
| `get_image_features(...).pooler_output` | 上述 soft tokens 经 `embed_vision` 投影为 512 维，按图片拆成 tuple | 这里名字带 pooler，但不是每图一个向量；仍未进入共享文本主干 |
| 复合模型 `last_hidden_state` | 文本与媒体占位序列共同经过文本主干后，每 token 768 维 | 可作为下游适配起点；官方默认会继续均值池化 |
| Sentence Transformers `encode()` | 每个输入一个归一化 768 维向量 | 整图、整段或组合输入级表示，不是区域列表 |

以上依据 [视觉实现 L632–696、L1977–2042][vision-code]、[图文实现 L565–591、L664–692、L781–852][embedding-code] 和 checkpoint 的 [modules.json][modules]、[Pooling 配置][pooling]。均值池化配置为 `include_prompt=true`。

`output_hidden_states` 提供中间层记录机制，但**不能未经核实就把 `hidden_states[-1]` 当成最后一层 raw patch**：固定版本的 `capture_outputs` 默认将最后一项替换为 `last_hidden_state`，而本视觉模型的后者已经池化并去 padding。取最后一层池化前特征需要明确截取 encoder 输出或验证专用记录方式。本次没有运行这些访问路径。[output_capturing.py L234–307][capture-code]

只取 `vision_tower` 在模块结构上可行；官方选择性加载示例则是去掉音频而保留文本+视觉（440M）。只保留 170M 的独立加载、state-dict 提取及与现有项目的运行兼容性本轮未验证。**独立塔的 768 维特征和完整模型的 768 维检索向量维数相同，但所在空间和语义不同，不能直接与完整模型的文本向量做有保证的跨模态相似度比较。**[开发指南][guide]、[图文实现][embedding-code]

## 4. 训练目标与能力证据的边界

能确认的是：官方将其训练为多模态 embedding 模型，使用任务前缀和 Matryoshka Representation Learning，支持 768/512/256/128 维；训练数据介绍包含多种模态及跨模态配对，时间截止为 2025 年 1 月。**已查阅材料未给出第二代完整预训练 loss、各 loss 权重、教师模型身份、训练阶段冻结策略或区域级监督配方。** 不应把第一代论文中的 NCE/蒸馏细节直接写成第二代事实。[模型卡][card]

Google 的多模态微调教程展示 `MultipleNegativesRankingLoss`，以 text anchor、positive image、negative image 做对比学习。这说明官方提供了检索式图文微调路径，**并不证明预训练时使用完全相同的目标，也不等于带框区域绑定训练**。[多模态微调教程](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-multimodal-embeddinggemma-with-sentence-transformers)

官方全精度、768 维公开指标包括 MIEB lite 64.64（Mean(TaskType)）、MMEB v2 Image 57.28（Hit@1）和 VisDoc 67.84（NDCG@5）。它们支持完整模型有图像与视觉文档检索能力。[官方评测表][card]

本轮未找到独立视觉塔的同条件 SigLIP2 对照，也未找到 EmbeddingGemma 2 对左右镜像题、同图多对象文字指代、对象交换、颜色绑定反事实或 OCR 字符识别的分项验收结果。VisDoc 检索不能直接换算为 OCR 字符准确率；MIEB/MMEB 总分不能证明按文字选区域。官方所述提高视觉 token 预算可能改善细节，是值得实验验证的方向，尚无本项目收益证据。参数更大或发布日期更新也不是这些能力的证明。

作为基线，Google 的 [SigLIP2 官方模型卡](https://huggingface.co/google/siglip2-base-patch16-256) 已明确列出 decoder loss、global-local 与 masked prediction 等训练目标，并将定位和稠密特征列为训练目的。因此，“SigLIP2 只学整图语义、新模型才学习局部信息”不符合现有官方材料。这里仅比较训练目的，不能反向断言当前 SigLIP2 已足够解决 deVision 的全部真实图问题。

## 5. 软件与精度支持

官方开发指南要求 Sentence Transformers **6.1.0 或以上**；checkpoint 保存时记录 Transformers **5.18.0.dev0**。这只是保存版本，不应冒充最小稳定依赖版本。本轮固定的 Transformers 源码已包含所需模型与 processor；Google 发布页还列出 MLX、LiteRT、llama.cpp 等生态支持，但没有在本地逐一验证其多模态、中间层提取、训练和 CPU 路径。[开发指南][guide]、[保存环境配置][sentence-config]、[发布公告][launch]

模型卡要求 bf16 或 fp32，明确不支持 fp16 激活范围；做 CPU 试验应先采用其建议的 fp32 路径。Pixel 11 Pro 的量化内存声明只适用于对应设备、量化形式和运行库，不能作为本项目 Mac CPU 的时延或内存实测。[模型卡][card]、[AI Edge 发布说明](https://developers.googleblog.com/en/google-ai-edge-with-embeddinggemma-2/)

## 6. 尚缺的决定性证据

1. 独立 170M 视觉塔对本项目真实图区域绑定、空间关系、小字或小物体的受控收益；完整模型检索分数不替代这一项。
2. 70/280 预算在目标 CPU 上的端到端时延、峰值内存及图文适配成本；本轮只有公式分析。
3. 视觉权重初始化来源及第二代完整训练配方；公开结构不证明单独塔学到了与整机相同的能力。
4. 中间层提取与项目连接层的实际 tensor/mask/数值尺度兼容性；需由获准执行者在隔离环境做最小验证。

## 来源索引

[launch]: https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/
[guide]: https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/
[card]: https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
[config]: https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/config.json
[processor-config]: https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/processor_config.json
[modules]: https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/modules.json
[pooling]: https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/1_Pooling/config.json
[sentence-config]: https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/config_sentence_transformers.json
[embedding-code]: https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py
[text-config]: https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/models/embedding_gemma2/configuration_embedding_gemma2.py#L39-L47
[vision-code]: https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/models/gemma4/modeling_gemma4.py
[image-code]: https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/models/gemma4/image_processing_gemma4.py
[capture-code]: https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/utils/output_capturing.py#L234-L307

## deVision 的适配判断（主代理补充）

本节观察时间：2026-10-09（Asia/Shanghai）。主仓库代码 `50df607`；R13 worktree 此次读到 `144c127`。对象是当前发布的 v0.2 / v12-ext；R13 的 S1 / S2 只作为适配边界，不声称本次核实其训练进度。

### 我们实际在用什么

当前不是拿 SigLIP2 的整图检索向量回答问题。[model.py](../../src/devision/model.py) 的 `VisionDecisionModel` 关闭视觉 pooling head，取 256 个、每个 768 维的 patch 特征，再通过 2×2 拼接和 MLP，得到 64 个、每个 1024 维的视觉 token。ModernBERT 与 Laya 决策头再结合问题和候选项输出分数。因此，EmbeddingGemma 2 的最终 768 维整图 embedding 与当前接口用途不同；相同的向量宽度不等于相同的特征语义。

只读取已发布 safetensors 的 48,256 字节张量头、按 shape 求积，本次没有加载模型：视觉模块保存 92,930,304 个参数，其中不参与当前前向的 `vision.head` 为 7,087,104；实际视觉塔为 85,843,200。投影层 4,202,496，决策部分 421,293,830；完整发布权重 518,426,630。这些是本地 checkpoint 的参数统计，不是耗时测量。

SigLIP2 本来就是图文编码器；新模型的价值不能概括为“以前只懂文字，现在终于能看图”。SigLIP2 的原始训练还专门包含改善局部特征的任务，官方明确支持把视觉塔用于 VLM，见 [SigLIP2 论文](https://arxiv.org/abs/2502.14786) 和 [当前型号模型卡](https://huggingface.co/google/siglip2-base-patch16-256)。

### 可选路线与代价

| 路线 | 技术可行性 | 对本项目的意义 |
|---|---|---|
| 最终单个 768 维 embedding → 现有决策模型 | 需要重新设计接口并训练，不能直接塞进当前 projector | 将局部视觉序列先压成全局向量；可能用于粗分类，但没有证据能改善点名物体、空间关系和计数，不作为优先替换方案。不能据此断言全局向量完全不含关系信息。 |
| 仅约 170M 的 Gemma4 视觉塔 → 适配层 → 现有 ModernBERT / Laya | 可作为新视觉塔候选；需要修改实现及重训对齐 | 最接近“替换 SigLIP2”。完整模型粗估约 0.59–0.60B 参数，取决于连接层；不会自动获得原模型图文融合的全部能力。 |
| 保留约 440M 的图文模型，再接现有 ModernBERT / Laya | 可研究中间 token，但会串联两个文本模型 | 粗估约 0.86–0.87B 参数；若输入含问题，融合特征逐题变化，不能复用整段图文前向。原始视觉塔仍可按图缓存。优先级低。 |
| 约 440M 图文主干 + 新决策头，整体替换当前主干 | 独立架构实验，已不是只换 SigLIP | 能利用已经联合训练的图文融合，这是更有研究价值的另一个方向；需要重做题目编码、选项评分、训练和校准，不能沿用 Laya 权重后声称功能等价。参数数目较少不保证 CPU 更快。 |

以上参数是根据本地统计和官方近似模块规模的粗加总；未加载新模型逐参数计数，也未测内存、吞吐或延迟。

### 若只换视觉塔，需要改到哪些地方

- **预处理和张量接口**：[image.py](../../src/devision/image.py) 目前固定 letterbox 到 256，再返回 `[B,3,H,W]`。新塔应按官方 processor 提供 patch 与 `image_position_ids`，正确处理 padding、不同图片长度以及恢复 batch；不能只换 checkpoint 名。
- **连接层与网格**：[model.py](../../src/devision/model.py) 的 `Projector.forward` 用平方根恢复正方形网格。若走正方形、70 soft-token 预算路线，官方算法给 384 像素、576 个有效 raw patch、64 个 soft token，应取消当前额外的 2×2 合并并适配 768→1024。若走正方形、280 预算，256 个有效 soft token 在形状上仍可经 2×2 到 64，但视觉塔处理的是 2,304 个有效 raw patch，成本不同。这些算例不是本机执行结果。
- **训练入口不能漏改**：[train/align.py](../../src/devision/train/align.py) 与 [train/rlcd.py](../../src/devision/train/rlcd.py) 还直接调用 `model.vision(...).last_hidden_state`，只改推理的 `encode_image` 不够。
- **保存加载**：[infer.py](../../src/devision/infer.py) 当前按 `SiglipVisionConfig` 重建视觉塔，需保存编码器类型和 processor 配置，确保离线加载与训练预处理一致。
- **R13 的空间监督**：S1 的 `box_grid` 假定 letterbox 后的 8×8 网格；S2 读取未压缩的 SigLIP patch。采用保长宽比的非方形新输入时，必须重新定义每个 token 对应的图像区域和 mask；不能把数量相近的 token 当作同一个网格。
- **重新对齐与训练**：即使保留 768 维且外观 shape 一致，新特征分布也不同。原投影层不能当作已经适配的新模型；先用图文数据训练适配层，再验证决策训练与旧能力保持。旧温度不应直接作为新模型已校准的证明。

可以保留英文单图、`noul` / `choice` 及 Jev 输出接口；需要改变的是内部表示和加载训练路径。这些都是建议，尚未实施。

### 为什么现在不建议直接切换主线

本次直接读取 [已有诊断结果](../../runs/diag/binding_v12.json)：v12 端到端颜色绑定 0.71、计数 0.49；从冻结的 64 个投影视觉 token 训练小读出器，记录的颜色成绩 1.00、数量 0.89。这支持“至少在该合成任务上有可读取的信息，下游未必学会使用”，不证明真实照片上视觉塔已经足够，也不保证新视觉塔无收益。探针结果是已有产物，本次未复跑。

[既往结构筛选 D](../experiments/2026-09-30-rlcd-plateau.md) 中，512 输入、4×4 合并到 64 token 的迁移，在其短对齐预算下没有改善主要示意图 / 计数目标，CPU 单题 P50 从 145ms 变成 302ms。记录明确承认新投影层对齐预算不公平；它不能证明更高分辨率或不同视觉塔无效。它只提醒这次不能以“更多参数、更高分辨率、检索分数更好”代替本任务证据。

### 最小验证顺序（仅建议，未执行）

1. **零训练能力筛查，低成本**：执行者单独加载约 440M 图文路径、关闭音频。用已冻结的 dev 样本（约 200–500 个场景族 / 图片，覆盖真实照片与合成绑定）比较图像和完整候选描述，例如“红圆、绿方”与“绿圆、红方”，不要把裸词 yes / no 的相似度当作判断。加入真实图 / 错配图、反事实改色 / 改数量、选项换序与不读图对照，报告数量及按图或场景族的配对区间。相似度不是校准概率。此步骤只筛查完整嵌入模型，失败也不能单独否定其视觉塔经训练后的潜力。
2. **冻结视觉塔的小读出对照，中低成本**：若重点仍是换塔，比较旧 SigLIP2 与新塔、相同问题条件读出能力及训练数据划分，同时明确 token 数 / 分辨率 / 预处理差异。先测 70 预算，再决定是否值得增加预算；若增加预算，单独记录收益与代价。此时仍不能当作端到端迁移已成功。
3. **只有出现与绑定相关的收益，再做独立短训练**：用新适配层做对齐，接回决策训练；把初始化和对齐预算差异写清楚。判断依据是实际 dev 绑定和真实照片表现、旧任务退化以及同机 CPU 单题 / 同图多题 P50/P95，不用检索榜单放行。临近优势需追加种子确认，不在一次微小差值上选型。探索使用 dev，封存 test 留到候选确定后一次评估。

资源成本只能分档，不能在未测新塔的情况下承诺几分钟或固定倍速；量化手机内存也不能外推本机 FP32 训练 / CPU 推理。建议将其列为下一条独立研究分支，不中途改动 R13 的已固定比较。

**当前建议：可以作为新视觉塔候选，但不直接替换现有发布模型。最值得先确认的是其完整图文表示在我们的绑定题上是否有优势，再决定取视觉塔迁移还是研究全新的图文决策主干。**

本次只核对官方网页 / 配置 / 源码、本地代码和已有 JSON 结果，并统计权重文件头；没有下载新权重、运行模型、训练、评测、pytest 或修改代码 / 数据 / 依赖，也未写 critic。
