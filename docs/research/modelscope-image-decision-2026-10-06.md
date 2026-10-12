# ModelScope 图片决策模型检索

检索日期：2026-10-06（Asia/Shanghai）。任务是寻找接近 deVision 的“图片 + 问题 + 自定义候选项 → 概率”模型；本文是外部资料调研，不是代码、训练或数据验收。未下载权重、运行模型或修改训练状态。

## 直接决策模型的检索结果

本次通过 ModelScope 域名限定检索 `laya-vision`、`ModernVBERT`、`thaitea`、`laya-v`、`henilchopada`、`GLiNER2.5`、`calibrated vision` 等关键词，核对实际模型卡。**确认了数个文本决策模型，但尚未确认下表三个图片决策 checkpoint 的 ModelScope 权重仓库。**“未确认”仅是本次检索结果，不能推断平台不存在其他镜像。下面把已确认的上游 checkpoint 与已确认的 ModelScope 模型分开记录。

| 模型与来源 | 架构与能力 | 原生候选概率 | 与本项目的关系和限制 |
| --- | --- | --- | --- |
| [thaitea/laya-vision-modernvbert-250m（上游 HF）](https://huggingface.co/thaitea/laya-vision-modernvbert-250m) | 约 250M；双向 ModernBERT-150M + SigLIP2 + pixel shuffle；图片和问题一次前向，支持自定义选项 | 是；`choice` / `noul`，模型卡说明有温度校准 | 结构与接口接近，适合直接对照；`score` 没有训练，不能按有效能力使用；512 像素、64 个图像 token；权重 CC BY-NC-SA 4.0。本次未确认 ModelScope 镜像 |
| [thaitea/laya-vision（上游 HF）](https://huggingface.co/thaitea/laya-vision) | 当前模型卡为 201M，来自 SmolVLM-256M，语言层从 30 层裁至 20 层；一次前向图片决策 | 是；`choice` / `noul` / `score` | 当前别名已经不是旧版本，比较时必须固定 revision；与上述 ModernVBERT checkpoint 不是同一架构。每个状态一张图，512 像素；权重 CC BY-NC-SA 4.0。本次未确认 ModelScope 镜像 |
| [henilchopada/laya-v（上游 HF）](https://huggingface.co/henilchopada/laya-v) | 约 0.75B；SigLIP2 So400m NaFlex + mmBERT-base；图片和多问题、可选文本 | 是；`choice` / `noul` / `score` | 较大的多语言候选；模型卡支持不同视觉 patch 预算，经池化为 128 个图像 token，再由两层 MLP 投影到文本宽度。模型卡标注 Apache-2.0。发布者给的是 T4 延迟，不能直接当成 CPU 性能。本次未确认 ModelScope 镜像 |

这些是模型发布者的能力说明，本次没有对其性能、校准或数据划分进行独立复核，也没有与 deVision 同口径评测。原生输出概率不代表在本项目分布上已经校准。

## ModelScope 已确认的同名或近名文本候选

| ModelScope 模型 | 模型卡声明 | 为什么不能直接当作图片决策模型 |
| --- | --- | --- |
| [convaiinnovations/laya-multilingual](https://www.modelscope.cn/models/convaiinnovations/laya-multilingual) | mmBERT-base，约 322M，文本或 JSON 状态加自定义问题，单次前向输出决策概率 | 模型卡接口和骨干为文本，不提供原生图片输入 |
| [convaiinnovations/laya-typed-decisions](https://www.modelscope.cn/models/convaiinnovations/laya-typed-decisions) | ModernBERT-large，约 421M，四种文本工作流的决策微调 | 不是视觉扩展；模型卡还披露其温度拟合存在问题，不能照搬“已校准”的结论 |
| [fastino/GLiNER2.5-Decide](https://www.modelscope.cn/models/fastino/GLiNER2.5-Decide) | DeBERTa-v3-large，一次前向，运行时提供标签集，多头文本分类；模型卡支持 CPU/GPU | 示例是 `classify_text`，没有图片输入；属于文本决策的可借鉴实现，不是视觉模型 |

基础 [convaiinnovations/laya 的 ModelScope 页面](https://www.modelscope.cn/models/convaiinnovations/laya) 也能确认存在；其文本属性同时得到同家族的上述模型卡说明支持。原始 [ModernVBERT/modernvbert](https://huggingface.co/ModernVBERT/modernvbert) 是视觉语言编码器，模型卡提供 MLM 用法和检索变体；不能把它本身等同于已经训练好的 Laya 视觉决策头。

## 对 deVision 的取舍

如果目标是找相同接口的直接对照，优先看已确认的上游 `laya-vision-modernvbert-250m` 和当前 `laya-vision`；`laya-v` 可作为更大、多语言版本的补充。它们的 CPU 延迟需要在固定硬件、线程、精度、图像分辨率与问题数量下测量，本次资料不能证明哪个在我们的 CPU 部署目标上更好。ModelScope 上的图文匹配与小型生成式 VLM 另列下节，不能直接归为同类校准决策模型。


## ModelScope 已确认的视觉候选

下表为本次实际读到模型卡的 ModelScope 仓库；能力描述来自发布者，最后一列为针对 deVision 需求的推断，不是同口径实测结论。

| 模型（ModelScope 链接） | 输入与原生输出 | 对 deVision 的参考价值与差异 |
| --- | --- | --- |
| [thomas/blip-itm-base-coco](https://www.modelscope.cn/models/thomas/blip-itm-base-coco) | 图片 + 文本陈述，返回图文匹配 logits；同时支持相似度分数 | 非生成式的图文联合判别，可把候选答案改写成陈述后打分。其匹配任务与自定义问题决策仍有差别，匹配概率不能直接当作已校准的任务正确概率 |
| [google/siglip2-base-patch16-512](https://www.modelscope.cn/models/google/siglip2-base-patch16-512) | 图文表征、候选标签打分；模型卡给出零样本图片分类示例 | 可作简单视觉分类/图文打分基线，也是我们视觉塔所属系列；完整双塔的图文匹配不等于 deVision 的问题条件化决策头 |
| [iic 中文 CLIP ViT-H/14](https://www.modelscope.cn/models/damo/multi-modal_clip-vit-huge-patch14_zh/) | 中文图文双塔，图文表征和匹配；页面现标识为 iic，旧 damo 链接可检索到该卡 | 可研究中文候选描述的匹配。没有我们这种多问题 `choice/noul` 概率接口；需自行组织候选并评测 |
| [ggml-org/SmolVLM-256M-Instruct-GGUF](https://modelscope.cn/models/ggml-org/SmolVLM-256M-Instruct-GGUF) | 图片和文字输入，生成文字；ModelScope 提供 GGUF，并链接 llama.cpp 集成 PR | 可作小型生成式基线、观察 CPU 部署路线；原生不提供 deVision 的候选概率协议。模型卡公布的原模型规模约 256M，实际 CPU 延迟未测 |
| [Qwen/Qwen3.5-0.8B](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) | 带视觉编码器的因果语言模型，图文输入、文字输出；模型卡标注总规模约 873M | 可作多模态任务微调或能力对照候选；仍是生成式接口。要比较选项概率需另行定义打分/校准，不能把生成一串概率数字等同于可靠校准 |

BLIP 的具体架构另外核对了 [Salesforce 官方实现](https://github.com/salesforce/BLIP/blob/main/models/blip_itm.py)：视觉编码结果进入 BERT 的跨模态注意力，取首位置输出给两类匹配头；这条 ITM 路径不生成文本。ModelScope 卡中通用的 captioning 文案不能替代实际 `BlipForImageTextRetrieval` 调用语义；[上游模型卡](https://huggingface.co/Salesforce/blip-itm-base-coco)也明确是图文匹配 checkpoint。

SmolVLM 能力另核对 [Hugging Face 官方模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)：支持图文问答、输出文字，代码使用 `generate`；每个 512×512 图块压成 64 个视觉 token。ModelScope 的 GGUF 页只用来确认模型资源及其上游来源，本次没有下载、核验量化精度或测试 llama.cpp。

**本次选型判断**：若研究“直接判别、不生成”的架构，ModelScope 上优先参考 BLIP-ITM 和 SigLIP2；若要与完整图片多问题概率接口直接对照，上节 HF 上的 Laya Vision / Laya-V 更接近。这个判断是任务与接口匹配，不是效果排名。中文 CLIP 是中文图文检索方向；SmolVLM / Qwen3.5-0.8B 适合作为另一类生成式对照。未查到镜像不能表述为平台不存在，也没有证明任何一个可以无需适配直接替代 deVision。

资料检索仅更新本 research 文档，不写 critic；没有下载权重、安装依赖、运行模型或启动评估。
