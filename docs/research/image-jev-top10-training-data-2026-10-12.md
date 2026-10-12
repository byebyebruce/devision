# 开放权重 Image Jev 模型前十与可复用训练数据调研

调研日期：2026-10-12，Asia/Shanghai。用途：为 deVision 从 v12 继续训练选择数据来源。本文是调研报告，不是实施计划或训练验收；未修改代码、配置、数据或权重，未下载训练数据本体，未启动训练、评测或数据生成。

## 1. 结论与建议阅读顺序

**能找到十个有共同图片决策评测结果的开放权重系统，但不能找到十套与这些发布模型完整对应、可以直接下载的图片训练集。** 有的只公开数据来源和构建程序，有的只公开模型；有的 Jev 微调完全使用文本，视觉能力来自基座。

对我们最有价值的两条路径是：

1. **借鉴 Imajev 的任务构造，再使用它公开的原始来源**：真实照片的属性、图文冲突、困难负例、可回答性，以及困难题与旧能力回放。ABO、VizWiz、VisA 等可以取得；其最终混合训练包本次未找到公开入口。[Imajev 技术说明](https://github.com/mohit67890/imajev/blob/main/docs/technical-specification.md)
2. **采用已经公开媒体和标注的数据**：OneJev-Data 的静态图片部分用于低成本试接入；PixMo-Points 补真实照片的指代和条件计数；CoSyn 补科学图解、图表和视觉数学。这些补充数据未必来自榜上前十，但可获取性更好。[OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data)、[PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points)、[CoSyn-400k](https://huggingface.co/datasets/allenai/CoSyn-400k)

**建议先形成最多约 3 万题的候选池，按能力分块验证，再决定扩量。** 这是本文的研究建议，不替换已经讨论的 v12 重训方案。原始数据规模、作者实际训练量、我们过滤后的可用量，全文分别标注；目前不能给出过滤后的真实总量，也不能承诺提升多少个百分点。

阅读顺序：第 2 节看排名；第 3–4 节看每个模型实际开放了什么；第 5–7 节看数据、下载入口、数量和收益；第 8 节看如何适配当前模型；第 9 节看建议优先级。

## 2. “能力前十”的统计口径

### 2.1 采用同一套 ImageJevBench，而不是混排不同模型的自测

本次采用 Benchmark Heaven 的 **ImageJevBench v0.3.0 / Core**，发布数据构建时间为 **2026-10-10 11:02:20 UTC**。页面显示 56 measured、54 accepted，并另外保留 dated-carry 信息。旧版本成绩不纳入本表。[榜单](https://benchmarkheaven.com/image-jev-bench)、[机器可读结果](https://benchmarkheaven.com/api/decision-results/imagejevbench/v0.3.0/json)、[结果数据卡](https://benchmarkheaven.com/jev-models/data)

本报告的筛选规则：

- 主表对象为同榜已接受、具有公开模型权重且榜上有明确自托管配置的图片决策系统，包括基座加决策读取程序的系统；不要求它必须额外做过视觉微调。开放基座的托管变体另列于第 2.4 节。
- 按 **Capability = (Intelligence + Calibration) / 2** 降序；不加入推理成本、延迟上限。这里是根据官方分数重新筛选的排名，**不是官网默认综合排名，也不是所有视觉模型的世界前十**。
- “开放权重”不等于全部训练数据开放，也不等于所有许可都属于宽松开源许可；例如有 Gemma 条款、非商用条款及需要接受条件才能取权重的模型。
- 同系列不同尺寸按榜单分别列出，复用的数据只算一次。没有把通用 VLM 的其他 benchmark 分数塞进来补位。

[Capability 计算源码](https://github.com/fstandhartinger/model-market-comparison/blob/main/lib/jevbench-capability.mjs)、[榜单转换源码](https://github.com/fstandhartinger/model-market-comparison/blob/main/lib/imagejev-board.mjs)

**指标解释**：Intelligence 是经过猜测基线修正等处理的能力轴，不是原始准确率；Calibration 是越大越好的校准轴，不是 ECE。Capability 同时奖励答对和概率质量；若只关心答对率，不能把它当准确率排名。Composite 则另外包括速度、成本，不能替代这里的能力分。

整次评测抽取 300 public + 1,200 sealed；Core 对这些本地系统对应 **235 + 939 = 1,174 题**，computer-use 单列。它与本项目的 ScienceQA、POPE、绑定切片不是同一分母，不能直接比较百分比。近似并列不代表显著差异，例如第 6、7 名只差约 0.01 分。[方法与原始结果](https://github.com/fstandhartinger/model-market-comparison/blob/main/data/imagejev-v03.json)

### 2.2 开放权重、自托管条目的能力前十：取消服务预算过滤

| 名次 | 模型与公开入口 | 规模／基座 | Capability ↑ | Intelligence ↑ | Calibration ↑ | 图片训练数据公开情况 |
|---:|---|---|---:|---:|---:|---|
| 1 | [Imajev-4B](https://huggingface.co/mohit67890/imajev-4b) | 4B / Qwen3.5-4B | **68.01** | 49.91 | 86.10 | 来源、转换和生成代码较完整；最终训练整包未找到 |
| 2 | [Imajev-9B](https://huggingface.co/mohit67890/imajev-9b) | 9B / Qwen3.5-9B | **67.07** | 48.33 | 85.81 | 与 4B 大量共用来源；不是第二套独立数据 |
| 3 | [Surogate Rune 26B-A4B v3](https://huggingface.co/surogate/rune-26b-a4b-GGUF) | 26B 总量、约 4B 激活 / Gemma 4 | **64.81** | 44.80 | 84.81 | 专用训练数据、规模、完整配比未找到；权重有 gate |
| 4 | [Bonsai-2-27B v2 PQ2_0 + Q8_0 MMProj](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | 27B / Qwen3.8-27B 衍生 | **63.07** | 43.23 | 82.91 | 压缩训练语料未找到；Jev 侧主要公开推理与校准 |
| 5 | [AutoJev-27B](https://huggingface.co/denis-pplx/autojev-27b) | 27B / Qwen3.8-27B | **62.15** | 37.56 | 86.74 | 73k 精确语料未发布；较早构建器含少量合成图片 |
| 6 | [Jevify Gemma 4 26B-A4B](https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b) | 26B 总量、约 4B 激活 / Gemma 4 | **58.73** | 38.50 | 78.97 | 约 47k 文本题配方公开；图片微调题为 0 |
| 7 | [JPT-9B](https://huggingface.co/kirp/jpt-9b) | 9B / Qwen3.5-9B | **58.72** | 44.29 | 73.16 | 49,221 道文本决策题；图片微调题为 0；成品未找到 |
| 8 | [Clef / Cloudflare](https://huggingface.co/Cloudflare/clef) | 27B / Qwen3.8-27B | **58.08** | 47.40 | 68.75 | 官方称内部合成数据；数量、图像占比和下载未公开 |
| 9 | [NeoHorse Jev 4B](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B) | 4B / NeoHorse-1-4B | **57.62** | 36.83 | 78.42 | 未找到该 checkpoint 的训练集和图片训练规模 |
| 10 | [Visual-Jev generic](https://huggingface.co/Qwen/Qwen3.5-4B) | 冻结 Qwen3.5-4B，zero-shot 读取 | **56.73** | 29.70 | 83.76 | 该配置没有额外图片决策微调集；基座完整训练数据未找到公开入口 |

分数统一来自上述 v0.3.0 结果，不使用模型卡里的旧版第一名、自测成绩或官网含成本的综合名次。Bonsai 的成绩对应特定量化与视觉投影文件；Visual-Jev 是特定推理配置，不能视作任意 Qwen3.5-4B 调用都有相同成绩。

官网把 Wity、Autoloops、System1 的受测提交归为 closed API，**这不等于它们的基座都闭源**；本次又核对了服务商原站，将开放基座的托管变体单列于第 2.4 节。Laya Vision 当前为作者要求 hold，未获得本版可比名次，放在第 7 节单独研究。[完整榜单的提交状态](https://benchmarkheaven.com/image-jev-bench)

### 2.3 为什么官网默认“能力榜”会显示另一组前十

官网 Jev-class Capability 额外要求成本和调整后中位延迟均不超过冻结的 Jev 1.13 参考值两倍：约 **$0.064595 / 1,000 决策、1.232957 秒**。在这个服务预算内，前十为：

| 名次 | 系统 | Capability | 相对上表变化 |
|---:|---|---:|---|
| 1 | Imajev-4B | 68.01 | 同上 |
| 2 | Rune v3 | 64.81 | 同上 |
| 3 | AutoJev-27B | 62.15 | 同上 |
| 4 | Jevify 26B-A4B | 58.73 | 同上 |
| 5 | NeoHorse Jev 4B | 57.62 | 同上 |
| 6 | [vjev-vision](https://huggingface.co/yah01/vjev-vision) | 55.66 | 有约 99k 视觉题配方，但成品未找到 |
| 7 | [JPT-4B](https://huggingface.co/kirp/jpt-4b) | 55.62 | 与 JPT-9B 同一份文本训练数据 |
| 8 | [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | 54.25 | 披露约 30k 微调问题；图片占比未知 |
| 9 | [Reflex 4B stable](https://github.com/kshetrajna12/reflex/blob/main/docs/results/frozen-vs-trained.md) | 53.34 | 实际为冻结基座，不使用旧 LoRA |
| 10 | [Shisa DE-1](https://huggingface.co/shisa-ai/shisa-de-1) | 51.46 | 披露 3,632 训练例；无已确认图片训练包 |

这五个补充模型也全部查了训练数据，见第 4 节。两张榜的差异来自明确的筛选规则，不是拿不同版本拼出的排名。[预算规则与分数](https://benchmarkheaven.com/api/decision-results/imagejevbench/v0.3.0/json)

### 2.4 开放基座的托管系统：避免因榜单标签漏掉强模型

| 托管条目 | 本版 Capability | 官方核实结果 | 对训练数据调查的影响 |
|---|---:|---|---|
| [Autoloops – Gemma 4 31B IT](https://autoloops.ai/models/gemma-4-31b-jev) | **68.35** | 底座 [Google Gemma 4 31B IT](https://huggingface.co/google/gemma-4-31B-it) 公开；Jev 路径输出校准概率；本次没找到与受测服务匹配的公开实现、adapter／校准参数和完整自托管配置 | 若只要求底座开放、允许托管配置，它应排在本报告主表的 Imajev 前；没有核实到其专用公开图片训练集，也不能仅下载底座就声称复现该成绩 |
| [System1 s1-vision](https://system1models.ai/legal/model-licences) | **58.27** | 官方称直接托管开放权重。榜单 10 月 6 日所对应的历史实现为 Rune v3，特定条件下有 Jev-Omni fallback；10 月 11 日后换成 Blink | 是服务变体，不重复计算一套独立训练数据；当前服务已变，不能把 58.27 归给 Blink |
| [Wity-1 reasoning auto](https://wity.alphanimble.com/) | **74.03** | 可访问托管接口，未核实到对应公开权重 | 不纳入开放权重候选；不据此断言其未披露基座的许可 |

因此“开放基座”“公开训练代码”“可定位自托管配置”“完整数据开放”是四件事。本报告保留这些强托管条目的信息，同时不将无法复核的托管分数直接赋给其原始基座。[Autoloops 基座说明](https://autoloops.ai/models/gemma-4-31b-it)、[System1 历史 NOTICE](https://system1models.ai/legal/NOTICE.txt)、[System1 当前模型页](https://system1models.ai/models)

## 3. 前十逐项调查：训练量、公开程度与复用价值

本文把公开程度分成四种：**成品可取**＝训练记录与媒体有公开入口；**可按代码构建**＝有来源与转换程序但仍需重建；**只有配方**＝知道来源或数字但没有完整生成路径；**未披露**＝本次官方资源未找到。它们不能互相替代。

### 3.1 Imajev-4B / 9B：最值得借鉴的数据工程

**已核实**：官方仓库公开 `scripts/v1/` 各来源转换器、后续教师标注与困难题生成脚本；HF 作者的数据仓库中可见的 benchmark、运行结果不能当训练集。没有确认可以直接下载作者完整的最终混合 train manifest 与媒体。[代码](https://github.com/mohit67890/imajev/tree/main/scripts)、[第一阶段转换器](https://github.com/mohit67890/imajev/tree/main/scripts/v1)、[技术规格](https://github.com/mohit67890/imajev/blob/main/docs/technical-specification.md)

| 阶段 | 作者披露数量 | 数量到底指什么 |
|---|---:|---|
| 第一阶段 | 504,000 决策，466,135 records | 200,000 文本 + 300,000 图片决策 + 4,000 商品照片与描述冲突题 |
| 4B 第一次混合训练 | 866,854 训练决策 | 包含上面的 504k；另加教师标签和图文／双图题，不能与 504k 相加当独立总量 |
| 后续 soft-target 阶段 | 39,515 records | 混合文本、教师分布、程序题与旧图片回放；不是 39,515 张新图 |
| 4B 新阶段第一轮 | 125,424 行 | 58,000 困难文本 + 32,427 困难图像 + 19,998 旧文本 + 14,999 旧图像 |
| 4B 新阶段第二轮 | 26,494 行 | 13,247 上轮错题 + 等量回放；不是全部新数据 |

历史阶段来源以技术规格为准；4B 后续更新以当前模型卡为准，9B 没有同一阶段更新，不能套用其数据量。[4B 当前训练说明](https://huggingface.co/mohit67890/imajev-4b#training)、[9B 模型卡](https://huggingface.co/mohit67890/imajev-9b)

第一阶段的图片来源及作者采用量如下。这里是 **Imajev 加工后的决策题数**，不是上游全量，也不是我们能直接取到的最终题数。

| 来源（链接为官方数据或作者转换入口） | Imajev 决策题数 | 主要增益方向 | 我们的复用判断 |
|---|---:|---|---|
| [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | 38,413，另有 4,000 冲突题 | 商品属性、照片和记录是否一致 | **优先**：新图片域、当前分辨率较友好 |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) | 25,926 | 属性、对象和关系 | 已有来源，优先补题型，不重复算新图 |
| [TDIUC 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_tdiuc.py) | 22,505 | 多种视觉问题、不存在项 | 多为 COCO，同源隔离后选缺口 |
| [masked_evidence 构造](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_masked_evidence.py) | 22,384 | 证据不足、依赖图像区域 | GQA/VG 派生，不是新图片源 |
| [state_aware 构造](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_state_aware.py) | 22,252 | 图片结合状态／条件 | 多源派生；必须保留决策所需上下文 |
| [VQAv2](https://visualqa.org/download.html) | 21,807 | 通用视觉、人工答案分布 | 已有，重点是软标签与选项构造 |
| [VG attributes 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_vg_attributes.py) | 19,535 | 细粒度属性、指代 | 已有 VG 图像；可补不同对象的区分 |
| [Fashion200k 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_fashion200k.py) | 18,804 | 服饰细粒度属性 | 新域，但文本标签未必都能从图上确认 |
| [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/) | 15,073 | 真实拍摄、可回答性 | **优先**：低质照片与不确定性 |
| [bool_abstain 构造](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_bool_abstain.py) | 11,232 | 存在／不存在／无法确定 | COCO 派生；不能把 unknown 当 false |
| [TextVQA 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_textvqa.py) | 11,423 | 图片文字 | 256 分辨率下先做可读性筛选 |
| [TallyQA](https://github.com/manoja328/TallyQA_dataset) | 11,126 | 条件计数 | 已有，核对复杂题和指代覆盖 |
| [UnkVQA 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_unkvqa.py) | 9,075 | 不确定／不可回答 | 先核实标签语义与现有接口匹配 |
| [VizWiz quality 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_vizwiz_quality.py) | 7,870 | 画质与可用性 | 与 VizWiz 同图可能重合 |
| [defects 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_defects.py) | 7,459 | 工业异常，VisA / DAGM / BTAD | 可选域扩展，不是绑定问题首选 |
| [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning) | 7,033 | 空间关系 | 已有 COCO 派生来源 |
| [DUDE 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_dude.py) | 5,830 | 文档理解 | 多页／细文字，当前适配成本高 |
| [A-OKVQA](https://github.com/allenai/aokvqa) | 5,564 | 视觉常识 | 已有，关注图像必要性 |
| [KonIQ](https://database.mmsp-kn.de/koniq-10k-database.html) | 5,563 | 真实图像质量 | 新任务域，不能直接解释为视觉推理提升 |
| [Marqo-GS 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_marqo_gs.py) | 5,563 | 商品匹配与属性 | 可候选，标注／图片来源分别记录 |
| [SQID / ESCI 转换](https://github.com/mohit67890/imajev/blob/main/scripts/v1/convert_sqid_esci.py) | 5,563 | 商品搜索相关性 | 依赖检索上下文，不能直接去掉文本 |

来源和数字：[Imajev 数据章节](https://github.com/mohit67890/imajev/blob/main/docs/technical-specification.md#2-training-data)。后续新图还用了 PD12M、Wikimedia Commons、Open Images；披露的 144,871、36,639、44,851 是对应来源保留的题量、包含多个划分，不全是训练量。

**值得借鉴的是方法，不是直接照抄它的混合比例**：按模型错误选困难题，给旧能力留回放；同图改变条件、属性或关系，使答案必须随视觉证据变化；有确定程序答案的题优先。Imajev 还支持双图、score、unknown，其所有任务都不能直接塞进我们当前单图 `noul/choice`。教师一致只能减少一部分噪声，不能证明图片标签必然正确。

### 3.2 Rune v3：强基座和公开引擎，不等于公开训练语料

公开权重有访问条件，训练／推理引擎开放；本次没有找到 Rune 专用数据清单、混合比例、最终训练文件和图片题数量。不能把作者同组织其他 Romanian VLM 数据自动归给 Rune。[模型卡](https://huggingface.co/surogate/rune-26b-a4b-GGUF)、[发布说明](https://invergent.ai/blog/rune/)、[引擎](https://github.com/invergent-ai/surogate)

特别注意：[decision-index-results](https://huggingface.co/datasets/surogate/decision-index-results) 是评测记录，发布方明确不允许用于训练。**本轮可确认取得的 Rune 专用图片训练量：未知。**

### 3.3 Bonsai：不要把校准请求数写成训练数据量

权重与特定 GGUF 文件公开；Jev 层提供 llama.cpp 修改、概率读取和温度校准。所披露的 **27,500 requests = 5,499 calibration + 22,001 held-out test** 是标定／评测，不是新增视觉微调题。压缩训练语料、图片占比和可下载训练清单未找到。[Jev 项目](https://github.com/kyr0/Bonsai-Llama-Jev)、[权重](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)、[白皮书](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)

工程上可参考 CPU 推理和校准方式；对本轮图片训练数据采购没有直接增量。

### 3.4 AutoJev：73k 成品未开放，旧构建器可以参考

发布模型披露 **73,000 unique rows**；训练元数据明确该精确语料及图片没有随仓库打包。公开的较早构建器目标为约 **96k**，其中 **95k 文本 + 1,000 自绘几何题／250 张图**，不能把二者说成同一配方。[模型卡](https://huggingface.co/denis-pplx/autojev-27b)、[训练元数据](https://github.com/denis-pplx/autojev/blob/main/configs/training.json)、[构建器](https://github.com/denis-pplx/autojev/blob/main/src/autojev/data.py)

几何题为 224 像素、左右对象的颜色／形状判断，重建成本低，适合管线检查和简单绑定控制。它不足以补真实照片、多物体遮挡或科学图解；本项目已有合成关系题，净信息增量可能很小。73k 中究竟多少图片题，仍未知。

### 3.5 Jevify：可重建的约 47k 题，全部为文本

公开配方为 14 个文本来源和 2 类合成文本，约 47,000 行，视觉能力保留自 Gemma 基座。来源包括 Banking77、CLINC、AG News、DBpedia、情绪、BoolQ、MNLI、ToxicChat 等；**没有图片训练题**。[模型卡](https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b)、[训练说明与来源](https://github.com/kushalpatil07/jevify/blob/main/train/README.md)、[数据构建器](https://github.com/kushalpatil07/jevify/blob/main/train/build_dataset.py)

可以参考多标注员概率目标与选项换序，不能把 47k 算成可补视觉多样性的训练集。

### 3.6 JPT-9B：49,221 文本决策题，不提供新的视觉监督

9B 与 4B 使用同一份数据，披露 **49,221 题／32,835 records**，并采用两份选项换序副本训练；原卡未明确该题数是否已经包含复制，副本不增加独立语义数据量，两个模型的数据也不能再相加。作者明确图片能力是 Qwen3.5 视觉路径的 zero-shot，图像微调为 0。[9B 模型卡](https://huggingface.co/kirp/jpt-9b)、[4B 模型卡](https://huggingface.co/kirp/jpt-4b)、[作者提交说明](https://github.com/fstandhartinger/jevbench/issues/78)

数据类别／配方有披露，但完整训练文件只见本地路径引用，本次未找到公开整包。作者的 `decision-index-results-jpt-*` 是评测结果，不能补作训练数据。对我们有参考价值的是选项换序和目标设计，不是新图片来源。

### 3.7 Clef：权重确实公开，训练数据仍是内部数据

核实到公开、无门禁的完整主干分片和 `joint_head.safetensors`。官方说明使用内部合成数据训练决策头及 LoRA；没有披露图片题数、总体样本量、可下载数据或完整生成器。[模型与文件](https://huggingface.co/Cloudflare/clef/tree/main)、[官方训练说明](https://blog.cloudflare.com/clef-decision-models/)

它能提供多问题／schema 扰动、概率目标等思路；目前不能直接给我们一份已公开图片训练集。

### 3.8 NeoHorse：未找到该发布 checkpoint 的训练数据

榜单实测权重为 `434cb21d…`，基座是 NeoHorse-1-4B。公开模型卡、作者提交和服务仓库未提供该版可获取的图片训练包、训练图片数或完整构建程序。[模型卡](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B)、[Jev 说明](https://github.com/TokenRhythm/NeoHorse/tree/main/jev)、[提交说明](https://github.com/fstandhartinger/jevbench/issues/131)

卡内 Image-NLI 的 8,000 项是评测规模，不能填进训练数据量。本报告结论是“未披露／未找到”，不是声称它没有训练。

### 3.9 Visual-Jev generic：冻结基座的参照系统

榜单明确这是 zero-shot Qwen3.5-4B baseline，不是额外训练了一套图片决策集的模型。Qwen 权重公开，但未找到与这一基座完整对应的预训练图片／问答语料整包。[榜单该系统](https://benchmarkheaven.com/image-jev-bench/visual_jev_generic_baseline)、[Qwen 官方卡](https://huggingface.co/Qwen/Qwen3.5-4B)

它说明“决策接口能取得多好成绩”也依赖基座已获得的视觉表征；不能把排名反推为有一份免费训练配方可直接让小模型复现。

## 4. 默认 Jev-class 前十中另外五个模型

| 模型 | 真实披露的训练量 | 图片部分和可获取性 | 我们可以借鉴什么 |
|---|---|---|---|
| [vjev-vision](https://huggingface.co/yah01/vjev-vision) | 文本阶段约 145k；视觉阶段约 84k COCO 几何题 + 15k VQAv2 | 约 99k 视觉题，混 25% 文本回放；只找到配方，未找到完整转换结果或生成器 | 框产生关系／大小／计数／不存在负例；VQA 保留 10 人答案分布 |
| [JPT-4B](https://huggingface.co/kirp/jpt-4b) | 与 9B 相同 49,221 题 | 图像训练为 0，不能重复计量 | 同第 3.6 节 |
| [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | 作者称约 30k 问题 | 来源、图像占比、成品未找到 | 暂无可直接采用的图片数据 |
| [Reflex stable](https://github.com/kshetrajna12/reflex/blob/main/docs/results/frozen-vs-trained.md) | stable 不加载历史 adapter | 榜上是冻结 Qwen3.5-4B；历史 7,850 例文字 LoRA 不是榜上配置 | 微调伤害迁移的负结果值得参考 |
| [Shisa DE-1](https://huggingface.co/shisa-ai/shisa-de-1) | 3,632 条训练例 | 所列任务偏文本；未找到整包，也未披露图片数量 | 不能把几千例后训解释成全部视觉学习量 |

vjev 的原始 [COCO](https://cocodataset.org/#download) 和 [VQAv2](https://visualqa.org/download.html) 可以获得，但我们已经使用这些图片域，价值主要是同图上更好的问题，而不是 99k 新图。Reflex 的历史 [recipe](https://github.com/kshetrajna12/reflex/blob/main/src/reflex/train/recipes.py) 会把 MMLU-Pro 的 `test` 重新划分训练；若照搬，其官方 test 就失去独立性。Jev-Omni 链接的 DecisionBench 也只是评测，不是所称 30k 训练包。

Shisa 视觉塔未更新本身不能证明所有训练例都无图，因此只记“图片份额未知”，不把推断写成零。[Shisa 发布卡](https://huggingface.co/shisa-ai/shisa-de-1)、[Reflex 旧 LoRA 记录](https://github.com/kshetrajna12/reflex/blob/main/docs/results/lora-mix3-qwen3.5-4b.md)

## 5. 实际可取的数据：入口、数量、媒体、成本与收益

### 5.1 优先候选清单

下表中的数量来自发布者数据卡或文件元信息，**不是本次全量下载验收的结果**。“优先”表示与当前目标契合，不表示已经证明可以提高 deVision 成绩。磁盘大小按发布包／标注区分，不包含缓存、解压和转换的额外空间。

| 优先级 | 数据与获取入口 | 已公开规模及单位 | 媒体与已知下载量 | 关联模型／来源 | 对 deVision 的潜在收益与限制 |
|---|---|---|---|---|---|
| P0 | [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points) · [文件](https://huggingface.co/datasets/allenai/pixmo-points/tree/main) | **2,376,222 条指代表达／点集记录**，不是独立图片数 | 图片 URL + SHA，另取图片；标注约 **198MB**，不含图 | Molmo 系列公开上游；补充来源，不冒充前十训练集 | **真实图像的文本指代、条件计数、多对象区分**；我们已用过 PixMo 相关数据，需核对未覆盖表达和新图 |
| P0 | [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) · [小图包](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar) · [商品信息](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar) | **147,702 商品、398,212 独立 catalog 图片**；尚无我们的决策题数 | max-256 小图包约 **3GB** + 信息约 **83MB**；原图约 110GB | Imajev 第一阶段使用 | **商品属性、细粒度类别、图文冲突**；同商品多视角要同组，商品属性不一定全在照片上可见 |
| P0 | [CoSyn-400k](https://huggingface.co/datasets/allenai/CoSyn-400k) · [文件](https://huggingface.co/datasets/allenai/CoSyn-400k/tree/main) | **401,571 训练图片记录**，每图多 QA | 图像内嵌，并有生成代码／数据；本次未确认全包总字节数 | AllenAI 公开数据；补充来源 | **science 图解、图表、视觉数学**；优先无需细小文字的图，程序答案也需核对渲染后是否可见 |
| P1 | [OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data) · [文件](https://huggingface.co/datasets/OmniJev/OneJev-Data/tree/main/data) · [许可与源](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/main/licenses.csv) | **94,707 决策题**，其中静态图片 **13,559**；其余见第 6 节 | 18 Parquet，合计 **17.716GB**，包含图像／视频帧；不是静态图片子集大小 | OneJev 的实际训练发布包，榜外补充 | **现成决策格式、快速对比不同新域**；单图／题型／同图去重后会少于 13,559 |
| P1 | [VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/) · [训练图片](https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip) · [标注](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) | **20,523 train 图片／问题对**；4,319 val、8,000 test 分开；每题 10 人答案 | 图片和标注公开；本次未量全包大小 | Imajev、OneJev 使用 | **真实低质照片、可回答性、概率分布**；不可回答与“否”不同，需单独设计目标 |
| P1/P2 | [PixMo-AskModelAnything](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) | **161,737 训练 QA** | URL + SHA；标注约 **63MB**，不含图 | Molmo 公开上游 | 开放自然问答、真实图多样性；长答案转唯一正确选项需要审核，成本高于闭式 QA |
| P2 | [VisA](https://github.com/amazon-science/spot-diff#data-download) | **10,821 图**：9,621 正常、1,200 异常，12 对象子集；这是全数据量 | 官方 AWS 图片、类别及像素 mask；本次未量下载大小 | Imajev defects 的来源之一 | **缺陷／缺件／异常判断**；是专门域；1-class 原划分训练只有正常图，需选合适监督划分 |
| P2 | [CoSyn-point](https://huggingface.co/datasets/allenai/CoSyn-point) | **68,051 train 图片记录**，另 1,000 val；每图可有多定位题 | 图像内嵌；包合计约 **12.4GB**，包含 val | AllenAI 公开上游 | 文档／界面定位；对细文字依赖强，与真实照片关系绑定不同 |
| P2 | [PD12M](https://huggingface.co/datasets/Spawning/PD12M) | 约 **1,240 万图片 URL／caption 行**；不是决策题 | 元数据 + 图片 URL；整图成本未核实 | Imajev 后续教师标注原料 | **扩展新图片域或图文对齐**；需要选择子集和新增标签，不能直接作为阶段 2 闭式题 |
| P2 | [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | 50 个来源；Laya 的 19 源当前共 **845,991 原始行** | 多数内嵌图像；这 19 源约 **65.976GB** | Laya / Idefics / SmolVLM 等 | 低成本重用标准视觉 QA；我们多数已用，先审缺口与重叠，不整包重复叠加 |
| P3 | [PlayJev 训练记录](https://huggingface.co/OmniJev/PlayJev-0.8B/tree/main/data) · [重建程序](https://github.com/OmniJev/PlayJev) | **2,166,984 决策、11,416 episodes** | 约 **853MB JSONL**；图片需浏览器重放重建，不在 JSONL 中 | PlayJev | 游戏动作、可控空间状态；连续帧高度相关，不能当 216 万独立自然场景 |
| P3 | [OpenJev-Vision-Research-v0.1](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1) | **12,832 图片记录全部划分；仅 6,536 train 图** | Parquet 含图片、来源、split、targets | 固定本体研究项目 | 合成概率 gold 依赖图片和显式先验，不能丢掉先验；Pets/CLEVR4 为固定本体硬标签。适合小型控制实验 |

P0 为当前优先调查与构建方向；P1 补充广度／可用性；P2 需要额外目标或转换投入；P3 仅作为特定研究工具。**没有把几十万 caption、上百万点集行和几万闭式问题加成一个“可直接训练题数”。**

### 5.2 几个最影响实际收益的细节

**PixMo-Points：最贴近“按文字挑对象”的缺口。** 原始点标注能把表达指向具体目标，也能支持条件计数。建议保留描述中的属性和关系，在一图中建立相近的候选对象／计数答案。点集不等于所有物体的穷尽清单：没标出的物体不能直接当负例。新数据应与已有 PixMo Count、R13 中曾准备的 PixMo 数据按图和表达核对，避免只是换了包装。[数据卡](https://huggingface.co/datasets/allenai/pixmo-points)

**ABO：扩展真实图片多样性的下载成本较低。** 官方小图包已经与我们 256 输入接近。适合颜色、形状、部件和类别选择；不适合把品牌、产地、内部材质等不可见字段直接当视觉标签。属性正负例应选同类、视觉上确实可分的商品，避免“椅子和鞋子”这种过于简单的负例。拆分按商品身份完成，避免同一个商品不同角度跨 train/dev。[ABO 官网](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)

**CoSyn：科学图和数学图比一般网页截图更值得先筛。** 已公开分支包含 chemical 8,942、circuit 10,470、diagram 34,963、math 66,714 条训练图片记录；四者合计 **121,089 图片记录**，这是候选上限，不是筛完能用的闭式题数。chart 116,814、table 46,518 也有价值，但大量问题需要读文字。其 QA 由模型基于生成代码产生，因此仍要检查答案是否依赖图片外信息。[CoSyn 数据卡及各分支](https://huggingface.co/datasets/allenai/CoSyn-400k)

**VizWiz：训练“看不清就不要装懂”的数据基础。** 在目前接口下，可以将可靠的 answerable 标签用于明确的可回答性判断；一般问答只取有足够支持的答案。不要把 `unanswerable` 直接转成普通真假题的 false，也不要未经定义就把它变成 0.5 gold。2026 年官网已放出自评 test 标注，公开不代表应该拿来训练。[当前数据说明](https://vizwiz.org/tasks-and-datasets/vqa/)

**PD12M：能补图片，但不能直接补决策标签。** 当前数据卡标的是 CDLA-Permissive-2.0 元数据许可，单图有公有领域／CC0 标记；不能简单把整包写成统一 CC0。它包含馆藏、历史照片、绘画、物品图等，应按目标域筛选，不直接拿 1,240 万条作自然照片分布。[PD12M 数据卡](https://huggingface.co/datasets/Spawning/PD12M)

### 5.3 许可按项目研究用途记录

本项目 [CLAUDE.md](../../CLAUDE.md) 已明确“研究、不商用”，本报告**不因非商用条款直接排除候选**。保留逐来源条款，方便复现和后续发布说明：

| 数据 | 发布方标注的许可／边界 |
|---|---|
| ABO、VizWiz、VisA | 官方页面为 CC BY 4.0；保留署名与原始来源 |
| PixMo / CoSyn | 数据卡 ODC-BY；URL 指向的原图仍保留原图来源信息，不能只记录标注许可 |
| OneJev-Data | 127 来源逐项许可；包含 research-only、non-commercial、未明确条款项；不能继承代码 Apache 2.0 |
| Cauldron | 各原始集分别适用条款；包装提示词许可不覆盖所有图片／标注 |
| PD12M | 集合元数据与每张图片的许可分别记录 |
| PlayJev | 代码、游戏、素材及生成记录分别看作者来源清单 |

上述为发布信息记录，不是新的法律结论，也没有在本次调研中改变任何模型发布许可。[OneJev 逐源清单](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/main/licenses.csv)、[Cauldron 许可说明](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron#licensing-information)、[VisA 许可](https://github.com/amazon-science/spot-diff#license)

## 6. OneJev-Data 详解：有成品，但图片只有其中一部分

这是本轮找到的、较适合拿来研究决策格式的真实训练发布包。**OneJev、Jev-Omni、tinnel123 的 OmniJev 是不同项目**，不能因为名字相近把训练包归给榜上的 Jev-Omni。

发布包为原训练 99,193 题中的 **94,707 题（约 95.5%）**，一部分因为来源限制没有发布。文件 API 核对为 18 个 Parquet、17,715,763,169 bytes；图像／视频帧已内嵌，但本次未实际解码检查。[数据卡](https://huggingface.co/datasets/OmniJev/OneJev-Data)、[文件树](https://huggingface.co/datasets/OmniJev/OneJev-Data/tree/main/data)、[官方解包器](https://github.com/OmniJev/OneJev/blob/main/train/unpack.py)

| 类型 | 已公开题数 | 当前是否适合 |
|---|---:|---|
| 静态图片 | **13,559** | 优先筛单图、英文、choice/noul；仍包含评分、比较等任务 |
| GUI | 33,405 | 暂单列；256 输入可能看不清界面字和控件 |
| 短视频 | 12,115 | 当前单图架构不直接兼容 |
| 长视频 | 6,779 | 当前单图架构不直接兼容 |
| 文本 | 16,849 | 不是新增图片监督 |
| 规则 | 12,000 | 不是新增图片监督 |
| **合计** | **94,707** | 不能称 9.47 万张图片 |

静态图片的 35 个来源及包内题数如下，合计恰好 13,559。下面的数量是 **包内抽样后的题数**，与上游原始全量不同。原始来源链接见作者 [licenses.csv](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/main/licenses.csv)。

| 能力方向 | 来源与包内题数 | 我们的主要判断 |
|---|---|---|
| 对象／关系／计数 | GQA 762；CLEVR 620；TallyQA 441；VQAv2 438；VSR 385；Visual7W 382 | 大部分已覆盖；筛真正新增的题型和图片 |
| 图像蕴含／视觉常识 | SNLI-VE 443；Flickr30K 442；A-OKVQA 440；NLVR2 438 | 注意 NLVR2 双图；图文匹配不等于对象绑定 |
| 科学与图表 | ScienceQA 437；IconQA 382；FigureQA 381；MapQA 343；AI2D 273；TQA 242 | 可作为现成小样本；多数不是新的来源 |
| 真实劣质图 | VizWiz 440 | 与原始 VizWiz 去重，不另算新图片 |
| 细粒度类别／属性 | FGVC-Aircraft 650；CUB200 315；Food101 315；Oxford Pets 314；Flowers102 312；Stanford Cars 310 | 扩展长尾域；小字／近似类别可能超出现分辨率 |
| 纹理、遥感与符号 | DTD 311；EuroSAT 311；GTSRB 308 | 净新增领域可能较多；先小样本检查用途 |
| 生成图质量／偏好 | ImageRewardDB 659；GenAI Arena 653；AGIQA3K 529 | 可能需多图或评分；不直接对应当前 choice/noul |
| 审美／画质 | AVA 272；AADB 269；KADID10k 195；KonIQ10k 192；FLIVE 191；LIVE Challenge 164 | 任务方向不同，暂不为解决绑定而大量混入 |

**现实定位**：它适合快速发现我们缺哪些域、参考题目表达和目标格式；每来源通常只有几百题，不是大规模补量的终点。下一步若确定某个域有效，应回原始 train split 扩量，避免反复重采样这 1.36 万题。

## 7. Laya 与更大公开数据池：哪些值得补充，哪些不急

### 7.1 Laya Vision：最接近我们的参考，但很多数据已重合

Laya 的 201M 版本披露 **28,414 steps × 64 = 1,818,496 次训练曝光**，其中 45% 为游戏帧。这不是 182 万独立图片。ModernVBERT 分支则实际加工出 **269,900 训练问答**；它继承图文预训练编码器，不是总共只学过 27 万题。[201M 模型卡](https://huggingface.co/thaitea/laya-vision)、[ModernVBERT 实验记录](https://github.com/r33drichards/laya-vision/blob/9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c/site-docs/reference/results/modernvbert-cauldron.md)

19 个 Cauldron 闭式来源和转换代码公开；最终准备文件在作者 Modal volume，未找到统一公开的最终 train JSONL。当前数据卡这 19 个来源的原始行合计 **845,991**，约 **65.976GB**；一行可能有多个 QA，不能与 269,900 直接比较。[数据说明](https://github.com/r33drichards/laya-vision/blob/9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c/site-docs/concepts/data.md)、[转换代码](https://github.com/r33drichards/laya-vision/blob/9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c/laya/cauldron.py)

| Laya 闭式来源 | 当前 Cauldron train 原始行数 | 对我们有什么用 |
|---|---:|---|
| AI2D / A-OKVQA / IconQA | 2,434 / 16,539 / 27,307 | 已覆盖科学图和自然图；检查难度与负例 |
| InterGPS / ScienceQA / TQA | 1,280 / 4,976 / 1,493 | 几何、科学、教材；细文字与上下文限制 |
| Visual7W / VSR | 14,366 / 2,157 | 真实图指代／空间，已有来源 |
| RAVEN / FigureQA | 42,000 / 100,000 | 矩阵关系与图形逻辑，合成域 |
| Hateful Memes / NLVR2 / VQA-RAD | 8,500 / 50,426 / 313 | 特定内容、多图、医学；不作为本次主要增量 |
| CLEVR / DVQA / MapQA | 70,000 / 200,000 / 37,417 | 合成关系、图表、地图；当前多数已有 |
| OCR-VQA / VQAv2 / ChartQA | 165,746 / 82,772 / 18,265 | OCR 分辨率限制，VQA 重合 |

这些是 Cauldron 包装行数，不是最终可用闭式题数；Laya 会限制每源行数、每图 QA 数，并跳过大量开放答案。**TallyQA 虽在 Cauldron 的 50 来源中，却不属于 Laya 这 19 个来源。** [Cauldron 当前卡](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)

此外 Laya 使用 [VLFeedback](https://huggingface.co/datasets/MMInstruction/VLFeedback)、[AVA 子集](https://huggingface.co/datasets/trojblue/AVA-aesthetics-10pct-min50-10bins)、[RichHF-18K](https://huggingface.co/datasets/Exploration/richhf_18k_with_images)、[CrisisMMD](https://huggingface.co/datasets/QCRI/CrisisMMD) 等评分来源，分别约 80k 图片、20k train、15.8k train、2.5k train，均为作者所述转换规模。它们偏评分／偏好／严重程度，适合未来 `score`，当前不宜为解决对象绑定而整批引入。[评分来源说明](https://github.com/r33drichards/laya-vision/blob/9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c/site-docs/concepts/score-data.md)

### 7.2 百万级候选池：公开了，也不应全拿

| 数据池 | 公开宣称规模及入口 | 适用方向 | 暂不整包采用的原因 |
|---|---|---|---|
| LLaVA-OneVision-1.5 | [85M 图文对齐](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)、[22M 指令数据](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data) | 新图域、科学、图文理解 | 85M/22M 是作者集合口径，不是同量英语单图决策；重叠、长答案和格式转换成本大 |
| LLaVA-OneVision-2 | [数据卡](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-2-Data)、[官方代码](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2)；约 4M spatial samples | 指代、定位、空间 | 含多图／3D／视频与 COCO/VG 派生；未核精确单图比例；viewer 的 10 行预览不是全量 |
| Cambrian | [Cambrian-10M](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M)，约 9.784M 数据点；7M 为其重平衡子集 | 视觉中心任务、多源覆盖 | 7M 不是额外 7M；与我们已有 VQA/图表源重合，宜按来源挑选 |
| Docmatix | [数据卡](https://huggingface.co/datasets/HuggingFaceM4/Docmatix)，约 2,444,750 图、9.5M QA | 文档图文对齐、OCR | 每记录最多 4 图、大量文字，当前单图256闭式决策适配差 |

这些来自公开通用 VLM 数据，不归为“Image Jev 前十的已公开训练集”。LLaVA 的 85M／22M 在本次仅核验数据卡与实际分片存在，未全量检查完整性和独立图片数。[LLaVA-OneVision-1.5 论文](https://arxiv.org/abs/2509.23661)

**数量不等于效果**：Cambrian 作者同预算配比实验中，重平衡的 7M 比原始 10M 更好，支持先控制组成再扩量的思路；这不是 deVision 必然获得同等收益的证据。[作者实验](https://cambrian-mllm.github.io/cambrian-1/)

## 8. 转成 deVision 训练数据时，哪些信息必须保留

依据当前 [CLAUDE.md](../../CLAUDE.md) 与 [规格](../spec/vision-decision-mvp.md)：我们主要处理英文、单图、256 letterbox、64 视觉 token、`noul/choice`。支持状态文本不代表所有多模态任务都可无损转换。

| 原始监督 | 建议转换方式 | 不能这样做 |
|---|---|---|
| 原生多选 QA | 保留问题、必要上下文与候选；统一选项身份，换序时同步 gold | 删除上下文后沿用原答案；拿测试 split 补训练 |
| 多标注员答案 | 合并同义答案，在定义清楚的候选集合上计算目标分布；记录集合外质量 | 把 10 人分歧强制变成不可靠的 one-hot；随意丢弃集合外答案再声称“原始概率” |
| 点／框／关系 | 由已确认对象和完整相关标注构造属性、指代、条件计数；同图相关题共组 | 根据“未被标注”断言“不存在”；用框中心代替所有 near/behind/overlap 语义 |
| unknown / unanswerable | 单独明确问题语义；有需要时作为普通 choice 选项或独立可回答性任务 | 当普通 false；未经定义直接设 noul=0.5 |
| ordinal score／偏好 | 本轮先单列；若改 choice，明确改成哪种分类语义 | 将原 score API 的概率／刻度直接当我们已有功能 |
| 多图／视频／动作历史 | 当前候选池先过滤；需要时另立多图课题 | 随便取一帧／一张图但沿用原任务 gold |
| caption／教师长回答 | 阶段 1 对齐候选，或另经问答生成与标签审核进入阶段 2 | 不生成可靠决策题就直接计入可用决策训练量 |

未来构建时，每条记录建议保留 `source`、上游 split、原始 ID、图片身份、原始问题／候选、标签来源、转换版本、许可、同图／同商品／同 episode 组。这里是建议元信息，不代表当前实现已增加这些字段。

**去重和评测隔离优先于混合**：统一 COCO 与 Visual Genome 映射，覆盖阶段 1 caption、历次训练和现有 dev/cal/test；ABO 同商品、游戏同 episode、CoSyn 同生成模板／内容族也要避免跨边界。Cauldron 为其自有 benchmark 做过过滤，不证明它与我们的评测隔离。

**不要训练评测结果集**：本轮遇到的 Rune `decision-index-results`、JPT 同名结果包、ImajevBench、Jev-Omni 的 DecisionBench、NeoHorse 的 Image-NLI 评测表，以及 ImageJevBench 公开／密封题，都不能因为“可以下载”就直接计作训练来源。对公开 benchmark 反复调数据后，应记作开发证据，不能继续叫独立最终测试。

**先看最终 256 输入**：保留能真正看清目标、细线、标签的样本；如题目的答案依赖无法辨认的小字，不应给当前模型硬灌更多同类题。原图能看清不等于经过当前预处理仍能看清。

## 9. 建议的数据采用顺序和收益验证

### 9.1 首轮候选池：最多约 3 万题，不是承诺可以产出 3 万题

下面是给后续执行者的**采样预算建议**，不是已生成数据，也不是要求立即开训。要与负责人已经讨论的 v12 重训数据块合并审查；不要在原方案之外无上限叠加。

| 数据块 | 首轮候选上限 | 要补的能力 | 数据准备成本性质 |
|---|---:|---|---|
| PixMo-Points 新表达／新图 | 10,000 决策题 | 真实照片文字指代、条件计数、相近对象区分 | 中：图片获取、完整标注检查、点集到闭式题转换 |
| ABO | 8,000 决策题 | 新物体域、颜色／形状／部件、图文冲突 | 中：约 3GB 小图和信息可直接获取；主要成本为可见属性审核 |
| CoSyn 科学／数学／图示 | 8,000 决策题 | science 与图示结构，减少只靠文字猜题 | 中：闭式化与256可读性过滤；不必新生成图片 |
| VizWiz | 2,000 决策题 | 真实劣质图、可回答性、软目标 | 低至中：有人工答案，但必须处理不可回答语义 |
| OneJev 新域子集 | 最多 2,000 决策题 | 纹理、遥感、符号及其他缺失域的快速试验 | 低至中：现成格式；去掉与上面和已有数据重合的图 |
| **合计上限** | **30,000 决策题** | 过滤不足就降低总量，不以复制简单题补齐 | 不包含旧能力回放，不是新训练总量 |

这些数字是为了控制第一次筛选与试验的成本，**没有经过最优配比搜索**。OneJev 的 2,000 上限在过滤后可能达不到；它与其他候选共享的图只归入一个统计来源。规模最终应按“净新图、净新题、困难负例、每图题数”报告。

暂不优先投入：纯文本 Jev 数据、百万游戏帧、全量 GUI／OCR、全量 PD12M 教师打标。它们可能有价值，但没有直接证据表明比上述候选更能解决我们当前绑定和 science 的问题。

### 9.2 用最小对照回答“新数据有没有用”

后续如获准实施，建议沿用 v12 的初始化与旧能力保护规则，先保持结构、训练预算和回放策略一致，只更换一个数据块。此处不另定学习率、不启动实验。

| 假设 | 最小对照 | 观察指标 | 不能据此宣称什么 |
|---|---|---|---|
| PixMo 新监督改善真实绑定 | 等步数基线 vs 替换同量旧题为 PixMo 候选；均从 v12 | 未见图绑定题 balanced accuracy、条件计数、按图配对差值区间；真实图与错配图差 | 只提升合成左右题，不等于真实绑定解决 |
| CoSyn 能补 science | 等预算 science 旧题 vs 旧题加可读 CoSyn | science 分来源 accuracy/NLL；图示题单列；无图／错配图对照 | 只提高文本可解题，不能归因为看图能力 |
| ABO 扩展域而不破坏旧能力 | 等预算加 ABO 与原混合对照 | 新商品图属性、相似负例，以及原存在性／关系／科学面板 | 商品识别改善，不等于复杂多对象关系改善 |
| VizWiz 改善概率可靠性 | 有／无少量 VizWiz 的相同训练与校准口径 | 可回答性、软 gold NLL/Brier、准确率、ECE、拒答误判 | ECE 单独下降，不证明识别能力提升 |

统计按图、商品或场景族分组；保持校准集与最终测试分离。旧能力是否可接受沿用已经冻结的项目判定规则，不为本调研临时放宽。一次小规模对照只能决定是否值得扩大，不能证明所有种子、所有部署域稳定受益。

### 9.3 进入正式混合前，执行者应给出的数据清单

这是后续实施建议，并未在本轮执行：

1. 每来源原始数量 → 可解码 → 单图英文且任务兼容 → 图片隔离 → 标签筛选 → 最终保留的数量；同时给净新图片数。
2. 保留按能力、来源、gold、选项数分布，防止所有否定题或所有某一颜色都来自同一模板。
3. 抽查最终 256 输入；对条件计数、不可回答、商品属性、科学图四类分别记录标签错误与不可见答案。
4. 交付固定数据版本、来源／许可、划分与生成规则。需要教师时另列模型、费用及复核方案；不把教师置信度当真值保证。

## 10. 证据快照、限制与交付范围

### 10.1 可复核快照

榜单结果来自官方 JSON 的已发布 Core 分数；本次导出中的 `source_sha256` 为：

`7213259ec49a93875bdee6bfca3d05f2a8c2ca095115af51ed29ae393f3be038`

这是官方源结果的摘要标识，不是本地文件自行计算的 hash。[固定版本导出](https://benchmarkheaven.com/api/decision-results/imagejevbench/v0.3.0/json)

部分关键资源观察到的版本如下，用于调查溯源，不是要求在使用接口里新增 revision 参数：

| 资源 | 查询到的 SHA |
|---|---|
| Imajev-4B HF | `27d4e54fc11efeea2ea4d10e910643440d2333b5` |
| Imajev-9B HF | `1f209705e30009d32a6ff027ac56425d1ebd14e4` |
| Rune GGUF HF | `c6b360d47895bb77bdf3805a13ee5a5557ab1921` |
| Bonsai GGUF HF | `b072e1d3b35a0a630cece372c2127528e0994386` |
| AutoJev HF | `6f5b557e037f5edb25c7dc92dbc6553e5a19c015` |
| Jevify HF | `d4c0d1d455892957d274f68ec64e9fc0881011c8` |
| JPT-9B HF | `9fda2c7f123d36c4789c46c93737e80fc3a6c9fe` |
| Clef HF | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` |
| OneJev-Data HF | `c991b20e70359acb4448147c2ce99eecccd08ac4` |
| Laya Vision GitHub | `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c` |

这些是调研时当前资源的版本；除榜单明确固定的条目外，不自动证明与榜单 runner 当时使用的每个文件逐字节一致。模型卡后来更新的说明也不能追溯改变已测成绩。

### 10.2 已验证与未验证

**已做**：在线核对榜单版本与计分源码、官方模型卡／训练说明、公开仓库文件树、数据卡／分片元信息、部分构建代码；逐项区分训练与评测、图片与文本、行数与题数；核对主要求和及我们当前输入约束。

**未做**：完整训练语料下载、媒体可解码验收、全量标签抽验、与 deVision 全历史图像身份清单比对、过滤后规模统计、训练／推理测速，以及收益实验。公开入口存在不保证每一张外链图仍能下载；标注大小不等于媒体和缓存总大小。

“未找到公开训练包”限定于观察日期查询到的官方资源，不是证明世界上不存在任何其他副本。报告中的收益全部是基于监督内容和本项目缺口的研究判断，未用榜单分数承诺迁移后的涨幅。

**本轮交付只是一份研究文档。** 没有更改训练方案或运行状态，也没有把调研写入 critic 或触发 Engineer。
