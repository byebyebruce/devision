# 小型开源视觉模型调研：deVision 可借鉴的架构与训练路线

记录时间：2026-10-24（Asia/Shanghai）。以下来源均于 **2026-10-24 联网核验**。本轮只调研公开原始资料、读取配置和推理代码；**没有在本地运行这些外部模型**。数值为作者报告或 Hub 元数据，不能视为 RTX 5070 实测。

## 结论与候选顺序

最接近我们接口与产品目标的是 **d3-lite**：直接输出每个选项概率、零文本生成，0.853B，具备原生图像输入。最有价值的小型通用视觉基座是 **Qwen3.5-0.8B、SmolVLM-500M / 256M**；InternVL3.5-1B 是略超 1B 的参考上界。FastVLM 的高分辨率低 token 思路值得学习，但公开权重的研究用途许可限制使它不适合直接成为我们的产品发布底座。Florence-2 更适合辅助构造区域、OCR、定位监督。

这是一份有边界的候选清单，不是“已穷尽市面所有模型”的榜单。尤其要区分三类：直接决策模型、可改造成决策模型的生成式 VLM、提供视觉监督的任务模型。生成式模型高分不能直接证明它具有经过校准的选项概率。

## 1. 模型身份、参数口径与开放情况

表中“精确总量”取官方 Hugging Face API 的 `safetensors.total`，是已登记权重元素数；不同模型可能含共享权重、独立读出或训练辅助组件，不能解释为完全相同口径的 FLOPs。模型名中的 0.8B、1B、2B 不一定是视觉与语言总参数。

| 模型 | 精确总量 / 公开分解 | 开放情况 | 我们优先参考什么 |
|---|---|---|---|
| d3-lite | 852,985,920；卡片称含视觉约 0.10B | Apache-2.0；权重、推理代码；未在模型卡找到完整训练配方 | 原生图像决策概率接口、预训练 VLM 改造为零生成读出 |
| Qwen3.5-0.8B | 873,438,784（含视觉及发布组件） | Apache-2.0；基座与后训练权重、配置公开 | 原生多模态初始化、混合注意力、小参数 OCR / GUI 能力 |
| Qwen3.5-2B | 2,274,069,824 | Apache-2.0 | 本地能力上界 / 教师候选，已超 2B 总量 |
| SmolVLM-256M-Instruct | 256,484,928；SmolLM2-135M + 约 93M SigLIP + 连接部分 | Apache-2.0；训练工具与公开数据混合 | 同等低 token 预算下提高原始图像分辨率 |
| SmolVLM-500M-Instruct | 507,482,304；SmolLM2-360M + 约 93M SigLIP + 连接部分 | Apache-2.0 | 与我们约 0.518B 最接近的生成式对照 |
| InternVL3.5-1B | 1,060,897,792；卡片四舍五入视觉 0.3B、语言 0.8B、合计 1.1B | Apache-2.0；分阶段权重公开 | 动态分辨率、分阶段训练与能力保持 |
| Qwen3-VL-2B-Instruct | 2,127,532,032 | Apache-2.0 | 成熟较强基线；不是当前最小的 Qwen 视觉模型 |
| Moondream2（2025-06-21） | 当前主分支 API 1,927,237,104；历史“1.86B”不可套所有版本 | Apache-2.0；权重、推理代码；完整训练数据不可据此认定公开 | 定位、指点、UI 与视觉问答联合训练 |
| Florence-2 base / large | 231,567,705 / 776,721,497 | MIT；权重、任务实现公开 | OCR 与区域绑定辅助任务，不直接替代通用决策模型 |
| FastVLM-0.5B / 1.5B | 名称指语言规模；FastViTHD 论文约 125.1M，还需连接器；不能称总量 0.5B | 代码 Apple 许可；**官方权重仅研究用途** | 视觉编码器实际延迟、高分辨率与 token 数联合优化 |
| MobileVLM V2-1.7B | 官方名 1.7B，语言 MobileLLaMA-1.4B；CLIP + LDPv2 | 代码 Apache-2.0；上游模型 / 数据仍遵原许可 | 早期端侧工程与轻量下采样连接器；非当前首选 |

参数元数据的原始来源：各模型 [Hub API](https://huggingface.co/docs/hub/api) 对应 `https://huggingface.co/api/models/<repo>`。关键身份固定如下，供后续下载 / 实测使用；此处仅读取元数据，未下载完整权重：

| repo | 本轮读取主分支 revision |
|---|---|
| vllm-sr/d3-lite | `03560c26f5199c3c04129a2ccb38960c913635e0` |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |
| Qwen/Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| HuggingFaceTB/SmolVLM-256M-Instruct | `7e3e67edbbed1bf9888184d9df282b700a323964` |
| HuggingFaceTB/SmolVLM-500M-Instruct | `a7da5b986cb59b408707209984f360a5f4ad7e47` |
| OpenGVLab/InternVL3_5-1B | `2f71cf52542334823e48a46ffba0e2bc9add3446` |
| Qwen/Qwen3-VL-2B-Instruct | `89644892e4d85e24eaac8bacfd4f463576704203` |
| vikhyatk/moondream2 | `5d6c926f44e26b07957b0dd315bbedcb4c17a5fe` |

## 2. 直接同类：d3-lite

**验证：2026-10-24。** 官方卡片描述其在 Qwen3.5-0.8B 上训练为决策模型，支持 choice / noul / score、图片和视频，每项返回概率，无文本生成。卡片报告 Jev Decision Index 视觉榜 **0.3.2：42.1**，公开 47.9、私人 36.3，榜单日期 2026-10-12。这是作者引用官方榜的陈述；本轮尚未独立执行该评测，**也不是我们 v0.5 文档旧 ImageJevBench 300 题协议的分数**。不能用它改写 46.1 的旧验收目标。[官方模型卡](https://huggingface.co/vllm-sr/d3-lite)

源码比模型卡更能说明适配路径：Qwen3.5 主干 + **255 × hidden 的 answer-code 读出矩阵**，在最后提示位置读取各候选 code 的 logits，FP32 读出与 softmax。`decision_config.json` 为 `pooling=last`、温度 1、`attention_mode=noncausal_full_attention`。**去因果作用于 full-attention 层，DeltaNet 层仍保持因果**；不能把它描述为整个主干全双向。每问有独立序列，可批处理；视觉特征可复用，不是所有问题一次融合后共同判定。[modeling_d3.py](https://huggingface.co/vllm-sr/d3-lite/blob/main/modeling_d3.py)、[decision_config.json](https://huggingface.co/vllm-sr/d3-lite/blob/main/decision_config.json)、[d3_runtime.py](https://huggingface.co/vllm-sr/d3-lite/blob/main/d3_runtime.py)

原图保持纵横比，处理器支持约 65,536～1,638,400 像素，每 32×32 对应一个输入 token；图像先于文本。这个输入上限远高于我们固定 256×256。它的视觉能力差异不能只归因于决策损失或读出头。[运行实现](https://huggingface.co/vllm-sr/d3-lite/blob/main/d3_runtime.py)

**可迁移推断：** 优先实测它在我们的同题同选项、换图和概率指标上是否真有优势；若优势稳定，证明“已有多模态主干 + 轻量决策改造”值得进入独立实验分支。其完整训练集、混合比例、超参和消融证据本轮未查到，不能声称已复现其训练方式。

## 2.1 新增精确同类：OneJev、Intern-Decision、JPT

**验证：2026-10-24。** 三者都由 Qwen3.5-0.8B 派生、直接返回候选概率；Hub API 均登记总量 **852,985,920**，不能因都叫0.8B就假定训练和推理实现相同。

| 模型 | 已核实的实现 / 训练区别 | 开放与限制 |
|---|---|---|
| [OneJev1.6-0.8B](https://github.com/OmniJev/OneJev) | 2026-10-17发布；单forward多字段；公开训练代码和94,707条可再分发训练数据，原始99,193条；README配方冻结视觉、1epoch、LR5e-6、CE+Brier | Apache-2.0；需检查数据源许可及我们评测图片交集；公开训练样本与所发布版本的完整重建关系仍须追踪 |
| [Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) | 因果forward，完整JSON骨架各`<decision>`前一位置读取候选symbol；多字段一次计算；温度2.74776独立校准 | Apache-2.0；注意骨架前一位置而非placeholder自身，避免引入答案或错位 |
| [JPT-0.8B](https://huggingface.co/kirp/jpt-0.8b) | LoRA训练后合并，视觉塔不变，单次prefill决策；独立温度1.140 | **CC-BY-NC-4.0，非商业许可**；可作研究对照，不直接作商业发布底座 |

这几项说明“已有多模态预训练 +候选读出 +校准”已经是实际存在的训练路线，不能把小型视觉概率模型视为空白领域。OneJev的公开数据与配方比仅发布推理权重更适合跟进；但公开文本Decision Index结果、视觉Decision Index与旧ImageJevBench不可混比。本轮不另列未经同口径复现的排名。

固定 revision：OneJev `0acd65660ac22ca42f40b8b6a6da3a039590b412`；Intern-Decision `42457fafd13d10d0b0d9be46cb30a23caa409994`；JPT `1431c0509bbc10772cd59964cc7af5835c8720d6`。数值来自各官方Hub API；实现与许可来自上表链接的原始发布者资料。

## 3. 同尺度参照：SmolVLM-256M / 500M

**验证：2026-10-24。** 两个版本都使用约 93M SigLIP-B/16 视觉塔，512×512 输入，SmolLM2 语言模型。已核对配置：`image_size=512, patch_size=16, scale_factor=4`，所以 **32×32=1024 视觉 patch，经 4×4 合并成为 64 个视觉 token / tile**。高分辨率图还可切成多个 tile，不能把整张任意尺寸图均计为 64 token。我们当前 256×256 ->256 patch ->64 token；两者 token 数相同，送入视觉编码器的像素信息相差四倍。[256M 配置](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/config.json)、[500M 配置](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/blob/main/config.json)

官方 500M 卡描述训练混合为 The Cauldron + Docmatix，强调文档、caption，并覆盖图表、推理和指令；文档约 25%、caption 约 18%。这提示我们学习**监督覆盖与信息密度**，而不是只把少数 UI 样本多次重复。公开工具可作为配方入口，但不能据此认定每个上游样本都符合本项目的数据隔离。[模型卡与训练入口](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)

论文表 1 报告（256M / 500M）：ChartQA **55.6 / 62.8**，DocVQA **58.3 / 70.5**，TextVQA **50.2 / 60.2**，MMMU **29.0 / 33.7**。均为原任务评测，不可直接与我们经过选项重构的 ChartQA 52.7% 比较。论文还观察到：小模型不一定受益于更大视觉塔；过量 CoT 文本可损害紧凑 VLM 视觉表现。这约束了“换更大塔 / 加长思维链即可解决”的假设，并不证明我们所有场景都应禁用推理监督。[技术报告 §§2–4](https://arxiv.org/html/2504.05299v1)

**可迁移推断：** 先比较 256 输入 +64 token 与 512 输入 +64 token，固定决策主干和评测；这是新的“像素信息”变量，与 exp83 的 256 输入 +256 token 不同。正式开训前必须检查位置编码 / 预训练分辨率迁移，不能直接改图像大小就认定可用。**历史2026-10-06结构筛选已经测过D=512输入+64token并失败；重试必须明确新的真实UI任务、预训练初始化或监督条件，不能把同条件当新实验。**

## 4. 当前轻量通用基座：Qwen3.5 与 Qwen3-VL

**验证：2026-10-24。** Qwen3.5-0.8B 支持原生视觉，默认非思考，可同时获得基座与后训练版本。其小模型是**稠密**模型，不能将模型族关于 MoE 的宣传套到 0.8B。已核对配置：语言宽 1024、24 层，每四层含三层 Gated DeltaNet +一层完整 attention；视觉 12 层、宽 768、patch16、spatial merge2。不是“把纯文本 0.8B 临时接上随机视觉连接器”。[配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)、[官方模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)

官方非思考结果（0.8B / 2B）：RealWorldQA **61.6 / 71.2**，MMMU **47.4 / 64.2**，AI2D **68.7 / 81.5**，OCRBench **79.1 / 85.4**，ScreenSpot Pro **46.5 / 54.5**。Thinking 与 non-thinking 必须分开。卡片明确采样设置；这些值用于筛选候选，不能直接与本项目 587 道 native-choice RealWorldQA 子集比较。[官方结果和评测设置](https://huggingface.co/Qwen/Qwen3.5-0.8B#benchmark-results)

Qwen3-VL-2B 仍值得作为成熟对照：官方结构改动包括视觉多层特征注入 DeepStack、位置编码与视频时间处理。它总量约 2.128B，且当前 Qwen3.5 已有更小视觉模型，所以不能只介绍 Qwen2.5-VL-3B 并据此说 Qwen 家族最小要 3B。[Qwen3-VL 官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、[技术报告](https://arxiv.org/abs/2511.21631)

**可迁移推断：** 对我们而言，优先复用经过视觉语言训练的表示，再保留概率决策接口；不必连同长文本生成习惯一起采用。模型卡没有公布足够完整、可在单张 12GB 卡复现的预训练配方；当前可执行的是局部适配与蒸馏，而非声称重建其预训练。

## 5. InternVL3.5-1B：动态细节与分阶段训练

**验证：2026-10-24。** 官方约 0.3B InternViT +0.8B Qwen3，总量约 1.1B。常规每个 tile 的 1024 patch token 压为 256；**Flash / ViR 才额外引入 64-token 路径和按内容路由**，不能把 Flash 的压缩或 4.05× 族级速度收益直接安到常规 1B 权重上。官方示例采用 448 tile、动态切图，tile 上限会显著影响成本。[模型卡架构章节](https://huggingface.co/OpenGVLab/InternVL3_5-1B#model-architecture)

公开 CPT、SFT、MPO、CascadeRL 阶段的权重便于研究监督路线；CascadeRL 为先离线 MPO、后在线 GSPO。它强调先形成可用多模态表示，再用后训练精炼，不支持“绕过基础视觉监督，先上 RL”这一结论。系列总体改进不能自动作为 1B 在某个具体任务上的收益。[训练阶段与分阶段 checkpoint](https://huggingface.co/OpenGVLab/InternVL3_5-1B#training-pipeline)、[技术报告](https://arxiv.org/abs/2508.18265)

**可迁移推断：** 一张图根据细节需求分配视觉预算值得做预算明确的对照；但单卡实时决策暂不需要复制其跨 GPU 视觉 / 语言拆分服务。

## 6. 其他高价值参照及限制

**FastVLM（验证：2026-10-24）。** FastViTHD 是卷积 / Transformer 混合高分辨率视觉塔，约 125.1M，核心是同时降低视觉编码延迟与后续 token 数。论文比较指出提高输入分辨率、保持适量输出 token 对文字密集场景有收益。85× TTFT 是指定高分辨率、同 0.5B 语言模型、M1 MacBook Pro 对照 LLaVA-OneVision 的结果，**不是对所有 VLM、所有硬件的通用倍数**。[论文 §§3–4](https://arxiv.org/html/2412.13303v1)、[官方仓库](https://github.com/apple-aiml-research/ml-fastvlm)

其官方 [`LICENSE_MODEL`](https://github.com/apple-aiml-research/ml-fastvlm/blob/main/LICENSE_MODEL) 明确仅限非商业科研，排除产品开发与商业产品服务；代码另有 Apple 许可。这里列为研究参考，不能标为 Apache / MIT 权重直接用于我们的发布。这是已核实的许可边界，不是泛化风险提示。

**Moondream2（验证：2026-10-24）。** 当前卡固定推荐 2025-06-21，包含 query / detect / point。版本说明报告通过高质量框监督与 RL 改进定位、UI，ScreenSpot F1@0.5 60.3 ->80.4；推理模式可选择显式空间 grounding，以延迟换精度。训练数据与完整配方未充分公开；不要将更新过的当前权重一律描述为最早 Phi-1.5 初始化版本。Moondream3-preview API 总量约 9.27B、许可证 `other`，不进入本轮小总量优先候选。[官方卡与版本说明](https://huggingface.co/vikhyatk/moondream2)、[Moondream3-preview](https://huggingface.co/moondream/moondream3-preview)

**Florence-2（验证：2026-10-24）。** DaViT 视觉编码器 +文本 encoder-decoder，将 OCR、定位、caption 等统一为任务提示和结构化文本输出；FLD-5B 约 126M 图、5.4B 标注。base 约 0.23B、large 约 0.77B、MIT。它的任务训练和区域标注很适合借鉴来建立“文字—框—区域”监督；原 checkpoint 不是任意问题的概率决策接口，不能据下游微调表称其开箱即用通用 Jev 模型。[论文](https://arxiv.org/abs/2311.06242)、[官方模型卡](https://huggingface.co/microsoft/Florence-2-base)、[配置](https://huggingface.co/microsoft/Florence-2-base/blob/main/config.json)

**MobileVLM V2（验证：2026-10-24）。** CLIP +轻量 LDPv2 +MobileLLaMA。官方两阶段均冻结视觉、训练连接器和语言；对齐阶段约 1.2M caption，多任务阶段约 2M 混合。V2-1.7B 作者报告 GQA59.3、TextVQA52.1。原论文的 Snapdragon / Orin tokens/s 属于特定端侧生成链路，不能转成我们单次概率决策速度；该方向有工程参考价值，当前优先级低于 d3-lite / SmolVLM / Qwen3.5。[官方仓库和训练数据入口](https://github.com/Meituan-AutoML/MobileVLM)、[V2 论文](https://arxiv.org/abs/2402.03766)

## 7. 速度与显存：可使用的证据边界

| 来源 | 作者报告 | 必须保留的条件 |
|---|---|---|
| d3-lite 卡片 | 新图片单问 warm p50 64.0ms、p90 64.9ms；AutoModel | 一张 AMD MI325X、ROCm7.2、PyTorch2.12、Transformers5.17，约 1.6MP，3 次预热、最多 20 次计时，单请求；FP32 读出、BF16 主干由运行代码规定 |
| d3-lite 服务卡片 | 518 requests/s、p99<1s | **文本单问**、批处理服务，不是图片吞吐；不得和前一行互换 |
| SmolVLM 论文表1 | batch1 显存约 0.8GB /1.2GB（256M/500M） | 作者推理配置，不是完整训练显存，也不是任意高分辨率最大 tile 的上界 |
| FastVLM 论文 | 指定对照最多 85× TTFT | Apple M1、高分辨率、同0.5B语言模型；不能外推RTX5070 |

来源：[d3-lite Serving performance](https://huggingface.co/vllm-sr/d3-lite#serving-performance)、[SmolVLM Table1](https://arxiv.org/html/2504.05299v1)、[FastVLM](https://arxiv.org/html/2412.13303v1)。对未明确一致的 batch、分辨率、精度、解码长度不做跨模型速度排行。

本地下一步统一测：同一 RTX 5070、batch1、固定问题 /选项、明确真实图片分辨率与模型视觉 token 数、同进程预热、CUDA 同步、p50/p95与峰值 allocated/reserved 显存。拆分预处理、视觉编码、决策主干及总端到端延迟；生成式基线分别测单步候选打分与自由生成，不能把长回答解码时间当成判别模型的公平速度基线。CPU / M4 性能另测，不从 CUDA 数据换算。

## 8. 对 deVision 下一方向的建议（推断，尚未执行）

1. **先测现成同类上界，再决定是否继续改当前主干。** d3-lite 是最直接的第一基线；SmolVLM-500M 与当前总规模相近，Qwen3.5-0.8B 是原生多模态备选。固定真实任务、换图对照和选项换序，联合看 accuracy /NLL/ECE与延迟；外部预训练数据未知意味着不能把已有本地集合当它们绝对干净的泛化测试。
2. **把像素可读性与 token 容量拆开。** 512输入 +64token 与256输入 +256token回答不同问题。先量化256缩图后文本高度与 OCR 可读率，再选高分辨率或局部视图；不能因 exp83 失败排除高分辨率路线，也不能忽略2026-10-06历史D已测512+64的失败条件。新的真实UI任务 /多模态预训练起点须提供新增信息收益。
3. **针对缺的绑定信息设计监督，而不只堆次数。** 在真实 UI /文档中增加文字、位置、区域配对，保存困难负例、同题不同图、选项位置平衡；Florence /Moondream的任务设计可参考，模型生成框只能作为待核验伪标签，不替代人工 gold。
4. **保留已有效的概率接口，先做监督与保持能力，再考虑复杂 RL。** 对旧任务保留回放 /教师约束，并监测逐域退化。使用经过视觉语言对齐的骨干可能比从纯文本编码器反复补能力更省实验预算，但须通过同等预算对照证明。

这些建议由主实验计划落实预算、对照、checkpoint和停止条件；本文件没有授权发布，也不声称这些路线已经通过 v0.5 验收。Laya /ModernVBERT /decider /Vision-JEV 的精确同类源码分析由本次主调查另行汇总，避免把生成式 VLM 清单误当成全部视觉决策模型市场。
