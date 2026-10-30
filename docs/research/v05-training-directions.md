# v0.5 训练方向调研：文档 / 图示 / 图表 / UI / 计数与表面检查

日期：2026-10-27。作者：Claude Code（网络调研 + 读本仓库已有实验）。状态：**调研，不是实验结论**；没有下载数据集、没有训练、没有占用 GPU（exp114 正在跑）。

阅读前提：已读 [v0.5 策略](../experiments/2026-10-24-v05-research-driven-strategy.md)、[下一步方案](../experiments/2026-10-24-next-training-options.md)、[执行计划](../experiments/2026-10-23-v05-execution-plan.md)、[exp114 预写](../experiments/2026-10-27-exp114-v4-realmix.md)、[encoder-data](encoder-data.md)、[data-quality](data-quality.md)、[laya-vision-gap](laya-vision-gap.md)、[小模型调研](2026-10-24-small-vlm-landscape.md)、[文档/UI 数据](2026-10-23-v05-document-ui-data.md)、[Image Jev 前十数据](image-jev-top10-training-data-2026-10-12.md) 和训练日志顶部。

标注约定：**【证据】** = 论文 / 数据卡 / 本仓库 run 里的数字；**【推断】** = 我据此做的判断，未在本项目验证。外部数字都是生成式 VLM 在开放式任务上的结果，迁移到我们的「编码器 + 选项打分头」只能作为方向依据，不能当作预期增益。

---

## 0. 一句话结论

本仓库已经证明：**同域真实监督能显著提升**（exp81 图表 +21pp），而**单纯改分辨率 / token 数没有帮助**（exp78、exp79、exp83），前提是评测题本身不需要读小字。外部证据最一致的两点是：(1) 文本密集的真实数据和广覆盖的混合数据是小模型最主要的增益来源；(2) 文档 / 图表 / 小字任务的分辨率增益很大，但要靠**原生分辨率切块（tiling）**拿到，不能靠插值位置编码。所以推荐的顺序是：**先用 FineVision 的现成子集把文档 / UI / 计数的真实监督扩量（不改结构、CPU 延迟不变），再用一次预写的短程对照检验「全局图 + 2×2 原生 256 切块」**，同时把教师软标签和分域温度作为提升概率质量的手段。

---

## (a) 推荐方向（按优先级）

### 1. 文本密集真实数据扩量：FineVision 子集 + 同图干扰项（首选）

- **目标用途**：文档检查、UI 操作、图表决策；顺带覆盖计数。
- **【证据】**
  - 本仓库：exp81 只加 4.8 万道真实 ChartQA 派生四选一（加等量回放），图表开发集 31.7→52.7%（+21.0 [+17.3, +24.7]），保护集没有触发退化（`runs/exp81/report.json`）。这是 v0.5 唯一一次过了门槛的训练变化，而且只改了数据。
  - [FineVision](https://arxiv.org/abs/2510.17269)（HF，2025）：2,400 万样本、185 个子集，对 66 个公开基准做了去污染。用 460M 的 nanoVLM（SigLIP2-B-512 + SmolLM2-360M）同一配方训练，11 项基准平均 FineVision 49.6，Cambrian-7M 44.6，Cauldron 38.9，LLaVA-OneVision 35.9。去污染后重训，基线掉 2.7–3.7pp，FineVision 只掉 1.6pp。论文的结论是「保留覆盖面比激进的质量过滤更好」，把同一张图的多题合成多轮对话没有收益（[HTML 版](https://arxiv.org/html/2510.17269v2)）。
  - [Molmo/PixMo](https://arxiv.org/html/2409.17146) 表 3c：只用学术数据 11 项平均 72.5，加上合成文档 / 图表 PixMo-Docs 74.0，全部 PixMo 76.9，去掉 PixMo-Docs 75.8。
  - [MultiUI](https://arxiv.org/abs/2410.13824)（ICLR 2025）：730 万条网页 UI 样本，VisualWebBench 最多 +48%，Mind2Web 元素准确率 +19.1%，**而且能迁移到文档、OCR 和图表**。
  - [SmolVLM](https://arxiv.org/html/2504.05299v1)：256M / 500M 的数据以 Cauldron + Docmatix 为主，偏重文档。256M 的 DocVQA 58.3、ChartQA 55.6，500M 分别为 70.5、62.8。
- **本仓库已知的数据缺口【证据】**：UI 实验（exp82–85）只用了 ScreenQA 包装库 train **shard0** 的 4,058 题 / 1,605 屏（`docs/research/2026-10-23-v05-document-ui-data.md`），初筛后可用的只有 1,337 题。但 [rootsautomation/RICO-ScreenQA](https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA) 的完整 train 有 **6.9 万题**（另有 val 8.6k、test 8.4k，CC-BY-4.0，3.3 GB），而且每个答案都带 `ui_elements` 的 `bounds` 和 `text`；FineVision 的 `screenqa` 子集有 80,761 行。exp82 里 512 题 ×4 次重复后错图对照仍有 67.6%，说明问题出在独立样本太少，而不是方法本身【推断】。
- **做法【推断】**
  1. 从 FineVision 取 `docvqa`、`infographic_vqa`、`screenqa`、`textvqa`、`st_vqa`、`chartqa`、`sroie`、`invoices_receipts`、`visualmrc`、`tallyqa`，以及 `CoSyn_400k_{document,chart,table,nutrition}` 的一部分（行数和大小见表 b）。这些子集已经统一成 parquet + 对话格式，走 hf-mirror 按子集下载，不必整包下载（整包约 5 TB）。
  2. 开放答案转成选择题时，**干扰项从同一张图里取**：同屏其他 `ui_elements.text`（ScreenQA 原始版）、同一文档其他题的答案或同类型词元（日期换日期、金额换金额、对数字做 ±10% / 换位扰动）、同一张图表的其他标签或数值。不要从全库随机抽词，那样只看文字就能排除（data-quality.md 记录过 47.4% 文字频率先验的教训）。
  3. 混合比例从 exp81 的「新任务 50% + 回放 50%」起步。每用途 2–4 万题，先文档 + UI 两块（图表已有 exp81）；每图最多 4–6 题，按文档 / 应用 / 模板分组切 dev。
  4. 必须看图：每个新子集都报告「只看题目」基线和配错图对照。错图准确率比真图低不到 5pp 的子集，不进混合。
- **预期增益【推断】**：参考 exp81，同域的开发集准确率可能涨 +10 到 +20pp。但 DocVQA / InfoVQA 这类必须读小字的题，在 256 输入下上限明显受限（PaliGemma2 在 224 输入下 DocVQA 只有 39.9，见方向 2）。UI 和收据中字号较大的题收益会更好。
- **成本**：数据准备在 CPU 上做，数小时；训练按 exp81 的速度，每 5 万题约 15–20 分钟，12 GB 够用。**CPU 推理延迟不变。**

### 2. 原生 256 切块：全局图 + 2×2 局部图（对照实验，要先测可读性）

- **目标用途**：文档、UI 密集屏、图表的小字和刻度、多面板图示。
- **【证据】**
  - Molmo 表 2d：单张裁剪 11 项平均 62.8，多裁剪无重叠 75.7，多裁剪有重叠 76.9。**这是论文所有消融里最大的单项差距。**
  - [PaliGemma 2](https://arxiv.org/pdf/2412.03555)（3B）从 224 到 448：DocVQA 39.9→73.6，InfoVQA 25.2→37.5，ChartQA 人工题 42.0→54.0（[模型卡](https://ai.google.dev/gemma/docs/paligemma/model-card-2)）。
  - SmolVLM 把大图切成子图再加一张缩小的全图，最长边 1920。可学习的子图位置 token 明显好于字符串 token：字符串 token 会出现「OCR loss plague」（Finding 5）。[NVLM](https://arxiv.org/pdf/2409.11402) 的结论类似：加 tile tag 显著提升 ChartQA、DocVQA、OCRBench。
  - [FastVLM](https://arxiv.org/abs/2412.13303)：在同样 144 个视觉 token 下，TextVQA 比 ConvLLaVA 高 8.4%，DocVQA 高 12.5%。说明高分辨率的收益可以和 token 预算分开取得。
- **为什么不算重复本仓库的失败【证据 + 推断】**：
  - exp78 把位置编码**插值**到 384 后训了 2 万步，结果更差，CPU 慢一倍；但它的评测面板几乎没有必须读小字的题，文档里也写了「没有排除原生分辨率」。
  - exp79 / exp83 是同样 256 像素换成 256 个 token，**没有新增像素**。
  - 切块让每块都按 SigLIP2-256 的原生分辨率编码，不插值，而且带来真正的新像素，问的是另一个问题。
- **最小设计【推断】**：
  - 全局 256 letterbox 后得到 64 个 token（不变）。图像长边 ≥ 512 时，再切 2×2 带少量重叠的块，每块 256 送 SigLIP2、pixel shuffle 后各 64 个 token，共 320 个视觉 token。
  - 每块加一个可学习的块位置嵌入，不用文字标记。投影层由四块共享，从 v0.4 初始化。
  - 小图不切块，行为与 v0.4 完全相同，照片类旧能力的风险最小。
- **先做零成本检查**：统计五个用途 dev 图片在 256 letterbox 后的字高分布（可以用 FUNSD / ScreenQA 的框高换算）。字高 < 6 px 的题占多数时才值得跑。
- **成本**：
  - 训练：每步视觉编码 ×5，文本序列从约 150 增加到约 400 token，估计每步慢 2.5–3 倍，12 GB 下要减小 micro batch 并加梯度累积。
  - **CPU 延迟【推断，必须实测】**：v0.4 在 256 下 P50 278 ms（exp78 表）；切块后估计 0.7–1.1 s，只对大图生效。可选缓解：块内改用 4×4 shuffle（每块 16 token）；或者只在文档 / 截图这类图上切块（宽高比和边缘密度的启发式），照片保持原路径。
- **NaFlex 备选**：[SigLIP 2](https://arxiv.org/pdf/2502.14786) 的 NaFlex 变体保持宽高比，在 OCR / 文档 / 屏幕类检索上优于固定方形版本，序列短时差距更大；但 **B 尺寸的 NaFlex 在自然图像上不如标准版**。因此它只适合当「文档路径」的编码器备选，不直接替换现有塔。

### 3. 大模型教师的选项软标签蒸馏（提升概率质量，并利用开放答案数据）

- **目标用途**：五个用途的校准，以及原本只有开放答案、难以给出可靠 gold 的数据（VizWiz、长答案 UI / 文档题）。
- **【证据，弱】**：
  - 软标签蒸馏在分类上普遍有效，例如 [VL2Lite](https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_VL2Lite_Task-Specific_Knowledge_Distillation_from_Large_Vision-Language_Models_to_Lightweight_CVPR_2025_paper.pdf)（CVPR 2025）从 VLM 蒸到轻量分类器，最高 +7%。[2508.20224](https://arxiv.org/pdf/2508.20224) 讨论了教师校准对学生的影响。
  - **没有找到「大 VLM → 小模型多选打分头、并报告 ECE」的直接文献。**
  - 本仓库：r21-soft 的软标签（0.9 / 0.1）让准确率最好，但存在类和 size 变得欠自信（训练日志 r21 一节）。说明软目标会直接改变输出尺度，必须按题目逐个给出，不能全局固定。
- **做法【推断】**：
  - 等 exp114 结束、GPU 空闲后，用 4-bit 的 Qwen2.5-VL-7B / Qwen3-VL-8B（或较小的 Qwen3-VL-4B）在原分辨率下，对每道题读取各选项字母的 logit 并归一化，得到教师分布。
  - 训练目标设为 `α·gold + (1−α)·teacher`，α 从 0.7 起；教师答错的题把 gold 权重提到 1。
  - 教师在高分辨率下能读小字，这正好补我们 256 输入的短板。
  - 负责人 2026-10-21 曾暂缓教师蒸馏（`docs/experiments/2026-10-21-v04-plan.md`），重启需要负责人确认。
- **成本**：教师推理估计每题 0.3–1 s，5 万题约 4–14 小时 GPU 时间，只能离线串行。学生训练成本不变，**CPU 延迟不变**。
- **风险**：教师自身的选项位置偏置。教师要做正序 / 倒序两次取平均。

### 4. 计数与表面检查的专项数据（补空白用途）

- **目标用途**：计数、表面检查（目前就绪矩阵里完全没有表面检查域）。
- **【证据】**：
  - Molmo 的计数消融：直接计数在 CountBenchQA / PixMo-Count 上为 87.9 / 80.2，先指点再计数为 89.4 / 86.3。点序有序比无序高（86.3 对 74.1）。
  - 本仓库 r15 解冻视觉顶层后 CountBench +16.7；r19 / r20 的 TallyQA / CountBench 有大幅提升，说明现有架构能学计数。
  - [MMAD](https://arxiv.org/abs/2410.09453)（ICLR 2025）基于 MVTec AD / VisA 等 4 个数据集，共 8,366 张图、39,672 道多选题、7 个子任务，GPT-4o 只有 74.9%。
- **做法【推断】**：
  - 计数：PixMo-Count（本地已有 `allenai/pixmo-count`）和 TallyQA（FineVision 有 98,680 行，本地 `data/ext/tallyqa`）出数字四选一，干扰项取 gold±1、±2，平衡大计数。
  - 表面检查：VisA（10,821 张图，其中 1,200 张异常，带像素 mask）和 MVTec AD 出「有无缺陷」是非题、「缺陷类型」选择题和「缺陷位置（九宫格）」选择题。
  - **MMAD 只能当评测**：它的图来自同一批数据集，训练时要按图片 id 把 MMAD 用到的图剔除，否则就泄漏。
- **成本**：数据准备在 CPU 上做，训练是短程，**CPU 延迟不变**。缺陷都很小，256 下可能看不见；这类题是切块路径（方向 2）的候选受益者。

### 5. 分域温度 + 不用全局 label smoothing（便宜的校准改进）

- **【证据】**：
  - 本仓库 r17 显示 POPE 的最优温度约 6.5，VQAv2 类约 2，「一个题型一个温度不能同时适合」。r18 的全局 label smoothing 中途停掉；r21 的软化造成欠自信。
  - 外部研究也显示 MLLM 普遍校准不良，温度缩放是主要的事后手段（[2412.14660](https://arxiv.org/pdf/2412.14660)、[2604.17274](https://arxiv.org/pdf/2604.17274)）。
- **做法【推断】**：在现有的（题型, 选项数）温度桶之外，再加「图像域」桶（照片 / 文档截图 / 图表 / 示意图），在每个用途独立的校准集上拟合。图像域由一个线性头预测，或用简单的图像统计判断，推理时额外开销可以忽略。这需要改 spec（温度查找规则），要先改 spec 再实现。
- **成本**：几乎为零，**CPU 延迟不变**。属于发布前的收口步骤，不是能力提升。

---

## (b) 数据集优先级表

「HF / FineVision」列中的行数取自 FineVision 数据卡（`hf-mirror.com/api/datasets/HuggingFaceM4/FineVision`，revision `3c380a7`，单位是**行 = 图片 + 多轮问答**，不是题数）。转换难度：易 = 原生就是选择或是非题；中 = 开放的短答案，可以从同图取干扰项；难 = 长答案或动作类，需要教师或人工整理。

| 优先 | 数据集 | 规模 | 覆盖用途 | 转换 | 许可（只记录） | 获取 | 备注 |
|---|---|---|---|---|---|---|---|
| P0 | **RICO-ScreenQA（完整版）** | train 6.9 万题 / val 8.6k / test 8.4k；FineVision `screenqa` 80,761 行（44.9 GB） | UI | 中：答案带 `ui_elements` 的文字和框，**同屏其他元素就是天然干扰项**（另可出「哪个元素在上 / 下」） | CC-BY-4.0 | `rootsautomation/RICO-ScreenQA`（3.3 GB） | 本仓库只用过 shard0 的 4k 题；按 app 切分 |
| P0 | **DocVQA** | FineVision 10,189 行（12 GB）；原始 1 万图 / 3.9 万题 | 文档 | 中：数字 / 日期 / 名称的类型化干扰 | 研究用途 | FineVision `docvqa`；`lmms-lab/DocVQA` | 256 下小字可读性差，是切块方向的主要测试集 |
| P0 | **SROIE + invoices_receipts** | 33,616 + 3,013 行 | 文档（收据） | 中：金额 / 日期 / 商户字段，用同票其他字段做干扰 | SROIE 研究用途 | FineVision | 正对「收据检查」 |
| P0 | **ChartQA（继续用）+ PlotQA** | ChartQA 18,265 行；PlotQA 157,070 行（8.9 GB），原始 2,890 万题 | 图表 | 中：数值 ±扰动、同图其他图例 | ChartQA GPL-3.0；PlotQA CC-BY-4.0 | 本地已有 ChartQA；FineVision `plotqa` | exp81 已证明 ChartQA 有效；PlotQA 是合成真实数据表，可补量 |
| P0 | **TallyQA / PixMo-Count** | TallyQA 98,680 行；PixMo-Count 在本地 | 计数 | 易：数字选项 gold±k | TallyQA 研究用途；PixMo ODC-BY | 本地 `data/ext/tallyqa`、`data/raw/hf/...pixmo-count` | 已部分用过（r19 / r20），这次扩到大计数 |
| P1 | **InfographicVQA** | 4,394 行（4.5 GB）；原始 2.4 万训练题 | 文档、图示 | 中 | 研究用途 | FineVision `infographic_vqa` | 小字多，单图很大，依赖切块 |
| P1 | **TextVQA + ST-VQA** | 21,943 + 17,247 行 | 真实照片里的文字（也对应 VizWiz 场景） | 中：同图 OCR 词作干扰 | CC-BY-4.0 / 研究用途 | FineVision | 真实照片 + 文字，对 RealWorldQA 有利【推断】 |
| P1 | **VisA / MVTec AD** | VisA 10,821 图（1,200 异常）；MVTec AD 约 5.4k 图 | 表面检查 | 易：有无缺陷、缺陷类型、九宫格位置（由 mask 算） | VisA CC-BY-4.0；MVTec CC BY-NC-SA | 官方站点（[VisA](https://github.com/amazon-science/spot-diff)） | 剔除 MMAD 用过的图；MMAD 作为评测 |
| P1 | **CoSyn-400k（chart / document / table / nutrition / diagram）** | 116,814 / 71,282 / 46,518 / 6,931 / 34,963 行 | 图表、文档、图示 | 中：问题由模型生成，题质量要抽查 | ODC-BY | FineVision `CoSyn_400k_*`；`allenai/CoSyn-400k` | [CoSyn](https://arxiv.org/abs/2502.14846)（ACL 2025）在 7 项文本密集基准上报告了 SOTA；合成数据，回放比例要控制 |
| P1 | **Misviz-synth / Misviz** | 合成 81,814 图，真实 2,604 图，12 类误导手法 | 误导图表 | 易：「图是否误导」是非题，「哪种误导」选择题 | 见仓库 | [GitHub UKPLab](https://github.com/UKPLab/arxiv2025-misviz)（[论文](https://arxiv.org/abs/2508.21675)） | 真实集只用于评测；合成集用于训练 |
| P2 | **Misleading ChartQA** | 3,026 题、21 类误导、多选 | 误导图表 | 易（原生多选） | 见论文 | 论文初版只公开样例 | [2503.18172](https://arxiv.org/abs/2503.18172)（EMNLP 2025），**仅评测** |
| P2 | **VisualMRC / SlideVQA / pdfvqa** | 3,027 / 1,919 / 8,593 行 | 文档、多面板 | 中到难 | 研究用途 | FineVision | 题少，单图大 |
| P2 | **screen2words / groundui** | 15,730 / 13,531 行 | UI | 中：屏幕摘要四选一（干扰取其他屏的摘要，同 app 更难）；groundui 的「指令→元素」可以出成元素四选一 | CC-BY-4.0 等 | FineVision | 正对「agent 操作」中的元素选择 |
| P2 | **MultiUI** | 730 万条 | UI → 文档 / 图表迁移 | 中 | 需在 HF 上同意条款 | `neulab/MultiUI` | 外部证据强；量大，先取 5–10 万 |
| P2 | **VizWiz** | FineVision 6,604 行；官方 train 20,523 | 真实照片（VizWiz 评测） | 难：开放答案；`unanswerable` 不能直接当 no | CC-BY-4.0 | FineVision / 官方 | 适合配合教师软标签（方向 3） |
| P3 | **DVQA / FigureQA / MapQA** | 20 万 / 10 万 / 37,417 行 | 图表、地图 | 易 | 研究用途 | 本地 Cauldron 已有 | **v12-ext 已训过，收益很小；MapQA 在 r19 伤了 ScienceQA**，不再加 |
| P3 | **Docmatix / DoclingMatix** | 240 万图 / 950 万问答；DoclingMatix 127 万行（961 GB） | 文档 | 中 | MIT / 见卡 | HF | 量过大；256 下多数文字读不清，等切块路径验证后再考虑 |
| 仅评测 | CharXiv、VisualWebBench、ScreenSpot、MMAD、RealWorldQA、NaturalBench、MMBench、MME、CV-Bench、SEED-Bench | — | — | — | — | — | 不进训练；FineVision 声称已对 66 个基准去污染，**我们仍要按图片 id / 哈希自己剔除** |

---

## (c) 不要做的事

**本仓库已经失败或已有结论的：**

1. **插值位置编码把输入放大到 384 / 512**：exp78 在 2 万步后更差（−2.5 [−3.5, −1.4]），CPU 慢 2 倍；2026-10-06 的 512+4×4 合并也失败过。要提高分辨率只走原生切块（方向 2）。
2. **同样像素下改成 256 个视觉 token**：exp79 P1 +0.13 [−0.78, +1.04]，exp83 在 UI 上 −1.95。不再重复。
3. **冻结视觉塔**（exp84）、**用少量 UI 题重复多次**（exp82 的 512 题 ×4、exp85 的 256 题 ×8）：能拟合但不泛化。
4. **D3 / SmolVLM 原生骨干的末层适配**（exp86、exp94/96、exp110–112）：科学图示落后 v0.4 9–11pp，已停止。
5. **RLCD 替代 CE**：exp80 没有检出差异，继续用 CE。
6. **全局 label smoothing / 统一软化硬标签**：r18 中途停止，r21 造成欠自信。软目标只在逐题有依据时用（教师分布）。
7. **旧块 KL λ=1 的自蒸馏**（exp18）：不原样重跑。方向 3 是外部教师，不是同一个方法。
8. **再加 DVQA / FigureQA / MapQA 这类合成两两比较图**：v12-ext 收益约 +1pp，MapQA 伤了 ScienceQA。

**外部证据不支持的：**

9. **混入纯文本 SFT 或较多 CoT**：SmolVLM 发现复用 LLM 的文本 SFT 会让小模型图像任务最多 −6.5%，CoT 超过 0.05% 就明显变差（Finding 7、8）。
10. **按质量分激进过滤**：FineVision 发现保留覆盖面更好。
11. **用文字串标记子图位置**：SmolVLM 发现这会导致 OCR loss plague，应该用可学习的位置嵌入。
12. **把 MMAD / CharXiv / VisualWebBench / ScreenSpot 当训练数据**：它们是评测集。MMAD 的图与 VisA / MVTec 重叠，必须剔除。
13. **整包下载 FineVision 或 Docmatix**：前者约 5 TB，后者大部分文字在 256 下不可读。只按子集取。

---

## 建议的下一步（等 exp114 出结果后，由负责人决定）

1. 若 exp114 通过：在它的基础上做方向 1 的「文档 + UI」扩量（完整 ScreenQA + DocVQA / SROIE / receipts + TextVQA，每块 2–4 万题，同图干扰），配方与 exp81 相同，预写判据。
2. 同时在 CPU 上做方向 2 的字高统计（零 GPU）。字高普遍 < 6 px 时，预写切块短程对照（与 v0.4 同数据、同曝光），并实测 CPU P50 / P95。
3. 计数 / 表面检查数据（方向 4）随第 1 步一起准备，单独出 dev 面板。
4. 方向 3（教师）和方向 5（分域温度）需要负责人确认；方向 5 要先改 spec。

## 来源

- FineVision：https://arxiv.org/abs/2510.17269 ，https://arxiv.org/html/2510.17269v2 ，数据卡 https://huggingface.co/datasets/HuggingFaceM4/FineVision
- SmolVLM：https://arxiv.org/html/2504.05299v1 ；博客 https://huggingface.co/blog/smolervlm
- Molmo / PixMo：https://arxiv.org/html/2409.17146
- PaliGemma 2：https://arxiv.org/pdf/2412.03555 ，https://ai.google.dev/gemma/docs/paligemma/model-card-2
- SigLIP 2（NaFlex）：https://arxiv.org/pdf/2502.14786
- NVLM（tile tag）：https://arxiv.org/pdf/2409.11402
- FastVLM：https://arxiv.org/abs/2412.13303
- MultiUI：https://arxiv.org/abs/2410.13824 ，https://huggingface.co/datasets/neulab/MultiUI
- CoSyn：https://arxiv.org/abs/2502.14846
- RICO-ScreenQA：https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA ；ScreenQA 论文 https://arxiv.org/abs/2209.08199
- Docmatix：https://huggingface.co/blog/docmatix
- InfographicVQA：https://arxiv.org/pdf/2104.12756
- ChartQA：https://arxiv.org/pdf/2203.10244
- Misleading ChartQA：https://arxiv.org/abs/2503.18172 ；Misviz：https://arxiv.org/abs/2508.21675
- MMAD：https://arxiv.org/abs/2410.09453 ；VisA：https://github.com/amazon-science/spot-diff
- NaturalBench：https://arxiv.org/abs/2410.14669
- VL2Lite：https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_VL2Lite_Task-Specific_Knowledge_Distillation_from_Large_Vision-Language_Models_to_Lightweight_CVPR_2025_paper.pdf ；教师校准：https://arxiv.org/pdf/2508.20224
- MLLM 校准：https://arxiv.org/pdf/2412.14660 ，https://arxiv.org/pdf/2604.17274
- SeeClick / ScreenSpot：https://arxiv.org/abs/2401.10935
- VizWiz：https://arxiv.org/abs/1802.08218

未核实：PixMo-Count 的确切行数、TallyQA 和 MVTec AD 的确切规模、InfographicVQA 许可、Misviz 是否有 HF 镜像，这些都以官方页面为准，下载前再查。
