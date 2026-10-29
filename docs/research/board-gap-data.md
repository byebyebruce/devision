# Decision Index 视觉榜差距：新建的训练数据

记录时间：2026-10-29。作者：Claude Code。状态：**只建了数据，没有混入训练、没有训练、没有评测**（r23 正在用 GPU，本次全部在 CPU 上做）。背景见 [decision-index-vision.md](decision-index-vision.md)（v0.4 公开视觉指数 8.7）。

**榜单题目和图片永远只评测**：本次所有数据都来自上游的 train 划分或其他来源，并逐图对照下面的排除索引；榜单行（`/home/bruce/ext/decision-index-data/vision`）只用来算哈希和读 id，没有一道题进训练文件。

## 1. 排除索引 `data/v05/board_exclusion.json`

- 脚本：`scripts/data/v05_board_exclusion.py`（公共函数在 `scripts/data/v05_board_common.py`）。
- 内容：榜单 12 个集合的全部 **11,542** 张图（含 BLINK / Winoground 双图题的第二张图）+ CharXiv **test** 的 1,323 张图（不在榜单里，但属于 CharXiv 的留出划分），每张记
  文件字节 sha256、解码后 RGB 像素 sha256、64 位 dHash、64 位 pHash；**32,417** 个来源 id：全部榜单条目 id 和图片名、CV-Bench 原始文件（COCO val2017 → `coco:<id>` 805 个、ADE20K val 633 个、Omni3D test 674 个，取自本地缓存的钉死 CV-Bench parquet）、Hateful Memes 全部 dev / test 行（dev_seen、dev_unseen、test_seen、test_unseen，3,640 个）、FUNSD test 文档 50 个、Mind2Web annotation id 142 个、CharXiv val + test 引用的 arXiv 论文 2,209 篇（`arxiv:<id>`，同一篇论文的其他图也排除）。
- 判定：字节相同、像素相同，或 **dHash ≤ 6 位且 pHash ≤ 8 位**（两个都要近）。榜单内部有 3,073 张图在这个规则下有近似孪生图（Mind2Web 同一页面不同标框、CV-Bench 同图多题、Hateful Memes 同底图换字、BLINK 成对图），说明规则抓到的是真重复而不是偶然相似。Hateful Memes 另加一条更松的规则（dHash ≤ 10 且 pHash ≤ 12），专门抓"同一张底图换了文字"的 confounder。
- `upstream_splits` 字段按基准列出榜单取题的上游划分（CV-Bench test = COCO val2017 / ADE20K val / Omni3D test；BLINK val；RealWorldQA test；CharXiv val；InfographicVQA val；Mind2Web test_website；Winoground test；CORD val + test；FUNSD test；Hateful Memes dev_seen + test_seen；R-Bench-M test；MMMU-Pro vision test）。各构建脚本只读 train 划分（Mind2Web 只读 `train`，不读 test_task / test_domain；Hateful Memes 四个 dev / test 划分都排除）。

### 审计：现有训练数据里已有榜单图

`--audit data/v05/exp132/train.jsonl`（**r23 正在用的训练文件**）：149,432 张图里 **88 张是榜单图，涉及 145 行**——79 张是 BLINK Spatial_Relation（BLINK 这部分取自 VSR，即 COCO train2014 图；来自我们的 coco-size 31、coco-exist 25、n2-spatial 20、vqav2 10、gqa 8 等来源），9 张是 CV-Bench（gqa 6、n2-spatial 2、tallyqa 1）。抽查的两对（COCO_train2014_000000015391 ↔ BLINK val_Spatial_Relation_138_1，tallyqa_0_4364 ↔ CV-Bench 711）是同一张照片。影响：BLINK 单图 387 题 / CV-Bench 2,038 题里各有不到 0.4% 的图在训练中见过（题目不同），对榜单分数影响很小，但**下一版混合数据应先用这个索引过滤**；r23 不动。

## 2. 新建的数据（`data/v05/board/<family>/{train,dev}.jsonl` + `MANIFEST.json`）

dev 是同一批 train 来源里按图（或按文档 / 任务 / 场景 / 论文）留出的部分，只用于训练中监测，不是榜单。MANIFEST 记录各步丢弃数、排除命中明细、按 kind 计数、金标位置分布和只看文字的基线（第一个 / 最长 / 最短选项、留一法的"同题型最常见答案"）。

| family | 对应榜单 | 来源（只用 train） | train | dev | 图片 | 排除命中 | 只看文字最好的基线 / 随机 |
|---|---|---|---:|---:|---:|---|---|
| `caption` | Winoground | COCO captions train2014 + 规则生成的难负例 | 103,602 | 4,315 | 66.9k | 98 张（BLINK 等，近似）+ 2,700 张项目留出图 | 0.522 / 0.50 |
| `hateful` | Hateful Memes | Hateful Memes train（`neuralcatcher/hateful_memes`） | 4,165 | 197 | 4,362 | **570 张**与榜单 meme 同底图（437 近似 + 133 松规则） | 0.502 / 0.50 |
| `kie` | KIE（CORD+FUNSD） | CORD v2 train 773 张收据 + FUNSD training_data 149 张表单 | 7,402 | 730 | 919 | **27 张** CORD train 与榜单 CORD val/test 像素相同 | 0.397 / 0.379 |
| `infographic` | InfographicVQA（也帮文档类） | InfographicVQA train（lmms-lab/DocVQA，2,507 张）+ DocVQA train（Cauldron，3,582 张） | 31,374 | 1,758 | 6,073 | 0 | 0.261 / 0.25 |
| `figure` | CharXiv（也帮 R-Bench-M / MMMU-Pro 的读图） | ArxivQA（GPT-4V 写的论文插图四选一） | 90,730 | 3,007 | 31.3k | 276 题属于 CharXiv 用过的论文 + 3 张近似图 | 0.397（同题型先验）、0.326（最长）/ 0.240 |
| `drive` | RealWorldQA | DriveLM-nuScenes train 前视相机 | 8,951 | 313 | 696 个场景 | 1 张（CV-Bench Omni3D nuScenes） | 0.389 / 0.346 |
| `som` | Mind2Web | Multimodal-Mind2Web `train`，按榜单格式画框标 A–D | 7,115 | 206 | 1,009 个任务 | 0 | 0.251 / 0.25 |
| `depth` | CV-Bench Distance、BLINK Relative_Depth | COCO train2014 室内照片 + CPU 上的单目度量深度教师 | 3,388（推理进行中） | 102 | 2.3k | 4 张 | 0.545 / 0.50 |

各 family 的题目怎么造：

- **caption**（Winoground 的 text 子任务形式："Which caption describes this image?"，两句选一）：真 caption + 由它改出的一句负例。五种规则（不用 LLM、不用 parser）：`swap_obj`（两个同单复数的名词在非对称关系词两侧互换，"a man riding a horse" → "a horse riding a man"；中间有 and / next to / with 等对称连接就不换），`swap_att`（同组属性互换，颜色对颜色、大小对大小，同一属性词出现两次不换），`replace_rel`（above↔below、in front of↔behind、left↔right…，句中已有两个方向不换；up/down 不用，"walking down the beach" 看不出来），`replace_att`（换颜色或反义属性），`replace_obj`（换成同超类、且**该图实例标注里没有**的 COCO 类别）。每图最多 2 题、每题用不同的真 caption，每类最多 3 万。种类分布：replace_att 28.8k、replace_obj 28.8k、replace_rel 26.8k、swap_obj 18.5k、swap_att 650（COCO caption 里两个"属性+名词"同时出现的很少）。
- **hateful**："Is this meme hateful?"，A = Yes、B = No（与榜单同顺序），不给 meme 文字（榜单也不给）。train 8,500 行里 1,756 张图 HF 仓库缺失；去掉同底图的 570 张后按标签配平（各 2,181）。
- **kie**：题型与榜单 kit 的公开转换器一致（MIT），但只用 train 文档。CORD：总额 / 小计 / 税 / 找零 / 某菜品价格四选一，干扰项是**同一张收据**上的其他金额，金标上下各几个按轮换安排（最大 / 最小不成为线索）；字段是否存在（税、折扣、服务费、现金、信用卡、找零、小计）是非题，按字段配平。FUNSD：表单字段值四选一（干扰项是同表单同类型——日期 / 数字 / 文字——的其他答案）和"字段是否填写"是非题（配平）。
- **infographic**：每个（问题，答案）一题，干扰项是**同一张图**上其他问题的答案（同为数字或非数字、同带或不带 %），且不是任何可接受答案的子串 / 母串；同图数字不够时最多补一个近邻数（补多了金标会成为"其他选项的中心"）。InfographicVQA 用 lmms-lab/DocVQA 的完整 train（Cauldron 版只有其中 2.1k 张图）；断言每行 `data_split == train`。同图答案不足 3 个的题丢弃（3 万题），所以 DocVQA 1 万张图只用上 3.6k 张。
- **figure**：ArxivQA 选项去掉 "A." / "B)" 前缀，标签（字母、"C) Four" 或选项原文）解析成唯一选项；金标位置轮换（原数据偏斜）；金标恰为最长选项的比例截到 28%（丢 5,740 题）。仍有文字线索：最长选项 0.326、同题型先验 0.397（随机 0.240）——GPT 生成的选择题的已知问题，混合时要配合对照（配错图 / 只看文字）看收益。
- **drive**：只用前视相机一张图。`describe`：红框标出车辆 / 行人 / 骑行者，"Which description matches the object in the red box?"，干扰项是同类别、**与金标没有共同颜色词**的其他描述（不会出现 "White truck." 对 "White commercial vehicle." 这种同义）；`light`：红框里的信号灯颜色（红 / 绿 / 黄）；`element`："Is there any traffic element in the front view?" 是非题（配平）。
- **som**：与榜单 Mind2Web 完全同格式（1080 px 高视窗、目标 + 3 个互不重叠的负候选、5 色框 + A–D 字母标签、Arial Bold、同样的 "Task / Previous actions / Which marked element…" 文本），只用 Mind2Web 官方 `train`。金标字母轮换，所以只看文字的基线就是随机。这和训练集里已有的 `mind2web`（动作短语选择）是同一批 train 任务的另一种问法。
- **depth**：COCO train2014 中只含室内类别、至少两个类别各只出现一次的照片，用 Depth-Anything-V2-Metric-Indoor-Small（CPU，6 线程约 0.6–0.8 s/张）估深度，取每个物体多边形内的中位深度。三种题：`distance`（CV-Bench 原句式，红框锚点、蓝 / 绿框两个候选，60° 视场反投影，远 / 近距离比 ≥ 1.5 且差 ≥ 0.4 m），`points`（BLINK 原句式，两个圈出的点 A / B，深度比 ≥ 1.4、局部平坦），`camera`（哪个框更靠近相机，比值 ≥ 1.3）。框超过画面 35% 的物体不问。**这是教师伪标签**，只保留差距明显的情况。12,000 张的推理还在后台跑（约 2 小时），跑完会自动重建 `build`（`data/v05/board_depth_build.out`），表里是 2,300 张时的部分结果。

## 3. 按榜单基准：覆盖了什么、还缺什么

| 基准 | 我们的 skill | 现在可用的训练数据 | 还缺的 / 为什么 |
|---|---:|---|---|
| CV-Bench（Count / Relation / Distance） | 14.4 | 已有：tallyqa、pixmo-count、coco-count、coco-relation、n2-spatial；新：`depth`（distance 363 题，部分） | Distance 只靠伪深度，题量小（3 个唯一类别、比值门槛下大部分图出不了题）；没有真 3D 数据（SUN RGB-D / Hypersim / ARKitScenes 体量大，且 Omni3D test 在榜单里，需要按 id 剔除）。计数数据已有，缺口更可能在混合比例而不是数据量。 |
| BLINK | −19.0 | 单图部分：新 `depth`（points 2.2k，部分）；Spatial_Relation 已有 VSR / coco-relation | 574 道双图题（Correspondence、Multi-view）**结构上要多图输入**，单图数据补不了。 |
| RealWorldQA | 4.0 | 新 `drive` 8,951（物体描述、信号灯、交通元素） | DriveLM 的问题多是开放式或依赖多帧（运动状态、规划），能单图判定的只有这几类；RealWorldQA 里的车道数、能否转弯、读路牌没有覆盖。DriveLM 标注和 nuScenes 图像都是非商用许可（CC BY-NC-SA 4.0）。 |
| CharXiv | 13.8 | 已有 chartqa 24k；新 `figure` 90.7k（ArxivQA） | 读小字需要分辨率，256 输入下大部分刻度 / 图例不可读（见 exp133 的切块实验）；ArxivQA 有文字线索。 |
| InfographicVQA | 16.4 | 新 `infographic` 31.4k（InfoVQA train 全量 + DocVQA） | 信息图很大、字很小，256 letterbox 后基本不可读，数据只能教格式和先验，真正的提升要靠分辨率 / 切块。 |
| Mind2Web | 5.7 | 已有 mind2web 动作短语 6k；新 `som` 7.1k（榜单同格式） | 同上，1280×1080 的截图缩到 256 后框和字母都很小。 |
| Winoground | 18.5 | 新 `caption` 103.6k | 规则负例里 swap_obj 有些不合常理（"a car stuck in a cat"），只看文字就能排除；没有 LLM 改写，Winoground 那种语义细的对照（"some plants surrounding a lightbulb" / "a lightbulb surrounding some plants"）覆盖有限。 |
| KIE（CORD+FUNSD） | 6.0 | 已有 funsd_binding 414；新 `kie` 7.4k | CORD train 只有 800 张、FUNSD 149 张；SROIE 在 HF 上是脚本数据集（`darentang/sroie` 只有加载脚本），没取。收据也要读小字。 |
| Hateful Memes | 4.5 | 新 `hateful` 4,165 | train 图在 HF 镜像缺 1,756 张；判断依赖读 meme 文字 + 语义；MAMI / HarMeme 没取。 |
| R-Bench-M | 2.7 | 已有 ScienceQA / AI2D / TQA；`figure` 间接相关 | 是需要多步推理的理科题，选项 A–F；没有找到同分布的 train 数据，本轮没建。 |
| MMMU-Pro vision | 0.8 | 无专门数据 | 题干和 10 个选项都**印在图里**、criteria 只是字母，要在高分辨率下读整段文字；256 输入下没有可行的数据方案。权重只有 0.25。 |

## 4. 来源与下载记录

- 下载脚本 `scripts/data/v05_board_fetch.py`（断点续传，进度写 `data/raw/board/fetch.log`），文件放 `data/raw/board/<name>/`，每个目录有 `REVISION`。HF 来源都经 `HF_ENDPOINT=https://hf-mirror.com`（目前 308 跳转到 huggingface.co，负责人已接受）：`naver-clova-ix/cord-v2@7f0115a4`（train 4 个分片）、`HuggingFaceM4/the_cauldron@847a98a7`（infographic_vqa、docvqa）、`lmms-lab/DocVQA@539088ef`（InfographicVQA train 24 个分片）、`princeton-nlp/CharXiv@f441eb63`（val + test，只用于排除索引）、`neuralcatcher/hateful_memes@d201c488`、`MMInstruction/ArxivQA@85a6dca0`（images.tgz 8 GB，解压到 `data/raw/board/arxivqa/images/`）、`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf@8078d68a`（模型，缓存在 `data/raw/hf`）。DriveLM 在 HF 上是 gated（403），改从 **ModelScope `OpenDriveLab/DriveLM`** 用 `curl -C -` 取，按列出的大小校验。COCO、FUNSD、Mind2Web 用本地已有的文件。
- 许可（只记录，不作为排除条件；本项目只做研究）：Hateful Memes 是 Facebook 的研究用许可（`data/raw/board/hateful/LICENSE.txt`）；ArxivQA CC BY-SA 4.0（数据卡）；DriveLM 标注 CC BY-NC-SA 4.0（数据卡），nuScenes 图像同为非商用；FUNSD 仅限研究；CORD、InfographicVQA / DocVQA、Depth-Anything-V2-Metric-Indoor-Small 的条款以各自数据卡 / 模型卡为准（本次没有逐一核对）；COCO 标注 CC BY 4.0。

## 5. 复现

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 uv run --no-sync python scripts/data/v05_board_fetch.py --root data \
    --only cord infovqa_full docvqa charxiv_test hateful arxivqa      # + drivelm（ModelScope，不需要 HF_ENDPOINT）
tar --no-same-owner -xzf data/raw/board/arxivqa/images.tgz -C data/raw/board/arxivqa
uv run --no-sync python scripts/data/v05_board_exclusion.py --root data [--audit data/v05/exp132/train.jsonl]
uv run --no-sync python scripts/data/v05_board_caption.py --root data
uv run --no-sync python scripts/data/v05_board_hateful.py --root data
uv run --no-sync python scripts/data/v05_board_kie.py --root data
uv run --no-sync python scripts/data/v05_board_infovqa.py --root data --subsets infovqa_full docvqa
uv run --no-sync python scripts/data/v05_board_arxivqa.py --root data
uv run --no-sync python scripts/data/v05_board_drive.py --root data
uv run --no-sync python scripts/data/v05_board_som.py --root data
CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/data/v05_board_depth.py infer --root data --limit 12000   # CPU，约 2.5 h
uv run --no-sync python scripts/data/v05_board_depth.py build --root data
uv run --no-sync pytest scripts/data -q -k board
```

排除索引变了（例如加了新的留出集）要先重建它，再重跑各 family。

## 6. 下一步（未做）

1. 混合：这些 family 体量差别很大（caption 10 万、figure 9 万 vs hateful 4 千），直接拼接会把训练拉向图文匹配和 ArxivQA。建议按能力配额取样（每个 family 几千到一两万），并保留现有 r23 数据的回放；混合前先用排除索引过滤现有数据（第 1 节的 145 行）。
2. 预写一个只加这些数据的短程对照（沿用 exp 编号），dev 用各 family 的 `dev.jsonl`，最终在榜单上评一次（只评测）；对 `figure`、`depth` 要看配错图对照，确认收益来自看图而不是文字 / 伪标签先验。
3. OCR 类（InfographicVQA、KIE、Mind2Web、MMMU-Pro）的上限受 256 输入限制，数据能做的是格式和先验，真正的提升要和分辨率 / 切块方案一起评估。
