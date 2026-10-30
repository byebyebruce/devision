# Decision Index 视觉榜：v0.4 的外部位置

记录时间：2026-10-28。只评测、不训练；不提交榜单、不向 kit 仓库提 PR。**这些题目和图片永远不进我们的训练数据**（数据放在仓库外 `/home/bruce/ext/decision-index-data`）。

## 1. 榜单与 kit

- Kit：<https://github.com/apolinario/decision-index>（MIT），本地 clone 在 `/home/bruce/ext/decision-index`，commit `25c6ec7`（2026-10-20），独立 venv（`uv venv`，阿里云 PyPI 镜像，`pip install -e '.[vision]'`）。说明见 kit 的 `docs/vision.md`。
- 线上是 Decision Index 0.4；视觉榜 = 50% 公开视觉指数 + 50% 私有同技能测试（只有维护者能跑）。0.4 的公开视觉套件与 0.3 完全相同，kit 里 `EDITION = "0.3"`。我们只能算**公开视觉指数**（public vision index）。
- 公开套件：11 个基准、12,210 道多选题，每题是一道 `choice` 问题（选项键 A、B、C…），附 1–4 张图。`suite rebuild --vision` 从钉死的上游 revision 重建（逐文件 sha256 校验），`vision verify` 核对条目数、items 文件哈希、图片摘要、每个基准的 run id 与 payload 哈希。

| 基准 | 行数 | 权重 | 内容 | 选项数 | 对我们的模态 |
|---|---:|---:|---|---|---|
| CV-Bench | 2,038 | 1 | 2D/3D 空间：Count、Relation、Distance（去掉 Depth） | 2–6 | 单图 |
| BLINK | 961 | 1 | Counting、Relative_Depth、Spatial_Relation（单图 387）；Visual/Semantic/Functional Correspondence、Multi-view（双图 574） | 2–4 | **574 道双图 → 不支持** |
| RealWorldQA | 589 | 1 | 真实驾驶/街景问答（字母选项 + 是非） | 2–4 | 单图 |
| CharXiv | 532 | 1 | 科研图表推理，4 选，干扰项来自同图描述题答案 | 4 | 单图（高分辨率图表） |
| InfographicVQA | 244 | 1 | 信息图问答，4 选，干扰项是同图其他题的答案 | 4 | 单图（大图，需 OCR） |
| Mind2Web | 973 | 1 | 网页截图 set-of-marks，A–D 四个框选目标元素 | 4 | 单图 |
| Winoground | 800 | 1 | 只计 text 子任务：给一张图选两句 caption 之一 | 2 | 单图（image 子任务 800 道双图不计分） |
| KIE（CORD+FUNSD） | 2,465 | 1 | 收据/表单字段值选择 + 字段是否存在（是非，按字段配平） | 2–4 | 单图（需 OCR） |
| Moderation（Hateful Memes） | 1,213 | 1 | "Is this meme hateful?" 是非 | 2 | 单图 |
| R-Bench-M | 665 | 0.5 | 推理题，A–F | 6 | 单图 |
| MMMU-Pro vision | 1,730 | 0.25 | 题干和最多 10 个选项都在图里，criteria 是字母到字母 | 最多 10 | 单图 |

整个套件里我们结构上无法作答的只有 BLINK 的 574 道双图题；没有视频题。

### 计分

- 每个基准：`skill = (accuracy − chance) / (1 − chance)`，chance = 该基准各行 `1/选项数` 的均值；**没答的（unsupported、error、缺失）算错**。
- 指数：skill 截到 [0, 1] 后按权重（9 个基准权重 1，R-Bench-M 0.5，MMMU-Pro vision 0.25）加权平均 × 100。负 skill 照样显示，但在指数里算 0。
- 图像上限：每张图原生分辨率、**最多 1,638,400 像素（1.6 MP）**，超过就按比例缩小；作者自设的更低上限要放开到 1.6 MP（我们的模型结构上固定 letterbox 到 256×256，属于"固定图像预算"，服务端再缩）。
- 上榜还要求在一张 RTX PRO 6000 上中位 / 平均 / P80 延迟 ≤ 1,000 ms/请求；本次是 CPU，不适用，只作参考。

## 2. 重建与校验

- 命令：`HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 decision-index suite rebuild --vision --work /home/bruce/ext/decision-index-data --font /mnt/c/Windows/Fonts/arialbd.ttf`（Mind2Web 的框标字母要 Arial Bold，用 Windows 自带字体，Mind2Web 图片摘要因此对上）。数据在仓库外 `/home/bruce/ext/decision-index-data/vision`（约 4.4 GB），不进仓库、不进训练。
- **hf-mirror.com 目前对所有请求 308 跳转到 huggingface.co**，所以 HF 来源实际是经镜像跳转后从 huggingface.co 下的（不是我们改成直连）。FUNSD 来自官方 zip（`guillaumejaume.github.io`，kit 钉死的地址，无镜像）。
- **Winoground**：HF 上 `facebook/winoground` 是 gated，我们的账号没有通过授权（403），没有在 HF 上申请。改从 **ModelScope `facebook/winoground`** 取两个源文件，sha256 与 kit 钉死的完全一致（`examples.jsonl` `055f6866…`，`images.zip` `062f0a8c…`），再用 kit 自己的 builder 构建（`/home/bruce/ext/scripts/build_winoground_ms.py` 只替换下载函数，校验照旧）。许可仍是 Winoground 的 research-only 协议，本项目只做研究、只评测。
- **InfographicVQA**：第一次用 Pillow 13.0.0 构建，56 张超过 4096 px 的图重新编码后字节不同，`vision verify` 报 `images_sha256` / `payloads_sha256` 不一致；按 kit 文档降到 Pillow 12.3.0 重建该集合后一致。
- 最终 `vision verify`：**12 个源集合、11 个基准全部一致**（条目数、items 哈希、图片摘要、run id、payload 哈希），12,210 行（`runs/ext-decision-index/v0.4/vision-verify.json`）。没有取不到的集合或行。

## 3. 接入方式：自定义 Engine

- kit 的 `http` engine（`HttpSystemOne`）只发 `{"model", "state", "questions"}`，**不带图片**，签名也不接受 `images`；视觉榜的三个内置 engine 是 `vllm-vl`、`decide`（`/v1/decide` 或 `/v1/decisions`，图放在 `state` 列表或顶层 `images` 数组）和 `openai-vlm`，都与我们的 `/v1/systemone` 不兼容。
- 所以在仓库外写了 `/home/bruce/ext/devision_engine/devision_engine.py`（`DevisionSystemOne`，不进 `devision` 包），驱动脚本 `/home/bruce/ext/scripts/run_devision.py` 调 kit 自己的 `vision.run.run` 与 `vision.score.score_run`（计分逻辑未改）。映射：
  - kit 调用 `engine(state, questions, images=[路径], prompt=...)` → `POST /v1/systemone`，body `{"state": "", "questions": <原样>, "image": <data URI>}`；
  - 行里的 `state` 都是空对象 `{}`，按 Jev 语义会被序列化成文本 `"{}"` 多一个 token，这里改发空字符串（不含任何题目信息）；
  - 图片用 kit 的 `images.encode(..., "data_url", max_pixels=1.6MP)`：超过 1.6 MP 的按 kit 规则缩小，其余原字节；服务端再 letterbox 到 256×256；
  - `questions` 原样转发（criteria 是 字母 → 选项文本；MMMU-Pro vision 是 字母 → 字母）；
  - `prompt.readout`（`"The answer is ("`，给生成式模型的读出前缀）对编码器决策模型没有意义，忽略；
  - 图片数 ≠ 1 → `Unsupported`（计分时算错，这是公平规则）；服务器返回 422（如选项塞不进 512 token）→ `Unsupported`；
  - 返回体原样作为 kit 的 response（Jev 结构，`answers.answer.choice` + `probabilities`），由 kit 的 `validate` 检查。
- 服务：`uv run --no-sync devision-serve --checkpoint runs/r21-soft/train --device cpu --port 8011 --no-demo`，`OMP_NUM_THREADS=8`、`CUDA_VISIBLE_DEVICES=` （GPU 在训练，完全不用）。kit 端 1 个 worker，一次一个请求。

## 4. 运行记录

- 全量 12,210 行都跑了（没有抽样）。先跑了 100 行冒烟（5 个集合各 20 行），然后全量；kit 的 `results.jsonl` 支持续跑，中途改服务配置只影响延迟、不影响答案（同一 checkpoint、同一请求）。
- 机器同时在跑别人的 GPU 训练、评测和多组 pytest（load 35–45，20 核），而我们的进程是 nice 5。8 线程单服务在争抢时出现 10–20 s 的长尾（OpenMP 线程被抢占后整组等待），4 线程更差（P50 6.8 s）；最终改为 **4 个单线程服务（端口 8011–8014，共 4 个线程）+ kit 4 个 worker**，每个 worker 固定一个服务。
- 全量评测墙钟约 2 h 10 min（2026-10-28 22:13–00:20 北京时间，含两次重启）；重建套件约 45 min（主要是下载）。
- 结果：`/home/bruce/ext/decision-index-runs/devision-v0.4/`（`results.jsonl` 含题目，不进仓库；`vision-scores.json` / `scores.json`、`environment.json`、`summary/`）；不含题目内容的汇总复制到仓库内 `runs/ext-decision-index/v0.4/`（`summary.json`、`vision-scores.json`、`table.md`、`environment.json`、`vision-verify.json`、engine 源码副本；`runs/` 不进 git）。
- 状态：10,036 ok、574 unsupported（全部是 BLINK 双图题）、0 error；没有 422（没有选项塞不进 512 token 的题）。kit 记录的 `effective_image_budget.max_image_pixels` = 1,638,400。

### CPU 延迟（每个成功请求，kit 侧 HTTP 墙钟，含图片编码与传输）

| 服务配置 | 请求数 | P50 | P95 | 均值 |
|---|---:|---:|---:|---:|
| 8 线程 × 1 服务（开头，负载较轻时） | 960 | 288 ms | 12.9 s | 1.33 s |
| 4 线程 × 1 服务 | 128 | 6.8 s | 22.3 s | 8.2 s |
| **1 线程 × 4 服务（主体）** | 10,548 | **1.65 s** | **3.06 s** | 1.82 s |

这些都是在一台满载的共享机器上测的，只能当作上界；不满载时 8 线程单请求约 0.3–0.4 s（冒烟前单独 curl 0.38 s）。上榜规则（RTX PRO 6000 上 ≤ 1,000 ms）不适用于这次 CPU 运行。

## 5. 结果

公开视觉指数 **8.71**（`complete: true`，0 缺失）。下表我们的列是 acc / chance / skill（%）；其他列是官方视觉榜（`multimodalart/decision-index-vision-leaderboard`，2026-10-28 11:55 UTC 生成）上全部 ≤ 1.2B 参赛者的每基准 skill，按公开指数从高到低。d3-edge 模型卡里的"内部评测"数（CV-Bench 54.2、BLINK 35.1、RealWorldQA 24.8、CharXiv 40.9、InfographicVQA 60.7、Mind2Web 41.2、Winoground 29.0、KIE 86.4、Hateful Memes 6.8、R-Bench-M 10.3、MMMU-Pro 4.9）与榜上的数略有出入（卡里标了"近似重建"），表里用榜上的官方数；它的结果数据集 `vllm-sr/d3-edge-decision-index` 只有文本榜的运行，没有视觉的 `benchmark-summary.json`。

| 基准 | 权重 | 我们 acc | chance | **我们 skill** | 不支持 | QEV-0.8B | OneJev1.5-0.8B | vLLM-SR d3-lite (0.8B) | Saina Helm 2 Omni (0.8B) | JPT-0.8B | Intern-Decision-0.8B | vLLM-SR d3-edge (0.6B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CV-Bench | 1.0 | 48.1 | 39.4 | **14.4** | 0 | 59.7 | 62.7 | 65.0 | 56.9 | 51.2 | 51.7 | 54.0 |
| BLINK | 1.0 | 23.1 | 35.4 | **-19.0** | 574 | 39.3 | 53.8 | 38.1 | 33.0 | 33.6 | 24.9 | 34.8 |
| RealWorldQA | 1.0 | 41.3 | 38.8 | **4.0** | 0 | 40.9 | 41.5 | 36.2 | 43.1 | 40.9 | 38.1 | 25.6 |
| CharXiv | 1.0 | 35.3 | 25.0 | **13.8** | 0 | 54.4 | 57.4 | 53.1 | 48.6 | 52.1 | 48.4 | 36.1 |
| InfographicVQA | 1.0 | 37.3 | 25.0 | **16.4** | 0 | 75.4 | 77.6 | 73.2 | 77.0 | 74.9 | 71.6 | 62.8 |
| Mind2Web | 1.0 | 29.3 | 25.0 | **5.7** | 0 | 73.4 | 46.4 | 50.7 | 50.8 | 37.9 | 21.2 | 38.5 |
| Winoground | 1.0 | 59.2 | 50.0 | **18.5** | 0 | 48.0 | 55.0 | 54.5 | 50.8 | 48.0 | 43.8 | 27.8 |
| KIE (CORD+FUNSD) | 1.0 | 42.9 | 39.3 | **6.0** | 0 | 88.8 | 83.8 | 76.3 | 75.4 | 73.9 | 63.0 | 62.5 |
| Moderation (Hateful Memes) | 1.0 | 52.3 | 50.0 | **4.5** | 0 | 31.9 | 24.6 | 13.1 | 11.5 | 12.8 | 12.4 | 9.2 |
| R-Bench-M | 0.5 | 18.9 | 16.7 | **2.7** | 0 | 12.3 | 10.1 | 9.4 | 16.4 | 13.7 | 5.8 | 9.4 |
| MMMU-Pro vision | 0.25 | 13.5 | 12.8 | **0.8** | 0 | 9.2 | 8.8 | 7.2 | 7.1 | 8.6 | 7.7 | 4.2 |
| **公开视觉指数** | | | | **8.71** | 574 | **53.36** | **52.32** | **47.88** | **46.89** | **44.55** | **38.96** | **36.61** |

- 其他小模型：LiquidAI d1-omni-600M 不在官方视觉榜，d3-edge 模型卡里用官方 kit 跑的公开分是 **4.8**（`system_one` 接口）；Sieve-0.8B、jiwo（`eljiwo/decision-index-results`）只有文本榜结果，没有视觉运行；`bynn-intelligence-com/decisionnode-edge`（2.2B）自报公开视觉 59.13，不属于 < 1B。
- 位置：v0.4 的 **8.71** 在官方视觉榜最低的 d3-edge（36.61，57 名中第 57）之下很远，只高于榜外的 LiquidAI 4.8。11 个基准里每一个都低于所有 ≤ 1.2B 的上榜模型，只有 Winoground（18.5）和 Hateful Memes（4.5）接近 d3-edge（27.8 / 9.2）。

### 分项观察

- **BLINK −19.0**：574 道双图题按规则算错（Visual / Semantic / Functional Correspondence、Multi-view 全为 0）。只看能答的 387 道单图题：准确率 57.4%（Spatial_Relation 73.4、Relative_Depth 51.6、Counting 44.2），skill 26.2，与小模型相当。支持多图是这个基准唯一的大杠杆（但权重只占 1/9.75）。
- **CV-Bench 14.4**：Relation 64.3%（chance 50），Count 35.7%（chance 22.5），Distance 47.0%（chance 50，低于随机）。我们自己评 `bench_cvbench` 的口径不同（那是我们转换的集合），这里是 kit 的转换和 2D/3D 全部子任务。
- **OCR 类（InfographicVQA 16.4、CharXiv 13.8、KIE 6.0、Mind2Web 5.7）**：256×256 letterbox 后文字基本不可读；KIE 的字段是否存在题 50.9%（随机），字段值 33.3%（chance 26.4）。小模型在这些项上是 50–89，这是差距的主体。
- **RealWorldQA 4.0、Hateful Memes 4.5、R-Bench-M 2.7、MMMU-Pro 0.8**：接近随机。R-Bench-M 的 `other_correct`（正确答案不是内容选项的 165 题）只有 1.2%。
- **Winoground 18.5**：单图选 caption 是我们训练的形式（图文匹配），相对最好。

### 注意事项

- 只有公开一半；视觉榜的 Full 还要私有一半，只有维护者能跑，我们不提交。
- 我们的模型固定 256×256 输入、单图、英文，属于"固定图像预算"的架构；kit 照 1.6 MP 上限发图，服务端再缩。
- 行里的 `state` 是 `{}`，我们发的是空字符串（见第 3 节）；`readout` 前缀忽略。kit 的计分代码没改。
- CV-Bench 的部分图来自 COCO；v0.4（r21）的训练数据用到 COCO train2014 等，未逐图核对与 CV-Bench 的重叠。v0.5 方向的 GUI / Mind2Web 类数据不在 v0.4 里。
- **这些题目和图片不得用于训练**；数据只放在 `/home/bruce/ext/decision-index-data`。
