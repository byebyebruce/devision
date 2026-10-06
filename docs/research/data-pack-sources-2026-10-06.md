# 扩展数据包候选来源（2026-10-06）

负责人要求：架构不变，从 Hugging Face、ModelScope、Kaggle 找优质的图片决策数据，做一个量大、多样的扩展数据包，从 v11 接着训。本文是三路网上调研的汇总：自然照片类、示意图 / 图表 / 科学类、ModelScope / Kaggle 与大型混合数据集。仓库 ID 都用 HF API（`huggingface.co/api/datasets/<id>`）或 ModelScope API 核实过；许可证标"未核实"的要在使用前再查。调研只读了页面和 API，没有下载数据。另见同日 Codex 写的 `modelscope-image-decision-datasets-2026-10-06.md`（ModelScope 镜像入口，国内下载可用）。

## 先避开的污染

- **ScienceQA validation 是我们的留出集**：laya-vision 的 ScienceQA 评测集（`bench_lv_scienceqa`）取自 `derek-thomas/ScienceQA` 的 validation（`scripts/data/lv_bench.py`）。所以 ScienceQA 只能用 train。
- `lmms-lab/LLaVA-OneVision-Data` 和 `HuggingFaceM4/FineVision` 的 `scienceqa(nona_context)`：19,208 道全带图，比 ScienceQA 全部带图题还多，必然混有 val/test，不用。
- LLaVA-OV 的 `ai2d(gpt4v)` 4,864 行，接近 AI2D 全部图片，可能含 test，不用。
- LLaVA-OV 的 `tqa(cauldron,llava_format)` 实为 IconQA，标错了名。
- `LightChen2333/M3CoT` 从 ScienceQA 重新切分，`Xkev/LLaVA-CoT-100k` 含 ScienceQA，都不用。
- M3IT 的 vqa-v2 val 是 COCO val2014，只能用 train。
- **COCO 2017 train 里约有 3.5 万张是 val2014 的图**，VG 含 GQA val 的图。凡是用 COCO / VG 图片的来源都要按图片身份和 dHash 排除留出图（现有 `v9_mix.py` 的规则）。混合数据集里的图片不带原始 id，只能靠 dHash 或从原始 JSON 重建。
- IconQA、TabMWP 与 ScienceQA 同一作者团队，按图片哈希再查一遍。

## 候选来源

| 来源 | 获取入口（已核实） | 可转换的量 | 转成 | 许可证 | 图片 / 留出风险 | 说明 |
|---|---|---|---|---|---|---|
| **Objects365** | FineVision `objects365_qa`（166.6 万行，约 200 GB）或原始标注 | 上百万 | 存在（是非，有真负例）、计数（选择）、左右（选择，从框算） | 标注 CC BY 4.0，仅限学术；图片不得再分发 | 自有图片，不是 COCO | 物体标注齐全，负例可靠；量最大的新图来源 |
| **PixMo-Points / PixMo-Count** | `allenai/pixmo-points`（238 万行，约 43 万张图）、`allenai/pixmo-count`（3.7 万） | 上百万 | 计数（选择）、存在（是非，含"没有"） | ODC-BY | 网络图片按 URL 下载，部分失效；不是 COCO | 人工逐个点出物体，正对"新图计数"短板 |
| **TallyQA** | GitHub 原始 JSON（带图片路径）；Cauldron `tallyqa` | 约 25 万 | 计数（选择） | Apache-2.0 | COCO + VG：按 id 排除 | 含"复杂计数"题 |
| **CLEVR / Super-CLEVR / CLEVR-Math** | ModelScope `OmniData/CLEVR`、Kaggle `timoboz/clevr-dataset`；`RyanWW/Super-CLEVR`；`dali-does/clevr-math` | 数十万 | 比较（"A 比 B 多吗"，是非）、属性（选择）、计数（选择） | CC BY 4.0（Super-CLEVR MIT） | 合成，无风险 | 标签由场景算出，绝对准确；合成图要限比例 |
| **IconQA** | Cauldron `iconqa`、ModelScope `OpenDataLab/IconQA` | 约 3 万 | 原生选择题；填空题多为计数 | CC BY-NC-SA 4.0 | 按哈希对 ScienceQA test / val 去重 | 小学数学图，专练"比较多少""数数" |
| **VisOnlyQA_Train** | `ryokamoi/VisOnlyQA_Train`（7 万，约 15 GB） | 7 万 | 对 / 错（是非）、a–e（选择） | GPL-3.0 | 合成 | 纯感知："哪条线更长、哪个角更大"，最接近"比较图中两处" |
| **FigureQA** | Cauldron `figureqa`（10 万张图，133 万题） | 133 万 | 全是是非题 | 微软研究许可，条款未核实 | 合成 | 是非各半；比较两条曲线 / 两根柱子 |
| **DVQA / MapQA** | Cauldron `dvqa`、`mapqa` | 数百万 | 是非、"哪个最大 / 最小"（选择） | DVQA CC BY-NC 4.0；MapQA CC BY-SA 4.0 | 合成 / 地图 | 字偏小，先抽查 256 输入下能否辨认 |
| **SpatialSense** | GitHub `princeton-vl/SpatialSense`；FineVision `spatialsense` | 1.7 万 | 关系真假（是非） | 代码 BSD-2，图片 Flickr / NYU | 无 | 刻意设计成只看文字猜不出来；标注质量高 |
| **SNLI-VE** | `HuggingFaceM4/SNLI-VE`；修正版 `J1mb0o/e-SNLI-VE` | 约 35 万（只取蕴含 / 矛盾） | 是非 | BSD-3，图片按 Flickr30k 条款 | Flickr30k，无 | "中立"标签错误多，丢掉；要验证只看文字答不出来 |
| **Vision-Flan** | `Vision-Flan/vision-flan_191-task_1k`（18.6 万，36 GB） | 约 4.6 万有选项或是非，另约 10 万短标签 | 选择、是非 | 按任务各自的许可证 | 部分任务用 COCO train2014，文件名带 id | 191 个人工标注任务，多样性最高（颜色、材质、运动、场景、情绪、卫星、医学……） |
| **M3IT 选择题子集** | `MMInstruction/M3IT` 的 vcr / snli-ve / coco-itm / coco-goi（只用 train） | 约 10 万 | 原生选择题 | 来源各自许可证；VCR 仅限研究 | coco-itm / coco-goi 是 COCO：按 id 排除 | VCR 选项是长句子，适配性差 |
| **Open Images V7 已验证标签** | 官方下载；`vikhyatk/openimages-bbox` | 可很大 | 存在（是非，人工确认的负例）、材质属性 | 标注 CC BY 4.0，图片 CC BY 2.0 | Flickr，无 | 全量太大，取子集 |
| **分类数据集** | Places365、Food-101、CUB、Oxford Pets、Stanford Cars、iNat21 | 可很大 | "这是哪种……"（选择，用相近类别作干扰项） | 各自许可证 | 无 | 标签干净，但偏细粒度冷知识，限比例 |

## 不用

- 字太小、256 输入读不出来：PlotQA、ChartQA、TabMWP、ArxivQA、SciGraphQA、TextVQA。
- 答案由模型改写成长推理或长回答：MAmmoTH-VL、ALLaVA、LLaVA-665k、Infinity-MM、MMInstruct（大部分）。
- 两张图一题：NLVR2、RAVEN、MLLM-CompBench（需要拼图，在 256 输入下太小）。
- 只有测试集：What'sUp、CV-Bench、ARO、SugarCrepe、Winoground、CountBenchQA、MMStar、MMMU。
- 许可证需另行确认：InternSpatial（CC BY-ND）。

## 结论

- 没有公开数据复现 ScienceQA 的磁铁、烧杯、动能那几类模板；能练的是背后的能力"比较图中两处"（IconQA、VisOnlyQA、FigureQA、CLEVR）。
- Kaggle 和 ModelScope 上基本都是已知数据集的镜像，没有新来源；ModelScope 镜像在国内下载更方便。
- 量最大、最干净的新图来源是 Objects365 和 PixMo，它们也最贴近"新图计数"和"存在判断"。
