# v0.4 读取链证据盘点（V04-READER-EVIDENCE-PREP-02；只用 CPU，没有加载模型）

日期：2026-10-19。脚本 `scripts/v04_evidence/inventory.py`，输出 `data/v04_evidence/inventory.json`。

## A. 29 个集合与全部 extras 的路由覆盖和成本

- 路由用产品读取链的 devision.reading.domain_of（温度 / 磁力走新读取链，其余是普通 decide）。请求数按 devision-eval 分组（同一张图、同一段文字、最多 8 题）。
- **对照**：29 个集合按 v0.3 全评的口径，都有配错图；含选择题的还有选项倒序。extras 按 configs/v04-extra-evidence.yaml 所列。
- **用时**：粗估 = 请求数 × 实测每请求时间（普通 0.334 秒，用的是 T 的 P50；路由 0.43 秒，用的是 EXP71 实测）× 运行次数（含对照）。这不是实测的全评。

| 范围 | 题数 | 走新读取链的题 | 粗估用时 |
|---|---|---|---|
| 29 个集合（含对照） | 55,900 | 222（science 20、dev_mix 20、bench_lv_scienceqa 182） | 约 503 分钟 |
| extras | 12,468 | 504（sqa_temp / force 及错图 322、lv_scienceqa_raw 182） | 约 37 分钟 |
| **合计** | | | **约 9.0 小时**（与交付计划此前的 7.7–9.0 小时一致） |

- **独立确认项全部走普通路径**：confirm_property 75 题（及错图、倒序）、confirm_chartqa 80 题（及错图、倒序），新读取链 0 题。它们保留为旧能力和一般图表的证据，但**不能独立确认温度 / 磁力读取的新能力**。
- 29 个集合和全部 extras 按原角色保留（dev16、擦除控制 syn_erased、校准 calib_val_*、lv_scienceqa_raw 等都在），这次开发迁移不替代其中任何一项。
- 新读取链在 29 集合里只有 222 题（0.4 %），而且全部来自已多次使用的集合：science / dev_mix 的 20 题是 ScienceQA 训练图，bench_lv_scienceqa 的 182 题是验证集图（见 B）。

## B. 真实温度 / 磁力图有没有可用的确认来源

- **方法**：对本地原始 ScienceQA（HF derek-thomas/ScienceQA，固定快照 f18b0a70）三个划分里，所有被路由到读取域、且带图的题，按「划分 + 行号」对照 data/ 里所有 JSONL 的题 id（v3-scienceqa:train-N、v3-sqa-test:test-N、lv-scienceqa:val-N），并核对题目文字（不符 0）。按文件归类：训练、开发 / 监控、基准、JevBench。
  - 数据文件里的图是重新编码过的，不能按字节匹配；第一次按字节匹配全部「未使用」，是这个原因，已改用 id 匹配。

| 划分 | 走读取链的题（温度 / 磁力） | 暴露情况 | 完全未用过 |
|---|---|---|---|
| train | 168 / 365 | 训练文件 159 / 354；另有部分在开发 / 监控文件（v3 / v4 / v5 dev_mix、panel_science） | **0** |
| validation | 68 / 114 | 全部是 laya-vision 基准 bench_lv_scienceqa（每轮都评，配置里注明它的切片曾用来选研究方向） | **0** |
| test | 41 / 120 | 全部是 dev_sqa（exp21 / v10，反复用于开发和方法选择） | **0** |

- **结论**：本地没有合格的真实温度 / 磁力确认余量。全部 876 道题都已用于训练、开发或反复评测，不能把已经看过的开发题改名为 heldout。
- **补充来源的方案**（只是方案，没有获取，也没有看任何成绩）：
  1. 同一出版物风格的新题：ScienceQA 原始来源（IXL 的 K-12 题库）里可能还有未收录的同类图。需要确认许可和获取方式，并按图像近重复（感知哈希）对已有 876 题去重；
  2. 独立绘制：由未参与开发的人，按 ScienceQA 图的版式另画一批真实风格的图（不同字体、画布和间距），题目和金标按规则生成，绘制参数在看任何模型成绩之前冻结；
  3. 上述任一来源都要先写抽样规则（数量、温度 / 磁力配比、去重阈值），在候选组合锁定之前不看确认成绩，也不缩减原有确认项。
  - 如果都拿不到，就在交付记录里明确写「新读取链没有独立确认」，不用开发集代替。

## C. 完整组合身份（发布 IDENTITY 只绑 model.safetensors 已经不够）

- 同一份 EXP57 权重可以有 off / on 两种组合（EXP71），两者的筛查和全评证据不能互借。EXP71-on 实际由这些东西决定：
  - 决策：model.safetensors 962503fb…、config.json 09a5b447…、tokenizer（树哈希）；
  - 读取组件：selector.pt c92b7b7e…、head.pt 0e275bdf…、corrector.pt 199b5367…；
  - 规则：POLICY "templates-v1"、GRID（12 / 0.25 / 256 / 16）、CORRECTOR（两个速度模板、上下 0.5 个框高、左右 8 px、只用局部视图）、TEMPLATES、表头 HEADER；
  - 代码：src/devision/reading.py、image.py、infer.py。
  - bundle_on 的 reading.json（d8faab97…）已经绑定了组件文件和决策端点；规则和代码目前不在其中。
- **接线方案**（未实现，供交付流程采用）：
  - 发布的 IDENTITY 加一项 reading_combination = {reading.json 的 sha256，selector / head / corrector 的 sha256，POLICY / GRID / CORRECTOR / TEMPLATES / HEADER 的规范 JSON 哈希，reading.py / image.py / infer.py 的哈希}；
  - check_eval 的 freeze / preflight / seal / check 每一步都核对这一项，任何筛查、全评或确认结果只属于记录的那一个组合；
  - 包内也随附 bundle（三个组件和 reading.json），模型卡写明组合身份。

## D. 交付缺口与剩余工时（如实更新，不回填按时完成）

- 原定 2026-10-18 的期限已经错过。
- 仍需逐项处理：EXP57 的 fit 失败和组合资格；全量校准概率（data/r13b/val.jsonl 3,892 题）；29 集合和 extras 的全评（约 9 小时，粗估）；七个保护面板（EXP73 约 32 分钟）；新读取链的独立确认（目前没有来源，见 B）；CPU 延迟；API / 安装；包和模型卡；远端发布核验。
- 剩余工时粗估：全评约 9 小时；校准、保护和确认约 2–3 小时（不含获取新确认来源）；CPU / API / 包 / 模型卡 / 发布核验约 3–4 小时；合计约 14–16 小时机器加人工时间，前提是组合资格先有结论。研究包不冒充 v0.4。
