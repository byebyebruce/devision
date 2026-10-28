# exp131：ChartQA 官方 test 划分做图表能力的未见测试（预写，2026-10-28 23:20，只评测，不算轮次）

**背景**：v0.5 目标包含图表能力，但当前 32 个评测集里没有图表测试集——图表只有 `data/v05/exp81/chart_dev.jsonl`（ChartQA val，训练中监测、选模型用，不能作为最终结论）。exp81 起训练混合一直带 ChartQA train 24k 题，图表收益从未在未见测试上确认。

**做法**：`scripts/v05/chart_test.py` 从 HuggingFaceM4/ChartQA 同一 revision 的 test 划分（之前未下载、未用过）按 exp80/exp81 的协议转成选择题：干扰项池与频率匹配取自 exp81 使用的同一批 train 行（eligible 数核对等于 exp81 MANIFEST），是非题为 noul；与 train / val 像素相同的 test 图剔除。输出 `data/v05/chartqa_test/test_chartqa.jsonl` + `MANIFEST.json`（图片存在 `data/v05/chartqa_test/images/`）。集合名 `test_chartqa`，对所有被评模型都是 heldout。

**被评模型**：v0.4（`runs/r21-soft/train`）、exp129（`runs/exp129-v4-scale-size-seed23/train`）；exp130 若选出 α 也评。结果写到各自轮次的 `eval/test_chartqa.*`（exp130 在 `runs/exp130-interp/eval/`）。

**判据**（`devision-compare` 按图 95%，对 v0.4；按 kind 拆分另报，human / machine 拆分用 sample_id 对回数据另算）：
1. exp129（及之后的 r23 候选）对 v0.4 在 test_chartqa 上无显著退步 → 图表维持条件满足。
2. 显著提升 → 记为 v0.5 图表能力证据，写进发布材料。
3. 显著退步 → r23 配方须加回图表比例（对照 exp81 的 24k），再定。

**预算**：约 2.5k 题 × 3 个模型，每个几分钟 GPU；挂在 exp130 链之后（`scripts/v05/chain_exp131.sh`），不与正在跑的评测抢 GPU。

**数据已生成（23:25）**：test 原始 2,500 题，与 train / val 像素相同剔除 21，无法按协议出 3 个频率匹配干扰项的 264 题不收，得 **2,215 题**（number 1,757 / text 189 / year 144 / yesno 102 / color 23；human 1,051 / machine 1,164；正确项位置 591 / 587 / 516 / 521），sha256 `5bb5bc97…`（`data/v05/chartqa_test/MANIFEST.json`）。`chain_exp131.sh 685941` 23:07 已挂在 exp130 链之后。
