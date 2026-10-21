# v0.4 评测冻结与验收规则（2026-10-21，r19 开训前写下）

方案：`docs/experiments/2026-10-21-v04-plan.md`。本文在任何 v0.4 候选训练之前写下，结果出来后不改；之后新增的集合只作报告，不进入判定。

## 评测集（26 个；sha256 前 16 位）

| 集合 | 文件 | 题数 | sha256 |
|---|---|---|---|
| dev_mix | `data/v5/dev_mix.jsonl` | 5178 | `4f63fce93e7b07a6` |
| test_vsr | `data/v5/test_vsr.jsonl` | 902 | `d202690d1821ef84` |
| test_v7w | `data/v5/test_v7w.jsonl` | 1000 | `df2bc055eec90a4b` |
| bench_pope | `data/v2/bench_pope.jsonl` | 9000 | `8f3310e117409d94` |
| test_vqa_yesno | `data/v2/test_vqa_yesno.jsonl` | 1000 | `4c8e43a98d231ac5` |
| test_vqa_choice | `data/v2/test_vqa_choice.jsonl` | 1453 | `b9dc9bb203f02ce4` |
| test_gqa | `data/v2/test_gqa.jsonl` | 986 | `e5299e4aa555d2d1` |
| test_exist | `data/v2/test_exist.jsonl` | 1120 | `6ff0bf6ed07d3784` |
| test_position | `data/v2/test_position.jsonl` | 1276 | `439170a8dc80d99d` |
| test_relation | `data/v2/test_relation.jsonl` | 2366 | `cf9ac02402d2378f` |
| test_size | `data/v2/test_size.jsonl` | 1198 | `e374f4ace05782be` |
| test_position_flip | `data/v6/test_position_flip.jsonl` | 640 | `22304b8e89a32668` |
| test_relation_flip | `data/v6/test_relation_flip.jsonl` | 1786 | `a6cc429291480d4c` |
| bench_lv_vqav2_yesno | `data/lv_bench/vqav2_yesno.jsonl` | 4887 | `1357ec3b72a8dde2` |
| bench_lv_aokvqa | `data/lv_bench/aokvqa.jsonl` | 1138 | `977aa7176ebccc54` |
| bench_lv_scienceqa | `data/lv_bench/scienceqa.jsonl` | 2097 | `7505832ea7cd07c0` |
| bench_lv_pope | `data/lv_bench/pope.jsonl` | 9000 | `8f3310e117409d94`（与 bench_pope 相同，判定时只算一次） |
| bench_whatsup | `data/r13/bench/whatsup.jsonl` | 820 | `e243b872eb2cc249` |
| bench_countbench | `data/r13/bench/countbench.jsonl` | 491 | `705af08d578a87ab` |
| bench_cogent_b | `data/r13/bench/cogent_b.jsonl` | 2000 | `986fee2dd6a10c90` |
| test_count_coco | `data/n2/test_count_coco.jsonl` | 1100 | `9f6ad34900c147fb` |
| bench_mmbench | `data/n2/bench/mmbench.jsonl` | 4294 | `99e128acc22e580c` |
| bench_cvbench | `data/n2/bench/cvbench.jsonl` | 2638 | `baa8ebc477ad39b8` |
| bench_mme | `data/n2/bench/mme.jsonl` | 2374 | `a631144074325186` |
| bench_naturalbench | `data/n2/bench/naturalbench.jsonl` | 7570 | `cb4498905be66cf9` |
| dev_sqa | `data/n2/dev_sqa.jsonl` | 2017 | `2a84a47df6e0b79c` |

说明：
- `dev_mix`、`dev_sqa` 在 r19 训练中用于监测（monitoring），只报告，不参与判定；其余 23 个（POPE 算一次）参与判定。
- 本机没有重建 `test_bind_gqa`、`test_count_fresh`（依赖旧机器的整条训练历史文件），由 `test_count_coco`（COCO val2014，v0.3 谱系整体留出）和外部评测替代。
- 外部评测（MMBench / CV-Bench / MME / NaturalBench / CountBench）没有可追溯的图片 id，按 dHash 从所有 data-n2 训练源剔除（`scripts/data/n2/held.py`）；v0.3 当年的训练数据没有按它们剔除，这一点对 v0.3 和候选一样，不影响配对比较。

## 验收（候选 vs v0.3，同题配对，按图片重采样 2,000 次，95% 区间）

1. **不退步**：参与判定的 23 个集合，没有一个准确率差值的上界 < 0。
2. **整体提升**：至少 15 个集合点估计 > 0。
3. **弱项提升**：以下 5 项中至少 4 项下界 > 0——ScienceQA（`bench_lv_scienceqa`）、VSR（`test_vsr`）、计数（`bench_countbench` 与 `test_count_coco` 至少一个）、A-OKVQA（`bench_lv_aokvqa`）、V7W（`test_v7w`）。
4. **校准不变差**：没有集合的 NLL（各自 checkpoint 拟合的温度）升高 > 0.02 且区间下界 > 0。

四条都满足才作为 v0.4 发布候选；否则不发布，结果和取舍交负责人。补洞（r20 或与 v0.3 的权重插值）按同一规则重新判定；插值比例只在 `dev_*` 上选，测试集只看一次。
