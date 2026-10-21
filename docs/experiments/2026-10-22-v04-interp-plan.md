# v0.4 补洞：v0.3 与 r20 的权重插值（2026-10-22，运行前写下）

依据：`docs/experiments/2026-10-21-v04-plan.md` 第 4 阶段（"个别集合小幅退步：v0.3 与候选之间权重插值，开发集上选比例，测试集只看一次"）。r20-science 通过验收规则 1–3，规则 4（校准）未过：POPE NLL +0.23、size +0.09、MMBench +0.05、MME +0.03（`runs/r20-science/eval/compare/accept_v04.md`）。离线分析显示只重拟合温度不够（POPE 0.60 → 0.48，v0.3 0.37）。

## 做法

- θ(α) = θ_v0.3 + α·(θ_r20 − θ_v0.3)，α ∈ {1.0, 0.8, 0.6, 0.4}（`scripts/exp17/interp.py`；α = 1.0 就是 r20 的权重）。
- 每个 α 的温度都在 `data/v5/dev_mix.jsonl` 上重拟合（各来源均衡的开发集，`devision-calibrate --out`），不再用 data-n1 那份以 VQAv2 / GQA 为主的开发集。
- 选择只看开发集：`data/v2/dev_coco_exist.jsonl`（存在题，COCO train2014，与 POPE 同样带共现负例；其中约一半题也在 v5 dev_mix 里，是温度拟合集的一部分）的 NLL，以及 `data/n2/dev_sqa`、`dev_spatial`、`dev_count`、`dev_v7w` 的准确率（只报告）。

## 选择规则

取 dev_coco_exist 的 NLL ≤ v0.3 的 NLL + 0.02 的**最大** α。都不满足：停在这里，把曲线报告给负责人（下一步候选是 r21：存在题标签软化续训）。

选中的 α 做一次全量评测（26 个集合，同 `configs/r20-science.yaml` 的评测部分），按冻结规则 `docs/experiments/2026-10-21-v04-eval-freeze.md` 判定，测试集只看这一次，结果出来后不再换 α。
