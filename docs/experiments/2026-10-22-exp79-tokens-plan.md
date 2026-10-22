# exp79：输入不变、视觉 token 从 64 加到 256（2026-10-22，运行前写下）

负责人 2026-10-22："分辨率不提升，图片 token 增加一些呢？" exp78 显示 384 输入适应 2 万步后反而更差（细节面板 −2.5 [−3.5, −1.4]，CPU 慢一倍）。这里不改输入（256×256、SigLIP 计算量不变），只是不再把 2×2 个 patch 合并成一个：语言模型看到全部 256 个 patch，而不是 64 个 token。不算一轮。

## 做法

- 起点：v0.4 经 `scripts/exp20/fold.py` 折叠投影层（`visual_shuffle 1`；一个 2×2 邻域内 4 个 patch 相同时，新的 4 个 token 与原来的 1 个完全相同；其余权重和温度不变），`runs/exp79/v04-tok256`。
- 第 0 步：零训练评开发集，测 CPU 延迟（同 exp78 的测法）。
- 第 1 步：与 exp78 的 256 臂（`runs/exp78-res256`，64 token）完全相同的训练（`data/exp78/train.jsonl`、种子 0、20,000 步、同学习率、温度在 `data/v5/dev_mix.jsonl` 拟合），只差起点的 token 数。对照直接复用 exp78-res256。

## 判定（与 exp78 相同）

- P1：`dev_count`、`dev_tallyqa`、`dev_sqa` 自然科学、`dev_v7w` 四个面板合并，256 token − 64 token 的准确率差下界 > 0。
- P2：`dev_mix`、`dev_spatial`、`dev_coco_exist` 没有一个 ≤ −2 且上界 < 0。
- 报告：CPU 单题延迟、训练速度、各面板差值。

已知风险：ModernBERT 的局部注意力窗口是 128，256 个视觉 token 时选项位置在局部层看不到大部分视觉 token，只能靠全局层（exp20 的记录）。
