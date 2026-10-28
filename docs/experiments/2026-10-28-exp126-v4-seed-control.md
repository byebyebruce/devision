# exp126：exp122 数据 + seed 124（种子噪声对照；预写，2026-10-28 12:05，exp125 训练中）

**背景**：exp124（exp122 + 9k 存在回放，seed 124）修了 POPE（对 v0.4 −0.41 含 0），但 test_gui_state 对 exp122（seed 122）−2.47 [−4.29, −0.58]、test_gui −1.95、ScienceQA −1.24 显著。exp124 与 exp122 同时差了数据和种子，无法区分回放代价与种子噪声；exp125 的预写规则也要求"GUI 退步以同种子复现确认"。

**改动**：`configs/exp126-v4-seed-control.yaml` = exp122 的配置只把 seed 改为 124（= exp124 的配置只把训练文件换回 `data/v05/exp122/train.jsonl`）。不新造数据。

**对照**：exp124（同种子，只差 9k 回放）、exp122（同数据，只差种子），逐题配对；exp125 结果另作剂量参考。

**判据**（`devision-compare` 按图 95%）：
1. **种子噪声**：exp126 对 exp122 在 test_gui_state / test_gui / bench_lv_scienceqa 上的差值，若与 exp124 对 exp122 同向且区间不含 0 → 这些退步主要是种子造成的，回放（exp124 或 exp125 剂量）无 GUI 代价。
2. **回放代价**：exp124 对 exp126（同种子）在 test_gui_state 下界 > −2pp 且 POPE 点估计 > 0 → 9k 回放定入 r22 配方。
3. 若 exp124 对 exp126 test_gui_state 仍显著 < −2pp → 回放有真实 GUI 代价，r22 用 exp125 剂量（若 exp125 判据 1、2 过），否则把回放题限定在 COCO 存在题、不加 VQAv2。
- 结果出来后定 `r22-*`（多种子不再继续加，最多这一个对照）。

**预算**：训练约 55 分钟 + 评测约 55 分钟。`scripts/v05/chain_exp126.sh 626715 623743` 12:03 挂在 exp125 流水线之后（日志 `runs/v05-chain/chain_exp126.log`、`runs/exp126-v4-seed-control.out`），dry-run 通过，预计 15:10 前后评完。不拟合发布温度。
