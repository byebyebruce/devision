# exp127：r22 数据 + 第一人称 / 手机 GUI 扩量（预写，2026-10-28 16:05，r22 评测中）

**背景**：负责人 10-27 方向是扩量真实手机 / 网页操作和第一人称数据。exp120 的 +12k EPIC-VISOR 让 test_ego +3.6；r22（= exp124 数据）用了 EPIC-VISOR 24k，训练参与者池里还剩 19,617 题没用；AITW-general（多 App，test_gui_state 的 AITW 部分是未见 App）还剩约 19k。

**改动**（`scripts/data/v05/exp127_mix.py`，seed 127，`data/v05/exp127/MANIFEST.json`）：r22 训练文件 `data/v05/exp124/train.jsonl` 不改，加 EPIC-VISOR 剩余全部 19,617 题（contact 6,191 / touching 1,555 / action 11,871）+ AITW-general 屏幕状态改写题 9,000（2,098 个 episode；完全没用过的 episode 只有 2.6k 题，所以允许部分用过的 episode 的剩余步骤，都属于训练划分的 App）。共 192,657 题（约 24.1k 步）。shopping 保持 26k（exp121：再加 shopping 让 MME / MMBench 约 −1pp），Mind2Web 不变（256 下不可读）。init、超参、**seed 22**、评测集（32 集 + controls）全部同 r22。

**对照**：r22（同种子，只差 28.6k 新增题，逐题配对）、v0.4。

**假设**：新用途数据扩量继续提升第一人称与手机 GUI，且旧能力不被稀释。

**判据**（`devision-compare` 按图 95%）：
1. **扩量有效**：test_ego 对 r22 显著提升，或 test_gui_state 的 aitw-general 组对 r22 显著提升（至少一项）。
2. **不伤其他新用途**：test_gui_shop_state、test_gui、bench_egothink 对 r22 无显著退步（下界 > −2pp）。
3. **旧能力**：r22 判据 2 的全部旧集合与外部 bench 对 v0.4 无显著退步，且对 r22 无显著退步。
- 1–3 都过 → exp127 配方作为 `r23-*` 正式轮次候选（换种子复现后定）。
- 1 不过 → 这两类数据在当前量已饱和，不再扩同源数据，改找新来源（如 AndroidControl、更多场景的第一人称）。
- 3 不过 → 记录退步集合，按 exp121/123 的经验减量或加回放，不直接定 r23。

**预算**：训练约 65 分钟 + 评测约 90 分钟。`scripts/v05/chain_exp127.sh 637167` 16:03 挂在 r22 链（流水线 + lv_report）之后（日志 `runs/v05-chain/chain_exp127.log`、`runs/exp127-v4-gui-ego-scale.out`），dry-run 通过，预计 17:35 开训、20:10 前后评完。不拟合发布温度。
