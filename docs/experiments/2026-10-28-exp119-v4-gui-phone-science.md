# exp119：exp117 配方 + 手机 GUI 改写题扩量（预写，2026-10-28 02:05，exp118 评测尚未全部结束）

**背景**：exp118 的 GUI 部分已评完（[exp118](2026-10-28-exp118-gui-screen-state.md)）：`test_gui_state` 上 AITW（手机）真图 56.6 vs 配错图 34.8（+21.8pp），**Mind2Web（网页）35.6 vs 35.3，无看图增益**。Mind2Web 截图是 1024×800 的桌面页面，letterbox 到 256 后文字约 3px，屏幕读不出来；手机竖屏截图在 256 下仍能分辨界面状态。原题型 `test_gui` 从 exp116 的 80.6 掉到 66.3（主要是 Mind2Web 78.2→65.9，它丢掉了历史操作这条文字线索，又没有补上视觉）。

**改动**（`scripts/data/v05/exp119_mix.py`，seed 119，`data/v05/exp119/MANIFEST.json`），以 exp117 的混合（含科学回放）为底：
- AITW-general 6,000 题 → 改写版（与 exp118 同样的转换）；
- 新增 AITW web_shopping 12,000 题（1,481 个 episode，按 episode 整段取），改写版；
- Mind2Web 6,000 题保持原题型（带历史操作）——改写版在 256 下没有视觉信号；
- 其余与 exp117 相同；共 104,889 题。init、学习率、视觉解冻、CE、1 epoch 同 exp117/118。
- 新评测集 `test_gui_shop_state`（AITW-shopping test 774 题，5 个未见网站分组，改写版，词重叠基线 0.417→0.269）与监测集 `gui_shop`（dev 256）。

**对照**：v0.4、exp117、exp118 在 `test_gui_shop_state` 上（只评测 YAML `configs/*-guishop.yaml`，接续脚本先跑）；exp117（科学 / 旧能力底座）、exp118（GUI 改写）逐题配对；所有集合的配错图 / 倒序对照。

**判据**（`devision-compare` 按图 95%）：
1. **手机 GUI 看屏幕**：`test_gui_state` 的 AITW 部分、`test_gui_shop_state` 真图 − 配错图各 ≥ 10pp；`test_gui_shop_state` 相对 exp118 下界 > 0（扩量有效）。
2. **网页不退**：`test_gui` 的 Mind2Web 部分相对 exp117 不低于 −3pp（点估计）。
3. **科学与旧能力**：ScienceQA / dev_sqa 相对 exp117 无显著退步；其余集合相对 exp117 无显著退步；CV-Bench 只记录 3D 子项与 exp117 的差。test_ego / EgoThink 相对 exp117 无显著退步。
- 1–3 都过 → 作为 `r22-*` 正式轮次的候选配方（再与 v0.4 全面逐题配对、跑 lv_report）。
- 1 过、2 不过 → Mind2Web 改用切块 / 更高分辨率另做实验，不在 256 下加量。
- 3 不过 → 回到比例（GUI 占比 22.9%→ 下调）。

**预算**：基线评测约 3 分钟 + 训练约 35 分钟（13.1k 步）+ 评测约 50 分钟。`scripts/v05/chain_exp119.sh 565110`（exp118 的接续脚本）02:02 已挂上，exp118 退出后自动开跑（日志 `runs/v05-chain/chain_exp119.log`、`runs/exp119-v4-gui-phone-science.out`）。不拟合发布温度。

## 结果与裁决（2026-10-28 04:05）

| 集合 | v0.4 | exp117 | exp118 | exp119 | exp119 配错图 |
|---|---|---|---|---|---|
| test_gui_shop_state（未见网站） | 21.4 | 36.4 | 49.6 | **59.8** | 31.1 |
| test_gui_state AITW | 21.6 | | 56.6 | **58.9** | 32.3 |
| test_gui_state Mind2Web | 28.6 | | 35.6 | 36.4 | 33.7 |
| test_gui 原题型 Mind2Web | 47.4 | 77.4 | 65.9 | **77.4** | 73.7 |
| test_gui 原题型 AITW-general | 28.5 | 83.2 | 66.8 | 70.1 | 48.9 |

1. **过**：AITW 真图 − 配错图 +26.6pp、shopping +28.7pp；shopping 对 exp118 **+10.21 [+6.72, +13.70]**（AITW-general 对 exp118 +2.22 [−0.83, +5.26]）。
2. **过**：Mind2Web 原题型对 exp117 +0.00 [−2.08, +2.20]。（AITW-general 原题型对 exp117 −13.16，是改成看屏幕题型后丢了历史线索的文字先验，不计退步。）
3. **过**：对 exp117 无显著退步（CountBench −2.44 [−5.91, +1.22]、EgoThink −1.49、MME −0.84 均含 0）；CV-Bench **+1.71 [+0.45, +2.92]**（3D-depth +3.83 [+1.33, +6.33]），exp117/118 的 3D 退步未复现（底座相同的几次实验之间 CV-Bench 摆动 ±2，像种子噪声）。**对 v0.4 15 个旧集合全部含 0**（ScienceQA −0.14、dev_sqa −0.05、CV-Bench +0.34、CountBench −1.02、What'sUp −1.22、MMBench +0.23、MME +0.08、POPE +0.27、RealWorld 0.00 等）；新用途 test_ego +29.86 [+28.37, +31.48]、test_gui +35.39、test_gui_shop_state +38.37 [+33.72, +42.89]。
- **裁决：1–3 全过，exp119 配方是 `r22-*` 候选。** 逐题配对存于 `runs/exp119-v4-gui-phone-science/eval/compare/`（对 v0.4 与 exp117），lv_report 见同目录 `laya_vision.md`。下一步：exp120（深度 + ego 扩量）、[exp121](2026-10-28-exp121-v4-gui-scale.md)（手机 GUI 再扩 24k）判完后合并定 r22。
- lv_report（对 laya-vision 201M 逐题配对）：VQAv2 是非 +3.4 [+1.9, +4.8]、A-OKVQA +3.8 [+0.7, +7.0]、ScienceQA −0.5 [−2.5, +1.4]（与 exp117 的差值基本不变）；校准上 VQAv2 是非 ECE 0.115 vs 对方 0.062、ScienceQA 0.066 vs 0.028 —— 本实验未拟合发布温度，r22 需按独立校准集重拟合后再比。
