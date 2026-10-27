# exp118：要看屏幕才能答的 GUI 下一步操作题（预写，2026-10-28 00:10，exp117 结果出来之前）

**背景**：exp116 的 GUI 题准确率 80.6%，但配错图仍有 78.9%（Mind2Web 配错图反而略高），模型靠 goal + 已做操作推断下一步，不看屏幕（见 [exp116 裁决](2026-10-27-exp116-v4-gui-ego.md)）。原因在题目：干扰项是同一屏上的其他元素，大多和 goal 无关，gold 的文字与 goal 重合；历史操作又直接给出了流程走到哪一步。

**改题**（`scripts/data/v05/exp118_gui_state.py`，seed 118，`data/v05/exp118/MANIFEST.json`）：同一题、同一截图，
- 去掉历史操作（state_text 置空）；
- 把至多 2 个同屏干扰项换成**同一 episode 其他步骤的 gold 操作**（都和 goal 相关，只有屏幕能说明现在处在哪一步，例如搜索框还没打开就不该 `type`）；选项数不变、位置重新打乱。
- 词重叠基线：test AITW 0.431→0.260、Mind2Web 0.446→0.276（随机 0.20）；test 1,540 题中 1,373 题换了 2 个、135 题 1 个、32 题 0 个。dev / test 的 episode 池只取各自文件，与训练不重叠。

**假设**：用这种题训练，GUI 的看图增益（真图 − 配错图）明显变大，且原题型 test_gui 不比 exp116 差太多。

**唯一变化**（相对 exp116）：训练混合里 12,000 道 GUI 题换成上述版本（id、图片、其余 65,048 题不变）；seed 116→118。init、学习率、视觉解冻、CE、1 epoch 不变。监测加 `gui_state`（dev 512 题的同一改写），评测加 `test_gui_state`。

**对照**：v0.4、exp116 在 `test_gui_state` 上（只评测 YAML `configs/{r21-soft,exp116-v4-gui-ego}-guistate.yaml`，接续脚本在 exp118 之前跑）；所有集合的配错图 / 倒序对照；exp116 在原 `test_gui` 上。

**判据**（`devision-compare` 按图 95%）：
1. **看屏幕**：`test_gui_state` 上真图 − 配错图 ≥ 10pp（两个来源各自 ≥ 5pp），且相对 exp116 区间下界 > 0。
2. **原题型**：`test_gui` 相对 exp116 不低于 −5pp（点估计），配错图增益不小于 exp116 的 1.7pp。
3. **ego 与旧能力**：test_ego 相对 exp116 无显著退步；其余集合判据同 exp116 判据 3（科学退步承自底座，只记录与 exp116 的差）。
- 1 过 → GUI 用改写题型扩量（加 aitw-shopping），并与 exp117 的科学修复合并成候选配方，再定正式轮次。
- 1 不过但 exp116 → exp118 真图准确率掉很多、配错图接近随机 → 256 letterbox 下屏幕不可读（文字 2–4px），转查分辨率 / 切块，不加步数。
- 3 不过 → 不是 GUI 题型问题，回到比例。

**预算**：基线评测约 5 分钟 + 训练约 25 分钟 + 评测约 50 分钟。`scripts/v05/chain_exp118.sh <exp117 pipeline pid>` 在 exp117 退出后自动开跑（日志 `runs/v05-chain/chain_exp118.log`、`runs/exp118-gui-screen-state.out`）。不拟合发布温度。

## 初步结果（2026-10-28 02:05，GUI / ego / 科学已评完，其余集合评测中）

| 集合 | v0.4 | exp116 | exp118 | exp118 配错图 |
|---|---|---|---|---|
| test_gui_state 总 | 25.3 | 39.0 | **45.5** | 35.1 |
| — AITW | | 42.4 | **56.6** | 34.8 |
| — Mind2Web | | 35.9 | 35.6 | 35.3 |
| test_gui 原题型 | 38.6 | 80.6 | 66.3 | 59.6 |
| test_ego（choice） | 30.5 | 67.0 | 66.6 | 46.5 |

- 判据 1：总体真图 − 配错图 +10.4pp、对 exp116 +6.5pp，但 **Mind2Web 0.3pp < 5pp → 按预写不过**；AITW +21.8pp，手机屏幕确实被看了。
- 判据 2：test_gui −14.3pp（< −5pp）不过；配错图增益 6.7pp ≥ 1.7pp。
- 按预写分支：Mind2Web 属于"屏幕不可读"（1024×800 桌面页 → 256 letterbox 文字约 3px），转查分辨率 / 切块，不加步数；手机部分可扩量 → [exp119](2026-10-28-exp119-v4-gui-phone-science.md)。
- 区间：test_gui_state 对 exp116 总 +6.49 [+3.83, +9.09]、AITW +14.27 [+9.97, +18.56]、Mind2Web −0.37 [−3.79, +3.06]；对 exp117：test_ego +0.73 [−0.10, +1.58]、EgoThink −2.32 [−4.64, +0.17]、ScienceQA −2.53 [−3.72, −1.38]、dev_sqa −2.28 [−3.56, −0.94]（exp118 无科学回放，预期内）、RealWorld −1.19 [−3.75, +1.37]。判据 3 其余集合（03:00 补，对 exp116 按图配对）：**CV-Bench −2.05 [−3.26, −0.87] 显著退步**；NaturalBench +1.25 [+0.66, +1.86]；What'sUp −2.20 [−5.00, +0.49]、CountBench +0.41、MMBench −0.70、MME −0.21、POPE −0.16、lv AOKVQA +0.53、lv VQAv2 是非 −0.65，均含 0。→ 判据 3 除 CV-Bench 外通过；CV-Bench 3D 退步与 exp117 同向，交给 [exp120](2026-10-28-exp120-v4-depth-ego.md) 用深度回放检验。
