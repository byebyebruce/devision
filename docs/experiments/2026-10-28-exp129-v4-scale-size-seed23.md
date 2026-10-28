# exp129：exp127 数据 + 尺寸回放，新种子 23（预写，2026-10-28 18:05，exp127 评测中）

**背景**：exp127（r22 + EPIC-VISOR 19.6k + AITW-general 状态题 9k）已评的 5 个新用途集合满足判据 1、2（test_ego 对 r22 +3.79 [+2.76, +4.85]，test_gui_state 的 aitw-general 组 +3.46 [+0.69, +6.09]，其余新用途无显著退步）；旧集合待评。exp128（r22 + coco-size 回放）排在 exp127 之后。两项都过时 r23 的候选是「exp127 数据 + 尺寸回放」，exp127 文档要求换种子复现后再定，所以这一步把组合和换种子合在一起提前跑，避免 GPU 在 exp128 之后空转。

**改动**：`scripts/data/v05/exp128_mix.py --seed 129 --base v05/exp127/train.jsonl --out v05/exp129`，exp127 训练文件不改，加 soft_mix 里其余 4,202 道 coco-size（与 exp128 同一批来源），共 196,859 题（`data/v05/exp129/MANIFEST.json`）。init、超参、监测集、32 个评测集 + controls 同 exp127；**seed 23**（r22 / exp127 / exp128 都是 22）。

**对照**：v0.4（r21-soft，逐题配对，这是 r23 的正式门槛）；exp127 / exp128 同时差数据和种子，只作参考，不据此单独归因。

**判据**（`devision-compare` 按图 95%，对 v0.4）：
1. 新用途：test_gui_state、test_gui_shop_state、test_ego 对 v0.4 显著提升。
2. 旧能力：其余全部旧集合与外部 bench 对 v0.4 无显著退步（含 test_size）。
3. 种子稳定：test_ego 与 test_gui_state aitw-general 组的点估计不低于 r22（即 exp127 的扩量收益在新种子下仍在）。
- 1–3 都过 → 以本配方写 `r23-*` 正式轮次（数据与超参照此，种子另定），评完与 v0.4 配对、跑 lv_report。
- 2 不过 → 记录退步集合，对照 exp128（同回放、r22 数据）判断是扩量还是种子所致。
- 3 不过 → 扩量收益不稳定，r23 回到 exp128 配方。

**预算**：约 24.6k 步训练约 60 分钟 + 评测约 90 分钟。`scripts/v05/chain_exp129.sh 655362` 18:03 挂在 exp128 链之后（日志 `runs/v05-chain/chain_exp129.log`、`runs/exp129-v4-scale-size-seed23.out`），dry-run 通过；评完自动 `lv_report --prev runs/r21-soft/eval`。预计 exp128 约 22:30 评完，exp129 约次日 01:00 评完。不拟合发布温度。

## 开训前修订（2026-10-28 19:15，exp128 训练中、exp129 未开始）

exp127 评完判据 3 不过：test_vqa_yesno 对 r22 −3.00、对 v0.4 −2.80，POPE 对 v0.4 −0.80（均显著），定位为自然照片是非题被新用途数据稀释（见 [exp127 结果](2026-10-28-exp127-v4-gui-ego-scale.md)）。原计划的 exp129 只加尺寸回放，会带着这两项退步，所以在开训前把数据改为：

- `scripts/data/v05/exp128_mix.py --seed 129 --base v05/exp127/train.jsonl --out v05/exp129r --extra vqav2/yesno=0 --extra coco-exist=3000`（脚本加了 `--extra`，默认行为不变，coco-size 那批与原 exp129 相同）：exp127 训练文件 + coco-size 4,202 + soft_mix 其余全部 vqav2 是非题 5,307 + coco-exist 3,000，共 205,166 题（约 25.6k 步）。新增题与 32 个评测集图片重叠 0（`data/v05/exp129r/MANIFEST.json`）。原 `data/v05/exp129/` 保留不用。
- YAML 只改 `data` 路径；seed 23、超参、评测集、判据不变。判据 2 的旧集合明确包含 test_vqa_yesno、POPE、test_size。
- 增加的分支：判据 2 中 test_vqa_yesno / POPE 仍显著退步 → 是非题回放不足以抵消，r23 改用 exp128 配方（r22 数据 + 尺寸回放）或把 EPIC-VISOR 新增量减半。
- 预算：训练约 65 分钟 + 评测约 90 分钟，exp128 评完（约 21:30）后由 `chain_exp129.sh` 自动启动，约次日 00:30 评完。
