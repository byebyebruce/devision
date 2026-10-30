# exp139-infographic：全量信息图 / 文档数据（InfographicVQA train + DocVQA，31k）加进 r24 配方（预写 2026-10-29 22:30）

**目的**：exp137 的 infographic 只用了 Cauldron 的 2.1k 图子集（5,733 题），dev +4.9 [−0.4, +10.1] 不显著、榜单 InfographicVQA skill 25.1 → 36.6，按判据没有进 r24。之后 `data/v05/board/infographic/` 换成完整版（InfographicVQA train 2,507 图 + DocVQA 3,582 图，31,374 题，同图答案作干扰项）。本实验验证：数据量放大约 5 倍后，信息图 / 文档读图是否显著提升、旧能力是否不退——目标是 v0.5 之后的下一轮（或 r24 不过时的替补配方）。

**改动（单因素，对 r24）**：`scripts/data/v05/r24_mix.py --families caption,kie,figure,depth,drive,hateful,infographic --out v05/exp139 --avoid v05/exp137/dev_board_infographic.jsonl v05/exp139/dev_board_infographic_full.jsonl` → `data/v05/exp139/train.jsonl` = r24 的全部数据（同种子同抽样）+ infographic 全部行。全部按 `board_exclusion.json` 过滤；新增的 `--avoid` 把两个 infographic dev 的图（按字节 / 像素 / 近似哈希，同榜单规则）排出训练——exp137 的 dev 图来自 Cauldron 子集，同一张信息图在完整版 train 里换了文件名，不按哈希排会泄漏。过滤行数写进 MANIFEST。训练参数、seed（2323）、初始化（v0.4）与 r24 完全相同；温度拟合集若与 exp139 训练图无重叠则直接用 r24 的 `calib.jsonl`（否则用脚本重抽的，并在此注明）。

**实际数据（22:15）**：299,432 行 = r24 全部 268,583 行（id 完全相同）+ infographic 30,849 行 / 5,671 图（全族 31,374 行里 525 行 / 97 图被近似哈希排除，即与两个 dev 的同一张图）；基础文件过滤 145 行、A-OKVQA 回放过滤 5 行，与 r24 相同。两个 dev 与训练图零重叠；r24 的 `calib.jsonl` 与 exp139 训练图零重叠，直接沿用（脚本重抽的保留为 `calib.script.jsonl`，MANIFEST `calib.replaced_by`）。dry-run 通过。

**评测**：r24 的 41 集 + 对照，另加 `dev_board_infographic_full`（完整版族内按图留出的 1,758 题：InfoVQA 777 / DocVQA 981，与训练图不重叠）；r24 与 v0.4 也在接续脚本里评这一集，供配对。

**判据**（预写，按图配对 95%）：
1. **有效**：`dev_board_infographic_full` 对 r24 下界 > 0，且 Decision Index InfographicVQA skill 对 r24 点估计上升（`judge_board.py exp139-infographic infographic_full --ref r24-v05`）。另记旧的 `dev_board_infographic`（285 题）与 KIE / CharXiv 的变化。
2. **旧能力不退**：r24 的其余集合上，exp139 对 r24 没有显著退步 > 1 点；POPE 对 v0.4 ≥ −1.0。
3. 校准：各集 ECE 不劣于 r24 超过 0.01。
三条都过 → 下一正式轮次（r25）加入 infographic；只过 1 → 查是哪类旧能力退，减量或加回放再试；1 不过 → 记为「256 输入下信息图数据量不是瓶颈」，不再加量（负责人已定不提高分辨率 / 不切块）。

**对照**：r24（同配方同种子、无 infographic）；另记 v0.4。

**预算与接续**：`scripts/v05/chain_exp139.sh 834364`（等 r24 链——当前唯一 / 最后一条链，包括它的配对、验收表和榜单）：r24 / v0.4 在新 dev 上的评测（约 3 分钟）→ 训练（约 30 万行，约 110 分钟）→ 42 集 + 对照（约 65 分钟）→ lv_report、对 r24 / v0.4 配对、验收表 → 榜单（约 10 分钟）→ 判据 1 机器口径写到 `runs/v05-chain/judge_exp139.json`。日志 `runs/v05-chain/chain_exp139.log`。预计 r24 链 23:15 前后结束，exp139 次日 02:30 前后评完。
