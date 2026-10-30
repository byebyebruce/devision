# exp140-board-scale：已判有效的榜单数据加量（预写 2026-10-30 00:15）

**目的**：r24 里的六类榜单数据都只取了 exp137 / exp138 试过的量（caption 20k / 103.6k、figure 12k / 90.7k、depth 10k / 17.1k、drive 6k / 9.0k；kie、hateful 已是全部），各自 dev 对 r23 涨 +5 到 +31，Decision Index 8.71 → 15.56。本实验检验：在 256 输入、同配方下，**这些类继续加量是否还涨**（数据量是否仍是瓶颈），以及加量是否开始挤占旧能力 / 校准（r24 的 NaturalBench NLL 已比 v0.4 显著差 +0.029）。不同于 exp139（加一类新数据 infographic），这里不加新来源，只改配比。

**改动（单因素，对 r24）**：`scripts/data/v05/r24_mix.py --families caption,kie,figure,depth,drive,hateful --amounts caption=40000,figure=30000,depth=all,drive=all --out v05/exp140`（新加的 `--amounts` 只覆盖行数，每图上限不变）。基础文件、A-OKVQA 回放、board_exclusion 过滤与 r24 相同；抽样种子相同但量不同，所以新增类的行不是 r24 的超集（记在 MANIFEST）。训练参数、seed 2323、初始化 v0.4 与 r24 相同；温度拟合集直接用 r24 的 `data/v05/r24/calib.jsonl`（构造后核对与 exp140 训练图零重叠，结果写在下面）。约 +46k 行（≈ 31.5 万行，约 39k 步 × 0.117 s ≈ 80 分钟）。

**评测**：与 exp139 相同的 42 集 + 对照；对 r24、v0.4 逐题配对；lv_report；验收表；Decision Index 全量。

**判据**（预写，按图配对 95%，对 r24）：
1. **加量有效**（每类分开判，`judge_board.py exp140-board-scale caption,figure,depth,drive --ref r24-v05`）：该类 dev 对 r24 下界 > 0，且对应榜单 skill 点估计上升。
2. **旧能力不退**：其余集合对 r24 没有显著退步 > 1 点；POPE 对 v0.4 ≥ −1.0。
3. **校准**：各集 ECE 不劣于 r24 超过 0.01；NaturalBench NLL 不比 r24 更差（点估计）。
1–3 都过的类 → 下一正式轮次（r25，若 v0.5 之后还有）用加量后的配额；1 不过 → 记为「该类在当前量已饱和」，不再加；2 / 3 不过 → 只保留不伤旧能力的类，回到 r24 配额。

**对照**：r24（同配方同种子，原配额）；另记 v0.4、exp139。

**接续**：`scripts/v05/chain_exp140.sh 885119`（等 exp139 链——当前唯一 / 最后一条链），日志 `runs/v05-chain/chain_exp140.log`。预计 exp139 链 01:45 前后结束，exp140 约 04:30 前后评完 + 榜单。

**不占 GPU 的同期工作**：release/v0.5 改指 r24 的材料（模型卡说明、CPU 延迟、hf.py build / verify），等负责人决定。
