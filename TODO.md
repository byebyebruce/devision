# TODO

只记录已经确定要做的具体事项；方向、目标和备选想法不记在这里。完成后删除。

## 评测集统一命名：test / bench

在 v4 评测结束后、v5 开始前做（现在改会打断正在跑的评测）。

- **test**：我们自己数据里留出的测试划分（与训练、dev 图片不重叠），回答"这一轮比上一轮好多少"。现在的 `data/v2/eval_*`（POPE 除外）改名 `test_*`，与 LookFirst 的 `test` 划分对应。
- **bench**：外部公开评测，用来和别人比：laya-vision 的四个集合（现在的 `data/lv_bench/*`、`lv_*`、`*-lvbench.yaml`）和 POPE（现在的 `eval_pope`），统一叫 `bench_<名字>`，如 `bench_lv_aokvqa`、`bench_pope`。
- SwanLab 的评测汇总分 `test/`、`bench/`，训练中监测过或拟合过温度的集合仍记 `ref/`。
- 一个脚本改完：数据文件、`configs/*.yaml`、已有评测结果和逐题明细的文件名（`runs/*/eval/`，含 `compare/`）、文档里的引用、`data/v2/MANIFEST.json` 的键；改后对 v2 / v3 / v4 各跑一次只评测配置的 `--dry-run`，确认全部显示"已完成、跳过"，旧结果不用重跑。
