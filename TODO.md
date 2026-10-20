# TODO

只记录已经确定要做的具体事项；方向、目标和备选想法不记在这里。完成后删除。

## 1. 换设备后先做 V 阶段验证（EXP76 之前）

背景：EXP76 两次 MPS 运行都被 GPU 故障（InnocentVictim）打断，没有结论；原机器上的故障会悄悄改变输出。方案和判据见
`docs/experiments/2026-10-20-exp76-device-plan.md`，代码在 `scripts/device_v/`（已验收，没有真实运行过）。

- **同一台 Mac 重启后**：
  1. 只读核对新的开机时间、代码和证据的身份、没有模型进程；
  2. 取得单次许可后运行 `nohup bash scripts/device_v/run_v.sh > runs/exp76-device-v.out 2>&1 &`。
  只运行一次，总上限 2,700 s。V1 与 EXP76-RUN-02 的 step0 逐题一致（1e-6），V2 在 250 次更新后由预定截停结束，V3 做保存和加载检查。
- **换到 CUDA 机器**：现有的 V 代码按 macOS / MPS 编写（开机周期用 `sysctl kern.boottime`，系统事件读 `/Library/Logs/DiagnosticReports`，配置写的是 `device: mps`）。要先补：
  1. CUDA 的一致性判据：跨设备不能要求逐位一致，阈值先写好、交评审，再运行；
  2. Linux 上对应的开机和系统事件检查（或者明确写成不适用）；
  3. 配置只改 `device`，其他都不动（R 起点、数据、seed、`amp: off`、1,054 步日程）。
- 新机器要从本机拷贝的文件（`runs/`、`data/` 不在 git 里），拷完逐个核对 SHA：
  - R 起点 `runs/exp75-panel-r/r/`：model.safetensors b7f471fc…、config.json 7fb24ad3…、align_report.json 2721ecd6…；
  - EXP75 证据 `runs/panel-number/{compare.json,report.json,r/numbers.jsonl}`（EXP76 预检要用）；
  - V1 参照 `runs/exp76-panel-decision-r2/d2/monitor/step0/*.details.jsonl`（8 份，SHA 见方案第 2 节）；
  - 数据 `data/panel_learn/`（含 MANIFEST）、`data/protect_p/slice_K_text.jsonl`、`data/r13b/panel_{exist,science}.jsonl`，以及这些题引用的图片（`data/exp21/images/` 等）；
  - HF 模型缓存（laya、siglip2），运行时设置 `HF_HUB_OFFLINE=1`。

## 2. V 通过后：EXP76 完整运行（单独许可）

- 配置 `configs/exp76-panel-decision-r2.yaml`（只为换设备改 `device`，其余不变）。用来回答：从 R 起点开始，普通的决策题能不能学会。
- 时间预算：先用 V 的实测值重算（各部分耗时的上界 × 1.15 + 148 s；按 MPS 的速度约 140 分钟）。在运行之前把上限写进 runner，不能先运行再改截止。原来的 95 分钟即使不出故障也跑不完。
- 全程用已验收的守卫：出 GPU 错误就停，不自动重试。

## 3. v0.4 两份受影响的输出要补测

- `test_gqa.mismatched`（992 题）和 `test_bind_gqa` 真图（1484 题）运行时日志里有 GPU 错误，执行证据待补。见 `docs/experiments/2026-10-20-v04-gpu-evidence-errata.md`。
- 用同一冻结身份（`runs/v04-k/eval/IDENTITY.json` c14863e2…、`v04-freeze` 分支的代码）完整重测这两份，再加 `test_gqa` 真图作为逐位复现对照。原封存文件不改，新结果另存，再重算依赖它们的比较（相对 v0.3 的配对、看图收益、倒序翻转率）。
- 设备尽量与其余 92 份一致（重启后的 MPS）。换设备重测时，配对比较要另加对照。

## 4. 发布 v0.4 前还缺

- RELEASE-GAP-01：dev16 读数 36.8%、比较 41.5%，发布要求 ≥ 80%。EXP76 就是为这一项做的。
- 第 3 项补测完成之前，`test_bind_gqa` 相对 v0.3 的 −2.2 不能作为发布依据。

## 5. 本机待负责人决定

- 磁盘：旧实验 checkpoint 46 个目录约 93 GB（建议只删权重、保留结果），`data/raw/hf` 约 118 GB（只在重建数据时需要）。
- `devision-v04reader` worktree 里有一处未提交的修改：`docs/experiments/2026-10-18-v04-local-release-prep.md`。
- 主仓库 develop 工作区里另有未提交的改动：demo 小游戏、`serve.py`、若干文档。这次 push 不包含它们。
