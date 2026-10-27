# exp87：D3-lite 原生决策基线

2026-10-24 21:04 +08:00，预写，尚未取得模型结果。

假设：原生图文预训练并经决策训练的骨干，在真实UI同题对照上具有可用优势，且原生候选读出减少exp86的选项位置敏感性。初始化为vllm-sr/d3-lite固定revision 03560c26f5199c3c04129a2ccb38960c913635e0，Apache-2.0；零训练，不生成checkpoint。不是deVision已完成集成的候选，不据此发布。

本轮先从国内HF镜像下载固定代码、配置并检查依赖与本地兼容性，随后下载权重，在现有CUDA/uv环境使用发布者D3.system_one原生概率接口，batch=1，BF16主干/FP32读出、T=1、默认图像像素预算，不做permutation_average。不生成文本、不把gold放进输入。保留manifest、实际runtime_info、输入哈希与逐题结果；与exp81并非相同参数/像素预算，收益不能归因单一因素。外部预训练图片交集未知。

固定输入data/v05/exp85/ui_dev.jsonl、ui_dev_rotated.jsonl各256题，额外原图倒序256题；64四元组bootstrap5000次，seed86。主要门槛沿用exp86：相对exp81至少+5pp且95%区间下界>0，真图减轮换错图至少+3pp且下界>0；同时披露倒序准确率和语义答案改变率，>20%则不能直接扩大训练，先处理协议。通过仅晋级图表、照片、旧能力面板，未通过停止该初始化的扩量。原始NLL/Brier仅描述，正式概率验收另设隔离校准集。

预算：下载+加载+768题总墙钟1800秒，逐题落盘，每16题报告，外层timeout1850秒。镜像下载失败、非有限概率、候选键/概率和不符、OOM或前向异常立即退出并保留目录；不原样重试、不安装依赖。启动前GPU0%、144MiB，无活跃GPU任务；不干预其他任务。输出runs/exp87-d3-ui，日志supervisor.log，退出码EXIT_CODE；首题成功后核验实际进度。若依赖不兼容，记录最小故障并决定下一个候选，不占卡等待。

正式ImageJevBench300、五用途、独立校准和性能验收仍阻塞。周日12:00决策路线、18:00冻结训练、23:00回归与校准/性能、周一08:00完成材料并保留09:00前发布缓冲；正式榜单受理不能由本地执行保证。

## 21:08 兼容性预检与修订

首个独立目录runs/exp87-d3-ui下载结束后在manifest校验失败：下载过滤遗漏chat_template.jinja与merges.txt，尚未加载模型/占用GPU。保留全部目录与traceback。补全过滤，不跳过manifest校验。

CPU实际AutoProcessor预检发现缺少torchvision（视频组件强制导入）；不安装环境。增加静态图适配scripts/v05/d3_image_processor.py，原生Qwen3VLProcessor的图片路径配Qwen2VLImageProcessorPil，从同revision读取全部预处理参数，保留原像素上限；仅允许video_processor=None，明确拒绝视频。本地实际图片处理检查PASS：输入token范围13–248056、510视觉占位与image_grid计算一致、pixel_values[2040,1536]。不声称与torchvision后端逐位一致；这是披露的预处理适配因素。

修订输出runs/exp87-d3-ui-pil；其余固定集/门槛/1800秒预算不变。运行身份增加适配器源码哈希。原始失败未删除，无训练checkpoint产生。

## 21:11 完成与决策

修订独立运行实际exit0、768条完整逐题，GPU释放后实查0%、144MiB，无残留进程。真实256/256，错图0/256，倒序255/256；语义改变率1/256=0.390625%。相对exp81 +59.765625pp，64四元组5000次bootstrap95%区间[55.078125,64.453125]pp；真图减错图+100pp。P1/P2及顺序诊断均通过，晋级其他面板。全部结果runs/exp87-d3-ui-pil/native_report.json，实际身份IDENTITY.json；脚本report_exp87.py复算。极高准确率仅支持该开发面板的视觉文字绑定，外部预训练暴露未知，不作为正式泛化或五用途达标结论。

已预写exp88并于本任务退出且核对GPU空闲后，启动图表/照片/旧能力全量真图与轮换错图，独立runs/exp88-d3-panels。没有扩大Smol提示搜索，没有训练或发布新权重。原exp87失败目录、exp86失败目录与exp81控制均保留。
