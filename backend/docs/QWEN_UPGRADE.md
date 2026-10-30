# Qwen 语音与保守校对

本次试验使用 Qwen3-ASR-1.7B 识别原声，使用 Qwen3.5-4B 校对中文转写。
两个模型串行运行。问答与概念提取继续使用原 Qwen3-4B。

## 模型与来源

| 角色 | 模型 | 固定配置 | 输出 |
| --- | --- | --- | --- |
| 原声识别 | Qwen/Qwen3-ASR-1.7B，BF16 | `configs/qwen_asr.json` | 原生识别全文、预测时间 |
| 时间对齐 | Qwen/Qwen3-ForcedAligner-0.6B，BF16 | 同上 | 预测词级时间，不是人工标注 |
| 保守校对 | Qwen/Qwen3.5-4B，Q4_K_M | `configs/qwen35_llm.json` | 结构化候选、接受修改、拒绝原因 |

Qwen3.5 的 GGUF 来自 unsloth 转换仓库。
配置固定仓库 revision、文件大小和 SHA-256，并保存上游模型卡。
这次只做推理，没有训练或微调。

## 当前四个测试样例

主课堂为“认识氨基酸”的完整 90 秒原声。
另三段为已冻结的 Chinese-LiPS 官方测试音频。
新识别只接收音频，语言指定中文，没有课件词汇或人工稿提示。
官方人工参考只进入评分器。

旧识别与新识别各接入旧 Qwen3 和新 Qwen3.5，形成四组对照。
两种文字模型采用相同的保守提示和生成预算。
主课堂旧识别曾使用同课 OCR 提示，外部旧识别使用 SeacoParaformer。
因此，主课堂的识别差异属于具体配置对照。

评分定义、失败处理与复跑命令见 [评测说明](QWEN_UPGRADE_EVALUATION.md)。
实际输出保存在 `artifacts/verification/qwen-upgrade-20261028/`。
其中 `evaluation/comparison.md` 和 `comparison.json` 为可复核结果。
`status=partial` 表示仍有缺失或失败，不代表完整评测。

## 下载与独立识别

本次实际语音环境为 `.venv-qwen-asr-local`，与现有媒体环境分开。
实际使用 PyTorch 2.6.0+cu124、torchaudio 2.6.0+cu124 和 SymPy 1.13.1。
完整版本见 `asr/environment-freeze.txt`，固定依赖见 `requirements-qwen-asr-local.txt`。
下载 PyTorch 2.7.1 的尝试遇到网络失败，未用于本次推理。
当前环境只复制已有 CUDA 安装中的必要包，不继承系统 site-packages。
复制的关键文件通过源包 RECORD 校验，原环境未修改。
`asr/setup-replay.ps1` 保存了复建步骤；本轮只检查脚本语法，没有重新复建第二套环境。
运行前应确保显卡空闲。Windows 运行器持有项目模型锁。
原声识别结束后先卸载，再加载时间对齐模型。

从课堂复盘目录执行：

```powershell
.\.venv-media\Scripts\python.exe scripts/fetch_qwen35_llm.py
.\.venv-qwen-asr-local\Scripts\python.exe scripts/qwen_asr_remote_inputs.py --output artifacts/verification/qwen-upgrade-20261028/asr/model-inputs.json
.\.venv-qwen-asr-local\Scripts\python.exe scripts/qwen_asr_remote_download.py --config configs/qwen_asr.json --cache artifacts/verification/qwen-upgrade-20261028/asr/model-cache --output artifacts/verification/qwen-upgrade-20261028/asr
.\.venv-qwen-asr-local\Scripts\python.exe scripts/qwen_asr_remote_run.py --config configs/qwen_asr.json --inputs artifacts/verification/qwen-upgrade-20261028/asr/model-inputs.json --output artifacts/verification/qwen-upgrade-20261028/asr --cache artifacts/verification/qwen-upgrade-20261028/asr/model-cache --local-models --gpu 0 --min-free-mib 7000
```

最后一条为本机 8 GB 显卡的实跑设置。
显存记录包含设备型号、PyTorch 分配峰值及整卡采样，两者分别解释。
服务器连接不可用时，本机结果不能标成 A6000 测量。

## 发布独立课堂

`scripts/build_qwen_upgrade_demo.py` 将新识别包装为独立课堂。
脚本核验原声音频的 SHA、时长、完整覆盖和零偏移。
识别原文和预测时间保持原样，校对另存。

新的课堂编号为 `qwen35-chemistry-001-9`。
原课堂 `unified-chemistry-001-9` 保留。
同一录像的课件、行为预测和帧图像按原数据复用，并逐一核对图像哈希。

```powershell
.\.venv-media\Scripts\python.exe scripts/build_qwen_upgrade_demo.py --video 'E:\Research\Class Detection Dataset\TS-Interact\Chemisty\video\Chemistry_001_part9.mp4'
.\start_trained_demo.ps1 -LessonId qwen35-chemistry-001-9 -RestartApi
```

校对时会切换到 Qwen3.5，结束后恢复问答模型。
启动脚本检查已记录的进程身份，只管理项目配置的服务。

“语音与课件”中有折叠的“新模型试验”。
三列分别显示原识别、新识别与保守校对。
它们共享原声时钟，保留各自分段。
点击任一片段可回听对应原声，完整数据导出保留试验记录。

本课没有独立人工逐字稿，因此不计算本课字错率。
接受修改数不是纠错准确率，来源规则的修改也单列计数。
学生行为仍为可见动作预测，不因语音模型升级而获得新的心理积极度证据。
