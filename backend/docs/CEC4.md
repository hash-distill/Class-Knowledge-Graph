# CEC4 转写校对与实测对比

## 模型分工

转写校对使用 `configs/correction_llm.json`，对应 ChineseErrorCorrector4-4B 的 Q4_K_M 文件。原始语音识别模型继续保留，CEC4 接收识别后的文字。问答和概念提取仍使用 `configs/local_llm.json` 中的 Qwen3-4B。

CEC4 权重来源为 [固定版本 GGUF 仓库](https://huggingface.co/mradermacher/ChineseErrorCorrector4-4B-GGUF/tree/33952accc8b460769e3820ba0a794f558eec8f0b)。文件为 `ChineseErrorCorrector4-4B.Q4_K_M.gguf`，SHA-256 为 `b86247b60c844140944610cec31806f3c21150c9d53ca91d914dbc23a5c188db`。本项目没有重新训练权重。

本机只有 8 GB 显存。生成校对稿时，程序核对进程身份，卸载问答模型后再加载校对模型。校对完成或失败后会恢复问答模型。模型任务期间持有系统文件锁，并发请求返回“模型忙”，避免中途切换。

## 查看当前课堂

打开 [课堂复盘](http://127.0.0.1:5175/?lesson=unified-chemistry-001-9)。在“语音与课件”展开“校对模型对比”。可以切换守卫后结果与模型直接输出，按时间回听原声。页面分别显示明确模型贡献和来源规则贡献。

模型直接输出只供核验。图谱和问答使用通过校验的校对证据。原始 ASR、时间戳、源音频和录像不覆盖。完整 JSON 导出保留对比、请求记录及测量来源。

## 复跑

已有权重与运行环境时，不必重新下载：

```powershell
cd 'D:\全球校园人工智能精英大赛\4C\课堂复盘'
.venv-media/Scripts/python.exe scripts/fetch_correction_llm.py --verify-only
./scripts/start_local_llm.ps1
./start_review.ps1
```

单独生成新的校对比较文件：

```powershell
.venv-media/Scripts/python.exe scripts/prepare_proofreading.py `
  --media-dir artifacts/unified-chemistry-001-9 `
  --video 'E:\Research\Class Detection Dataset\TS-Interact\Chemisty\video\Chemistry_001_part9.mp4' `
  --output artifacts/verification/cec4-comparison/manual-cec4-run.json
```

测量脚本会串行运行两个模型，并暂时占用本机模型服务。脚本只写比较目录，保留课堂原始文件：

```powershell
.venv-media/Scripts/python.exe scripts/run_correction_benchmark.py `
  --media-dir artifacts/unified-chemistry-001-9 `
  --video 'E:\Research\Class Detection Dataset\TS-Interact\Chemisty\video\Chemistry_001_part9.mp4' `
  --output-dir artifacts/verification/cec4-comparison `
  --external-dir artifacts/verification/cec4-comparison/external-chinese-lips
```

比较协议及评分字段见 [纠错评测说明](CORRECTION_COMPARISON.md)。显存单位为 MiB，来自 `nvidia-smi` 整卡采样。报告同时保留加载前基线、加载后占用和运行峰值。采样峰值可能遗漏瞬时变化，不能解释为模型自身的精确分配量。

## 参考稿与数据范围

当前 90 秒化学课没有找到可确认的人工逐字稿。同名 TS-Interact 文本标签谈论兔子种群，与氨基酸视频不符，因此不参与文字准确率评分。主课只报告逐项改动、术语组出现、规则拦截和实测资源占用。

外部小样本来自 [Chinese-LiPS 官方数据集](https://huggingface.co/datasets/BAAI/Chinese-LiPS)。本次选择 3 位讲者的 3 段原声，总长约 37 秒。数据含配套课件视频、讲者视频及官方人工转写。`PPT.mp4` 是幻灯片视频，不能称为可编辑 PPTX。

外部 ASR 独立识别原始 WAV，人工稿只交给评分器。先汇总字符错误数，再除以参考字符总数；同时保留严格字错率和去除空白标点的字错率。正确参考字符受到的损伤单列，新增字符错误另列。没有建立训练数据排除证据，因此这不是独立保留集基准。

Chinese-LiPS 的许可为 CC-BY-NC-SA-4.0。原始样本保存在本机核验目录，不随项目源码打包。检索证据、文件哈希与配套核对记录见 `artifacts/verification/cec4-comparison/external-chinese-lips/`。
