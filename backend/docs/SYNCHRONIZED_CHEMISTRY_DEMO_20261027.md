# 同课三模态化学演示

## 数据与用途

新课堂包位于 `artifacts/unified-chemistry-001-9`。语音、课件画面和可见行为均来自 `Chemistry_001_part9.mp4`，时长 90 秒。视频的 SHA-256 为 `bd955b05ac65bcae85890ecda939e8e027389f3e739f1ea3775299db5231df1d`。

本片属于 TS-Interact 的训练组，只用于功能演示。行为包沿用已完成的同源推理，不计入新留出评测。ARIC 数据未用于本片在线推理。原课堂包 `ts-chemistry-001-9` 保留原样。

## 语音转写

`asr.json` 保存新推理的 28 段机器转写。模型从本机缓存加载，固定版本为 `Systran/faster-whisper-small@536b0662742c02347bc0e980a01041f333bce120`。脚本校验模型文件哈希，不下载模型。

新推理使用同一录像的课件 OCR 词汇：认识氨基酸、甘氨酸、丙氨酸和谷氨酸。词汇直接来自原分辨率分区 OCR，只合并标签内部的空白。`asr_prompt_vocabulary.json` 记录词汇、来源秒数和证据 ID。这些词汇作为解码提示，不是参考台词。

本次选择的配置为 CPU int8、4 线程、`beam_size=5`、`vad_filter=True`、`condition_on_previous_text=False` 和 `repetition_penalty=1.15`。`initial_prompt=None`，`hotwords` 为上述实际 OCR 词汇。一次完整解码的实测耗时为 76.5 秒，包含本地模型加载。耗时会随电脑负载变化。

`asr.unguided.json` 逐字节保留原版 36 段无提示转写。`bundle.json` 的 `transcript_unguided` 字段也保留这 36 段，以便完整导出。第一次提示试验单存为 `asr.initial-prompt-first-pass.json`。该试验产生重复“氨”，未选作展示版本。

新版本仍含“二发氨基酸”“缩疾”等错词。没有人工改写台词，也没有计算词错误率或字错误率。界面应标为“课件词汇提示的机器转写”。提示结果与课件共享词汇来源，不能据此宣称独立验证了转写准确率。

## 实际课件画面

脚本每秒抽样一次，并检查视频末尾，共处理 91 帧。全片 OCR 的原始记录保存在 `raw_ocr.jsonl`，未筛选的画面分段保存在 `slides.dense-unfiltered.json`。

只有 43.5 秒至 47.5 秒显示投影课件。脚本额外读取半秒间隔的边界画面，并保存在 `screen_observations`。43.0 秒和 47.5 秒是学生镜头，43.5 秒至 47.0 秒可见同一页投影。保留区间的抽样误差为 0.5 秒。

课件标签使用原分辨率的小区域 OCR。程序按区域 OCR 平均置信度之和选择代表帧，本次选中 44.5 秒。该帧识别出“认识氨基酸、甘氨酸、丙氨酸、谷氨酸”。OCR 置信度仅用于选择代表帧，不代表字符准确率。化学结构式以 `slides/S0001.jpg` 的原始画面为准。

`slides.json` 只保留这一页实际投影。其他镜头没有课件证据，语音与课件匹配结果使用 `no_slide_evidence`。该页不会延伸到学生镜头。原始课件文件未随数据提供，另外整理的演示 PPTX/PDF 应保留“录像观测页整理”的来源说明。

## 行为证据

脚本先校验视频哈希，再复用原课堂的行为结果。181 张行为帧逐文件复制并核对 SHA-256，完整清单位于 `activity/copied_frames.sha256.json`。程序只替换课例媒体 URL 前缀并追加复用说明，保留原预测、检测框、跟踪编号和时间。

181 帧中，27 帧含有效规则指数，范围为 80 至 81.5012；154 帧的指数为 `null`。镜头切到课件时，当前学生行为不可见。规则指数描述可见行为线索，不能解释为心理投入、知识掌握或教学效果。

## 图谱展示规则

原模型提议了 12 个候选，其中包含“二发氨基酸”和“缩疾”等 ASR 错词。新脚本提供 `filter_graph_with_observed_courseware` 纯函数，只保留已经由模型提议、同时在实际视频课件 OCR 中出现的标签。匹配只忽略空白，不替换字符，也不引入新术语。

对本片的只读实测保留了 5 个候选：甘氨酸、丙氨酸、谷氨酸、氨基酸和氨基，共 10 条同段出现关系。函数同步过滤悬空边、时间匹配中的概念 ID 和报告概念章节，更新概念计数、关系计数和摘要。语音原文和行为结果保持相等。

函数返回新课堂对象与原候选归档对象。主流程在课堂写入锁内保存 `graph-candidates-original.json`，再发布过滤后的课堂。原候选继续可查，展示规则写入图谱 provenance。字面支持只说明候选出现在课件文字中，尚未教师确认，也没有验证化学术语准确率。

脚本的 `build()` 默认执行这项规则。首次构建归档完整候选，并将过滤结果写入 `graph.json` 缓存。后续缓存命中仍展示 5 个候选，保留原 12 个候选的归档。有独立课件时，程序在同一个写入锁内过滤原课堂、重新叠加课件、再次过滤并发布课堂。

```python
from scripts.build_unified_chemistry_demo import filter_graph_with_observed_courseware

# base 为未叠加独立课件的原课堂对象。函数本身不写文件。
filtered_base, archive = filter_graph_with_observed_courseware(base)

# 主流程在同一个课堂写入锁内保存 archive 和 filtered_base。
# 如需随后叠加独立课件，再过滤一次，避免新增非原模型候选。
final, _ = filter_graph_with_observed_courseware(overlaid, candidate_archive=archive)
```

## 复跑与核验

在课堂复盘项目目录，用本机原视频的绝对路径运行：

```powershell
& '.\.venv-media\Scripts\python.exe' '.\scripts\build_unified_chemistry_demo.py' `
  --video '<本机 Chemistry_001_part9.mp4 的绝对路径>'
```

脚本默认使用新目录，不覆盖原课堂包，也不改 `configs/lessons.json`。原生构建器所需的注册操作写入临时目录，随后删除临时目录。完成产物核验后，可由主流程登记新 ID 并设置入口。

可单独运行 `--stage prepare`、`--stage asr`、`--stage build` 或 `--stage verify`。准备阶段需要媒体依赖和本机 Tesseract 中文语言包。构建图谱需要已启动的本地语言模型服务。脚本的解码部分使用 [faster-whisper 官方接口](https://github.com/SYSTRAN/faster-whisper)，本次通过 Context7 核对了本地加载、提示词和条件解码参数。

已有完整 ASR 和课件抽样时，`--stage build` 不重跑语音，也不使用 GPU。2026 年 10 月 27 日的实测把真实课堂证据复制到临时新目录，使用已存模型候选缓存连续构建两次，再核验完整课堂。结果仍为 5 个展示候选、12 个归档候选和 181 张行为帧。两次构建加核验共耗时 3.749 秒，ASR 和实时语言模型调用次数均为 0，语音文件哈希保持相同。这项检查验证了缓存重建；从零提取候选时仍需要语言模型服务。

只核验已有包时运行：

```powershell
& '.\.venv-media\Scripts\python.exe' '.\scripts\build_unified_chemistry_demo.py' `
  --video '<本机 Chemistry_001_part9.mp4 的绝对路径>' --stage verify
```

`unified.verification.json` 记录实际计数。核验检查同源哈希、零偏移时基、完整媒体抽样、课件区间、行为图文件哈希和原 36 段转写保留。检查通过说明课堂包满足这些工程约束，不代表 ASR、OCR、行为分类或概念选择已通过人工准确率评测。
