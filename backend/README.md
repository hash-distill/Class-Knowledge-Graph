# 精析智评后端

后端使用 FastAPI、Uvicorn 和 SQLite，为 Vue 工作台提供 `/api/review` 接口。完整应用从 [根目录 README](../README.md) 启动；本目录说明数据、处理链路和模型配置。

## 应用环境

在应用根目录执行：

```powershell
uv venv backend/.venv-media --python 3.11
uv pip install --python backend/.venv-media/Scripts/python.exe -r backend/requirements-app.txt
python scripts/workspace_setup.py
python scripts/run_workbench.py --api-only --port 8000
```

前端构建后，去掉 `--api-only` 即可从同一端口打开界面。接口文档在 `/docs`，健康检查为 `/api/review/health`。启动应用不会加载 GPU 模型；推理库和模型按任务单独安装。

应用配置如下。相对课程路径以登记文件所在目录为基准；启动参数优先于环境变量时，以具体入口说明为准。

| 配置 | 用途 |
| --- | --- |
| `configs/lessons.json` | 私有课堂登记，默认空对象，不随源码发布 |
| `REVIEW_LESSON_REGISTRY` | 直接后端或桌面入口使用的登记文件路径 |
| `REVIEW_DATA_DIR` | 任务、教师记录、复核及索引数据库目录 |
| `REVIEW_FRONTEND_DIR` | 已构建界面目录 |
| `REVIEW_RUNTIME_ROOT` | 模型与公开配置根目录 |
| `REVIEW_LLM_BASE_URL` | 兼容模型接口，默认 `http://127.0.0.1:18081/v1` |
| `REVIEW_LLM_MODEL` / `REVIEW_LLM_API_KEY` | 服务端模型别名及可选密钥，不放到前端 |

根目录 `run_workbench.py` 使用自己的默认登记、状态和运行时路径；自定义这些路径时传入 `--registry`、`--data-dir`。已有课程登记和证据不被初始化覆盖。

## 生成一节课堂

新录像处理需要 FFmpeg、Tesseract（`chi_sim` 与 `eng`）和额外媒体依赖。以下命令从本目录执行，路径为用户自行准备的本地素材：

```powershell
uv pip install --python .venv-media/Scripts/python.exe -r requirements-media.txt -r requirements-app.txt
& .venv-media/Scripts/python.exe scripts/fetch_local_llm.py --runtime cpu
./scripts/start_local_llm.ps1 -Backend cpu
./run_lesson.ps1 -Video ./local-media/my-lesson.mp4 -LessonId my-lesson -Title 我的课堂 -Roi 0.49,0.055,0.91,0.47 -SkipModelStart
```

需要 CUDA 引擎时按 [本地模型说明](docs/LOCAL_LLM.md) 选择下载和启动配置。下载模型需要网络、磁盘空间及其适用许可。上面的 ROI 仅是参数格式示例，必须改为本课真实屏幕区域，不能直接用于所有机位。该一键媒体入口目前面向 Windows PowerShell。

流程先完成主转写与录像屏幕 OCR，再核验视频摘要、时间基准和处理范围，生成校对、证据、概念与报告，最后登记课堂。`status != complete` 或 `.partial` 文件不作为完成结果。换录像或配置时使用新课堂 ID，避免混用旧缓存。

主要文件如下：

| 文件 | 内容 |
| --- | --- |
| `asr.json` | 原始转写、时间戳、模型版本和源文件摘要 |
| `slides.json` | 实际录像课件画面段与 OCR |
| `raw_ocr.jsonl` / `ocr_frames/` | 可回看的采样与原始 OCR |
| `transcript.proofreading.json` | 独立机器校对，原始 ASR 保留 |
| `graph.json` | 有证据的概念及共同出现关系 |
| `bundle.json` | 前后端 v3 课堂数据合同 |
| `report.json` / `report.md` | 生成时的课堂报告 |

录像内屏幕与独立 PDF/PPTX/PPT 分开保存。时间重合不自动证明语义匹配；未匹配、歧义与画面缺口保留状态。独立课件的渲染依赖与操作见 [课件说明](docs/COURSEWARE.md)。

## 模型配置

主转写默认 faster-whisper Small CPU int8。Qwen3-ASR-1.7B 和 Qwen3.5-4B 是独立升级实验，见 [Qwen 说明](docs/QWEN_UPGRADE.md)。CEC4 校对角色见 [CEC4 说明](docs/CEC4.md)，它的机器修订不是人工逐字稿。

Seaco Paraformer 仍作为可选中文二次识别通道，配合 FSMN-VAD 和 CT-Punc。脚本不再内置服务器地址、用户目录或 GPU 编号。使用前明确设置：

```powershell
$env:REVIEW_ASR_SSH_HOST = 'user@your-server'
$env:REVIEW_ASR_REMOTE_PYTHON = '/path/to/asr-env/bin/python'
$env:REVIEW_ASR_REMOTE_ROOT = '/tmp/jingxi-asr'
$env:REVIEW_ASR_GPU = '0'
./run_lesson.ps1 -Video ./local-media/my-lesson.mp4 -LessonId my-lesson -Title 我的课堂 -ChineseReference
```

远程环境须另行安装 `requirements-asr-chinese.txt`。启用后该课 WAV 会通过 SSH 发送到指定主机，产物为独立的 `asr.secondary.json`，绑定录像和 WAV 摘要。未配置主机时脚本在读取媒体和连接服务器之前退出。

概念提议和问答通过独立本地服务运行 Qwen3-4B；服务端检索本课证据，限制并复核回答引用。离线、超时、截断和无效引用返回对应错误。若显式配置外部接口，检索文本将发送到该接口。

## 学生行为与人工复核

视觉分析由 [视觉模块](../vision/README.md) 离线生成，工作台核验来源后读取。概览以学生画面为主，当前帧、趋势与导出共享 [整体积极度规则](docs/CLASS_ENGAGEMENT_20261029.md)。无有效观测不填零，低支持估计带提示。

[视觉复核](docs/VISUAL_REVIEW_20261029.md) 支持改标签、角色、框、误检排除、补漏检、版本恢复和样本 ZIP 导出。复核保存在 `workspace.sqlite3` 的独立表中，不覆盖模型结果。只有明确确认的学生已知行为会生成分类裁剪样本；部分标注帧不能直接作为完整人物检测训练标签。

教师目标、核对稿、改进计划、实际作答和执行结果独立入库。后台课件任务由 SQLite 队列与子进程处理，启动使用单个 Uvicorn worker。恢复机制见 [本地架构](docs/LOCAL_ARCHITECTURE.md)。

## 测试与数据边界

```powershell
& .venv-media/Scripts/python.exe -m pytest tests -q
```

单元测试主要使用合成夹具，冻结后端测试需要明确提供已构建二进制。它们验证实现，不证明真实课堂识别准确率。真实验收记录存于本地 `artifacts/verification`，不随源码分发。

图谱关系表示有来源的共同出现，尚未验证先修或因果关系。整体积极度是可见行为规则估计，未校准为心理投入、掌握程度或教学效果。原视频、时间戳和原始转写始终作为核对来源。
