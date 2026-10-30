# 精析智评

面向教师的本地课堂复盘辅助系统，将课堂录像、语音转写、课件与学生可见行为放到同一条时间轴，支持证据回看、人工纠错、带引用问答和教学改进记录。

仓库地址：[hash-distill/Class-Knowledge-Graph](https://github.com/hash-distill/Class-Knowledge-Graph)。仓库沿用历史名称，应用名称统一为“精析智评”。

文件夹和文件名使用英文，系统界面与说明文档正文使用中文。

## 页面预览

以下截图来自本地运行的当前程序。课堂文本、人物示意与行为分布均为合成演示数据，不包含真实学生影像，不代表模型实测准确率或教学效果。

### 课堂概览

在同一条时间轴上回看课堂，重点展示学生状态、整体积极度和同刻转写。

![精析智评课堂概览](docs/screenshots/overview.png)

### 学生行为与视觉复核

查看当前对象与状态，按原始图像坐标补框、改标签或排除误检，复核结果独立保存。

![精析智评学生行为与视觉复核](docs/screenshots/student-review.png)

### 知识图谱

从本课概念查看关联关系，点击节点或证据回到对应片段。

![精析智评知识图谱](docs/screenshots/knowledge-graph.png)

截图来源与展示范围见 [页面截图说明](docs/screenshots/README.md)。

## 已有功能

| 工作区 | 可以完成的操作 |
| --- | --- |
| 课堂概览 | 以学生画面为主，联动当前状态、整体积极度和同刻转写 |
| 语音与课件 | 对照原始转写、机器校对、录像课件和独立 PDF/PPTX/PPT，点击证据定位原录像 |
| 学生行为 | 查看人物框、状态来源、趋势和评分支持情况，人工补框、改标签、排除误检 |
| 知识图谱 | 查看本课概念及其原文证据，共同出现关系可回到来源 |
| 复盘报告 | 导出 JSON 和 Markdown，保留原始观测与独立的派生评分 |
| 教学改进 | 保存目标、教师核对稿、改进计划、执行结果、练习和实际作答 |

人工复核保存版本和来源摘要，支持冲突检测、撤回、恢复和分类样本 ZIP 导出。后台课件任务使用 SQLite 持久化，重启后保留中断状态并支持重试。模型问答会检查引用编号；模型离线、证据缺失或数据不足时给出对应状态。

## 技术结构

| 层次 | 当前实现 |
| --- | --- |
| 界面 | Vue 3、TypeScript、Vite、Pinia，蓝色主题和响应布局 |
| API | Python 3.11、FastAPI、Uvicorn，保留 `/api/review` 接口 |
| 本地存储 | SQLite 保存教师记录、任务、检索索引和人工复核；JSON 保存原始课堂证据 |
| 媒体 | FFmpeg、OpenCV、Tesseract；媒体通过 API 提供 Range/HEAD 读取 |
| 本地问答 | 本课文本与时间检索、引用校验、独立的 llama.cpp 兼容服务 |
| 桌面端 | Tauri 2 与 Python sidecar；模型运行时和权重单独安装 |

浏览、编辑记录和回看已处理课堂不需要加载大模型。视频处理、视觉分析和语言模型使用独立脚本或进程。面向约 8 GB 显存的电脑，应按任务串行加载；显存占用取决于具体模型、上下文长度和运行配置。

## 模型与实际接入状态

| 任务 | 已有路径 | 说明 |
| --- | --- | --- |
| 主转写 | faster-whisper Small，CPU int8 | `prepare_media.py` 的默认处理路径 |
| 中文语音升级 | Qwen3-ASR-1.7B | 独立实验配置，见 [Qwen 升级说明](backend/docs/QWEN_UPGRADE.md) |
| 中文二次识别 | FunASR Seaco Paraformer、FSMN-VAD、CT-Punc | 可选 SSH 参考通道，需明确配置远程主机 |
| 人物与姿态 | YOLO26n、YOLO26n-pose，整图与切片融合 | 人物定位与行为分类分开；检测框数量不是班级人数 |
| 七类可见行为 | MobileNetV3 Small 裁剪分类器 | 已有约 3.15 MB 权重实验；低置信和冲突状态保留未知 |
| 概念与问答 | Qwen3-4B GGUF，llama.cpp | 本地模型服务，生成结果需通过引用检查 |
| 校对 | 来源规则、ChineseErrorCorrector4-4B、Qwen3.5-4B 对照 | 原始转写始终保留，机器校对待教师核验 |

运行时清单中的 Qwen3-ASR-0.6B、PP-OCRv6 ONNX 和 BGE-small 属于推荐安装项或待完成适配项，不能仅凭清单出现就认为已经部署。当前问答使用文本检索，没有把向量数据库或 GraphRAG 作为默认实现。

## 从源码启动

需要 Python 3.11、[uv](https://docs.astral.sh/uv/) 和 Node.js 22 或 24。以下命令在仓库根目录执行，先启动应用壳和 API；这一步不下载模型或课堂录像。

```powershell
git clone https://github.com/hash-distill/Class-Knowledge-Graph.git
cd Class-Knowledge-Graph
uv venv backend/.venv-media --python 3.11
uv pip install --python backend/.venv-media/Scripts/python.exe -r backend/requirements-app.txt
cd frontend
npm ci
npm run build
cd ..
python scripts/workspace_setup.py
python scripts/run_workbench.py --port 8000 --open-browser
```

打开 [本机工作台](http://127.0.0.1:8000/)。接口文档位于 [API 文档](http://127.0.0.1:8000/docs)，健康检查为 `/api/review/health`。服务只监听 `127.0.0.1`。

首次启动会创建空的课程登记和本地数据目录；页面没有课程是正常状态。源码不附带课堂视频、转写、学生图像、模型权重或个人工作区。已有登记不会被初始化脚本覆盖。

Linux/macOS 安装依赖时将解释器路径换成 `backend/.venv-media/bin/python`。源码启动器会识别两种虚拟环境路径；Windows 安装包及 PowerShell 媒体入口的支持范围见 [桌面与运行时](backend/docs/DESKTOP.md)。

### 前后端开发

在一个终端运行 `python scripts/run_workbench.py --api-only --port 8000`，在另一个终端进入 `frontend` 运行 `npm run dev -- --host 127.0.0.1 --port 5175`。Vite 将 `/api/review` 代理到后端。`npm run preview` 只用于检查静态构建；完整应用使用上面的同源启动入口。

若 API 在前端首次构建前已启动，构建后先在原终端按 `Ctrl+C` 结束，再启动完整应用。启动器会拒绝复用缺少静态页面的旧服务，并保留它运行。

`python scripts/run_workbench.py --check` 只检查路径、工具和构建状态。可使用 `--python`、`--registry`、`--data-dir` 指定解释器、课程登记和状态目录。

## 使用自己的课堂

已处理课程通过私有文件 `backend/configs/lessons.json` 登记，例如：

```json
{
  "my-lesson": {
    "directory": "../artifacts/my-lesson",
    "source_video": "../local-media/my-lesson.mp4"
  }
}
```

相对路径以登记文件所在目录为基准，也支持绝对路径。`directory` 需要包含完成且来源一致的 `bundle.json` 和对应证据文件；只登记一个视频路径不会自动生成分析。原始素材和登记文件请保存在本地。

处理新录像需要额外安装媒体依赖、FFmpeg、Tesseract 中文/英文语言包及相应模型，并配置真实的课件区域。具体命令见 [后端与课堂处理](backend/README.md)。人物行为分析见 [视觉模块](vision/README.md)。上传 PDF/PPTX/PPT 是向已登记课堂补充课件，不等同于上传录像后自动完成全部推理。

本地问答配置见 [本地模型说明](backend/docs/LOCAL_LLM.md)。若在服务端显式设置外部 `REVIEW_LLM_BASE_URL`，检索证据会发送到该服务。可选中文二次识别会把该课 WAV 发往明确配置的 SSH 主机；默认不选择远程主机。

## 整体积极度如何计算

当前界面使用 `overall_behavior_engagement_v1`：书写 85、阅读 80、抬头 85、转头 30、举手 95、站立 90、讨论 75。将有效行为分归一化后，计算 `100 × 均值 × exp(-0.6 × 总体标准差)`，连续同镜头再作时间平滑。概览、详情、曲线和导出共用同一实现。

至少一个可信行为对象即可给出估计分；不足 3 个对象或覆盖率低于 60% 时标记“低支持估计”。无人物、无有效行为、几何姿态或未识别对象不按零分处理。人工复核独立保存，尚不会自动重算全课模型曲线。权重、平滑、覆盖和导出字段见 [评分说明](backend/docs/CLASS_ENGAGEMENT_20261029.md)。

评分是检测对象的可见行为规则估计，尚未完成人工校准，不能解释成学生个人成绩、心理投入或知识掌握。分类器现有固定裁剪测试存在相邻画面重叠，其 Macro-F1 不能作为独立课堂端到端准确率；详见 [视觉改造记录](backend/docs/VISION_IMPROVEMENT_20261029.md)。

## 验证与发布

```powershell
& backend/.venv-media/Scripts/python.exe -m pytest backend/tests -q
cd frontend
npm test
npm run build
cd ..
python scripts/prepare_release.py --check
python scripts/prepare_release.py --output release/github-ready
```

源码导出生成 `source/`、`jingxi-insight-source.zip` 和逐文件 SHA-256 清单，过滤本地登记、数据库、课堂产物、权重、环境、构建目录和嵌套 Git 元数据。输出目录已存在时拒绝覆盖，常见凭据格式命中时中止导出。它不能替代人工检查文件内容和素材授权。

[GitHub 更新说明](docs/PUBLISHING.md) 说明统一仓库布局、历史视觉目录迁移和上传前检查。GitHub Actions 配置会执行后端测试、前端测试、生产构建和源码审计；模型识别准确率与真实课堂验收仍需单独评测。

## 目录与文档

```text
scripts/              初始化、源码启动与发布导出
backend/             FastAPI、媒体处理、证据、教师工作区与测试
frontend/           Vue 工作台
desktop/              Tauri 壳和打包脚本
vision/             人物定位、状态分类与历史视觉管线
docs/                 发布说明
```

- [人工视觉复核与训练样本导出](backend/docs/VISUAL_REVIEW_20261029.md)
- [本地任务与数据架构](backend/docs/LOCAL_ARCHITECTURE.md)
- [独立课件导入与对齐](backend/docs/COURSEWARE.md)
- [前端开发](frontend/README.md)
- [贡献约定](CONTRIBUTING.md)
- [许可与第三方材料](NOTICE)

现有课堂素材、训练数据、第三方代码和模型分别遵守其原有许可。仓库尚未声明覆盖全部代码的统一开源许可证；公开源码不自动授予课堂素材与模型权重的再分发权。
