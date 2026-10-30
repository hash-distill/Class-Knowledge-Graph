# 真实课堂复盘前端（v3）

首页默认从 `GET /api/review/lessons` 加载真实课堂列表，再读取对应课程包。没有课程显示空态；服务失败显示错误与重试，不回落到演示数据。当前语音、课件与图谱证据使用同一课程包中的原始时间。

## 使用

1. 启动复盘后端 `127.0.0.1:8000`，运行 `npm run dev -- --host 127.0.0.1 --port 5175`。
2. 打开前端首页。选择课程后，视频、整课时间轴与语音/课件面板联动。
3. 点击语音段落、课件页段、知识概念或证据按钮，可定位原录像；概念的多个独立区间逐条保留，不合并成首末时间之间的一整段。
4. 复盘助手支持“整堂课”和“当前片段”两种范围。请求保留提问时的课程、时间与范围；失败后重试使用同一上下文。
5. 回答展示真实的 `source`、`model` 和 `request_id`。引用必须存在于本课证据中，且时间与原记录一致。HTTP 错误、超时或引用校验失败只显示错误，不生成本地模板答案。
6. 完整 JSON 导出保留整个课程包；Markdown 导出保留报告、原文件与 SHA-256、证据时间、生成方式、处理来源与限制。

新增「独立课件对齐」：直接上传 PDF/PPTX/PPT，异步查看处理状态，预览原页，按实际观测区间人工确认或排除页面，移除关联后恢复原始证据。未匹配时跟随模式显示缺失，不回退到第一页。课件辅助修订单独展示原文、建议、页码与回听入口，JSON/Markdown 均保留来源。`coursewareValidation.ts` 校验课件引用、修订字符范围、原始转写和完整检索段的对应关系。后端要求与运行方式见 [COURSEWARE.md](../../backend/docs/COURSEWARE.md)。

可选 `evidence_blocks` 是连续原生 ASR 组成的检索段（B 编号）。机器校对单独保存为 P 段及 PB 证据块，保留原始 T 段、替换文字、依据与疑点。默认显示校对稿，点击“对照原始转写”查看识别原文。引用必须与对应证据的文字和时间一致。

课程时长直接取 `lesson.duration_sec`，不依赖人物快照数量。`lesson1-real` 当前课程包以音频时间为基准为 1145.092063 秒，页面显示 19:05；视频流约 1144.433313 秒，差约 0.66 秒。具体处理范围与限制以当前后端课程包为准。

没有确认同课同步的行为录像时，行为面板明确显示缺失，不生成参与曲线、投入分数或掌握度。机器转写、OCR、来源提取概念和模型提议关系都不能视作人工标签真值。

## 结构

- `src/components/review/LessonReviewWorkspace.vue`：教学工作台布局和课堂选择。
- `LessonPlayer.vue`：实际视频、独立整课时钟与证据时间条。
- `LessonEvidence.vue`：转写搜索、跟随播放和课件页段。
- `LessonGraph.vue`：直接展示课程包提供的节点、关系、状态及证据，不随机生成活动度。
- `LessonAssistant.vue`：实际模型问答、来源、引用与错误重试。
- `LessonReport.vue`：报告与来源导出。
- `src/stores/lessonReview.ts`：课程、播放、选择和问答状态；证据索引及概念区间按课程缓存，避免全量课堂反复建立索引。
- `src/utils/lessonReview.ts`：来源合同、时间与引用校验、证据索引、Markdown 导出。
- `src/api/reviewService.ts`：同源 API；模型密钥和配置只存在后端。

## 开发和生产

Vite 开发服务器将 `/api/review` 原路径转发到 `http://127.0.0.1:8000`。媒体只允许同源 `/api/review/media/...`，不把其他课程录像拼接进来。

运行 `npm run build` 进行类型检查与生产构建。生产目录 `dist` 只包含页面脚本、样式及品牌图标。媒体由复盘后端按课程提供。部署时配置同源 `/api/review` 反向代理，并支持媒体 Range 请求。

当前工作台通过 `lesson` 和 `view` 查询参数保存课程与功能视图。切换视图保留同一视频节点、播放时刻与表单状态。

## 验证

```powershell
npm run test:review
npm run test:activity
npm run build

# 实际后端、媒体与本地模型：需要相关服务已启动
python scripts/verify-proofread-students.py
python scripts/verify-unified-workbench.py
```

合同测试覆盖无快照真实时钟、空课程、来源完整性、重复概念之间的空白区间、引用跨课/时间错误、模型失败无假回答及重试上下文。浏览器测试覆盖语音→视频/PPT同步、概念证据回跳、实际模型来源与引用、导出、390 px 布局。`browser-verification.json` 明确区分合同模拟与实际 API/模型验证；这些通过不证明 ASR/OCR 或自动知识关系的人工准确率。
