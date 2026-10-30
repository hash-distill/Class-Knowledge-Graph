# 视觉复核与纠错 Implementation Plan

> **For agentic workers:** Use executing-plans to implement the checked tasks sequentially. User has selected the visual review workflow; keep this release bounded to that workflow.

**Goal:** 在当前帧逐对象确认、改标签、排除误检和补漏检，保存可撤回记录，导出带来源的人工确认样本

**Architecture:** 新增独立 SQLite 复核表，不修改课堂 bundle 和模型分数。记录绑定录像摘要、整帧模型结果和原始帧图片摘要，使用逐帧版本处理并发。前端锁定正在编辑的帧，草稿按课堂和帧隔离，导出 ZIP 中的完整标注清单及已确认学生状态裁剪图。

**Tech Stack:** Existing FastAPI, SQLite, Pillow, Vue 3, TypeScript and SVG pointer events

- [x] `tests/test_vision_review.py` 先验证保存、并发冲突、来源变更、撤回历史、无效坐标和只导出人工确认样本
- [x] `review/vision_review.py` 实现来源绑定、逐帧版本记录、历史与 ZIP 导出；`review/api/vision_review.py` 暴露接口；`review/api/application.py` 注册服务
- [x] `src/types/visionReview.ts` 和 `src/api/visionReview.ts` 提供类型与服务；`src/components/review/VisionFrameReview.vue` 提供蓝色复核工作台，锁定帧、拖框、编辑、撤回复核、错误和冲突处理
- [x] `LessonStudentState.vue` 加入复核入口，保留模型输出与原始帧；自动化数据只写入独立测试数据库
- [x] 运行后端回归、前端构建和浏览器操作验证，独立代码审查后修复问题，更新本地可运行版本与使用说明

导出规则：仅显式确认的学生且状态非 unknown 才生成七类状态样本。未知身份、未知状态及忽略对象保留在清单中，不作为分类真值；未复核对象不导出为真值。整帧标注是部分标注，不能作为完整人物检测训练标签。记录过期时不应用、不导出，历史仍可查。复核不自动改变全课积极度曲线，也不代表模型精度已通过验收。
