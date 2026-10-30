# 精析智评前端

Vue 3、TypeScript、Vite 和 Pinia 构建的课堂复盘工作台。完整启动、模型与数据准备见 [应用 README](../../README.md)。

## 开发与构建

使用 Node.js 22 或 24，在本目录执行：

```powershell
npm ci
npm run dev -- --host 127.0.0.1 --port 5175
```

根目录另开终端运行 `python scripts/run_workbench.py --api-only --port 8000`。Vite 将 `/api/review` 代理到 `127.0.0.1:8000`。生产构建后由 FastAPI 同源提供页面与 API：

```powershell
npm test
npm run build
```

`npm run build` 包含 Vue/TypeScript 检查与 Vite 构建。`dist/` 只含应用与品牌资源，历史 `public/` 课堂图片和快照不会复制进生产包。`npm run preview` 用于静态预览，完整功能请使用后端同源入口。

## 页面与数据

课堂概览以学生画面、可见状态、整体积极度和同刻转写为主；课件在语音与课件页面查阅。学生行为支持来源查看、时序、人工补框、标签纠正、误检排除、版本历史和样本导出。另有知识图谱、复盘报告与教学改进页面。

默认课程来自 `/api/review/lessons`。空数据、服务故障、模型离线与未知观测显示对应状态，不自动切换演示。前端不保存模型密钥，不直接启动 GPU 推理。

| 代码 | 职责 |
| --- | --- |
| `src/components/review/` | 课堂工作台与六个页面 |
| `src/api/` | API 客户端、错误处理与视觉复核请求 |
| `src/utils/classEngagement.ts` | 当前帧、趋势和导出共用的评分 |
| `src/utils/studentStates.ts` | 标签与来源解释 |
| `scripts/test-*.mjs` | 合同、记录、行为、视觉草稿和评分测试 |

当前派生评分以独立 `derived_engagement` 字段导出，不修改原始观测。未知对象不按零分处理；低支持估计与人工复核分开展示。规则见 [评分说明](../../backend/docs/CLASS_ENGAGEMENT_20261029.md)，人工纠错见 [复核说明](../../backend/docs/VISUAL_REVIEW_20261029.md)。

## 验证

`npm test` 依次运行 `test:review`、`test:activity`、`test:vision-review` 和 `test:engagement`。浏览器检查脚本在 `scripts/verify-*.py`，需要单独的 Playwright 环境及对应本地课堂；浏览器证据留在忽略的 `output/` 目录，不打入源码包。

历史数据合同与接口说明保留在 [LESSON_REVIEW.md](LESSON_REVIEW.md)。其中既有课堂数据与本机测试路径属于开发记录，不是新克隆随附素材。
