# 更新 GitHub 仓库

目标仓库为 [hash-distill/Class-Knowledge-Graph](https://github.com/hash-distill/Class-Knowledge-Graph)。应用标题使用“精析智评”，保留已有仓库地址。简介建议为“精析智评｜面向教师的本地课堂复盘辅助系统，整合语音转写、课件对齐、学生可见行为分析、人工纠错、知识图谱与教学改进记录，支持 FastAPI、Vue 3 和 Tauri 桌面端”。

## 生成统一源码

原工作区内的视觉、前端和历史图谱目录有各自 Git 元数据。应用根目录尚不是统一 Git 仓库，不要直接在根目录执行 `git add .`。

在应用根目录运行：

```powershell
python scripts/prepare_release.py --check
python scripts/prepare_release.py --output release/github-ready
```

第二条命令要求输出目录不存在。生成的 `source/` 是统一源码，ZIP 是同一份快照，`manifest.json` 记录原始路径、大小和 SHA-256。不会改动旧仓库、源视频、环境、数据库或模型。

源码过滤覆盖环境、缓存、构建、权重、课堂产物、私有登记和前端历史媒体导出。提交前仍需查看清单及文本中的个人路径、SSH 配置和材料权利；凭据规则只识别常见格式。

## 保留远程历史

使用单独的新目录克隆目标仓库并建立发布分支，将干净源码覆盖进去。现有远程根目录的 `Class_Detection/` 属于旧视觉布局；统一布局为 `vision/Class_Detection/`。迁移时保留 Git 历史，将已有视觉文件移入模块目录后再应用新的视觉源码；同时移入旧视觉 requirements、训练手册和 GPU 检查脚本。

旧远程 `.idea` 和快照 JSON 属于早期 IDE/课堂输出，不再纳入新的源码树。它们的历史版本仍保留在旧提交中。若使用自动准备好的独立 checkout，先核对 `git status --short`、`git diff --stat` 与 `git diff --cached --stat`，不要对原工作区执行清理或重置。

推荐先推送独立发布分支，在 GitHub 比较目录、测试结果与 README 后再合并。不要强制推送，不要把新工作区的数据目录或模型路径纳入提交。克隆、提交和推送使用维护者自己的 Git 身份及登录方式，不在脚本中写入访问令牌。

## 本地验收

在全新源码目录安装应用依赖，执行后端测试、前端 `npm ci`、`npm test` 和 `npm run build`。运行 `scripts/workspace_setup.py` 后应得到空登记，使用 `scripts/run_workbench.py` 启动时应能访问健康接口、空课程列表和界面。

CI 工作流只验证代码与构建，不下载课堂模型、不访问私有素材，也不发布安装包。真实录像识别、语言模型输出与 Windows 安装向导分别验收。确认整体许可证前，README 不添加未经授权的开源许可徽章。
