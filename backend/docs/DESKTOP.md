# 精析智评本地桌面与运行时

桌面壳使用 Tauri 2 和 Windows 已安装的 WebView2。Python 后端作为独立子进程运行，只监听 `127.0.0.1`。页面、课程证据、工作区和模型运行时分开存放。

## 直接运行源码

在 `4C` 目录执行：

```powershell
& .\desktop\start_source.ps1
```

入口使用 `backend/.venv-media`，打开系统浏览器。首次运行前须在 `frontend` 执行 `npm ci`、`npm run build`，并安装 `backend/requirements-app.txt`。完整新环境安装见 [应用 README](../../README.md)。

该便捷入口默认把工作区保存在 `desktop/workspace/source-state`，也可传入 `-DataDir` 或设置 `REVIEW_DATA_DIR`。

也可以明确指定登记表、工作区和前端：

```powershell
& .\backend\.venv-media\Scripts\python.exe .\backend\scripts\desktop_entry.py `
  --port 0 --registry .\backend\configs\lessons.json `
  --frontend-dir .\frontend\dist --reuse --open-browser
```

`--port 0` 由操作系统分配空闲端口。进程启动后打印 `4C_DESKTOP_READY` 和本次地址。默认工作区位于 `%LOCALAPPDATA%/org.4c.workbench/state`，可以用 `--data-dir` 更改。源码入口默认只读现有 `configs/lessons.json`。关闭命令窗口或按 `Ctrl+C` 可停止自己启动的后端。

`--reuse` 只复用同一工作区内、启动参数指纹一致且通过私有随机令牌握手的桌面后端。程序不会根据端口、进程名称或 PID 猜测归属。握手禁止代理和重定向。复用入口退出时保留原服务。未匹配的服务不受影响。

## 构建 Windows 版本

需要 Node.js、Rust MSVC 工具链、Visual Studio C++ 桌面开发工具和 WebView2。构建过程不要求管理员权限。推荐先使用现有工具链；没有 Rust 时，可以按 [Tauri 官方前置条件](https://v2.tauri.app/start/prerequisites/)安装用户工具链。也可将 `CARGO_HOME`、`RUSTUP_HOME` 分别指向 `desktop/.toolchains/cargo` 和 `desktop/.toolchains/rustup`，并对安装器使用 `--no-modify-path`。这些环境变量只需设置在当前 PowerShell 会话。

在 `4C` 目录建立独立的打包环境：

```powershell
& .\backend\.venv-media\Scripts\python.exe -m venv .\desktop\.venv-build
& .\desktop\.venv-build\Scripts\python.exe -m pip install `
  -r .\desktop\requirements-build.txt `
  -r .\backend\requirements-review.txt `
  -r .\backend\requirements-courseware.txt
```

然后构建前端和桌面：

```powershell
Push-Location .\frontend
npm ci
npm run build
Pop-Location
Push-Location .\desktop
npm ci
& .\scripts\build.ps1
Pop-Location
```

脚本先用 PyInstaller 构建 `desktop/src-tauri/binaries/4c-review-backend-x86_64-pc-windows-msvc.exe`，再构建 Tauri。后端只收集公开配置白名单，不收集 `lessons.json`、课堂产物、视频或模型权重。Tauri 直接把当前 `frontend/dist` 映射为安装包的 `frontend` 资源，避免暂存副本残留旧页面。仅需构建便携主程序时，使用 `build.ps1 -NoBundle`；后端已构建时可再加 `-SkipBackend`。

Tauri 输出在 `desktop/src-tauri/target/release`；NSIS 安装包在其 `bundle/nsis` 子目录。默认按当前用户安装。安装包复用系统 WebView2；没有 WebView2 的机器须先安装它。构建失败时，应保留终端中的 Rust、链接器或网络错误；配置文件存在不代表已得到可运行的桌面包。

## 运行已打包程序

桌面壳启动自己的后端，等待就绪后才打开工作台。退出时先通过标准输入请求后端正常关闭，超时才清理自己创建的子进程组；不会查找或结束外部 Python、模型服务或同名进程。

在源码项目中，`desktop/start_desktop.ps1` 可直接打开已构建的桌面程序，并连接现有课程登记表和模型文件清单。它使用可写的 `desktop/workspace/desktop-state` 作为独立工作区；可通过 `-Registry`、`-DataDir`、`-RuntimeRoot` 更改。需要浏览器回退时使用 `start_source.ps1`。

打包程序首次使用独立空登记表。要查看现有课程，可以在启动前设置：

```powershell
$env:REVIEW_LESSON_REGISTRY = (Resolve-Path .\backend\configs\lessons.json).Path
& .\desktop\src-tauri\target\release\four-c-desktop.exe
```

如本机 AppData 目录受限，可额外设置 `REVIEW_DESKTOP_DATA_DIR` 为自己可写的专用工作区目录。桌面壳会把数据库、后端临时目录和 WebView 用户数据都放在该目录内。该变量只影响桌面工作区，不改变课程源文件位置。

登记表仍可指向本机原始证据。工作区数据库单独写入应用的本机数据目录。模型放在该工作区的 `runtime/artifacts/models` 下，公共配置首次复制到 `runtime/configs`。也可通过 `REVIEW_RUNTIME_ROOT` 指向已有运行时根目录；目录中应包含 `configs` 与 `artifacts`。程序不自动下载模型。

## 模型与运行时清单

`review.runtime.inspect_runtime()` 是给 API 使用的只读函数，根应用可将其映射到 `GET /api/review/runtime`。它只检查配置及非空文件，不导入模型库、不连接网络、不读取视频、不初始化 GPU。返回的路径只包含运行时根目录内的相对路径，拒绝绝对路径、目录穿越和指向根目录外的链接。

`configs/runtime_profiles.json` 覆盖 ASR、OCR、嵌入、问答和校对。现有 Qwen3、Qwen3.5 和校对模型配置按本机文件检查；配置存在、模型文件存在、执行程序存在分别报告。`installed_unverified` 表示文件检查通过，推理仍未验证。`service_status=not_checked`、`inference_status=not_verified` 和 `available=false` 不应在界面中改写为“模型可用”。这里没有启动或调用模型。

Qwen3-ASR-0.6B、PP-OCRv6 ONNX、BGE-small 为推荐安装项。OCR/嵌入的 ONNX 文件命名是预留约定，适配器尚未完成。缺少文件时明确返回 `missing`，不把预留方案列为已实现推理。现有 ASR 配置可能来自另一台机器的实验，本机没有权重时仍显示缺失。

模型与 CPU/CUDA 推理引擎单独安装、单独启动，保留上游模型卡、原始许可证、版本、校验和及下载来源。安装包不包含这些模型，也不自动接受或替换其许可证。API 主进程不承担 GPU 模型加载。

## 后台任务与边界

冻结后的后台任务入口为：

```text
4c-review-backend.exe --run-worker --db <jobs.sqlite3> --registry <lessons.json> --job <id> --token <token>
```

`--run-worker` 必须位于首项。入口移除此项后，原样交给 `review.job_worker.main()`，不经过桌面服务启动逻辑。数据库路径、令牌和课程参数由任务队列生成，不在网页中拼接命令行。

课件图像匹配还使用独立的本机原生库进程。冻结入口只兼容固定的 `-m review.courseware_alignment` JSON 协议，不开放任意 Python 模块执行。PPT/PPTX 转 PDF 的原有 PowerPoint 转换脚本随后端打包；实际转换仍要求本机安装 PowerPoint。文字型 PDF 不要求安装 PowerPoint。

Tauri capability 不授予网页 shell、文件系统或自定义原生命令权限，未暴露全局 Tauri 对象。原生层只运行固定的后端 sidecar。窗口导航限制在应用内置启动页和本次后端地址；后端为页面设置同源 CSP、禁止嵌入对象与外部脚本。完整课堂证据仍由 API 的路径检查与版本约束管理。

## 验证

```powershell
& .\backend\.venv-media\Scripts\python.exe -m pytest .\backend\tests\test_desktop_runtime.py -q
```

测试覆盖缺失模型、空文件、损坏配置、路径脱敏、推荐配置未安装、启动参数、冻结 worker 参数，以及真实 HTTP 服务的启动、复用与关闭。符号链接测试在 Windows 未授予权限时跳过。这里的测试不代表模型准确率或推理速度评估。

要对实际后端可执行文件重跑生命周期测试，在同一 PowerShell 会话设置 `REVIEW_DESKTOP_BINARY` 为构建出的后端绝对路径，再执行上述测试命令。桌面壳将 PyInstaller 临时展开目录放在自己工作区的 `backend-temp` 下，避免依赖受限或不可写的系统临时目录。

Tauri 接口核对使用[官方 sidecar 文档](https://v2.tauri.app/develop/sidecar/)、[权限说明](https://v2.tauri.app/security/capabilities/)与[CSP 说明](https://v2.tauri.app/security/csp/)。本次 Context7 解析工具不可调用，因此使用官方文档作为回退来源。

## 本次构建记录（2026-10-28）

已生成实际 Windows x64 桌面主程序、独立 Python 后端和 NSIS 安装包。主程序使用 Tauri 2.12.5，复用本机 WebView2；Rust 1.99.1 和构建缓存位于项目内，未修改系统 PATH。

实际冻结后端的桌面测试为 **13 通过、1 跳过**。测试完成 PDF 上传、持久化任务、冻结 worker、独立图像匹配与结果发布，并检查最后的任务恢复取消边界已进入打包代码。跳过项是 Windows 符号链接权限测试。

`start_desktop.ps1` 已连接当前真实登记表验证：6 节可见课堂、7 个运行时配置；原生窗口加载课堂，录像时间实际推进。窗口正常关闭后，登记表和证据包哈希保持一致，原视频大小与修改时间未变。视频没有重新计算整文件哈希。

最终文件、源代码输入、前端输入及 SHA-256 见 `desktop/build/release-manifest.json`；原生验收见 `desktop/build/native-real-registry-check.json`。安装包已生成，本次未运行安装向导；模型推理能力仍按清单中的 `not_verified` 展示。
