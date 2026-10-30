# 本地公开语言模型

本项目已经在本机实际运行官方 Qwen3-4B Q4_K_M，以 llama.cpp 的 OpenAI 兼容接口供后端检索问答调用。没有训练或微调，没有创建付费服务。模型接收后端提供的文字证据，不直接识别视频、音频或课堂行为。

## 固定资产

| 项目 | 固定值 |
|---|---|
| 权重仓库 | `Qwen/Qwen3-4B-GGUF`（Qwen 官方） |
| 权重 revision | `bc640142c66e1fdd12af0bd68f40445458f3869b` |
| 文件 | `Qwen3-4B-Q4_K_M.gguf`，2,497,280,256 bytes |
| 权重 SHA-256 | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` |
| 架构 / 参数 / 量化 | `qwen3` / 4B / Q4_K_M；服务报告 4,022,468,096 parameters |
| 权重许可证 | Apache-2.0，原文保存在 `artifacts/models/Qwen3-4B-GGUF/LICENSE` |
| 运行时 | `ggml-org/llama.cpp` 官方 Windows x64 release `b11871` |
| 运行时 revision | `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9` |
| 运行时许可证 | MIT，原文保存在对应 runtime 目录的 `LICENSE` |

CPU、CUDA 12.4 运行时压缩包及 CUDA DLL 压缩包的官方 SHA-256 和字节数均固定在 `configs/local_llm.json`。下载器验证全部字节后才解压；原压缩包保留于 `artifacts/runtime/downloads`。模型卡和许可证也保留为本地文件。资产校验结果在 `artifacts/verification/llm/asset_manifest_cpu.json` 和 `asset_manifest_cuda.json`。

官方来源：

- [固定版本模型页](https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/bc640142c66e1fdd12af0bd68f40445458f3869b)
- [官方模型卡与许可证](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [llama.cpp b11871 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11871)
- [固定版本 server 文档](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/tools/server/README.md)

llama.cpp 参数已通过 Context7 官方仓库文档和下载后的 `llama-server.exe --help` 双重核对。

## 启动和停止

在本项目目录运行，命令不要求 API key。默认配置使用 CUDA，CPU 运行时也已下载并验证过健康检查。

```powershell
python scripts/fetch_local_llm.py --runtime cuda
python scripts/fetch_local_llm.py --runtime cuda --verify-only
.\scripts\start_local_llm.ps1 -Backend cuda
```

脚本使用隐藏窗口启动，启动前重验权重 SHA-256。日志、PID、完整可执行文件路径和进程开始时间写在 `artifacts/verification/llm/server_state.json` 及同目录日志中。停止时核验 PID 对应程序和开始时间，避免误停 PID 已被复用的其他程序：

```powershell
.\scripts\start_local_llm.ps1 -Stop
```

改配置或切换运行后端应先停止。CPU 回退：

```powershell
python scripts/fetch_local_llm.py --runtime cpu
.\scripts\start_local_llm.ps1 -Stop
.\scripts\start_local_llm.ps1 -Backend cpu
```

CPU 健康检查已经成功，但本轮中文生成性能数据来自 CUDA；不能把 GPU 数据当成 CPU 性能。

## 接口合同与资源设置

- Base URL：`http://127.0.0.1:18081/v1`
- 生成：`POST /v1/chat/completions`
- 模型列表：`GET /v1/models`
- 健康：`GET /health`（在主机根路径）
- 模型 alias：`classroom-local`
- 24,576 tokens 上下文、单并行槽、生成与提示处理各 4 CPU 线程；CUDA 使用全层卸载。
- 已通过服务返回信息确认实际 `n_ctx=24576`；本机 RTX 4060 Laptop 8 GB 上观测全卡显存约 6,174 MiB。其他 GPU 进程和更大的上下文会影响可用显存。
- 后端应限制输入与输出之和不超过上下文；字符数不能直接当 token 数。服务是单槽，建议请求串行或设置队列与 180 秒超时。

默认请求：

```json
{
  "model": "classroom-local",
  "messages": [
    {"role": "system", "content": "只根据给出的课堂证据回答；证据不足时明确说明。引用时完整照抄方括号及其中的完整编号，不得缩写或改写。"},
    {"role": "user", "content": "在这里放检索到的证据和用户问题。"}
  ],
  "temperature": 0.1,
  "max_tokens": 1000,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Qwen3 支持思考和非思考模式。此部署使用 `--jinja --reasoning off` 全局关闭思考，请求再传 `chat_template_kwargs.enable_thinking=false`。当前 llama.cpp 文档仍支持该请求字段；启动参数 `--chat-template-kwargs` 中设置 `enable_thinking` 已标为弃用，因此本脚本使用 `--reasoning off`。不要只删除输出中的 `<think>` 标签冒充已关闭思考。

`temperature=0.1` 是本项目为了证据问答采用的参数，模型卡给非思考模式建议的 temperature 为 0.7；本部署没有声称该参数最优。可为概念提取覆盖 `max_tokens=1700`，前提是保留足够上下文余量。

## 已完成的实际检查（2026-10-25）

1. 三个离线测试：固定来源、同尺寸内容篡改导致哈希失败、ZIP 越界路径拒绝。这些不评价模型答案准确率。
2. CPU runtime 真实加载并返回健康状态；后续已停止 CPU 服务。
3. CUDA runtime 加载、`/health` 和 `/v1/models` 成功，报告正确 alias、24,576 上下文和 Q4_K 模型信息。
4. 两条中文问题由本地模型实时生成。最终测试分别耗时约 0.72 秒、0.24 秒，正确保留测试引用，缺失作业证据时明确说明不足，未返回思考标签或 `reasoning_content`。
5. 19,000 中文字符的**合成负载**以 `max_tokens=1700` 请求成功，实际输入 14,186 tokens、输出 4 tokens，耗时约 5.91 秒。这只验证该长度输入与参数能处理，没有验证 1,700 token 实际生成质量或所有 24K 边界输入。

可重放中文接口测试：

```powershell
python artifacts/verification/llm/smoke_local_llm.py
python -m unittest discover -s tests -p test_llm_assets.py -v
```

原始请求、返回和耗时：`live_smoke_cuda.json`；长输入：`long_context_cuda.json`。两问资料是明确标记的**合成接口测试证据**，不是该节课实际转写，不能据此宣称真实课堂问答或课堂引用已验收。

首次测试中模型把 `[probe_E1]` 简写为 `E1`，保留于 `live_smoke_cuda_initial.json`。增加“完整照抄引用编号”的提示后通过，但模型仍可能遗漏或改写引用。后端必须只接受已检索到的本课证据 ID，验证引用后再给前端，不能仅凭模型字符串认定来源真实。真实课堂语料上的准确性、引用忠实度和概念关系质量需要独立验收。

服务错误应由后端以真实错误返回。不得用预设文本、合成测试回答或云端余额失败时的模板冒充模型结果。
